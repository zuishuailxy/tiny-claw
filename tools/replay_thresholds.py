"""离线重放调参 — 用一次跑分录下的语料，网格搜索压缩阈值，API 成本为 0。

为什么存在这个脚本
------------------
阈值（COMPACT_TOKENS_*）若靠「改数字 → 真跑一遍 → 看效果」来调，
每轮迭代都要烧钱。本脚本把调参搬回本地：

    语料 = 跑分时录下的「每轮压缩前的原始上下文快照」
    重放 = 对每份快照，用候选阈值重跑一遍 Compactor.compact()
    产出 = 每个阈值下的触发率 / 压缩后峰值 / 误伤统计 + 推荐值

选阈值的判据（写在 report 里，不是拍脑袋）：
  1. 正常任务零压缩：阈值应高于语料中的上下文峰值 × 校准系数，
     否则 agent 每次正常干活都在丢历史。
  2. 不超过窗口安全水位：1M 窗口下取 ≤ 25%（约 256k token），
     给模型输出、突发大输出留余量。
  3. 校准系数 real/est > 1 时，本地估算偏低，必须按系数放大阈值，
     否则「8000 token」实际是「12000 token」级别的误判。

Usage:
    python tools/replay_thresholds.py
    python tools/replay_thresholds.py --grid 8000,20000,50000,100000,200000
"""

import argparse
import copy
import json
import sys
from pathlib import Path

# 保证脚本从项目根任意位置都能跑
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_claw.context.compactor import Compactor  # noqa: E402
from tiny_claw.context.tokens import estimate_messages_tokens  # noqa: E402
from tiny_claw.eval.instrument import RunTrace  # noqa: E402

# 1M 窗口下允许上下文占用的安全水位上限
CONTEXT_WINDOW = 1_000_000
SAFE_WATERMARK = 0.25

# 默认档位覆盖：现值 8000 / 一档放大 / 中间档 / 保守上限
DEFAULT_GRID = [8000, 16000, 32000, 64000, 128000, 256000]


def load_traces(trace_dir: Path) -> list[RunTrace]:
    paths = sorted(trace_dir.glob("test_*.json"))
    if not paths:
        print(f"未找到语料：{trace_dir}/test_*.json")
        print("请先跑一次 python run_benchmark.py 录制语料。")
        raise SystemExit(1)
    return [RunTrace.load(p) for p in paths]


def fit_calibration(traces: list[RunTrace]) -> tuple[float, float, float]:
    """最小二乘拟合：real_prompt ≈ b + a×est + c×n_messages

    为什么要三参数而不是单系数 real/est：
    - b（截距）：每次请求都要随 messages 一起发送但本地不参与压缩计量的
      固定开销 —— 主要是工具定义 JSON Schema + system 模板之外的协议开销。
      实测它就有几百 token，单系数模型会把它错摊到「低估比例」里。
    - a（斜率）：tokenizer 本身的低估系数（代码/JSON 场景实际
      0.3~0.4 token/字符，我们按 0.25 计）。
    - c：每条消息的协议结构开销（role/name 包裹等）。

    用正规方程解 3×3 线性方程组，不引入 numpy —— 校准工具不该有自己的依赖。
    """
    xs: list[list[float]] = []
    ys: list[float] = []
    for t in traces:
        for call in t.api_calls:
            if call.real_prompt <= 0:
                continue
            xs.append([1.0, float(call.est_tokens), float(call.n_messages)])
            ys.append(float(call.real_prompt))

    if len(xs) < 8:
        # 样本太少时最小二乘会过拟合，退化为单系数
        total_est = sum(x[1] for x in xs) or 1.0
        total_real = sum(ys)
        return 0.0, max(total_real / total_est, 1.0), 0.0

    # 正规方程 (XᵀX) w = Xᵀy
    xtx = [[sum(row[i] * row[j] for row in xs) for j in range(3)] for i in range(3)]
    xty = [sum(row[i] * y for row, y in zip(xs, ys)) for i in range(3)]

    # 高斯消元（列主元）
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(xtx[r][col]))
        xtx[col], xtx[pivot] = xtx[pivot], xtx[col]
        xty[col], xty[pivot] = xty[pivot], xty[col]
        for r in range(col + 1, 3):
            f = xtx[r][col] / xtx[col][col]
            for k in range(col, 3):
                xtx[r][k] -= f * xtx[col][k]
            xty[r] -= f * xty[col]
    w = [0.0] * 3
    for r in (2, 1, 0):
        w[r] = (xty[r] - sum(xtx[r][k] * w[k] for k in range(r + 1, 3))) / xtx[r][r]

    # 拟合出的截距和消息开销理论上应为正；数值噪声可能给出负值，钳到 0
    return max(w[0], 0.0), max(w[1], 0.5), max(w[2], 0.0)


def predict_real(b: float, a: float, c: float, est: int, n_msgs: int) -> int:
    """给定压缩后的 est 与消息数，预测真实 prompt token。"""
    return int(b + a * est + c * n_msgs)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线重放压缩阈值网格搜索")
    parser.add_argument(
        "--traces-dir",
        default="reports/traces",
        help="语料目录（run_benchmark.py 的产物，默认 reports/traces）",
    )
    parser.add_argument("--grid", default=",".join(map(str, DEFAULT_GRID)))
    args = parser.parse_args()

    grid = [int(x) for x in args.grid.split(",")]
    traces = load_traces(Path(args.traces_dir))

    # --------------------------------------------------------------
    # 1. 校准：最小二乘拟合 real ≈ b + a×est + c×n_messages
    # --------------------------------------------------------------
    b, a, c = fit_calibration(traces)

    print("=" * 78)
    print(f"语料: {len(traces)} 个用例 | 快照 "
          f"{sum(len(t.snapshots) for t in traces)} 份 | "
          f"API 调用 {sum(len(t.api_calls) for t in traces)} 次")
    print(f"校准模型: real ≈ {b:.0f} + {a:.2f}×est + {c:.1f}×消息数")
    print(f"  截距 {b:.0f} tk ≈ 工具定义等固定开销（每次请求都带，但不参与压缩计量）")
    print(f"  斜率 {a:.2f} × 本地估算（tokenizer 系统性低估的修正）")
    print("=" * 78)

    # --------------------------------------------------------------
    # 2. 逐快照重放：对每个候选阈值统计触发率与压缩后体积
    # --------------------------------------------------------------
    # 所有快照的原始上下文摊平（用例间隔离，互不影响）
    snapshots = [
        (trace.case_id, snap)
        for trace in traces
        for snap in trace.snapshots
    ]

    print(f"\n{'阈值':>8} | {'触发率':>7} | {'压缩后峰值(est)':>16} "
          f"| {'压缩后峰值(校准预测)':>20} | {'误伤条数':>8}")
    print("-" * 78)

    results: list[dict] = []
    for threshold in grid:
        compact = Compactor(max_tokens=threshold, retain_last=6)
        triggered = 0
        peaks_est: list[int] = []
        peaks_real: list[int] = []
        masked_total = 0

        for _case_id, snap in snapshots:
            messages = [_msg_from_dict(m) for m in snap.messages]
            before = snap.est_tokens
            if before < threshold:
                peaks_est.append(before)
                peaks_real.append(predict_real(b, a, c, before, len(messages)))
                continue

            # compact 不修改输入（内部逐条拷贝），但重放必须保护原始语料：
            # 同一份快照要在多个阈值下反复使用
            after_msgs = compact.compact(copy.deepcopy(messages))
            after = estimate_messages_tokens(after_msgs)
            triggered += 1
            peaks_est.append(after)
            peaks_real.append(predict_real(b, a, c, after, len(after_msgs)))
            # 误伤统计：原来没有省略标记、压缩后才出现的消息
            masked_total += sum(
                1
                for orig, new in zip(messages, after_msgs)
                if "...[" in new.content and "...[" not in orig.content
            )

        results.append(
            {
                "threshold": threshold,
                "trigger_rate": triggered / len(snapshots) if snapshots else 0,
                "peak_after_est": max(peaks_est, default=0),
                "peak_after_real_predicted": max(peaks_real, default=0),
                "masked_total": masked_total,
            }
        )
        r = results[-1]
        print(
            f"{threshold:>8} | {r['trigger_rate']:>6.1%} "
            f"| {r['peak_after_est']:>16,} "
            f"| {r['peak_after_real_predicted']:>20,} "
            f"| {masked_total:>8}"
        )

    # --------------------------------------------------------------
    # 3. 推荐值：语料真实需求（无压缩时校准预测峰值），取首个 ≥ 需求的档位
    # --------------------------------------------------------------
    peak_est = max(snap.est_tokens for _c, snap in snapshots) if snapshots else 0
    peak_n = max(len(snap.messages) for _c, snap in snapshots) if snapshots else 0
    need = predict_real(b, a, c, peak_est, peak_n)
    ceiling = int(CONTEXT_WINDOW * SAFE_WATERMARK)

    # 推荐档 = 网格中第一个 ≥ 需求的档位；若全部低于需求则报出并建议扩网格
    recommended = next((g for g in grid if g >= need), None)

    print("\n" + "=" * 78)
    print(f"无压缩上下文峰值(est): {peak_est:,} tk → 校准预测真实需求 "
          f"{need:,} tk（含 {b:.0f} tk 固定开销）")
    print(f"安全水位上限: {ceiling:,} tk（1M 窗口 × {SAFE_WATERMARK:.0%}）")
    if recommended:
        print(f"👉 推荐阈值: {recommended:,} token "
              f"（正常任务零压缩，且距水位尚有 "
              f"{ceiling / max(recommended, 1):.1f} 倍余量）")
    else:
        print(f"⚠️ 语料需求 {need:,} 超过全部候选档位，"
              f"请用更大的 --grid 重跑（如 512000）")
    print("=" * 78)

    # 持久化：报告可 diff，阈值变更要有据可查
    out = Path(args.traces_dir) / "threshold_report.json"
    out.write_text(
        json.dumps(
            {
                "calibration": {"intercept": round(b, 1), "slope": round(a, 3),
                                "per_message": round(c, 2)},
                "context_need_est": peak_est,
                "context_need_calibrated": need,
                "safe_ceiling": ceiling,
                "recommended_threshold": recommended,
                "grid_results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"报告已保存: {out}")


# ---- 快照反序列化（与 instrument 保持同构；单独实现避免依赖私有函数） ----


def _msg_from_dict(d: dict) -> "Message":  # noqa: F821
    from tiny_claw.schema import Message, ToolCall, Usage

    usage = d.get("usage")
    return Message(
        role=d["role"],
        content=d.get("content", ""),
        tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
        tool_call_id=d.get("tool_call_id", ""),
        usage=Usage(**usage) if usage else None,
        reasoning=d.get("reasoning", ""),
    )


if __name__ == "__main__":
    main()
