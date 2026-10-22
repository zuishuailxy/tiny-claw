"""基准测试/评估系统

每个测试用例完全物理隔离：独立沙箱目录、独立引擎、独立 Session。
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tiny_claw import config
from tiny_claw.engine import AgentEngine
from tiny_claw.engine.session import Session
from tiny_claw.eval.instrument import (
    InstrumentedCompactor,
    InstrumentedRegistry,
    InstrumentedTracker,
    RunTrace,
)
from tiny_claw.provider import DeepSeekProvider
from tiny_claw.schema import Message, Role
from tiny_claw.tools import RegistryImpl
from tiny_claw.tools.builtin import BashTool, EditFileTool, ReadFileTool, WriteFileTool

logger = logging.getLogger("tiny-claw.eval")


def _percentile(sorted_values: list[int], p: float) -> int:
    """线性插值分位数。输入必须已升序排列。"""
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return int(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo))


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════


@dataclass
class TestCase:
    """一个需要 Agent 去完成并验证的独立任务"""

    id: str  # 用例唯一标识
    name: str  # 用例名称
    task_prompt: str  # 发送给 Agent 的任务指令
    setup_script: str = ""  # 【可选】Agent 运行前执行的 bash 脚本
    validate_script: str = ""  # 【核心】Agent 结束后执行的校验脚本，exit 0=成功
    max_turns: int = 30  # 允许 Agent 尝试的最大轮数


@dataclass
class TestResult:
    """单次跑分结果"""

    case_id: str
    case_name: str
    passed: bool
    duration_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    error_msg: str = ""
    # ---- 顺滑度指标 ----
    error_turns: int = 0  # 包含至少一次工具调用失败的 Turn 数 (0 = 一发入魂)
    total_turns: int = 0  # 总 Turn 数（辅助计算比例）
    first_error_token: int = 0  # 首次错误时的累计 Token 数
    # ---- 阈值基线指标（定阈值用，口径见 eval/instrument.py）----
    est_tokens_total: int = 0  # 本地估算 token 累计（与真实值对比得校准系数）
    peak_prompt_tokens: int = 0  # 单次请求的真实 prompt token 峰值
    calibration_ratio: float = 0.0  # 真实 / 本地估算，>1 表示本地低估（危险）
    tool_output_max_tokens: int = 0  # 单次工具输出的最大体积
    tool_output_p90_tokens: int = 0  # 工具输出体积的 90 分位
    n_offloaded: int = 0  # 走 Offloading 落盘的次数
    n_snapshots: int = 0  # 录到的上下文快照数（离线重放语料量）
    trace_path: str = ""  # 语料落盘位置

    @property
    def waste_ratio(self) -> float:
        """试错浪费比：首次错误出现后消耗的 Token 占总 Token 的比例。

        0.0  = 零浪费（完美执行或首轮直接失败）
        0.5  = 一半的 Token 花在错误之后
        """
        total = self.prompt_tokens + self.completion_tokens
        if total == 0 or self.first_error_token == 0:
            return 0.0
        waste_tokens = total - self.first_error_token
        return max(0.0, waste_tokens / total)


@dataclass
class BenchmarkReport:
    """汇总报告"""

    total: int = 0
    passed: int = 0
    failed: int = 0
    total_cost_cny: float = 0.0
    total_duration_ms: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_est_tokens: int = 0
    # 预算相关的元信息：报告必须自证「这次跑分花了多少、是否被截断」
    budget_cny: float | None = None
    budget_tokens: int | None = None
    stopped_by_budget: bool = False
    model: str = ""
    results: list[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def calibration_ratio(self) -> float:
        """整体校准系数 = 真实 prompt token / 本地估算 token。

        >1 表示本地低估 —— 压缩会比预期晚触发，有撑爆窗口的风险。
        这是判断 estimate_tokens 能否继续用作阈值地基的关键指标。
        """
        if self.total_est_tokens == 0:
            return 0.0
        return self.total_prompt_tokens / self.total_est_tokens

    @property
    def peak_prompt_tokens(self) -> int:
        return max((r.peak_prompt_tokens for r in self.results), default=0)


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════


class BenchmarkRunner:
    """自动化回归测试执行器。

    每个用例拥有独立的沙箱目录、引擎实例、Session，确保完全隔离。
    """

    def __init__(
        self,
        model: str | None = None,
        base_work_dir: str | None = None,
        *,
        budget_cny: float | None = None,
        budget_tokens: int | None = None,
        trace_dir: str | None = None,
        report_dir: str | None = None,
        enable_thinking: bool = False,
        enable_recovery: bool = True,
    ):
        # 默认取统一配置，避免调用方各自硬编码一个可能不一致的模型名
        self._model = model or config.MODEL
        root = Path(base_work_dir or Path.cwd())

        # 两类产物必须分开落盘，早先混在一起导致证据被 gitignore 一并吞掉：
        #   - 用例沙箱：agent 在里面读写文件，属**临时物** → .claw/ 下（已忽略）
        #   - 报告与语料：属**可提交的证据**，是项目材料数字的出处 → reports/
        self._base_dir = root / ".claw" / "sandbox"

        # 双轨预算止损：金额 + token。
        # 为什么不能只看金额：模型不在 PRICING 表时成本会算成 0，
        # 金额止损直接失效，而余额已经真实扣了。token 不依赖价格表，是兜底。
        self._budget_cny = budget_cny
        self._budget_tokens = budget_tokens
        # 慢思考是否开启：默认 False（跑分追求成本可控）。
        # 注意 CLI 默认是**开**的 —— 所以"跑分口径"与"生产口径"并不相同，
        # 要对比两者就跑两次（`--thinking`），别把一份数据当成两边的证据。
        self._enable_thinking = enable_thinking
        # 错误自愈开关：默认开。关掉即为 A/B 的**基线组**（拿裸错误、无救援提示、
        # 无死循环打断），用来证明自愈机制到底有没有用。
        self._enable_recovery = enable_recovery
        self._report_dir = Path(report_dir) if report_dir else root / "reports"
        self._trace_dir = Path(trace_dir) if trace_dir else self._report_dir / "traces"

    # ------------------------------------------------------------------
    # 预算
    # ------------------------------------------------------------------
    def _is_over_budget(self, report: BenchmarkReport) -> bool:
        """金额与 token 任一超限即止损。"""
        if self._budget_cny is not None and report.total_cost_cny >= self._budget_cny:
            return True
        if self._budget_tokens is not None:
            spent = report.total_prompt_tokens + report.total_completion_tokens
            if spent >= self._budget_tokens:
                return True
        return False

    # ------------------------------------------------------------------
    # 套件入口
    # ------------------------------------------------------------------
    async def run_suite(self, cases: list[TestCase]) -> BenchmarkReport:
        """批量执行测试套件"""
        logger.info("=" * 50)
        logger.info("🚀 启动自动化 Benchmark 评估 | 模型: %s", self._model)
        logger.info("=" * 50)

        report = BenchmarkReport(
            total=len(cases),
            budget_cny=self._budget_cny,
            budget_tokens=self._budget_tokens,
            model=self._model,
        )

        for tc in cases:
            # 跑之前先查一次：上一轮已经超预算了就别再烧了
            if self._is_over_budget(report):
                report.stopped_by_budget = True
                logger.error(
                    "💰 预算止损：已花 ¥%.4f / %d token，达到上限 "
                    "(¥%s / %s token)，剩余 %d 个用例未执行。",
                    report.total_cost_cny,
                    report.total_prompt_tokens + report.total_completion_tokens,
                    self._budget_cny,
                    self._budget_tokens,
                    len(cases) - len(report.results),
                )
                break

            logger.info("\n>>> ⏳ 正在执行用例 [%s]: %s", tc.id, tc.name)
            r = await self._run_single(tc)
            report.results.append(r)
            report.total_duration_ms += r.duration_ms
            report.total_cost_cny += r.cost_cny
            report.total_prompt_tokens += r.prompt_tokens
            report.total_completion_tokens += r.completion_tokens
            report.total_est_tokens += r.est_tokens_total

            if r.passed:
                report.passed += 1
                logger.info(
                    ">>> ✅ [%s] 通过 | 耗时: %dms | 花费: ¥%.6f",
                    tc.id,
                    r.duration_ms,
                    r.cost_cny,
                )
            else:
                report.failed += 1
                logger.info(
                    ">>> ❌ [%s] 失败 | 错误: %s",
                    tc.id,
                    r.error_msg[:80],
                )

        # 终极报表
        logger.info("\n================ 🏆 跑分终极报告 ================")
        logger.info(
            "总用例: %d | 通过: %d | 失败: %d | 成功率: %.1f%%",
            report.total,
            report.passed,
            report.failed,
            report.pass_rate * 100,
        )
        logger.info(
            "Token: 输入 %d + 输出 %d = %d | 成本: ¥%.6f",
            report.total_prompt_tokens,
            report.total_completion_tokens,
            report.total_prompt_tokens + report.total_completion_tokens,
            report.total_cost_cny,
        )
        logger.info(
            "校准系数(真实/估算): %.3f | 单次 prompt 峰值: %d tk"
            " | 上下文快照: %d 份",
            report.calibration_ratio,
            report.peak_prompt_tokens,
            sum(r.n_snapshots for r in report.results),
        )
        if report.stopped_by_budget:
            logger.warning("⚠️ 本次跑分被预算止损截断，结果不代表完整套件。")
        logger.info("==================================================")

        return report

    # ------------------------------------------------------------------
    # 单用例执行（核心）
    # ------------------------------------------------------------------
    async def _run_single(self, tc: TestCase) -> TestResult:
        start = time.monotonic()

        # 语料录制容器：本次跑分的所有计量都灌进这里，结束时统一落盘。
        # 之后所有阈值实验都拿它离线重放，不再调 API —— 省钱的要害在这里。
        trace = RunTrace(
            case_id=tc.id,
            model=self._model,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        def finish(**kw) -> TestResult:
            """收尾：先落盘语料，再把从语料派生的阈值基线指标补进结果。"""
            kw.update(self._derive_metrics(trace))
            return TestResult(case_id=tc.id, case_name=tc.name, **kw)

        # ---- 1. 创建物理隔离沙箱目录 ----
        sandbox = self._base_dir / f"{tc.id}_{int(time.time())}"
        sandbox.mkdir(parents=True, exist_ok=True)
        work_dir = str(sandbox)

        # ---- 2. Setup 脚本 ----
        if tc.setup_script:
            ok, output = await _run_bash(tc.setup_script, work_dir)
            if not ok:
                return finish(passed=False, error_msg=f"Setup 失败: {output[:200]}")

        # ---- 3. 为每个用例创建独立引擎（完全物理隔离）----
        raw_provider = DeepSeekProvider()
        cost_tracker = InstrumentedTracker(
            raw_provider, model=self._model, trace=trace
        )

        inner_registry = RegistryImpl()
        inner_registry.register(ReadFileTool(work_dir))
        inner_registry.register(WriteFileTool(work_dir))
        inner_registry.register(BashTool(work_dir))
        inner_registry.register(EditFileTool(work_dir))
        registry = InstrumentedRegistry(inner_registry, trace)

        # 显式构造压缩器：档位与 AgentEngine 在 enable_thinking=False 时的默认
        # 行为保持一致（NORMAL），唯一区别是把 compact() 的输入上下文录成语料。
        # 压缩阈值要跟着思考模式走，否则慢思考下用的还是快思考的阈值 ——
        # 推理文本会把上下文顶得更高，阈值不跟着放大就过早触发压缩。
        if self._enable_thinking:
            _max_tokens, _retain = config.COMPACT_TOKENS_THINKING, config.RETAIN_LAST_THINKING
        else:
            _max_tokens, _retain = config.COMPACT_TOKENS_NORMAL, config.RETAIN_LAST_NORMAL
        compactor = InstrumentedCompactor(
            trace,
            max_tokens=_max_tokens,
            retain_last=_retain,
        )

        engine = AgentEngine(
            cost_tracker,
            registry,
            enable_thinking=self._enable_thinking,
            plan_mode=False,
            compactor=compactor,
            max_turns=tc.max_turns,
            enable_recovery=self._enable_recovery,
        )

        session = Session(tc.id, work_dir)
        cost_tracker.bind_session(session)

        def usage_kw() -> dict:
            """Session 上的计费与顺滑度指标（三处返回共用）"""
            return {
                "cost_cny": session.total_cost_cny,
                "prompt_tokens": session.total_prompt_tokens,
                "completion_tokens": session.total_completion_tokens,
                "error_turns": session.error_turns,
                "total_turns": session.total_turns,
                "first_error_token": session.first_error_token or 0,
            }

        # ---- 4. 驱动 Agent ----
        await session.append(Message(role=Role.USER, content=tc.task_prompt))

        try:
            await engine.run(session)
        except Exception as e:
            return finish(passed=False, error_msg=f"Agent 崩溃: {e}", **usage_kw())

        # ---- 5. 验收成果 ----
        if tc.validate_script:
            ok, output = await _run_bash(tc.validate_script, work_dir)
            if not ok:
                return finish(
                    passed=False,
                    error_msg=f"校验失败: {output[:200]}",
                    **usage_kw(),
                )

        # ---- 6. 返回结果 ----
        duration = int((time.monotonic() - start) * 1000)
        return finish(passed=True, duration_ms=duration, **usage_kw())

    # ------------------------------------------------------------------
    # 语料落盘与指标派生
    # ------------------------------------------------------------------
    def _derive_metrics(self, trace: RunTrace) -> dict:
        """落盘语料，并从中派生「定阈值」需要的基线指标。

        指标口径说明：
        - calibration_ratio：真实 prompt token / 本地估算。>1 = 本地低估，
          压缩会晚触发，是危险方向；<1 = 保守，会提前压缩。
        - tool_output_p90：工具输出体积的 90 分位。这是定 READ_MAX_CHARS /
          BASH_MAX_OUTPUT 的直接依据 —— 阈值若低于 p90，等于每 10 次读文件
          就有 1 次要截断或卸载。
        """
        path = trace.save(self._trace_dir / f"{trace.case_id}_{int(time.time())}.json")
        toks = sorted(trace.tool_output_tokens)
        return {
            "est_tokens_total": sum(c.est_tokens for c in trace.api_calls),
            "peak_prompt_tokens": trace.peak_real_prompt,
            "calibration_ratio": round(trace.calibration_ratio, 4),
            "tool_output_max_tokens": max(toks, default=0),
            "tool_output_p90_tokens": _percentile(toks, 0.9),
            "n_offloaded": sum(1 for t in trace.tool_outputs if t.offloaded),
            "n_snapshots": len(trace.snapshots),
            "trace_path": path,
        }

    # ------------------------------------------------------------------
    # 报告持久化
    # ------------------------------------------------------------------
    def save_report(self, report: BenchmarkReport, path: str | None = None) -> str:
        """保存报告为 JSON（默认落 reports/，即「可提交的证据」目录）"""
        if path is None:
            path = str(self._report_dir / "benchmark_report.json")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": round(report.pass_rate, 4),
            "total_cost_cny": round(report.total_cost_cny, 6),
            "total_duration_ms": report.total_duration_ms,
            "model": report.model,
            "total_prompt_tokens": report.total_prompt_tokens,
            "total_completion_tokens": report.total_completion_tokens,
            "total_est_tokens": report.total_est_tokens,
            "calibration_ratio": round(report.calibration_ratio, 4),
            "peak_prompt_tokens": report.peak_prompt_tokens,
            # 预算自证：读报告的人必须能看出这次跑分有没有被钱截断
            "budget_cny": report.budget_cny,
            "budget_tokens": report.budget_tokens,
            "stopped_by_budget": report.stopped_by_budget,
            "results": [
                {
                    "case_id": r.case_id,
                    "case_name": r.case_name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "cost_cny": round(r.cost_cny, 6),
                    "error_msg": r.error_msg,
                    "error_turns": r.error_turns,
                    "total_turns": r.total_turns,
                    "waste_ratio": round(r.waste_ratio, 4),
                    "est_tokens_total": r.est_tokens_total,
                    "peak_prompt_tokens": r.peak_prompt_tokens,
                    "calibration_ratio": r.calibration_ratio,
                    "tool_output_max_tokens": r.tool_output_max_tokens,
                    "tool_output_p90_tokens": r.tool_output_p90_tokens,
                    "n_offloaded": r.n_offloaded,
                    "n_snapshots": r.n_snapshots,
                    "trace_path": r.trace_path,
                }
                for r in report.results
            ],
        }
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("[Benchmark] 报告已保存: %s", path)
        return path


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════


async def _run_bash(script: str, cwd: str) -> tuple[bool, str]:
    """执行 bash 脚本，返回 (成功?, 输出)"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = stdout.decode("utf-8", errors="replace")
        return proc.returncode == 0, output
    except asyncio.TimeoutError:
        return False, "脚本执行超时(60s)"
    except Exception as e:
        return False, str(e)
