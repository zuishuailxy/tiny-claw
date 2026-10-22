"""Benchmark 插桩层 — 只测量，不改变任何业务行为。

为什么需要这一层
----------------
调阈值（压缩阈值、read_file 上限、bash 上限）最烧钱的做法是：
「改一个数字 → 真跑一遍 → 看效果 → 再改」。每跑一遍都要真金白银。

本层把它换成**一次录制、无限重放**：
一次真实跑分把三样东西录成离线语料，之后所有阈值组合都在本地重放，
API 成本恒为 0：

    ① 每次 API 调用：真实 prompt token vs 本地 estimate_tokens
        → 校准系数。本地估算是整个压缩阈值体系的地基，
          如果它系统性低估，所有阈值都是虚的。这个必须实测。
    ② 每次工具输出：字符数 / token 数 / 是否被卸载
        → 定 READ_MAX_CHARS 与 BASH_MAX_OUTPUT（分布的分位数）
    ③ 每次 compact() 的输入上下文快照
        → 语料。重放时用不同 max_tokens / retain_last 重跑压缩逻辑，
          看触发次数、降幅、误伤了什么。

设计约束
--------
- 插桩不得改变执行结果：Tracker/Compactor/Registry 都只在调用前后「旁听」，
  不修改入参与返回值。
- 计量口径与生产代码完全共用 `context.tokens.estimate_messages_tokens`，
  否则校准出来的系数对不上压缩逻辑，等于白测。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tiny_claw import config
from tiny_claw.context.compactor import Compactor
from tiny_claw.context.tokens import estimate_messages_tokens, estimate_tokens
from tiny_claw.schema import Message, ToolCall, ToolDefinition, ToolResult, Usage
from tiny_claw.tools import Registry
from tiny_claw.tools.base import BaseTool
from tiny_claw.tracing import CostTracker

logger = logging.getLogger("tiny-claw.eval.instrument")


# ═══════════════════════════════════════════════════════════════
# 数据记录
# ═══════════════════════════════════════════════════════════════


@dataclass
class APICallRecord:
    """一次 LLM 调用的计量快照。

    `est_tokens` 是本地估算（压缩器判定阈值用的同一个函数），
    `real_prompt` 是厂商返回的真实计费 token。两者之比就是校准系数：

        ratio = real_prompt / est_tokens

    - ratio > 1：本地**低估**，压缩触发得比预期晚，有撑爆窗口风险 → 危险
    - ratio < 1：本地**高估**，压缩提前触发，偏保守 → 安全但浪费上下文
    """

    seq: int  # 本次运行内的调用序号（0 起）
    est_tokens: int  # 本地估算（含 system 与 tool_calls 参数）
    real_prompt: int  # 厂商返回的 prompt token
    real_completion: int
    n_messages: int  # 上下文消息条数
    latency_ms: int

    @property
    def ratio(self) -> float:
        """真实 / 估算。>1 表示本地低估。"""
        if self.est_tokens == 0:
            return 0.0
        return self.real_prompt / self.est_tokens


@dataclass
class ToolOutputRecord:
    """一次工具输出的体积快照"""

    name: str
    chars: int
    tokens: int
    is_error: bool
    offloaded: bool  # 是否走了 Offloading（落盘后只回了摘要+路径）


@dataclass
class CompactSnapshot:
    """一次 compact() 调用的输入上下文快照 —— 离线重放的语料单元。

    注意：即使没触发压缩（未超阈值），compact 也会被调用，
    因此这里记录的是**每轮未经压缩的原始上下文**，
    重放时才能完整模拟「第 N 轮用阈值 T 会怎么样」。
    """

    seq: int
    messages: list[dict[str, Any]]  # 序列化的 Message 列表
    est_tokens: int  # 输入时的估算 token（重放基准）


@dataclass
class RunTrace:
    """一次用例运行的完整语料"""

    case_id: str
    model: str
    started_at: str = ""
    api_calls: list[APICallRecord] = field(default_factory=list)
    tool_outputs: list[ToolOutputRecord] = field(default_factory=list)
    snapshots: list[CompactSnapshot] = field(default_factory=list)
    passed: bool = False
    total_turns: int = 0
    error_msg: str = ""

    # ------------------------------------------------------------------
    # 派生指标
    # ------------------------------------------------------------------
    @property
    def total_real_prompt(self) -> int:
        return sum(c.real_prompt for c in self.api_calls)

    @property
    def peak_real_prompt(self) -> int:
        return max((c.real_prompt for c in self.api_calls), default=0)

    @property
    def calibration_ratio(self) -> float:
        """整体校准系数（用总量算，避免单次小样本噪声）"""
        est = sum(c.est_tokens for c in self.api_calls)
        if est == 0:
            return 0.0
        return self.total_real_prompt / est

    @property
    def tool_output_tokens(self) -> list[int]:
        return [t.tokens for t in self.tool_outputs]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(asdict(self), indent=2, ensure_ascii=False)

        # 语料是「可提交的证据」，必须脱掉机器路径：
        # agent 的 bash 输出里会带 pwd / 绝对路径（如 ls、find 的结果），
        # 原样入库会暴露本机目录结构。两层替换——先仓库根，再兜底家目录。
        raw = raw.replace(str(Path.cwd()), "<repo>")
        raw = re.sub(r'/Users/[^/"]+/', "~/", raw)

        p.write_text(raw, encoding="utf-8")
        logger.info("[Trace] 语料已保存: %s", p)
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> "RunTrace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            case_id=data["case_id"],
            model=data["model"],
            started_at=data.get("started_at", ""),
            api_calls=[APICallRecord(**c) for c in data.get("api_calls", [])],
            tool_outputs=[
                ToolOutputRecord(**t) for t in data.get("tool_outputs", [])
            ],
            snapshots=[CompactSnapshot(**s) for s in data.get("snapshots", [])],
            passed=data.get("passed", False),
            total_turns=data.get("total_turns", 0),
            error_msg=data.get("error_msg", ""),
        )

    def replay_messages(self, seq: int) -> list[Message]:
        """取回某次 compact 的原始上下文（反序列化为 Message）"""
        for snap in self.snapshots:
            if snap.seq == seq:
                return [_dict_to_message(m) for m in snap.messages]
        raise KeyError(f"快照 seq={seq} 不存在")


# ═══════════════════════════════════════════════════════════════
# 插桩组件
# ═══════════════════════════════════════════════════════════════


class InstrumentedTracker(CostTracker):
    """旁听每次 LLM 调用，记录「本地估算 vs 真实 token」。

    继承而非替换 CostTracker：计费逻辑一行不改，只在前后加计量。
    """

    def __init__(self, provider, model: str, trace: RunTrace):
        super().__init__(provider, model)
        self._trace = trace

    async def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> Message:
        start = time.monotonic()
        est = estimate_messages_tokens(messages)
        resp = await super().generate(messages, available_tools)
        latency_ms = int((time.monotonic() - start) * 1000)

        real_prompt = resp.usage.prompt_tokens if resp.usage else 0
        real_completion = resp.usage.completion_tokens if resp.usage else 0

        self._trace.api_calls.append(
            APICallRecord(
                seq=len(self._trace.api_calls),
                est_tokens=est,
                real_prompt=real_prompt,
                real_completion=real_completion,
                n_messages=len(messages),
                latency_ms=latency_ms,
            )
        )
        logger.info(
            "[Trace] #%d 估算 %d tk / 真实 %d tk → 校准系数 %.2f",
            len(self._trace.api_calls) - 1,
            est,
            real_prompt,
            (real_prompt / est) if est else 0.0,
        )
        return resp


class InstrumentedCompactor(Compactor):
    """记录每轮 compact() 的输入上下文，作为离线重放语料。

    只「旁听」：先拷一份输入再交给父类，压缩结果原样返回。
    """

    def __init__(self, trace: RunTrace, max_tokens: int, retain_last: int):
        super().__init__(max_tokens=max_tokens, retain_last=retain_last)
        self._trace = trace

    def compact(self, messages: list[Message]) -> list[Message]:
        seq = len(self._trace.snapshots)
        est = self._estimate_tokens(messages)
        self._trace.snapshots.append(
            CompactSnapshot(
                seq=seq,
                messages=[_message_to_dict(m) for m in messages],
                est_tokens=est,
            )
        )
        return super().compact(messages)


class InstrumentedRegistry(Registry):
    """代理 Registry，记录每次工具输出的体积。

    为什么用代理而不是 middleware：Registry 的中间件只能拿到 ToolCall、
    拿不到 ToolResult，量不到输出大小。而体积分布恰恰是定
    READ_MAX_CHARS / BASH_MAX_OUTPUT 的依据。
    """

    def __init__(self, inner: Registry, trace: RunTrace):
        self._inner = inner
        self._trace = trace

    def register(self, tool: BaseTool) -> None:
        self._inner.register(tool)

    def use(self, mw) -> None:
        self._inner.use(mw)

    def get_available_tools(self) -> list[ToolDefinition]:
        return self._inner.get_available_tools()

    async def execute(self, call: ToolCall) -> ToolResult:
        result = await self._inner.execute(call)
        self._trace.tool_outputs.append(
            ToolOutputRecord(
                name=call.name,
                chars=len(result.output),
                tokens=estimate_tokens(result.output),
                is_error=result.is_error,
                offloaded=config.OFFLOAD_DIR in result.output,
            )
        )
        return result


# ═══════════════════════════════════════════════════════════════
# 序列化 — Message ↔ dict
# ═══════════════════════════════════════════════════════════════


def _message_to_dict(msg: Message) -> dict[str, Any]:
    """Message → 可 JSON 化的 dict。

    用 asdict 递归展开 dataclass（ToolCall / Usage 都是 dataclass）；
    role 字段原样保留（schema.Role 是字符串常量）。
    """
    return asdict(msg)


def _dict_to_message(data: dict[str, Any]) -> Message:
    """dict → Message。重放时把语料还原成生产环境的真实对象。

    ⚠️ `is_system_reminder` 必须一起还原。它是压缩器区分
    「系统提醒」与「用户原始指令」的**唯一**依据（后者永不淘汰）——
    丢了它，重放时提醒会被当成用户指令永久保留，压缩的触发次数与降幅
    全都算错，而「重放」正是调阈值的依据。这类"少还原一个字段"的失误
    不会有任何报错，只会安静地把结论带偏，所以由单测守。
    """
    usage = data.get("usage")
    return Message(
        role=data["role"],
        content=data.get("content", ""),
        tool_calls=[ToolCall(**tc) for tc in data.get("tool_calls", [])],
        tool_call_id=data.get("tool_call_id", ""),
        usage=Usage(**usage) if usage else None,
        reasoning=data.get("reasoning", ""),
        is_system_reminder=data.get("is_system_reminder", False),
    )
