"""并发工具调用的协议完整性单测 — 对应 engine/loop.py 的并发执行段

核心契约：**每一个 tool_call 都必须有一条配对的 tool 结果消息**。

一旦缺失（孤儿 tool_call），下一轮请求里 assistant.tool_calls 会少一条
配对的 tool 消息，LLM 侧直接以 400 拒绝，整个会话当场死掉。

因此引擎对并发段的异常必须是「降级」而不是「放弃」：
1. Registry 抛异常 → 降级为 error ToolResult
2. Reporter 抛异常（如飞书卡片发送失败）→ 只记日志，不影响工具执行
3. gather 自身异常 → return_exceptions 兜底，不穿透整轮

本文件用会抛异常的 Registry / Reporter 把上述路径逐个逼出来验证。
"""

import asyncio

import pytest

from tiny_claw.engine.loop import AgentEngine
from tiny_claw.engine.reporter import Reporter
from tiny_claw.engine.session import Session
from tiny_claw.provider.base import LLMProvider
from tiny_claw.schema import (
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from tiny_claw.tools.base import BaseTool
from tiny_claw.tools.registry import Registry, RegistryImpl


# ═══════════════════════════════════════════════════════════════
# 测试替身
# ═══════════════════════════════════════════════════════════════


class _FakeProvider(LLMProvider):
    """第一轮吐出指定 tool_calls，第二轮收尾结束循环。"""

    def __init__(self, tool_calls: list[ToolCall]):
        self._tool_calls = tool_calls
        self.rounds = 0

    async def generate(self, messages, available_tools=None) -> Message:
        self.rounds += 1
        if self.rounds == 1:
            return Message(
                role=Role.ASSISTANT, content="", tool_calls=self._tool_calls
            )
        return Message(role=Role.ASSISTANT, content="任务结束")


class _ExplodingRegistry(Registry):
    """execute 直接抛异常 — 模拟 Middleware / 路由层的意外故障。"""

    def __init__(self, boom: Exception):
        self._boom = boom
        self.seen: list[str] = []

    def register(self, tool) -> None: ...

    def use(self, mw) -> None: ...

    def get_available_tools(self) -> list[ToolDefinition]:
        return []

    async def execute(self, call: ToolCall) -> ToolResult:
        self.seen.append(call.name)
        raise self._boom


class _EchoTool(BaseTool):
    """可控延迟的回声工具，用于验证并发下的顺序保证。"""

    def name(self) -> str:
        return "echo"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo", description="回声", input_schema={"type": "object"}
        )

    async def execute(self, arguments: dict) -> str:
        await asyncio.sleep(arguments.get("delay", 0))
        return f"ok:{arguments.get('tag', '')}"


class _FakeReporter(Reporter):
    """记录全部上报事件；可配置在 on_tool_call 时抛异常。"""

    def __init__(self, fail_on_call: bool = False):
        self.fail_on_call = fail_on_call
        self.results: list[tuple[str, bool]] = []

    async def on_thinking(self, content: str) -> None: ...

    async def on_tool_call(self, tool_name: str, args: dict) -> None:
        if self.fail_on_call:
            raise RuntimeError("上报通道故障（模拟飞书卡片发送失败）")

    async def on_tool_result(self, tool_name: str, output: str, is_error: bool) -> None:
        self.results.append((tool_name, is_error))

    async def on_message(self, content: str) -> None: ...


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════


def _tool_messages(history: list[Message]) -> list[Message]:
    return [m for m in history if m.tool_call_id]


# ═══════════════════════════════════════════════════════════════
# 用例
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_registry_exception_still_pairs_every_tool_call(tmp_path):
    """Registry 抛异常：不崩溃，且每个 tool_call 都有配对的 error 结果。"""
    calls = [
        ToolCall(id="c1", name="bash", arguments={"command": "ls"}),
        ToolCall(id="c2", name="read_file", arguments={"path": "a.py"}),
    ]
    registry = _ExplodingRegistry(RuntimeError("路由层炸了"))
    engine = AgentEngine(
        _FakeProvider(calls), registry, enable_thinking=False, reporter=_FakeReporter()
    )
    session = Session("s1", str(tmp_path))

    await engine.run(session)  # 不应抛出

    history = await session.get_working_memory()
    assistant_msg = next(
        m for m in history if m.role == Role.ASSISTANT and m.tool_calls
    )
    tool_msgs = _tool_messages(history)

    # 关键断言：结果 ID 与 tool_call ID 一一对应，无孤儿
    assert [m.tool_call_id for m in tool_msgs] == [tc.id for tc in calls]
    assert all(m.content.startswith("[ERR:") for m in tool_msgs)
    assert session.error_turns == 1  # 异常被计为失败轮，供容错追踪


@pytest.mark.asyncio
async def test_reporter_failure_does_not_break_the_turn(tmp_path):
    """Reporter 上报失败：只记日志，工具照常执行、结果照常回填。"""
    calls = [ToolCall(id="c1", name="echo", arguments={"tag": "x"})]
    registry = RegistryImpl()
    registry.register(_EchoTool())
    reporter = _FakeReporter(fail_on_call=True)

    engine = AgentEngine(
        _FakeProvider(calls), registry, enable_thinking=False, reporter=reporter
    )
    session = Session("s2", str(tmp_path))

    await engine.run(session)  # 不应抛出

    history = await session.get_working_memory()
    tool_msgs = _tool_messages(history)
    assert [m.tool_call_id for m in tool_msgs] == ["c1"]
    assert tool_msgs[0].content == "ok:x"  # 工具确实执行了
    assert reporter.results == [("echo", False)]  # 结果上报仍被记录


@pytest.mark.asyncio
async def test_results_keep_tool_call_order_under_concurrency(tmp_path):
    """并发执行下，结果写入顺序仍严格对齐 tool_calls 顺序（慢的先发起也先占位）。"""
    calls = [
        ToolCall(id="slow", name="echo", arguments={"tag": "slow", "delay": 0.05}),
        ToolCall(id="fast", name="echo", arguments={"tag": "fast", "delay": 0.0}),
    ]
    registry = RegistryImpl()
    registry.register(_EchoTool())
    engine = AgentEngine(
        _FakeProvider(calls), registry, enable_thinking=False, reporter=_FakeReporter()
    )
    session = Session("s3", str(tmp_path))

    await engine.run(session)

    history = await session.get_working_memory()
    tool_msgs = _tool_messages(history)
    assert [m.tool_call_id for m in tool_msgs] == ["slow", "fast"]
    assert [m.content for m in tool_msgs] == ["ok:slow", "ok:fast"]
