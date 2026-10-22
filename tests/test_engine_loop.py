"""AgentEngine 主循环用例

补空白的理由：引擎主循环此前是测试空白区（`pytest --collect-only | grep -E
"loop|engine"` 为空）。这里守的是循环的**控制流契约** ——
什么时候继续问模型、什么时候停、以及那条硬约束：

> 协议完整性优先：每个 `tool_call` 都必须回填一条配对的 `tool` 消息。
> 少一条，下一轮请求里就出现孤儿 tool_call，LLM 侧直接 400 打死整个会话。

所以最后一个用例专门用「一个工具成功 + 一个工具抛异常」来验证配对不丢。

另含子智能体（`run_sub`）用例：这里守的是**主/子循环的自愈路径必须同构**——
救援提示要真的进到模型看的上下文里、死循环打断要真的触发、
A/B 关掉自愈时子智能体也必须拿裸错误。
"""

from tiny_claw.engine.loop import AgentEngine
from tiny_claw.engine.session import Session
from tiny_claw.provider.base import LLMProvider
from tiny_claw.schema import Message, Role, ToolCall, ToolDefinition
from tiny_claw.tools.base import BaseTool
from tiny_claw.tools.registry import RegistryImpl


class _Provider(LLMProvider):
    """按脚本回放回复，并记录每次调用拿到的上下文规模与工具数"""

    def __init__(self, replies: list[Message]):
        self._replies = list(replies)
        self.calls: list[tuple[int, int]] = []

    async def generate(self, messages, available_tools=None):
        self.calls.append((len(messages), len(available_tools or [])))
        return self._replies.pop(0)


class _Tool(BaseTool):
    def __init__(self, name="t", output="ok", boom=None):
        self._name, self._output, self._boom = name, output, boom
        self.calls = 0

    def name(self):
        return self._name

    def definition(self):
        return ToolDefinition(
            name=self._name, description="测试用", input_schema={"type": "object"}
        )

    async def execute(self, arguments):
        self.calls += 1
        if self._boom:
            raise self._boom
        return self._output


class _Reporter:
    """静默 reporter：只记录，不打印（避免污染测试输出）"""

    def __init__(self):
        self.messages: list[str] = []
        self.tool_calls: list[str] = []
        self.tool_results: list[tuple[str, bool]] = []

    async def on_thinking(self, text):
        pass

    async def on_message(self, text):
        self.messages.append(text)

    async def on_tool_call(self, name, arguments):
        self.tool_calls.append(name)

    async def on_tool_result(self, name, display, is_error):
        self.tool_results.append((name, is_error))


def _mk_session(tmp_path, text="hi") -> Session:
    return Session("s1", str(tmp_path))


def _mk_engine(provider, registry=None, **kw) -> AgentEngine:
    # enable_thinking=False：跳过硬思考阶段，让每次 run 只消耗一条脚本回复
    return AgentEngine(
        provider,
        registry or RegistryImpl(),
        enable_thinking=False,
        reporter=_Reporter(),
        **kw,
    )


def _tool_results(session) -> list[Message]:
    """取出工具结果消息。

    ⚠️ 内部消息模型里**没有 tool 角色** —— `Role` 只有 system/user/assistant 三种
    （厂商中立），工具结果用 `user + tool_call_id` 表达，由 Provider 层翻译成各协议
    的具体格式（OpenAI 的 `role: tool` / Anthropic 的 tool_result content block）。
    所以判定条件是「role 是 user **且**带 tool_call_id」，不能按 role == "tool" 找。

    顺带一提：压缩器判断「这条是不是用户原始指令」也正是用
    「user 且无 tool_call_id」——之前那个「提醒被误判为用户指令、永不淘汰」的静默
    bug 就出在这个双关语义上。
    """
    return [m for m in session._history if m.role == Role.USER and m.tool_call_id]


def _assistant(content="", tool_calls=None) -> Message:
    return Message(
        role=Role.ASSISTANT, content=content, tool_calls=tool_calls or []
    )


# ══════════════════════════════════════════════════════════════
# 终止条件
# ══════════════════════════════════════════════════════════════


async def test_plain_reply_ends_in_one_turn(tmp_path):
    """模型没要工具 → 一轮就停"""
    p = _Provider([_assistant("done")])
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    await _mk_engine(p, max_turns=5).run(s)

    assert len(p.calls) == 1
    assert [m.role for m in s._history] == [Role.USER, Role.ASSISTANT]
    assert s._history[-1].content == "done"


async def test_max_turns_caps_the_loop(tmp_path):
    """模型一直要工具 → 必须在 max_turns 停下（防跑飞），且不消费多余回复"""
    reg = RegistryImpl()
    reg.register(_Tool(name="forever"))
    tc = ToolCall(id="c", name="forever", arguments={})
    p = _Provider([_assistant("", [tc]) for _ in range(10)])
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    await _mk_engine(p, reg, max_turns=3).run(s)

    assert len(p.calls) == 3
    assert len(p._replies) == 7  # 第 4 轮起的回复原封不动


async def test_max_turns_is_configurable_per_call(tmp_path):
    """max_turns 由调用方收紧（benchmark 靠它控成本）—— 曾经这个字段是死代码"""
    reg = RegistryImpl()
    reg.register(_Tool(name="forever"))
    tc = ToolCall(id="c", name="forever", arguments={})
    p = _Provider([_assistant("", [tc]) for _ in range(10)])
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    await _mk_engine(p, reg, max_turns=1).run(s)
    assert len(p.calls) == 1


# ══════════════════════════════════════════════════════════════
# 工具调用回路
# ══════════════════════════════════════════════════════════════


async def test_tool_call_round_trips_into_history(tmp_path):
    """工具调用 → 执行 → 结果回灌 → 再问一次模型"""
    reg = RegistryImpl()
    tool = _Tool(name="echo", output="tool-output")
    reg.register(tool)

    p = _Provider(
        [
            _assistant("", [ToolCall(id="c1", name="echo", arguments={"k": "v"})]),
            _assistant("finished"),
        ]
    )
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    await _mk_engine(p, reg, max_turns=5).run(s)

    assert tool.calls == 1
    assert len(p.calls) == 2  # 工具结果回灌后又问了一次
    tool_msgs = _tool_results(s)
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "tool-output"
    assert tool_msgs[0].tool_call_id == "c1"  # 配对 ID 必须对上


async def test_tool_exception_degrades_and_loop_continues(tmp_path):
    """工具抛异常 → 降级为错误结果，循环继续（一次异常不该打死整个会话）"""
    reg = RegistryImpl()
    reg.register(_Tool(name="bad", boom=RuntimeError("kaboom")))

    p = _Provider(
        [
            _assistant("", [ToolCall(id="c1", name="bad", arguments={})]),
            _assistant("recovered"),
        ]
    )
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    await _mk_engine(p, reg, max_turns=5).run(s)

    assert len(p.calls) == 2
    tool_msg = _tool_results(s)[0]
    assert "[ERR:" in tool_msg.content
    assert "kaboom" in tool_msg.content


async def test_every_tool_call_gets_a_paired_result(tmp_path):
    """协议完整性：N 个 tool_call 必须回填 N 条 tool 消息。

    故意让一个成功、一个抛异常 —— 异常路径最容易漏配对，而漏配对会让
    下一轮请求带上孤儿 tool_call，LLM 侧 400 打死会话。
    """
    reg = RegistryImpl()
    reg.register(_Tool(name="good", output="A"))
    reg.register(_Tool(name="bad", boom=RuntimeError("x")))

    p = _Provider(
        [
            _assistant(
                "",
                [
                    ToolCall(id="c1", name="good", arguments={}),
                    ToolCall(id="c2", name="bad", arguments={}),
                ],
            ),
            _assistant("done"),
        ]
    )
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    await _mk_engine(p, reg, max_turns=5).run(s)

    tool_msgs = _tool_results(s)
    assert sorted(m.tool_call_id for m in tool_msgs) == ["c1", "c2"]


async def test_unknown_tool_call_still_gets_result(tmp_path):
    """模型幻觉工具名（registry 里没有）也必须回填一条配对结果"""
    p = _Provider(
        [
            _assistant("", [ToolCall(id="c9", name="ghost", arguments={})]),
            _assistant("ok"),
        ]
    )
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    await _mk_engine(p, max_turns=5).run(s)

    tool_msgs = _tool_results(s)
    assert [m.tool_call_id for m in tool_msgs] == ["c9"]
    assert "不存在名为 'ghost' 的工具" in tool_msgs[0].content


# ══════════════════════════════════════════════════════════════
# 上下文组装
# ══════════════════════════════════════════════════════════════


async def test_system_prompt_is_prepended_to_context(tmp_path):
    """每轮都带 System Prompt；工具清单走 provider 参数而不是塞进 prompt"""
    reg = RegistryImpl()
    reg.register(_Tool(name="only_tool"))
    p = _Provider([_assistant("done")])
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    await _mk_engine(p, reg, max_turns=3).run(s)

    n_messages, n_tools = p.calls[0]
    assert n_tools == 1  # 工具清单通过 available_tools 传给 Provider
    assert n_messages >= 2  # 至少 system + user


async def test_reporter_receives_message_and_tool_events(tmp_path):
    """Reporter 是 UI 的唯一出口，事件不能漏"""
    reg = RegistryImpl()
    reg.register(_Tool(name="echo", output="out"))
    p = _Provider(
        [
            _assistant("", [ToolCall(id="c1", name="echo", arguments={})]),
            _assistant("done"),
        ]
    )
    s = _mk_session(tmp_path)
    await s.append(Message(role=Role.USER, content="hi"))

    engine = _mk_engine(p, reg, max_turns=5)
    await engine.run(s)

    rep = engine.reporter
    assert rep.tool_calls == ["echo"]
    assert rep.tool_results == [("echo", False)]
    assert rep.messages == ["done"]


# ══════════════════════════════════════════════════════════════
# 子智能体：自愈路径必须与主循环同构
# ══════════════════════════════════════════════════════════════
#
# 为什么单独守：`run_sub` 曾长期零覆盖，于是出过三个静默缺陷——
#   ① 救援提示算出来后只用于终端上报，注入模型的是裸 `result.output`；
#   ② 完全没有 ReminderInjector，"连续失败 3 次打断"对子智能体不生效；
#   ③ 不受 `_enable_recovery` 约束，A/B 基线组里仍带救援指南。
# 三条都不会让任何测试变红，只能靠断言注入内容来钉住。


class _SubProvider(LLMProvider):
    """回放 n 轮「一直调 boom」，最后一轮收尾；记录每轮拿到的上下文。"""

    def __init__(self, fail_rounds: int = 1):
        self._fail_rounds = fail_rounds
        self.seen: list[list[Message]] = []

    async def generate(self, messages, available_tools=None):
        self.seen.append(list(messages))
        if len(self.seen) <= self._fail_rounds:
            return _assistant(
                "再试一次",
                [ToolCall(id=f"c{len(self.seen)}", name="boom", arguments={"p": "x"})],
            )
        return _assistant("探索完毕")


class _BoomTool(BaseTool):
    """恒定失败且同参数的只读工具 —— 用来触发死循环指纹。"""

    def __init__(self, output="[ERR:FILE_NOT_FOUND] 路径不存在: x"):
        self._output = output

    def name(self):
        return "boom"

    def definition(self):
        return ToolDefinition(
            name="boom", description="必失败", input_schema={"type": "object"}
        )

    async def execute(self, arguments):
        return self._output


async def test_subagent_recovery_hint_reaches_the_model():
    """子智能体注入的失败结果必须带救援指南，而不是裸错误。

    这条直接对应曾经的 P0：`output`（增强）只喂了 display，`results` 回填的却是
    未增强的 `result`，于是模型看到的是 `[ERR:FILE_NOT_FOUND] 路径不存在: x`。
    """
    p = _SubProvider(fail_rounds=1)
    reg = RegistryImpl()
    reg.register(_BoomTool())

    await _mk_engine(p, reg).run_sub("找x", read_only_registry=reg)

    injected = p.seen[1][-1].content
    assert "[ERR:FILE_NOT_FOUND]" in injected
    assert "[系统救援指南]" in injected, "救援指南没进模型上下文——又退回result.output 了"
    # 原文也必须保留，救援指南是追加不是替换
    assert "路径不存在: x" in injected


async def test_subagent_deadloop_nudge_fires():
    """同参数连续失败 3 次 → 注入死循环打断消息（且带 is_system_reminder）。"""
    p = _SubProvider(fail_rounds=3)  # 3轮失败，第 4 轮收尾
    reg = RegistryImpl()
    reg.register(_BoomTool())

    await _mk_engine(p, reg).run_sub("找 x", read_only_registry=reg)

    last_round = p.seen[-1]
    nudges = [m for m in last_round if m.is_system_reminder]
    assert nudges, "连续 3 次同参数失败后没有注入打断消息"
    assert "死循环" in nudges[0].content
    # 打断消息不能带 tool_call_id，否则会被当成工具结果破坏配对
    assert not nudges[0].tool_call_id


async def test_subagent_respects_enable_recovery_off():
    """A/B 基线组（enable_recovery=False）里子智能体必须拿裸错误。

    守住的是"对照组干净"：若这条挂了，说明基线组其实开着自愈，
    之前那份错误自愈 A/B 报告（token 9181 → 6545）的两组就不成立。
    """
    p = _SubProvider(fail_rounds=1)
    reg = RegistryImpl()
    reg.register(_BoomTool())

    engine = _mk_engine(p, reg, enable_recovery=False)
    await engine.run_sub("找 x", read_only_registry=reg)

    injected = p.seen[1][-1].content
    assert "[ERR:FILE_NOT_FOUND]" in injected
    assert "[系统救援指南]" not in injected
