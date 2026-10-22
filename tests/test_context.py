"""上下文治理核心用例 — Compactor / Reminder / Session 工作记忆

这三处是长任务下最容易静默出错的地方：压缩破坏消息配对会导致协议 400，
死循环检测失效会白白烧 Token，孤儿工具响应同样会触发协议报错。
"""

import pytest

from tiny_claw.context.compactor import Compactor
from tiny_claw.context.tokens import estimate_tokens
from tiny_claw.engine.reminder import ReminderInjector
from tiny_claw.engine.session import Session
from tiny_claw.schema import Message, Role, ToolCall, ToolResult, Usage


# ═══════════════════════════════════════════════════════════════
# Compactor
# ═══════════════════════════════════════════════════════════════


def test_no_compaction_below_threshold():
    compactor = Compactor(max_tokens=10000, retain_last=5)
    msgs = [Message(role=Role.USER, content="很短")]
    assert compactor.compact(msgs) is msgs


def test_system_prompt_is_never_compacted():
    compactor = Compactor(max_tokens=10, retain_last=1)
    msgs = [
        Message(role=Role.SYSTEM, content="S" * 300),
        Message(role=Role.USER, content="U" * 300),
    ]
    out = compactor.compact(msgs)
    assert out[0].content == "S" * 300


def test_tool_calls_are_never_compacted():
    """tool_calls 维系着逻辑链，压缩时绝不能动"""
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls -la"})
    msgs = [Message(role=Role.ASSISTANT, content="A" * 300, tool_calls=[call])]

    out = Compactor(max_tokens=10, retain_last=0).compact(msgs)
    assert out[0].tool_calls[0].name == "bash"
    assert out[0].tool_calls[0].arguments == {"command": "ls -la"}


def test_old_tool_result_masked_but_keeps_length_hint():
    """远期工具结果全量掩码，但保留原始长度，让模型知道信息丢了"""
    compactor = Compactor(max_tokens=10, retain_last=1)
    msgs = [
        Message(role=Role.USER, content="X" * 500, tool_call_id="c1"),
        Message(role=Role.USER, content="recent"),
    ]
    out = compactor.compact(msgs)
    assert "已被系统强制清理" in out[0].content
    assert "500" in out[0].content


def test_recent_tool_result_is_head_tail_truncated():
    """近期工具结果掐头去尾：命令回显的头部有价値，报错堆栈在尾部"""
    compactor = Compactor(max_tokens=10, retain_last=2)
    body = "H" * 600 + "T" * 600
    msgs = [Message(role=Role.USER, content=body, tool_call_id="c1")]

    out = compactor.compact(msgs)
    content = out[0].content
    assert "已被系统截断" in content
    assert content.startswith("H" * 500)
    assert content.endswith("T" * 500)


def test_short_tool_result_is_left_untouched():
    # 阈值取 5：7 个中文字符约 7 token，会触发压缩但内容不够长，故原样保留
    compactor = Compactor(max_tokens=5, retain_last=2)
    msgs = [Message(role=Role.USER, content="很小的一段输出", tool_call_id="c1")]
    out = compactor.compact(msgs)
    assert out[0].content == "很小的一段输出"


def test_compaction_actually_reduces_length():
    msgs = [Message(role=Role.USER, content="X" * 3000, tool_call_id=f"c{i}") for i in range(5)]
    compactor = Compactor(max_tokens=100, retain_last=1)
    before =     compactor._estimate_tokens(msgs)
    after = compactor._estimate_tokens(compactor.compact(msgs))
    assert after < before


def test_compaction_does_not_mutate_original():
    original_content = "X" * 500
    msgs = [Message(role=Role.USER, content=original_content, tool_call_id="c1")]
    Compactor(max_tokens=10, retain_last=1).compact(msgs)
    assert msgs[0].content == original_content


# ── 按信息价值分级淘汰 ──


def test_failure_record_is_never_masked():
    """失败记录含错误码与救援提示，掩码掉会让模型重复踩同一个坑"""
    failure = "[ERR:FILE_NOT_FOUND] 找不到 a.txt\n\n[系统救援指南]: 请先读取文件再编辑"
    compactor = Compactor(max_tokens=10, retain_last=1)
    msgs = [
        Message(role=Role.USER, content=failure, tool_call_id="c1"),
        Message(role=Role.USER, content="recent"),
    ]

    out = compactor.compact(msgs)
    assert "[ERR:FILE_NOT_FOUND]" in out[0].content
    assert "系统救援指南" in out[0].content
    assert "已被系统强制清理" not in out[0].content


def test_long_failure_record_keeps_both_ends():
    """兜底截断时，错误码在头、救援提示在尾，两头都不能丢"""
    compactor = Compactor(max_tokens=100, retain_last=0)
    body = "[ERR:TOOL_FAILED] " + "X" * 10000 + "\n\n[系统救援指南]: 换一种方式重试"
    msgs = [Message(role=Role.USER, content=body, tool_call_id="c1")]

    out = compactor.compact(msgs)
    assert out[0].content.startswith("[ERR:TOOL_FAILED]")
    assert out[0].content.endswith("[系统救援指南]: 换一种方式重试")


def test_gradual_eviction_stops_once_under_threshold():
    """渐进式淘汰：一降到阈值以下就停手，近期内容不被误伤"""
    # 两段各 3000 字符英文 ≈ 1411 token，加消息开销总 ≈ 2942 > 2000 触发；
    # 掩码掉远期（降到 ≈1571）后即停手，近期原样保留。
    # （阈值数字为 2026-09-09 估算系数校准后重标定，下同）
    compactor = Compactor(max_tokens=2000, retain_last=1)
    recent_body = "R" * 3000
    msgs = [
        Message(role=Role.USER, content="X" * 3000, tool_call_id="c1"),
        Message(role=Role.USER, content=recent_body, tool_call_id="c2"),
    ]

    out = compactor.compact(msgs)
    assert out[0].content.startswith("...[为了节省内存")  # 远期已掩码
    assert out[1].content == recent_body  # 近期原样保留，未被顺手截断


def test_reasoning_outlives_tool_output():
    """同为远期内容，模型的推理链比工具输出更值得留下来"""
    compactor = Compactor(max_tokens=100, retain_last=0)
    msgs = [
        Message(role=Role.USER, content="X" * 3000, tool_call_id="c1"),
        Message(role=Role.ASSISTANT, content="我决定先读文件再改"),
    ]

    out = compactor.compact(msgs)
    assert out[0].content.startswith("...[为了节省内存")
    assert out[1].content == "我决定先读文件再改"


def test_user_instruction_is_never_compacted():
    """用户原始指令承载任务目标，任何情况下都不能被压缩"""
    compactor = Compactor(max_tokens=10, retain_last=0)
    instruction = "请帮我重构这个函数，注意保持对外接口兼容"
    msgs = [Message(role=Role.USER, content=instruction)]

    out = compactor.compact(msgs)
    assert out[0].content == instruction


def test_compaction_preserves_usage_and_reasoning():
    """拷贝消息时不能丢掉 usage 与 reasoning，否则计费与推理链会被静默吞掉"""
    compactor = Compactor(max_tokens=10, retain_last=1)
    msgs = [
        Message(
            role=Role.ASSISTANT,
            content="A" * 500,
            usage=Usage(prompt_tokens=120, completion_tokens=30),
            reasoning="先确认文件是否存在",
        )
    ]

    out = compactor.compact(msgs)
    assert out[0].usage is not None
    assert out[0].usage.prompt_tokens == 120
    assert out[0].reasoning == "先确认文件是否存在"


# ── token 计量口径 ──


def test_empty_text_costs_zero_tokens():
    assert estimate_tokens("") == 0


def test_cjk_is_counted_one_token_per_char():
    """中文 1 字 ≈ 1 token（略微高估；高估让压缩提前触发，是安全方向）"""
    assert estimate_tokens("中" * 100) == 101


def test_ascii_uses_calibrated_ratio():
    """英文按校准系数计：0.47 token/字符（benchmark 实测，旧值 0.25 低估 1.86 倍）"""
    assert estimate_tokens("a" * 1000) == 471


def test_same_budget_gives_english_more_room():
    """同一阈值下英文能存约 2.1 倍字符 —— 这正是字符口径的问题所在。

    校准前该比例约为 4 倍（0.25 token/字符）；实测修正到 0.47 后收窄，
    但中英文行为差异依然存在，字符口径的问题没有被消除，只是被如实计量。
    """
    compactor = Compactor(max_tokens=8000, retain_last=1)

    # 英文 16000 字符 ≈ 7521 + 60 消息开销，未达阈值，原样返回
    english = [Message(role=Role.USER, content="a" * 16000, tool_call_id="c1")]
    assert compactor.compact(english) is english

    # 中文 16000 字符 ≈ 16001 + 60 消息开销，远超阈值，必须压缩
    chinese = [Message(role=Role.USER, content="中" * 16000, tool_call_id="c1")]
    assert compactor.compact(chinese) is not chinese


def test_chinese_behaviour_unchanged_under_one_to_one_conversion():
    """1:1 保守换算：中文项目下 8000 字符仍触发，与改造前一致（不劣化）"""
    compactor = Compactor(max_tokens=8000, retain_last=2)
    msgs = [Message(role=Role.USER, content="中" * 8000, tool_call_id="c1")]

    out = compactor.compact(msgs)
    assert out is not msgs
    assert "已被系统截断" in out[0].content  # 近期 → 掐头去尾，不是掩码


def test_mask_hint_reports_chars_not_tokens():
    """阈值判定用 token，但提示语报字符数 —— 模型靠字符数判断值不值得重读"""
    compactor = Compactor(max_tokens=10, retain_last=1)
    msgs = [
        Message(role=Role.USER, content="X" * 500, tool_call_id="c1"),
        Message(role=Role.USER, content="recent"),
    ]

    out = compactor.compact(msgs)
    assert "500" in out[0].content  # 报字符数，不是 125 token
    assert "125" not in out[0].content


# ═══════════════════════════════════════════════════════════════
# ReminderInjector
# ═══════════════════════════════════════════════════════════════


def _err_call() -> tuple[ToolCall, ToolResult]:
    call = ToolCall(id="1", name="bash", arguments={"command": "ls"})
    return call, ToolResult(tool_call_id="1", output="boom", is_error=True)


def test_no_nudge_before_threshold():
    injector = ReminderInjector()
    call, err = _err_call()
    assert injector.check_and_inject(call, err) is None
    assert injector.check_and_inject(call, err) is None


def test_nudge_on_third_consecutive_failure():
    injector = ReminderInjector()
    call, err = _err_call()
    injector.check_and_inject(call, err)
    injector.check_and_inject(call, err)

    nudge = injector.check_and_inject(call, err)
    assert nudge is not None
    assert nudge.role == Role.USER
    assert "死循环" in nudge.content


def test_success_clears_all_counters():
    """一次成功说明走出了死胡同，计数器必须整体清零"""
    injector = ReminderInjector()
    call, err = _err_call()
    injector.check_and_inject(call, err)
    injector.check_and_inject(call, err)

    ok = ToolResult(tool_call_id="1", output="done", is_error=False)
    assert injector.check_and_inject(call, ok) is None

    assert injector.check_and_inject(call, err) is None
    assert injector.check_and_inject(call, err) is None


def test_fingerprint_ignores_argument_order():
    """参数键顺序不应影响指纹，否则模型换个写法就能绕过检测"""
    a = ReminderInjector._fingerprint("bash", {"a": 1, "b": 2})
    b = ReminderInjector._fingerprint("bash", {"b": 2, "a": 1})
    assert a == b


def test_different_arguments_produce_different_fingerprints():
    a = ReminderInjector._fingerprint("bash", {"command": "ls"})
    b = ReminderInjector._fingerprint("bash", {"command": "pwd"})
    assert a != b


def test_failures_are_counted_per_fingerprint():
    """换参数重试不应累加到原指纹上"""
    injector = ReminderInjector()
    call_a = ToolCall(id="1", name="bash", arguments={"command": "ls"})
    call_b = ToolCall(id="2", name="bash", arguments={"command": "pwd"})
    err_a = ToolResult(tool_call_id="1", output="boom", is_error=True)
    err_b = ToolResult(tool_call_id="2", output="boom", is_error=True)

    injector.check_and_inject(call_a, err_a)
    injector.check_and_inject(call_b, err_b)
    assert injector.check_and_inject(call_a, err_a) is None


# ═══════════════════════════════════════════════════════════════
# Session 工作记忆
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_working_memory_drops_leading_orphan_tool_response():
    """截断后若首条是没有对应 tool_call 的工具响应，必须剔除，否则协议报错"""
    session = Session("t1", "/tmp")
    for i in range(3):
        await session.append(Message(role=Role.USER, content=f"m{i}"))
    await session.append(Message(role=Role.USER, content="tool-out", tool_call_id="c1"))

    assert await session.get_working_memory(limit=1) == []


@pytest.mark.asyncio
async def test_working_memory_keeps_messages_when_head_is_normal():
    session = Session("t2", "/tmp")
    await session.append(Message(role=Role.USER, content="m0"))
    await session.append(Message(role=Role.USER, content="m1"))

    wm = await session.get_working_memory(limit=1)
    assert [m.content for m in wm] == ["m1"]


@pytest.mark.asyncio
async def test_error_turn_tracking_records_first_error_position():
    session = Session("t3", "/tmp")
    await session.record_usage(prompt=100, completion=50)
    await session.mark_error_turn()

    assert session.error_turns == 1
    assert session.first_error_token == 150


# ═══════════════════════════════════════════════════════════════
# 错误码体系的不变量
#
# 加这两条的理由：救援提示只在引擎层注入，而引擎靠 `[ERR:CODE]` 前缀识别错误。
# 于是"新增错误码忘了配提示"和"某条错误出口漏了前缀"这两类缺口都是**静默**的
# —— 功能看着正常，只是模型拿不到"下一步该做什么"。registry.py 就漏过两条
# （未知工具、中间件拦截），所以把不变量固化成用例。
# ═══════════════════════════════════════════════════════════════


def test_every_error_code_has_a_recovery_hint():
    """每个 ErrorCode 都必须配救援提示（UNKNOWN 例外，故意留给模型自行分析）"""
    from tiny_claw.context.recovery import _RECOVERY_HINTS, ErrorCode

    missing = [
        code
        for code in ErrorCode
        if code is not ErrorCode.UNKNOWN and code not in _RECOVERY_HINTS
    ]
    assert missing == [], f"这些错误码没有救援提示: {[c.value for c in missing]}"


def test_recovery_hint_tells_model_what_to_do_next():
    """提示不能是"出错了"这种废话——必须给出可执行的下一步"""
    from tiny_claw.context.recovery import RecoveryManager, format_error, ErrorCode

    # 用最容易让人空转的两个码来守：调错工具名、被策略拦截
    for code in (ErrorCode.TOOL_NOT_FOUND, ErrorCode.TOOL_BLOCKED):
        enhanced = RecoveryManager().analyze_and_inject(format_error(code, "原始报错"))
        assert "[系统救援指南]" in enhanced
        assert len(enhanced) > len(format_error(code, "原始报错")) + 10


def test_engine_guard_warns_on_error_without_code(caplog):
    """引擎守卫：失败结果缺 [ERR: 前缀时必须告警（这是防"静默丧失救援提示"的兜底）"""
    import logging

    from tiny_claw.engine.loop import _guard_missing_error_code
    from tiny_claw.schema import ToolResult

    with caplog.at_level(logging.WARNING):
        _guard_missing_error_code(
            ToolResult(tool_call_id="c1", output="裸错误文案", is_error=True)
        )
    assert any("缺少 [ERR:CODE] 前缀" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # 带前缀的失败、以及成功结果，都不该告警
        _guard_missing_error_code(
            ToolResult(
                tool_call_id="c1", output="[ERR:UNKNOWN] x", is_error=True
            )
        )
        _guard_missing_error_code(
            ToolResult(tool_call_id="c1", output="裸但成功", is_error=False)
        )
    assert caplog.records == []
