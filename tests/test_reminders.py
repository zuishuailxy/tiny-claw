"""System Reminders 测试

覆盖三件事：
1. ReminderBus 的触发条件与三条约束（冷却 / 限流 / 不误触发）
2. 提醒消息的身份标记（它走 user 通道，但**不是**用户指令）
3. Compactor 对提醒的分档：远期可清理、近期必须留、用户指令仍永不淘汰

第 3 点是修掉的那个静默 bug：提醒曾因 role=USER 且无 tool_call_id
被判为「用户原始指令」永不淘汰，于是每打断一次就多一条永生消息。
"""

from tiny_claw.context.compactor import Compactor
from tiny_claw.context.reminders import ReminderBus
from tiny_claw.engine.reminder import NAG_THRESHOLD, ReminderInjector
from tiny_claw.schema import SYSTEM_REMINDER_PREFIX, Message, Role, ToolCall, ToolResult

# ─────────────────────────────────────────────────────────────
# 1. ReminderBus：触发与约束
# ─────────────────────────────────────────────────────────────


def test_no_reminder_when_state_is_healthy():
    """状态健康时一条都不该发 —— 提醒本身也要花 token"""
    bus = ReminderBus()
    out = bus.collect(turn=1, max_turns=30, ctx_tokens=500, ctx_limit=16000)
    assert out == []


def test_turn_limit_reminder_fires_near_the_end():
    bus = ReminderBus()
    out = bus.collect(turn=26, max_turns=30, ctx_tokens=500, ctx_limit=16000)
    assert len(out) == 1
    assert "轮数预警" in out[0].content
    assert "26/30" in out[0].content


def test_context_pressure_reminder_fires_at_threshold_ratio():
    bus = ReminderBus()
    out = bus.collect(turn=2, max_turns=30, ctx_tokens=15_000, ctx_limit=16_000)
    assert len(out) == 1
    assert "上下文压力" in out[0].content
    assert "93%" in out[0].content  # 15000/16000


def test_context_pressure_skipped_when_limit_unknown():
    """阈值未知（ctx_limit=0）时不做压力判定，避免拿 0 做分母"""
    bus = ReminderBus()
    out = bus.collect(turn=2, max_turns=30, ctx_tokens=15_000, ctx_limit=0)
    assert out == []


def test_cooldown_suppresses_repeat_within_window():
    """同一类提醒在冷却窗口内不重复 —— 否则模型会麻木，且每轮白烧 token"""
    bus = ReminderBus(cooldown_turns=3)
    assert len(bus.collect(turn=26, max_turns=30)) == 1
    assert bus.collect(turn=27, max_turns=30) == []
    assert bus.collect(turn=28, max_turns=30) == []
    assert len(bus.collect(turn=29, max_turns=30)) == 1  # 29-26=3，出冷却


def test_max_per_turn_caps_the_noise():
    """多条提醒同时出现时模型往往一条都不听，所以限量"""
    bus = ReminderBus(max_per_turn=1)
    out = bus.collect(turn=27, max_turns=30, ctx_tokens=15_000, ctx_limit=16_000)
    assert len(out) == 1


def test_two_conditions_can_fire_together_when_allowed():
    bus = ReminderBus(max_per_turn=2)
    out = bus.collect(turn=27, max_turns=30, ctx_tokens=15_000, ctx_limit=16_000)
    assert len(out) == 2
    assert sum(1 for m in out if "上下文压力" in m.content) == 1
    assert sum(1 for m in out if "轮数预警" in m.content) == 1


# ─────────────────────────────────────────────────────────────
# 2. 身份标记：走 user 通道，但不是用户指令
# ─────────────────────────────────────────────────────────────


def test_reminders_are_marked_as_system_not_user():
    bus = ReminderBus()
    out = bus.collect(turn=26, max_turns=30)
    assert out and all(m.is_system_reminder for m in out)
    assert all(m.role == Role.USER for m in out)  # 协议不允许中途插 system
    assert all(m.content.startswith(SYSTEM_REMINDER_PREFIX) for m in out)


def test_nudge_from_reminder_injector_is_marked_too():
    """死循环打断走的是同一套身份约定"""
    injector = ReminderInjector()
    call = ToolCall(id="1", name="bash", arguments={"cmd": "ls"})
    failure = ToolResult(tool_call_id="1", output="[ERR:UNKNOWN] x", is_error=True)

    nudge = None
    for _ in range(NAG_THRESHOLD):
        nudge = injector.check_and_inject(call, failure)

    assert nudge is not None
    assert nudge.is_system_reminder is True
    assert nudge.role == Role.USER
    assert nudge.content.startswith(SYSTEM_REMINDER_PREFIX)


def test_success_clears_the_failure_counter():
    injector = ReminderInjector()
    call = ToolCall(id="1", name="bash", arguments={"cmd": "ls"})
    failure = ToolResult(tool_call_id="1", output="[ERR:UNKNOWN] x", is_error=True)
    success = ToolResult(tool_call_id="1", output="ok", is_error=False)

    injector.check_and_inject(call, failure)
    injector.check_and_inject(call, failure)
    assert injector.check_and_inject(call, success) is None  # 成功即清空
    # 清空后重新计数，两次失败还不到阈值
    assert injector.check_and_inject(call, failure) is None
    assert injector.check_and_inject(call, failure) is None
    assert injector.check_and_inject(call, failure) is not None  # 第 3 次


def test_argument_key_order_does_not_change_fingerprint():
    """sort_keys 是必须的：模型换个键顺序传同样的参数，指纹必须一致"""
    a = ReminderInjector._fingerprint("bash", {"cmd": "ls", "x": 1})
    b = ReminderInjector._fingerprint("bash", {"x": 1, "cmd": "ls"})
    assert a == b


# ─────────────────────────────────────────────────────────────
# 3. Compactor 分档：提醒可清理，用户指令永不淘汰
# ─────────────────────────────────────────────────────────────


def test_old_reminder_is_cleaned_but_recent_one_survives():
    """跑出工作记忆窗口的提醒已生效，可清理；还在窗口内的必须先让模型看见"""
    compactor = Compactor(max_tokens=1000, retain_last=2)
    old_reminder = Message(
        role=Role.USER, content="[SYSTEM REMINDER 警告] 早前的打断", is_system_reminder=True
    )
    big_output = Message(role=Role.USER, content="X" * 8000, tool_call_id="c1")
    recent_reminder = Message(
        role=Role.USER, content="[SYSTEM REMINDER 轮数预警] 快收尾", is_system_reminder=True
    )

    out = compactor.compact([old_reminder, big_output, recent_reminder])

    assert out[0].content == "...[早前的系统提醒已清理]..."
    assert out[2].content == recent_reminder.content  # 近期提醒原样保留


def test_user_instruction_still_never_evicted():
    """修提醒的分档不能误伤真正的用户指令"""
    compactor = Compactor(max_tokens=500, retain_last=1)
    instruction = Message(role=Role.USER, content="请帮我重构 src/engine 模块")
    big_output = Message(role=Role.USER, content="X" * 8000, tool_call_id="c1")

    out = compactor.compact([instruction, big_output])
    assert out[0].content == instruction.content


def test_compactor_preserves_the_system_reminder_flag():
    """拷贝时必须带上标记，否则压缩一次后提醒就退化成普通用户指令"""
    compactor = Compactor(max_tokens=100_000, retain_last=10)
    msg = Message(role=Role.USER, content="hi", is_system_reminder=True)
    out = compactor.compact([msg])
    assert out[0].is_system_reminder is True
