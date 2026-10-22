"""评测插桩层（`eval/instrument.py`）。

覆盖项目材料里「一次录制 + 离线重放，调参零 API 成本」这条。三条主线：

1. **只旁听、不改结果**：三个插桩代理（Tracker / Compactor / Registry）
   必须在调用前后记录，且原样透传入参与返回值。这条如果破了，
   跑分数据就不再代表生产行为，整套评测失去意义。
2. **语料可入库**：`RunTrace.save()` 必须脱掉机器路径（仓库根 → `<repo>`、
   家目录 → `~/`）。语料是要提交的，不能暴露本机目录结构。
3. **可离线重放**：落盘再读回后语义必须等价 —— 能真的把语料还原成
   生产环境的真实对象，否则"重放"和"真跑"结论不可比。
"""

import json
from pathlib import Path

from tiny_claw import config
from tiny_claw.eval.instrument import (
    APICallRecord,
    CompactSnapshot,
    InstrumentedCompactor,
    InstrumentedRegistry,
    InstrumentedTracker,
    RunTrace,
    ToolOutputRecord,
)
from tiny_claw.schema import (
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    Usage,
)
from tiny_claw.tools.base import BaseTool
from tiny_claw.tools.registry import RegistryImpl

# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────


def _call(est: int, real_prompt: int, real_completion: int = 0, seq: int = 0) -> APICallRecord:
    return APICallRecord(
        seq=seq,
        est_tokens=est,
        real_prompt=real_prompt,
        real_completion=real_completion,
        n_messages=3,
        latency_ms=100,
    )


def _trace(**kw) -> RunTrace:
    return RunTrace(case_id="case_x", model="deepseek-v4-flash", **kw)


class _EchoTool(BaseTool):
    """回显固定长度的工具，用来验证体积统计"""

    def __init__(self, payload: str = "x" * 123, name: str = "echo"):
        self._payload = payload
        self._name = name

    def name(self) -> str:
        return self._name

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="",
            input_schema={"type": "object", "properties": {}},
        )

    async def execute(self, arguments: dict) -> str:
        return self._payload


class _FakeProvider:
    """只返回一条带 usage 的 assistant 消息，不触网"""

    def __init__(self, prompt: int = 800, completion: int = 30):
        self.prompt = prompt
        self.completion = completion
        self.calls = 0

    async def generate(self, messages, available_tools=None):
        self.calls += 1
        return Message(
            role=Role.ASSISTANT,
            content="ok",
            usage=Usage(prompt_tokens=self.prompt, completion_tokens=self.completion),
        )


# ─────────────────────────────────────────────────────────────
# 1. 派生指标
# ─────────────────────────────────────────────────────────────


def test_ratio_is_zero_when_estimate_is_zero():
    """估算为 0 时不能除零崩掉——语料里出现空调用是可能的"""
    assert _call(est=0, real_prompt=500).ratio == 0.0


def test_ratio_is_real_over_est():
    assert _call(est=1000, real_prompt=1860).ratio == 1.86


def test_calibration_ratio_uses_totals_not_mean_of_ratios():
    """🔴 关键口径：整体校准系数 = 总真实 ÷ 总估算，**不是**逐次比值的平均。

    小样本单次估值的噪声不该等权进入系数 —— 用总量算才反映真实开销。
    本例中两者差得很明显：均值 1.5 vs 总量比 1.9。
    """
    t = _trace()
    t.api_calls = [
        _call(est=100, real_prompt=100, seq=0),    # 单次比值 1.0
        _call(est=900, real_prompt=1800, seq=1),   # 单次比值 2.0
    ]
    assert t.calibration_ratio == 1.9   # (100+1800) / (100+900)
    assert t.calibration_ratio != 1.5   # 不是 (1.0 + 2.0) / 2


def test_calibration_ratio_safe_on_empty():
    assert _trace().calibration_ratio == 0.0


def test_peak_and_total_prompt_are_tracked():
    t = _trace()
    t.api_calls = [
        _call(est=100, real_prompt=300, seq=0),
        _call(est=200, real_prompt=3506, seq=1),
        _call(est=150, real_prompt=900, seq=2),
    ]
    assert t.total_real_prompt == 4706
    assert t.peak_real_prompt == 3506   # 峰值决定阈值该定多高


def test_peak_is_zero_when_no_calls():
    assert _trace().peak_real_prompt == 0


# ─────────────────────────────────────────────────────────────
# 2. 落盘：可入库、可读回
# ─────────────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path):
    t = _trace()
    t.api_calls = [_call(est=1000, real_prompt=1860, real_completion=42)]
    t.tool_outputs = [ToolOutputRecord(name="echo", chars=123, tokens=60, is_error=False, offloaded=False)]
    t.snapshots = [CompactSnapshot(seq=0, messages=[{"role": "user", "content": "hi"}], est_tokens=99)]
    t.passed = True
    t.total_turns = 4

    path = t.save(tmp_path / "trace.json")
    back = RunTrace.load(path)

    assert back.case_id == t.case_id
    assert back.model == t.model
    assert back.passed is True
    assert back.total_turns == 4
    assert back.api_calls[0].real_prompt == 1860
    assert back.api_calls[0].est_tokens == 1000
    assert back.tool_outputs[0].chars == 123
    assert back.snapshots[0].est_tokens == 99
    # 重算派生指标必须与原始一致
    assert back.calibration_ratio == t.calibration_ratio


def test_save_redacts_repo_and_home_paths(tmp_path):
    """🔴 语料要提交，必须脱敏。

    agent 的 bash 输出里常带 `pwd` / 绝对路径（ls、find 的结果），
    原样入库会把本机目录结构一起提交上去。
    """
    repo = str(Path.cwd())
    t = _trace()
    t.snapshots = [
        CompactSnapshot(
            seq=0,
            messages=[{"role": "user", "content": f"{repo}/src 下有 {repo}/tests"}],
            est_tokens=10,
        )
    ]
    t.tool_outputs = [
        ToolOutputRecord(name="bash", chars=10, tokens=5, is_error=False,
                         offloaded=False)
    ]
    # 另造一个非仓库的家目录路径，验证兜底规则
    raw_path = t.save(tmp_path / "t.json")
    raw = Path(raw_path).read_text(encoding="utf-8")

    assert repo not in raw          # 仓库绝对路径必须消失
    assert "<repo>" in raw          # 换成占位符
    assert "~/src" in raw or "<repo>/src" in raw

    # 兜底：任意 /Users/<name>/ 都要被压成 ~/
    t2 = _trace()
    t2.snapshots = [
        CompactSnapshot(seq=0, messages=[{"role": "user", "content": "/Users/someoneelse/secret/x.txt"}], est_tokens=5)
    ]
    raw2 = Path(t2.save(tmp_path / "t2.json")).read_text(encoding="utf-8")
    assert "/Users/someoneelse" not in raw2
    assert "~/secret/x.txt" in raw2


def test_saved_file_is_valid_json(tmp_path):
    """报告要能被别的工具（离线重放脚本）直接读，必须是合法 JSON"""
    Path(_trace().save(tmp_path / "t.json"))
    data = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert data["case_id"] == "case_x"


# ─────────────────────────────────────────────────────────────
# 3. 离线重放：语料能还原成生产对象
# ─────────────────────────────────────────────────────────────


def test_replay_messages_restores_full_message_semantics(tmp_path):
    """🔴 重放保真度：还原出的 Message 必须与当初录进去的**语义等价**。

    这里刻意覆盖 `is_system_reminder` —— 它是压缩器区分
    「系统提醒」与「用户原始指令」的唯一依据（后者永不淘汰）。
    如果重放时这个标记丢了，压缩器就会把提醒当成用户指令永久保留，
    重放出来的触发次数与降幅全是错的 —— 等于语料白录。
    """
    original = [
        Message(role=Role.USER, content="用户指令"),
        Message(role=Role.ASSISTANT, content="", reasoning="想一下",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"a": 1})]),
        Message(role=Role.USER, content="[SYSTEM REMINDER] 上下文快满了",
                tool_call_id="c1", is_system_reminder=True),
    ]
    t = _trace()
    t.snapshots = [
        CompactSnapshot(
            seq=0,
            messages=[__import__("dataclasses").asdict(m) for m in original],
            est_tokens=42,
        )
    ]
    path = t.save(tmp_path / "t.json")

    restored = RunTrace.load(path).replay_messages(0)

    assert len(restored) == len(original)
    assert restored[1].reasoning == "想一下"
    assert restored[1].tool_calls[0].name == "echo"
    assert restored[1].tool_calls[0].arguments == {"a": 1}
    assert restored[2].tool_call_id == "c1"
    # ← 就是这一条：标记丢了，重放结论就不可信
    assert restored[2].is_system_reminder is True
    assert restored[0].is_system_reminder is False


def test_replay_messages_unknown_seq_raises():
    try:
        _trace().replay_messages(99)
    except KeyError:
        return
    raise AssertionError("取不存在的快照必须显式报错，不能静默返回空列表")


# ─────────────────────────────────────────────────────────────
# 4. 插桩必须「只旁听」，不改变执行结果
# ─────────────────────────────────────────────────────────────


async def test_registry_proxy_records_output_size():
    t = _trace()
    inner = RegistryImpl()
    inner.register(_EchoTool(payload="x" * 123, name="echo"))

    proxy = InstrumentedRegistry(inner, t)
    await proxy.execute(ToolCall(id="1", name="echo", arguments={}))

    assert len(t.tool_outputs) == 1
    rec = t.tool_outputs[0]
    assert rec.name == "echo"
    assert rec.chars == 123
    assert rec.tokens > 0
    assert rec.is_error is False
    assert rec.offloaded is False


async def test_registry_proxy_is_transparent():
    """透传：返回值必须与不走代理完全一致，工具清单也要一致"""
    inner = RegistryImpl()
    inner.register(_EchoTool(payload="payload-abc", name="echo"))
    proxy = InstrumentedRegistry(inner, _trace())

    direct = await inner.execute(ToolCall(id="1", name="echo", arguments={}))
    through = await proxy.execute(ToolCall(id="1", name="echo", arguments={}))

    assert through.output == direct.output
    assert [d.name for d in proxy.get_available_tools()] == \
           [d.name for d in inner.get_available_tools()]


async def test_registry_proxy_flags_offloaded_and_error():
    """体积分布是定 READ_MAX_CHARS 的依据，卸载与否也要能统计出来"""
    t = _trace()
    inner = RegistryImpl()
    inner.register(_EchoTool(payload=f"内容已卸载到 {config.OFFLOAD_DIR}/x.log", name="read_file"))
    await InstrumentedRegistry(inner, t).execute(
        ToolCall(id="1", name="read_file", arguments={})
    )
    assert t.tool_outputs[0].offloaded is True


def test_compactor_proxy_records_snapshot_even_without_compaction():
    """未超阈值也必须记：重放要能模拟「第 N 轮换阈值 T 会怎样」，
    只记被压缩的那几轮是样本偏差"""
    t = _trace()
    proxy = InstrumentedCompactor(t, max_tokens=10_000, retain_last=6)
    msgs = [Message(role=Role.USER, content="短消息")]

    out = proxy.compact(msgs)

    assert len(t.snapshots) == 1
    assert t.snapshots[0].seq == 0
    assert t.snapshots[0].est_tokens > 0
    assert t.snapshots[0].messages[0]["content"] == "短消息"
    # 未超阈值 → 原样返回
    assert [m.content for m in out] == [m.content for m in msgs]


def test_compactor_proxy_does_not_mutate_input():
    """旁听不得改动调用方传入的列表（否则生产行为被插桩改变）"""
    t = _trace()
    proxy = InstrumentedCompactor(t, max_tokens=10_000, retain_last=6)
    msgs = [Message(role=Role.USER, content="a"), Message(role=Role.USER, content="b")]
    before = list(msgs)

    proxy.compact(msgs)

    assert msgs == before
    assert len(msgs) == 2


def test_compactor_proxy_sequence_increments():
    """seq 是重放的索引键，必须连续且从 0 起"""
    t = _trace()
    proxy = InstrumentedCompactor(t, max_tokens=10_000, retain_last=6)
    for _ in range(3):
        proxy.compact([Message(role=Role.USER, content="x")])
    assert [s.seq for s in t.snapshots] == [0, 1, 2]


async def test_tracker_records_estimate_versus_real():
    """🔴 校准系数的唯一来源：同一次调用上「本地估算」与「真实计费」的对照。

    两者必须用**同一个**估算函数（`context.tokens`），
    否则校准出来的系数对不上压缩逻辑 —— 等于白测。
    """
    t = _trace()
    fake = _FakeProvider(prompt=800, completion=30)
    tracker = InstrumentedTracker(fake, "deepseek-v4-flash", t)

    await tracker.generate([Message(role=Role.USER, content="hi")])

    assert fake.calls == 1                    # 确实透传给了下一层
    assert len(t.api_calls) == 1
    rec = t.api_calls[0]
    assert rec.seq == 0
    assert rec.real_prompt == 800
    assert rec.real_completion == 30
    assert rec.est_tokens > 0                 # 本地估算真的算了
    assert rec.n_messages == 1


async def test_tracker_is_transparent():
    """插桩不能吃掉返回值（否则引擎拿到 None 直接崩）"""
    t = _trace()
    resp = await InstrumentedTracker(_FakeProvider(), "deepseek-v4-flash", t).generate(
        [Message(role=Role.USER, content="hi")]
    )
    assert resp.role == Role.ASSISTANT
    assert resp.usage.prompt_tokens == 800


async def test_tracker_tolerates_missing_usage():
    """厂商不返回 usage 时不能崩，只能记 0（真实世界里会发生）"""

    class _NoUsage:
        async def generate(self, messages, available_tools=None):
            return Message(role=Role.ASSISTANT, content="ok")

    t = _trace()
    await InstrumentedTracker(_NoUsage(), "deepseek-v4-flash", t).generate(
        [Message(role=Role.USER, content="hi")]
    )
    assert t.api_calls[0].real_prompt == 0
    assert t.api_calls[0].ratio == 0.0
