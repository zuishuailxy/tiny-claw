"""MCP 客户端测试（重构后：连接管理交给官方 ClientSessionGroup）

**全部使用假 group**，不发真实请求、不起子进程 —— 单测不该依赖网络与 npx。
真实连通性由 `tools/probe_mcp.py`（Phase 0 探针）负责。

这里守住的四件事：
1. 配置加载**坏配置不能崩启动**（外部依赖不可靠是常态）
2. SDK 字段名坑：属性是 `input_schema`，协议名 `inputSchema` 只是 alias
3. 命名空间：MCP 工具与内置工具同名时不能静默覆盖（Phase 0 实测撞上）
4. 单例 + 幂等 + 清理：因为引擎会被反复重建（飞书每条消息一次）
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from tiny_claw.mcp.client import MCPManager, _name_hook, tool_input_schema
from tiny_claw.mcp.loader import ServerConfig, load_servers

# ─────────────────────────────────────────────────────────────
# 假对象
# ─────────────────────────────────────────────────────────────


@dataclass
class FakeTool:
    name: str
    description: str = ""
    input_schema: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class FakeGroup:
    """模拟 ClientSessionGroup 的相关行为（tools 聚合 + sessions + 连接失败）"""

    def __init__(self, tools_by_marker: dict | None = None, fail_markers=(), hook=None):
        self._tools = tools_by_marker or {}
        self._fail = set(fail_markers)
        # 可注入真实 hook：用来验证「命名空间 + 审批策略记录」是同一次调用完成的
        self._hook = hook or (lambda n, si: _name_hook(n, si))
        self.tools: dict[str, FakeTool] = {}
        self.sessions: dict[str, object] = {}
        self.connected: list[str] = []
        self.session_params: list[object] = []

    async def connect_to_server(self, params, session_params=None):
        # 用 args[0] 当可识别的 server 标记
        marker = params.args[0] if params.args else params.command
        if marker in self._fail:
            raise RuntimeError(f"连不上 {marker}")
        self.connected.append(marker)
        self.session_params.append(session_params)
        self.sessions[marker] = object()
        for tool in self._tools.get(marker, []):
            # 用真实的 hook 生成名字：命名规则变了测试要能发现
            self.tools[self._hook(tool, SimpleNamespace(name=marker))] = FakeTool(tool)


def cfg(name: str, **kw) -> ServerConfig:
    """构造一个把 name 塞进 args 的配置，便于 FakeGroup 识别"""
    return ServerConfig(name=name, command="npx", args=[name], **kw)


def write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────
# 1. 配置加载：坏配置绝不能崩启动
# ─────────────────────────────────────────────────────────────


def test_load_servers_parses_community_format(tmp_path):
    p = write_config(
        tmp_path,
        {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {"FOO": "bar"},
                    "approval": "never",
                    "timeout": 120,
                }
            }
        },
    )
    servers = load_servers(p)
    assert len(servers) == 1
    s = servers[0]
    assert (s.name, s.command) == ("filesystem", "npx")
    assert s.args[-1] == "/tmp"
    assert s.env == {"FOO": "bar"}
    assert s.approval == "never" and s.timeout == 120
    assert s.enabled is True


def test_missing_file_is_not_an_error(tmp_path):
    """没配 MCP 是完全正常的用法，不该报错"""
    assert load_servers(tmp_path / "nope.json") == []


def test_broken_json_does_not_raise(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text("{ this is not json", encoding="utf-8")
    assert load_servers(p) == []


def test_missing_mcpservers_key_does_not_raise(tmp_path):
    p = write_config(tmp_path, {"servers": {}})
    assert load_servers(p) == []


def test_one_bad_server_does_not_kill_the_others(tmp_path):
    """单条配置写错只跳过它自己"""
    p = write_config(
        tmp_path,
        {
            "mcpServers": {
                "broken": {"args": ["x"]},  # 缺 command
                "notobj": "oops",
                "good": {"command": "npx"},
            }
        },
    )
    assert [s.name for s in load_servers(p)] == ["good"]


def test_illegal_approval_falls_back_to_safest(tmp_path):
    """approval 写错时必须回退到最保守的 always，不能变成放行"""
    p = write_config(
        tmp_path, {"mcpServers": {"x": {"command": "c", "approval": "yolo"}}}
    )
    assert load_servers(p)[0].approval == "always"


def test_bad_args_and_env_are_coerced(tmp_path):
    p = write_config(
        tmp_path,
        {"mcpServers": {"x": {"command": "c", "args": "not-a-list", "env": [1]}}},
    )
    s = load_servers(p)[0]
    assert s.args == [] and s.env == {}


def test_merged_env_appends_extra_path(monkeypatch):
    """npx 不在默认 PATH 时会连不上，且报错只有 No such file or directory"""
    monkeypatch.setenv("PATH", "/usr/bin")
    merged = ServerConfig(name="x", command="npx", env={"A": "1"}).merged_env(
        "/opt/node/bin"
    )
    assert merged["A"] == "1"
    assert merged["PATH"].startswith("/opt/node/bin")
    assert "/usr/bin" in merged["PATH"]


# ─────────────────────────────────────────────────────────────
# 2. 命名空间：交给官方 hook（Phase 0 实测：与内置工具同名）
# ─────────────────────────────────────────────────────────────


def test_name_hook_isolates_from_builtin_names():
    # 实测：filesystem server 的工具就叫 read_file / write_file / edit_file
    got = _name_hook("read_file", SimpleNamespace(name="secure-filesystem-server"))
    assert got == "mcp__secure-filesystem-server__read_file"
    assert got != "read_file"  # 不会顶掉内置工具


def test_name_hook_tolerates_missing_server_name():
    assert _name_hook("t", SimpleNamespace()) == "mcp__unknown__t"


# ─────────────────────────────────────────────────────────────
# 3. SDK 字段名坑（input_schema vs inputSchema）
# ─────────────────────────────────────────────────────────────


def test_input_schema_prefers_sdk_v2_field():
    tool = FakeTool(name="t", input_schema={"type": "object", "properties": {"p": {}}})
    assert tool_input_schema(tool)["properties"] == {"p": {}}


def test_input_schema_falls_back_to_protocol_name():
    """老/新 SDK 只暴露协议名时也要能取到，不能 AttributeError"""

    class LegacyTool:
        inputSchema = {"type": "object", "properties": {"legacy": {}}}

    assert tool_input_schema(LegacyTool())["properties"] == {"legacy": {}}


def test_input_schema_defaults_when_absent():
    assert tool_input_schema(object()) == {"type": "object", "properties": {}}


# ─────────────────────────────────────────────────────────────
# 4. MCPManager：聚合 / 降级 / 幂等 / 清理
# ─────────────────────────────────────────────────────────────


async def test_preload_aggregates_namespaced_tools():
    group = FakeGroup({"fs": ["read_file", "list_directory"]})
    mgr = MCPManager(servers=[cfg("fs")])
    names = await mgr.preload(group=group)
    # preload 对名字排序：工具清单顺序确定，便于复现与比对
    assert names == ["mcp__fs__list_directory", "mcp__fs__read_file"]
    assert mgr.discovered == names
    assert group.session_params[0] is not None  # 超时参数传给了 SDK
    await mgr.aclose()


async def test_preload_skips_disabled_server():
    group = FakeGroup({"fs": ["a"]})
    mgr = MCPManager(servers=[cfg("fs", enabled=False)])
    assert await mgr.preload(group=group) == []
    assert group.connected == []
    await mgr.aclose()


async def test_one_server_failing_does_not_break_the_other():
    """外部依赖不可靠是常态：一个连不上，其余照常，且启动不失败"""
    group = FakeGroup({"good": ["t"]}, fail_markers={"bad"})
    mgr = MCPManager(servers=[cfg("bad"), cfg("good")])
    names = await mgr.preload(group=group)
    assert names == ["mcp__good__t"]
    await mgr.aclose()


async def test_connect_timeout_is_skipped_not_fatal():
    """连接超时只 warning，不能让 Agent 起不来"""

    class TimeoutGroup(FakeGroup):
        async def connect_to_server(self, params, session_params=None):
            raise TimeoutError

    mgr = MCPManager(servers=[cfg("slow")])
    assert await mgr.preload(group=TimeoutGroup()) == []
    await mgr.aclose()


async def test_preload_is_idempotent():
    """飞书模式每条消息都会重建引擎 → 注册可重复，连接只建立一次"""
    group = FakeGroup({"fs": ["a"]})
    mgr = MCPManager(servers=[cfg("fs")])
    first = await mgr.preload(group=group)
    second = await mgr.preload(group=group)
    assert first == second == ["mcp__fs__a"]
    assert group.connected == ["fs"]  # 没有重复连接
    await mgr.aclose()


async def test_lifespan_cleans_up_after_exit():
    group = FakeGroup({"fs": ["a"]})
    mgr = MCPManager(servers=[cfg("fs")])
    async with mgr.lifespan(group=group) as m:
        assert m.discovered == ["mcp__fs__a"]
    assert mgr.discovered == []
    assert mgr.group is None


async def test_tools_snapshot_exposes_raw_objects_for_adapter():
    """Phase 2 的 adapter 需要拿到原始 Tool 对象来读 input_schema"""
    group = FakeGroup({"fs": ["read_file"]})
    mgr = MCPManager(servers=[cfg("fs")])
    await mgr.preload(group=group)
    snap = mgr.tools_snapshot()
    assert list(snap) == ["mcp__fs__read_file"]
    assert tool_input_schema(snap["mcp__fs__read_file"])["type"] == "object"
    await mgr.aclose()


def test_no_config_means_no_servers(tmp_path):
    """没配 MCP 时应该是个空壳，而不是报错"""
    mgr = MCPManager(config_path=tmp_path / "absent.json")
    assert mgr.servers == []
    assert mgr.discovered == []
    assert mgr.tools_snapshot() == {}


def test_singleton_is_shared_and_resettable():
    MCPManager.reset_instance()
    a = MCPManager.instance()
    assert a is MCPManager.instance()
    MCPManager.reset_instance()
    assert MCPManager.instance() is not a


# ─────────────────────────────────────────────────────────────
# 7. 适配层：MCP Tool → BaseTool（A1）
# ─────────────────────────────────────────────────────────────


@dataclass
class FakeBlock:
    text: str | None = None
    type: str = "text"
    data: str | None = None


@dataclass
class FakeCallResult:
    content: list
    isError: bool = False


class FakeCallGroup:
    """只实现 call_tool 的最小假 group"""

    def __init__(self, result=None, exc=None, delay_exc=None):
        self._result = result
        self._exc = exc
        self._delay_exc = delay_exc
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._exc is not None:
            raise self._exc
        if self._delay_exc is not None:
            raise self._delay_exc
        return self._result


def make_adapter(result=None, exc=None, tool=None, delay_exc=None):
    from tiny_claw.mcp.adapter import MCPToolAdapter

    tool = tool or FakeTool("read_text_file", "读取文本", {"type": "object", "properties": {"path": {}}})
    group = FakeCallGroup(result=result, exc=exc, delay_exc=delay_exc)
    return MCPToolAdapter("mcp__srv__read_text_file", tool, group), group


def test_adapter_definition_passes_schema_through():
    """Phase 0 的结论：MCP inputSchema 与 ToolDefinition.input_schema 同构，零转换"""
    ad, _ = make_adapter()
    d = ad.definition()
    assert d.name == "mcp__srv__read_text_file"
    assert d.input_schema["properties"] == {"path": {}}
    assert d.description == "读取文本"


async def test_adapter_renders_text_blocks():
    ad, group = make_adapter(result=FakeCallResult([FakeBlock(text="line1"), FakeBlock(text="line2")]))
    out = await ad.execute({"path": "x"})
    assert out == "line1\nline2"
    assert group.calls == [("mcp__srv__read_text_file", {"path": "x"})]  # 全名透传给 SDK


async def test_adapter_omits_binary_blocks():
    """MCP 会返回 base64 图片/音频块，绝不能原样灌进上下文"""
    huge = "A" * 5000
    ad, _ = make_adapter(
        result=FakeCallResult([FakeBlock(text="ok"), FakeBlock(type="image", data=huge)])
    )
    out = await ad.execute({})
    assert "ok" in out
    assert "image 内容已省略" in out
    assert huge not in out  # base64 本体没进上下文


async def test_adapter_clips_long_output():
    ad, _ = make_adapter(result=FakeCallResult([FakeBlock(text="x" * 20000)]))
    out = await ad.execute({})
    assert len(out) < 20000
    assert "已截断" in out  # 明确告知被截断，模型才知道可以换更窄的调用


async def test_adapter_maps_server_error_to_error_code():
    ad, _ = make_adapter(result=FakeCallResult([FakeBlock(text="EISDIR: illegal operation")], isError=True))
    out = await ad.execute({})
    assert out.startswith("[ERR:MCP_TOOL_ERROR]")  # registry 靠这个前缀判定失败
    assert "EISDIR" in out


async def test_adapter_never_raises_on_exception():
    """失败是返回值，不是异常 —— 抛出去会打断 agent 循环"""
    ad, _ = make_adapter(exc=RuntimeError("connection lost"))
    out = await ad.execute({})
    assert out.startswith("[ERR:MCP_TOOL_FAILED]")
    assert "connection lost" in out


async def test_adapter_maps_timeout():
    ad, _ = make_adapter(delay_exc=TimeoutError())
    out = await ad.execute({})
    assert out.startswith("[ERR:MCP_TIMEOUT]")


async def test_adapter_empty_content_is_explicit():
    ad, _ = make_adapter(result=FakeCallResult([]))
    out = await ad.execute({})
    assert "没有返回文本内容" in out


# ─────────────────────────────────────────────────────────────
# 8. register_into：把 MCP 工具注册进 Registry（A2）
# ─────────────────────────────────────────────────────────────


class FakeRegistry:
    def __init__(self):
        self.tools: dict[str, object] = {}
        self.register_count = 0

    def register(self, tool):
        self.register_count += 1
        self.tools[tool.name()] = tool


async def test_register_into_registers_all_discovered_tools():
    group = FakeGroup({"fs": ["read_file", "list_directory"]})
    mgr = MCPManager(servers=[cfg("fs")])
    await mgr.preload(group=group)

    reg = FakeRegistry()
    n = mgr.register_into(reg)
    assert n == 2
    assert sorted(reg.tools) == ["mcp__fs__list_directory", "mcp__fs__read_file"]
    # 注册进去的必须是 BaseTool，且能给出模型要看的 schema
    d = reg.tools["mcp__fs__read_file"].definition()
    assert d.name == "mcp__fs__read_file" and "type" in d.input_schema
    await mgr.aclose()


async def test_register_into_without_preload_is_noop():
    mgr = MCPManager(servers=[])
    reg = FakeRegistry()
    assert mgr.register_into(reg) == 0


async def test_register_into_can_run_repeatedly():
    """飞书模式每条消息都新建引擎 → 注册会被反复调用"""
    group = FakeGroup({"fs": ["a"]})
    mgr = MCPManager(servers=[cfg("fs")])
    await mgr.preload(group=group)
    reg = FakeRegistry()
    assert mgr.register_into(reg) == 1
    assert mgr.register_into(reg) == 1
    # 名字不叠加（registry 是 dict，重名覆盖而不是追加）
    assert list(reg.tools) == ["mcp__fs__a"]
    await mgr.aclose()


# ─────────────────────────────────────────────────────────────
# 9. 审批策略：第三方工具默认最保守（B1 + B2）
# ─────────────────────────────────────────────────────────────


def test_server_of_handles_separator_inside_server_name():
    """不要用 split("__", 2)[1]：server 自报名里可能有 __ 会截错"""
    from tiny_claw.mcp.client import server_of

    assert server_of("mcp__secure-filesystem-server__read_file") == "secure-filesystem-server"
    assert server_of("mcp__a__b__tool") == "a__b"
    assert server_of("read_file") is None          # 内置工具不是 MCP
    assert server_of("mcp__nodelimiter") is None   # 缺分隔符


async def test_preload_records_approval_policy_by_self_reported_name():
    """审批策略要按 server **自报名**索引，因为命名空间用的就是自报名"""
    mgr = MCPManager(servers=[cfg("fs", approval="never")])
    group = FakeGroup({"fs": ["t"]}, hook=mgr._hook)
    await mgr.preload(group=group)
    assert mgr.approval_for("mcp__fs__t") == "never"
    await mgr.aclose()


def _seed_manager(mapping: dict[str, str]):
    """把一个带审批映射的单例塞进去（供 approve.py 查询）"""
    MCPManager.reset_instance()
    mgr = MCPManager(servers=[])
    mgr._approval_by_server.update(mapping)
    MCPManager._instance = mgr


def test_mcp_tool_defaults_to_hanging_when_policy_unknown():
    """查不到策略必须按最保守处理 —— 查不到不等于可信"""
    from tiny_claw.feishu.approve import is_dangerous_command

    MCPManager.reset_instance()
    assert is_dangerous_command("mcp__whatever__write_file", "{}") is True


def test_mcp_policy_always_hangs():
    from tiny_claw.feishu.approve import is_dangerous_command

    _seed_manager({"srv": "always"})
    assert is_dangerous_command("mcp__srv__list_directory", "{}") is True
    MCPManager.reset_instance()


def test_mcp_policy_never_allows():
    from tiny_claw.feishu.approve import is_dangerous_command

    _seed_manager({"srv": "never"})
    assert is_dangerous_command("mcp__srv__read_file", "{}") is False
    MCPManager.reset_instance()


def test_mcp_policy_dangerous_falls_back_to_pattern_match():
    from tiny_claw.feishu.approve import is_dangerous_command

    _seed_manager({"srv": "dangerous"})
    assert is_dangerous_command("mcp__srv__run", '{"cmd": "rm -rf /tmp/x"}') is True
    assert is_dangerous_command("mcp__srv__run", '{"cmd": "ls -la"}') is False
    MCPManager.reset_instance()


def test_builtin_rules_unchanged_by_mcp_branch():
    """回归：MCP 分支不能改变内置工具的既有判定"""
    from tiny_claw.feishu.approve import is_dangerous_command

    assert is_dangerous_command("read_file", "{}") is False
    assert is_dangerous_command("write_file", "{}") is True
    assert is_dangerous_command("edit_file", "{}") is True
    assert is_dangerous_command("bash", '{"command": "rm -rf /"}') is True
    assert is_dangerous_command("bash", '{"command": "ls -la"}') is False


# ─────────────────────────────────────────────────────────────
# 10. 失败可见性：失败必须被记下来（plan §13 C1）
# ─────────────────────────────────────────────────────────────


async def test_preload_records_failed_server_with_reason():
    """连接失败不能只留一条 WARNING —— 要记下 (server, 原因)，让上层能播报具体原因。

    修之前的状态：npx 卡住 → 超时 236s → 只打一条 WARNING → agent 带着 0 个 MCP
    工具照常启动，用户与模型都不知道这批能力没接上。
    """
    group = FakeGroup({"good": ["t"]}, fail_markers=("bad",))
    mgr = MCPManager(servers=[cfg("bad"), cfg("good")])
    await mgr.preload(group=group)

    assert [n for n, _ in mgr.failures] == ["bad"]
    assert mgr.failures[0][1]          # 原因必须非空
    assert mgr.discovered              # 好的那个照常接入，不被拖累
    await mgr.aclose()


async def test_failures_empty_when_all_ok():
    group = FakeGroup({"fs": ["a"]})
    mgr = MCPManager(servers=[cfg("fs")])
    await mgr.preload(group=group)
    assert mgr.failures == []
    await mgr.aclose()


async def test_failures_cleared_on_aclose():
    group = FakeGroup({}, fail_markers=("bad",))
    mgr = MCPManager(servers=[cfg("bad")])
    await mgr.preload(group=group)
    assert mgr.failures
    await mgr.aclose()
    assert mgr.failures == []


async def test_connect_timeout_recorded_as_failure():
    """超时也是失败，同样要有记录与原因 —— 不是静默跳过"""

    class SlowGroup(FakeGroup):
        async def connect_to_server(self, params, session_params=None):
            await asyncio.sleep(30)

    mgr = MCPManager(servers=[cfg("slow", timeout=1)])
    await mgr.preload(group=SlowGroup())

    assert len(mgr.failures) == 1
    name, reason = mgr.failures[0]
    assert name == "slow"
    assert "超时" in reason
    await mgr.aclose()


# ─────────────────────────────────────────────────────────────
# 7. 子进程 stderr 路由：server 的自言自语不进用户终端
#
# 背景：SDK 的 `stdio_client(server, errlog=sys.stderr)` 是「定义时求值」的默认参数，
# 子进程 stderr 直接接到父进程 fd 2。实测后果是两个 server 的 banner 与告警
# 和 tiny-claw 自己的日志混在一起，且随 server 数量线性增长。
# ─────────────────────────────────────────────────────────────


def test_stderr_sink_writes_to_work_dir_log(tmp_path):
    """正常路径：日志落在 <work_dir>/.claw/mcp-logs/<server>.log"""
    from tiny_claw.mcp.client import _stderr_sink

    sink = _stderr_sink(tmp_path, "filesystem")
    try:
        sink.write("Client does not support MCP Roots\n")
        sink.flush()
        log = tmp_path / ".claw" / "mcp-logs" / "filesystem.log"
        assert log.is_file()
        assert "MCP Roots" in log.read_text(encoding="utf-8")
    finally:
        sink.close()


def test_stderr_sink_sanitizes_server_name(tmp_path):
    """server 名里的 / 与 .. 不能影响落盘位置（否则可能写到工作区外）"""
    from tiny_claw.mcp.client import _stderr_sink

    sink = _stderr_sink(tmp_path, "../../evil/name")
    try:
        sink.write("x")
        files = list((tmp_path / ".claw" / "mcp-logs").iterdir())
        assert len(files) == 1
        assert files[0].name == ".._.._evil_name.log"
        # 关键：没有跑到上级目录去
        assert not (tmp_path.parent / "evil").exists()
    finally:
        sink.close()


def test_stderr_sink_degrades_to_devnull_when_unwritable(tmp_path):
    """工作区不可写时降级为丢弃，**不能让 MCP 整体连不上**

    注入手法：把 `.claw` 位置先占成一个**文件**，`mkdir(parents=True)` 必然抛
    FileExistsError（OSError 子类），正好走到降级分支。不去 patch 全局的
    `Path.mkdir`——那会影响别的用例，属于测试之间互相污染。
    """
    from tiny_claw.mcp.client import _stderr_sink

    (tmp_path / ".claw").write_text("占位，不是目录", encoding="utf-8")
    sink = _stderr_sink(tmp_path, "fs")
    try:
        sink.write("noise")  # 不抛异常即可
        assert sink.name == os.devnull
    finally:
        sink.close()


async def test_preload_opens_stderr_log_before_connecting(tmp_path):
    """接线验证：preload 在**建连之前**就把该 server 的 stderr 日志接好了

    为什么必须"之前"：子进程是在 connect_to_server 那一刻被 fork 出来的，
    晚一步接就等于第一段输出仍然漏到终端。
    """
    from tiny_claw.mcp.client import _StderrRoutingGroup

    class SinkRecordingGroup(_StderrRoutingGroup):
        """继承真实类：`preload` 的 isinstance 判定必须命中，接线才真的被测到。

        只覆写建连行为（不启子进程），并记录"建连那一刻 errlog 是什么"。
        `set_errlog` 不覆写——它就是被测代码。
        """

        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.sink_at_connect = []

        async def connect_to_server(self, params, session_params=None):
            self.sink_at_connect.append(self._errlog)
            self.tools["mcp__fs__t"] = FakeTool("t")

    group = SinkRecordingGroup()
    mgr = MCPManager(servers=[cfg("fs")], work_dir=tmp_path)
    await mgr.preload(group=group)

    assert len(group.sink_at_connect) == 1
    assert group.sink_at_connect[0] is not None, "建连时 errlog 还没接上"
    log = tmp_path / ".claw" / "mcp-logs" / "fs.log"
    assert log.is_file()
    await mgr.aclose()
    # 连接关闭后句柄必须回收，避免句柄泄漏（飞书模式会反复 aclose）
    assert mgr._errlog_files == []


def test_instance_accepts_work_dir_on_first_call(tmp_path):
    """单例首次创建时能带上 work_dir；已有实例时参数被忽略（连接是进程级的）"""
    MCPManager.reset_instance()
    try:
        a = MCPManager.instance(servers=[], work_dir=tmp_path)
        assert a._work_dir == tmp_path
        b = MCPManager.instance(servers=[], work_dir="/somewhere/else")
        assert b is a, "已有单例不应被重建"
    finally:
        MCPManager.reset_instance()


def test_cli_drops_mcp_capability_noise_only():
    """过滤器只压 prompts/resources 两条噪音，**必须放行 Could not fetch tools**"""
    from tiny_claw.cli import _DropMcpCapabilityNoise

    f = _DropMcpCapabilityNoise()

    def rec(msg):
        return logging.LogRecord("root", logging.WARNING, __file__, 1, msg, None, None)

    assert not f.filter(rec("Could not fetch prompts: Method not found"))
    assert not f.filter(rec("Could not fetch resources: Method not found"))
    assert f.filter(rec("Could not fetch tools: Method not found")), "工具发现失败必须放行"
    assert f.filter(rec("其他日志"))


# ─────────────────────────────────────────────────────────────
# 8. MCP 日志落点：没给 work_dir 时的 fallback
#
# 这里的坑是「静默落错地方」：原写法 `Path(config_path or MCP_CONFIG_PATH).parent or "."`
# 看似有 fallback，但默认配置路径 "mcp.json" 是相对路径，`.parent` 得到 `Path(".")`
# 而非空值，于是 `or "."` 永远轮不到 —— fallback 链是断的。
# ─────────────────────────────────────────────────────────────


def test_mcp_log_root_defaults_to_cwd_when_no_config_path():
    """没给配置路径 → 落当前目录（而不是抛错、也不是落到某个意外位置）"""
    from tiny_claw.mcp.client import _default_mcp_log_root

    assert _default_mcp_log_root(None) == Path(".")


def test_mcp_log_root_ignores_relative_config_path():
    """相对路径 "mcp.json" 的父目录是 "."，不能因此把落点判成"已指定" """
    from tiny_claw.mcp.client import _default_mcp_log_root

    assert _default_mcp_log_root("mcp.json") == Path(".")


def test_mcp_log_root_uses_absolute_config_dir():
    """绝对路径配置 → 落在配置文件所在目录（其次是 work_dir 优先）"""
    from tiny_claw.mcp.client import _default_mcp_log_root

    assert _default_mcp_log_root("/etc/tiny/mcp.json") == Path("/etc/tiny")


def test_explicit_work_dir_wins_over_config_path():
    """显式给 work_dir 时，一切以它为准（`-d` 必须说话算数）"""
    import tiny_claw.mcp.client as client_mod

    mgr = MCPManager(servers=[], work_dir="/tmp/some-workspace")
    assert mgr._work_dir == Path("/tmp/some-workspace")
