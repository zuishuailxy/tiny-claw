"""MCP 连接管理 — 官方 `ClientSessionGroup` 的薄封装

**边界划清楚**（甄别标准见 `docs/MCP_INTEGRATION_PLAN.md` §12.9）：

| 谁负责 | 内容 |
|---|---|
| **官方 SDK** | 协议（JSON-RPC / stdio）、握手与发现、**多 server 生命周期**、**工具聚合**、**命名空间**、**按命名空间路由调用**、超时/进度/MRTR |
| **本项目** | 配置从哪来 + 坏配置降级、`Tool` → `BaseTool` 适配（Phase 2）、MCP 错误 → `[ERR:MCP_*]`、审批策略、以及「同步 `build_engine()` + 飞书每条消息新建引擎」的接缝 |

为什么不自写连接管理：SDK 的 `ClientSessionGroup` 已经做好且做得更多（实测见 plan §12.9）。
自己再写一遍 = 重复劳动 + 白白少掉 MRTR / 超时这些内建能力。

⚠️ **task 归属**：SDK 基于 anyio task group，`enter` 与 `exit` 必须在同一个 asyncio task 内。
所以本模块只暴露 `lifespan()`（同 task 成对）与 `aclose()`，不要跨 task 调用。

📁 **子进程 stderr**：server 的 banner 与告警默认会打到用户终端（SDK 的
`stdio_client` 默认 `errlog=sys.stderr`），实测两个 server 就能把 CLI 输出冲乱。
现由 `_StderrRoutingGroup` 引到 `<work_dir>/.claw/mcp-logs/<server>.log`。
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar

from tiny_claw import config
from tiny_claw.mcp.loader import ServerConfig, load_servers

# `mcp` 在 pyproject 里是**必装**依赖（`dependencies` 里的 `mcp>=2.0`），
# 所以这里直接顶层 import，不再逐个函数延迟 import —— 覆写 SDK 私有方法需要真类，
# 而 `_StderrRoutingGroup(ClientSessionGroup)` 这个继承关系在模块顶层就得成立。
#
# ⚠️ 别再把它退回 optional extra：CLI 顶层 import 本模块，SDK 又被继承依赖，
# 延迟加载做不到。退回 extra 的结果是 `pip install tiny-claw` 装完 `--help` 都起不来。
import mcp  # noqa: E402
from mcp.client.session_group import ClientSessionGroup  # noqa: E402

logger = logging.getLogger("tiny-claw.mcp.client")

# 子进程 stderr 的落盘目录（位于工作区内，与 Offload 同级约定）
MCP_LOG_DIR = Path(".claw/mcp-logs")


def _stderr_sink(work_dir: Path, name: str) -> Any:
    """给一个 server 子进程挑 stderr 的去处：日志文件，选不出来就丢进 /dev/null。

    **为什么必须换掉默认值**：SDK 的 `stdio_client(server, errlog=sys.stderr)`
    是「定义时求值」的默认参数，子进程 stderr 被接到**父进程的 fd 2** 上。后果是
    server 的 banner 与告警（如 filesystem server 的 "Client does not support
    MCP Roots"）直接打到用户终端，和 tiny-claw 自己的日志混在一起；server 越多越吵，
    最后「真出事的 WARNING」也淹没在噪音里、被当成常态忽略。

    为什么不做 fd 级重定向：`sys.stderr` 与 `sys.__stderr__` 是同一个对象，
    logging 的 StreamHandler 也绑着它——重定向 fd 2 会把项目自己的日志一起吞掉。

    日志文件**每次启动截断**（"w"）：这是排障直觉——出问题看的是本次运行，
    而不是把历次运行拼成一个大文件。
    """
    try:
        log_dir = Path(work_dir) / MCP_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        # 文件名只做保守替换，避免 server 名里的 / 或 .. 影响落盘位置
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name) or "server"
        return open(log_dir / f"{safe}.log", "w", encoding="utf-8", errors="replace")
    except OSError as e:
        # 工作区不可写（只读挂载等）**不能让 MCP 整体连不上**：降级为丢弃，
        # 连接本身远比看得见 server 的自言自语重要。
        logger.warning("[MCP] server '%s' 的 stderr 日志不可写，改写 /dev/null: %s", name, e)
        return open(os.devnull, "w", encoding="utf-8")


class _StderrRoutingGroup(ClientSessionGroup):
    """`ClientSessionGroup` + 把子进程 stderr 引到日志文件。

    为什么是覆写而不是自己写连接管理：SDK 的 `_establish_session` 里除了 stdio 建连，
    还有 SSE / streamable-http 两条分支、初始化、以及 session 栈的登记——整段抄过来
    等于把「用官方 SDK」这个前提作废。这里**只改「stderr 去哪儿」**，其余原样交给 SDK。

    ⚠️ 覆写的是私有方法。SDK 升级后若签名漂移，`preload()` 会捕获异常、把原因记进
    `failures` 并在日志里显式播报（`client.py` 的 logger.error），**不会静默退化**。
    """

    def __init__(self, component_name_hook: Any = None, errlog: Any = None) -> None:
        super().__init__(component_name_hook=component_name_hook)
        self._errlog = errlog

    def set_errlog(self, errlog: Any) -> None:
        """切换下一个 stdio 子进程的 stderr 去向。

        `connect_to_server` 是**逐个串行**调用的（`preload` 的 for 循环里 await），
        所以这里用一个可变字段在两次建连之间切换是安全的，不必每个 server 造一个 group。
        """
        self._errlog = errlog

    async def _establish_session(self, server_params: Any, session_params: Any) -> Any:
        # ⚠️ 这些名字的来处与 SDK 内部一致，照抄不要凭直觉改：
        # `SseServerParameters` / `StreamableHttpParameters` 定义在 session_group 自己里，
        # 不在 `mcp.client.sse` / `mcp.client.streamable_http`——导错就整条链 ImportError，
        # 而 `preload` 会把它记成"接入失败"，表现为所有 server 都连不上。
        from mcp.client.session_group import (
            SseServerParameters,
            StreamableHttpParameters,
        )
        from mcp.client.sse import sse_client
        from mcp.client.stdio import StdioServerParameters
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        session_stack = AsyncExitStack()
        try:
            # ── 与 SDK 唯一的差别：stdio 分支带上 errlog ──
            if isinstance(server_params, StdioServerParameters):
                client = mcp.stdio_client(server_params, errlog=self._errlog)
                read, write = await session_stack.enter_async_context(client)
            elif isinstance(server_params, SseServerParameters):
                client = sse_client(
                    url=server_params.url,
                    headers=server_params.headers,
                    timeout=server_params.timeout,
                    sse_read_timeout=server_params.sse_read_timeout,
                )
                read, write = await session_stack.enter_async_context(client)
            else:
                assert isinstance(server_params, StreamableHttpParameters)
                httpx_client = create_mcp_http_client(
                    headers=server_params.headers,
                )
                await session_stack.enter_async_context(httpx_client)

                client = streamable_http_client(
                    url=server_params.url,
                    http_client=httpx_client,
                    terminate_on_close=server_params.terminate_on_close,
                )
                read, write = await session_stack.enter_async_context(client)

            session = await session_stack.enter_async_context(
                mcp.ClientSession(
                    read,
                    write,
                    read_timeout_seconds=session_params.read_timeout_seconds,
                    sampling_callback=session_params.sampling_callback,
                    elicitation_callback=session_params.elicitation_callback,
                    list_roots_callback=session_params.list_roots_callback,
                    logging_callback=session_params.logging_callback,
                    message_handler=session_params.message_handler,
                    client_info=session_params.client_info,
                )
            )

            result = await session.initialize()
            # 以下是 SDK 的 session 栈登记方式，照抄以免生命周期跑偏
            self._session_exit_stacks[session] = session_stack
            await self._exit_stack.enter_async_context(session_stack)
            return result.server_info, session
        except Exception:
            await session_stack.aclose()
            raise


def tool_input_schema(tool: Any) -> dict:
    """取 MCP tool 的入参 schema。

    ⚠️ 字段名坑（Phase 0 实测）：MCP **线协议**上的字段叫 `inputSchema`，
    但 Python SDK v2 的模型属性是 **`input_schema`**（`inputSchema` 只是 pydantic
    alias），照协议名写会直接 `AttributeError: 'Tool' object has no attribute
    'inputSchema'`。这里两个都试，SDK 再改名也不至于整条链断掉。
    """
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return schema or {"type": "object", "properties": {}}


def _default_mcp_log_root(config_path: str | Path | None) -> Path:
    """没给 work_dir 时，MCP 日志往哪放：配置文件所在目录，否则当前目录。

    抽成函数是为了能单独测这条 fallback —— 它此前是坏的（见 `MCPManager.__init__`
    里的注释），而坏的形态是**静默落错地方**，不留神根本发现不了。
    """
    if config_path:
        parent = Path(config_path).parent
        if str(parent) not in ("", "."):
            return parent
    return Path(".")


def _name_hook(name: str, server_info: Any) -> str:
    """命名空间：`mcp__{server}__{tool}`。

    为什么必须有：Phase 0 实测 filesystem server 的工具叫
    `read_file` / `write_file` / `edit_file`，与本项目内置工具**完全同名**，
    而 `registry.register()` 遇重名是 warning 后**静默覆盖** —— 内置工具会凭空消失。

    为什么用 `server_info.name`（自报名）而不是配置里的 key：官方 hook 只传
    `(tool_name, server_info)`，**拿不到我们的配置 key**。自报名的代价是名字更长
    （`mcp__secure-filesystem-server__read_file`），但更准确、不依赖用户怎么命名。
    """
    server = getattr(server_info, "name", None) or "unknown"
    return f"mcp__{server}__{name}"


def _to_stdio_params(cfg: ServerConfig) -> Any:
    """ServerConfig → SDK 的 StdioServerParameters（含 PATH 补齐）"""
    from mcp import StdioServerParameters

    return StdioServerParameters(
        command=cfg.command,
        args=cfg.args,
        env=cfg.merged_env(config.MCP_EXTRA_PATH),
    )


def server_of(namespaced_name: str) -> str | None:
    """从工具全名反解 server 名。

    ⚠️ 不要写 `name.split("__", 2)[1]` —— server 自报名里可能含 `__`，
    那样会截错。用 `removeprefix` + `rsplit` 才稳。
    """
    prefix = "mcp__"
    if not namespaced_name.startswith(prefix):
        return None
    rest = namespaced_name[len(prefix):]
    if "__" not in rest:
        return None
    return rest.rsplit("__", 1)[0]


def _new_session_params(timeout: float | None) -> Any:
    from mcp.client.session_group import ClientSessionParameters

    return ClientSessionParameters(read_timeout_seconds=timeout)


class MCPManager:
    """MCP 连接的进程级持有者（单例）

    **为什么连接不挂在引擎上**（plan §3 矛盾 A）：`cli.py:43` 的 `build_engine()`
    是同步函数，而连接是异步且昂贵的（要拉子进程）；飞书模式还**每条消息都新建引擎**。
    → 连接提为进程级单例，`preload()` 只跑一次；引擎每次重建只做零成本的注册。

    用法（CLI 模式，同 task 生命周期）：

        async with MCPManager.instance(work_dir=work_dir).lifespan() as mgr:
            mgr.register_into(engine.registry)      # Phase 2
            await engine.run(session)

    `work_dir` 只影响子进程 stderr 日志落在哪（见 `_stderr_sink`），不影响连接本身。
    """

    _instance: ClassVar["MCPManager | None"] = None

    @classmethod
    def instance(cls, **kwargs: Any) -> "MCPManager":
        """取进程级单例；首次调用时可传构造参数（如 `work_dir`）。

        已有单例时参数被忽略——连接是进程级的，重建会白白重启子进程。
        """
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """测试用：丢弃单例"""
        cls._instance = None

    def __init__(self, config_path: str | Path | None = None, servers: list[ServerConfig] | None = None,
                 work_dir: str | Path | None = None):
        if servers is None and config.MCP_ENABLED:
            servers = load_servers(config_path or config.MCP_CONFIG_PATH)
        self._servers: list[ServerConfig] = servers or []
        # MCP 日志落盘位置：优先调用方给的工作区（CLI 传的就是 agent 的 work_dir）。
        # ⚠️ 不能写成 `Path(config_path or config.MCP_CONFIG_PATH).parent or "."`：
        # 默认 `MCP_CONFIG_PATH = "mcp.json"` 是**相对路径**，它的 `.parent` 得到
        # `Path(".")`（不是空值！），于是 `or "."` 永远轮不到，落点静默变成**进程 CWD**
        # ——只有「完全没给配置路径」时才会命中 fallback，而那恰恰是最常见的情况。
        self._work_dir = Path(work_dir) if work_dir else _default_mcp_log_root(config_path)
        self._stack = AsyncExitStack()
        self._group: Any | None = None
        self._names: list[str] = []
        self._preloaded = False
        # server 自报名 → 审批策略。为什么要把这个映射记下来：配置里的 key（用户随便写）
        # 与 server 自报名（如 secure-filesystem-server）不是一个东西，而命名空间用的是
        # 自报名，所以审批策略必须按自报名索引才查得到。
        self._approval_by_server: dict[str, str] = {}
        self._connecting: ServerConfig | None = None
        # 子进程 stderr 的落盘句柄（key = server 名）。连接期间必须保持打开，
        # 否则子进程往一个已关闭的 fd 写会直接 EBADF 崩掉。
        self._errlog_files: list[Any] = []
        # 接入失败的 server 与原因。为什么要**记下来**：此前失败只打一条 WARNING 就
        # 过去了，agent 带着 0 个 MCP 工具照常启动，用户与模型都不知道这批能力没接上
        # （plan §13 C1：失败不可见属运维级缺陷）。
        self._failures: list[tuple[str, str]] = []

    def _hook(self, name: str, server_info: Any) -> str:
        """命名空间 hook + 顺带记录"这个 server 的审批策略"

        官方 hook 只给 `(tool_name, server_info)`，拿不到我们的配置 key。
        但 hook 是**在 connect_to_server(cfg) 期间**被调用的，所以用
        `_connecting` 把"当前正在连的配置"对上——这是唯一能关联两者的时机。
        """
        server = getattr(server_info, "name", None) or "unknown"
        if self._connecting is not None:
            self._approval_by_server.setdefault(server, self._connecting.approval)
        return _name_hook(name, server_info)

    def _new_group(self) -> Any:
        """构造带 stderr 路由的 group（见 `_StderrRoutingGroup`）"""
        return _StderrRoutingGroup(component_name_hook=self._hook)

    # ------------------------------------------------------------------
    # 只读视图
    # ------------------------------------------------------------------
    @property
    def servers(self) -> list[ServerConfig]:
        return list(self._servers)

    @property
    def group(self) -> Any | None:
        """底层 ClientSessionGroup（Phase 2 的 adapter 与测试消费）"""
        return self._group

    @property
    def discovered(self) -> list[str]:
        """聚合到的工具全名（已带命名空间）"""
        return list(self._names)

    @property
    def failures(self) -> list[tuple[str, str]]:
        """接入失败的 server：`[(server 名, 原因)]`。空列表 = 全部成功。

        调用方（CLI / 飞书）用它给出**具体原因**，而不是只说一句"没有工具可用"。
        """
        return list(self._failures)

    def tools_snapshot(self) -> dict[str, Any]:
        """工具全名 → 原始 MCP `Tool` 对象（Phase 2 的 adapter 消费）"""
        if self._group is None:
            return {}
        return dict(self._group.tools)

    # ------------------------------------------------------------------
    # 预发现（异步，只跑一次）
    # ------------------------------------------------------------------
    async def preload(self, group: Any | None = None) -> list[str]:
        """接入所有启用的 server，返回聚合到的工具全名。

        **任何一个 server 失败都不影响其他 server，也不阻塞启动** ——
        外部依赖不可靠是常态（npx 下载失败、路径写错、server 崩了），
        不能让 Agent 起不来。

        Args:
            group: 供测试注入的假 group（不传则用官方 `ClientSessionGroup`）
        """
        if self._preloaded:
            return list(self._names)

        if group is not None:
            self._group = group
        else:
            # 交给自己的 exit_stack 管生命周期，保证 enter/exit 同 task
            self._group = await self._stack.enter_async_context(self._new_group())

        self._failures = []
        for cfg in self._servers:
            if not cfg.enabled:
                logger.info("[MCP] server '%s' 已禁用，跳过", cfg.name)
                continue
            timeout = cfg.timeout or config.MCP_CONNECT_TIMEOUT
            self._connecting = cfg  # ← hook 靠它把 server 自报名关联到配置
            # stderr 去哪：每个 server 一个日志文件（见 _stderr_sink）。
            # 必须在 connect_to_server **之前**装好——子进程是在那一刻 fork 出来的。
            if isinstance(self._group, _StderrRoutingGroup):
                sink = _stderr_sink(self._work_dir, cfg.name)
                self._errlog_files.append(sink)
                self._group.set_errlog(sink)
            try:
                async with asyncio.timeout(timeout):
                    await self._group.connect_to_server(
                        _to_stdio_params(cfg),
                        _new_session_params(config.MCP_CALL_TIMEOUT),
                    )
            except TimeoutError:
                reason = f"连接超时（{timeout}s）"
                logger.warning("[MCP] server '%s' %s，已跳过", cfg.name, reason)
                self._failures.append((cfg.name, reason))
                continue
            except Exception as e:  # noqa: BLE001 —— 外部依赖，失败即降级
                reason = f"{type(e).__name__}: {e}"
                logger.warning("[MCP] server '%s' 接入失败，已跳过: %s", cfg.name, reason)
                self._failures.append((cfg.name, reason))
                continue
            finally:
                self._connecting = None
            logger.info("[MCP] server '%s' 已接入", cfg.name)

        self._names = sorted(self._group.tools.keys())
        self._preloaded = True

        # 失败必须**显式**播报：只留一条 WARNING 的后果是「agent 照常启动、
        # 只是静默少了这批能力」，没人会注意到（plan §13 C1）。
        if self._failures:
            logger.error(
                "[MCP] ⚠️ %d/%d 个 server 接入失败：%s —— Agent 将不带这些工具启动",
                len(self._failures),
                len(self._servers),
                "；".join(f"{n}（{r}）" for n, r in self._failures),
            )
        logger.info(
            "[MCP] 就绪：%d 个 server / %d 个工具",
            len(getattr(self._group, "sessions", {}) or {}),
            len(self._names),
        )
        return list(self._names)

    # ------------------------------------------------------------------
    # 注册进 Registry（同步，可重复调用）
    # ------------------------------------------------------------------
    def register_into(self, registry: Any) -> int:
        """把已发现的 MCP 工具注册进 Registry，返回注册数量。

        **同步**是刻意的：`build_engine()` 是同步函数，飞书模式还每条消息都新建引擎。
        所以拆成「异步 preload 一次 + 同步注册 N 次」，这里只读缓存好的 `group.tools`。

        幂等：registry 遇重名是"覆盖 + warning"，重复调用不会重复累积。
        """
        if self._group is None:
            return 0

        # 延迟 import：避免 client ⇄ adapter 的模块级循环依赖
        from tiny_claw.mcp.adapter import MCPToolAdapter

        count = 0
        for full_name, tool in self._group.tools.items():
            registry.register(MCPToolAdapter(full_name, tool, self._group))
            count += 1
        if count:
            logger.info("[MCP] 已注册 %d 个 MCP 工具进 Registry", count)
        return count

    def approval_for(self, namespaced_name: str) -> str | None:
        """按工具全名查它所属 server 的审批策略（查不到返回 None）

        由 `feishu/approve.py` 消费。返回 None 时调用方应按**最保守**处理——
        第三方能力不该因为"查不到策略"而默认放行。
        """
        server = server_of(namespaced_name)
        if server is None:
            return None
        return self._approval_by_server.get(server)

    @classmethod
    def current(cls) -> "MCPManager | None":
        """已存在的单例，**没有就返回 None**（不像 instance() 会顺带建一个）"""
        return cls._instance

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def lifespan(self, group: Any | None = None) -> AsyncIterator["MCPManager"]:
        """CLI 模式：preload 与 aclose 在**同一个 task** 内成对出现"""
        await self.preload(group)
        try:
            yield self
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """关闭全部连接并回收子进程"""
        try:
            await self._stack.aclose()
        except Exception as e:  # noqa: BLE001 —— 关闭失败不应掩盖主流程
            logger.warning("[MCP] 关闭连接时出错: %s", e)
        # stderr 日志句柄：连接断开后再关，避免子进程还活着时句柄先没了
        for f in self._errlog_files:
            try:
                f.close()
            except Exception:  # noqa: BLE001 —— 关日志失败不该影响主流程
                pass
        self._errlog_files = []
        self._group = None
        self._names = []
        self._failures = []
        self._preloaded = False
