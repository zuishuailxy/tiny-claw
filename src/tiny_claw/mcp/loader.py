"""MCP 配置加载 — 解析社区 `mcpServers` 格式

设计原则：**坏配置不能让进程起不来**。
MCP server 是外部依赖（要 npx、要网络、路径还可能是别人机器上的），
任何一个写错都不该拖垮整个 Agent 的启动 —— 所以这里全部降级为 warning，
返回「能用的那一部分」，把决定权交给调用方。

配置沿用社区格式（Claude Desktop / Cursor 同款），这样用户可以直接复用
已有的 `mcp.json`；本项目只追加了几个可选扩展字段，对别的工具无害：

    enabled   bool   是否连接（默认 true）
    approval  str    "always" | "never" | "dangerous"（默认 always，最保守）
    timeout   int    覆盖全局调用超时（秒）
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("tiny-claw.mcp.loader")

_VALID_APPROVAL = {"always", "never", "dangerous"}


@dataclass(frozen=True)
class ServerConfig:
    """一个 MCP server 的连接配置"""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    # 审批策略：第三方能力默认最保守（always = 每次挂起等人工确认）
    approval: str = "always"
    timeout: int | None = None

    def merged_env(self, extra_path: str = "") -> dict[str, str]:
        """构造子进程环境：继承当前进程 → 叠加配置里的 env → 补 PATH。

        补 PATH 是必需的：`npx` 这类命令常不在默认 PATH 里，
        而失败时 SDK 只报 "No such file or directory"，极难排查。
        """
        merged = dict(os.environ)
        merged.update(self.env)
        if extra_path:
            merged["PATH"] = os.pathsep.join(
                [p for p in extra_path.split(os.pathsep) if p] + [merged.get("PATH", "")]
            )
        return merged


def load_servers(
    path: str | Path | None = None, *, default_path: str | Path = "mcp.json"
) -> list[ServerConfig]:
    """读取配置文件，返回**可用的** server 列表（坏的跳过并告警）。

    永远不抛异常：文件不存在、JSON 语法错、单条配置缺字段，都只 warning。
    """
    target = Path(path or default_path)
    if not target.is_file():
        logger.info("[MCP] 未找到配置文件 %s，跳过 MCP 能力", target)
        return []

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[MCP] 配置文件解析失败，已跳过全部 MCP server: %s", e)
        return []

    servers_raw = raw.get("mcpServers")
    if not isinstance(servers_raw, dict):
        logger.warning("[MCP] 配置缺少 'mcpServers' 对象，已跳过")
        return []

    servers: list[ServerConfig] = []
    for name, item in servers_raw.items():
        cfg = _parse_one(name, item)
        if cfg is not None:
            servers.append(cfg)
    logger.info(
        "[MCP] 配置加载完成：%d 个 server（其中 %d 个启用）",
        len(servers),
        sum(1 for s in servers if s.enabled),
    )
    return servers


def _parse_one(name: str, item: Any) -> ServerConfig | None:
    """解析单个 server 配置；不可用的返回 None（只告警）"""
    if not isinstance(item, dict):
        logger.warning("[MCP] server '%s' 配置不是对象，已跳过", name)
        return None

    command = item.get("command")
    if not isinstance(command, str) or not command.strip():
        logger.warning("[MCP] server '%s' 缺少 'command'，已跳过", name)
        return None

    args = item.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        logger.warning("[MCP] server '%s' 的 args 必须是字符串数组，已按空处理", name)
        args = []

    env_raw = item.get("env", {})
    if not isinstance(env_raw, dict):
        logger.warning("[MCP] server '%s' 的 env 必须是对象，已按空处理", name)
        env_raw = {}
    env = {str(k): str(v) for k, v in env_raw.items()}

    approval = item.get("approval", "always")
    if approval not in _VALID_APPROVAL:
        logger.warning(
            "[MCP] server '%s' 的 approval=%r 非法，回退为最保守的 'always'",
            name,
            approval,
        )
        approval = "always"

    timeout = item.get("timeout")
    if timeout is not None and not isinstance(timeout, int):
        logger.warning("[MCP] server '%s' 的 timeout 必须是整数，已忽略", name)
        timeout = None

    return ServerConfig(
        name=name,
        command=command,
        args=args,
        env=env,
        enabled=bool(item.get("enabled", True)),
        approval=approval,
        timeout=timeout,
    )
