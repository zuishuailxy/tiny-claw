"""MCP Tool → 本项目 BaseTool 的适配层

这是**跨两套系统的翻译**，四处必须对齐，否则"能连上"也"用不了"：

| 要对齐什么 | 为什么 |
|---|---|
| **命名空间名** | 由 SDK 的 `component_name_hook` 生成（`mcp__{server}__{tool}`）。不加前缀会**静默覆盖**内置工具——filesystem server 的工具就叫 `read_file`/`write_file`/`edit_file` |
| **schema 直通** | MCP 的 `inputSchema` 就是 JSON Schema，与 `ToolDefinition.input_schema` 同构，**零转换**（Phase 0 已实证） |
| **错误自带 `[ERR:` 前缀** | `registry.py` 的 `is_error` 判定只看前缀；靠 Registry 兜底（`Error executing …`）会**丢错误码**，`RecoveryManager` 就不会注入救援提示 |
| **输出必须截断** | MCP 可返回 base64 图片/音频块（如 `read_media_file`），不截断会直接撑爆上下文预算 |

审阅重点：`execute()` **绝不抛异常** —— 失败是返回值，异常会打断 agent 循环。
"""

import asyncio
import logging
from typing import Any

from tiny_claw import config
from tiny_claw.context.recovery import ErrorCode, format_error
from tiny_claw.mcp.client import tool_input_schema
from tiny_claw.schema import ToolDefinition
from tiny_claw.tools.base import BaseTool

logger = logging.getLogger("tiny-claw.mcp.adapter")


def render_content(result: Any) -> str:
    """把 MCP 的 content blocks 渲染成单一字符串。

    content 可能是混合块：`text` / `image` / `audio` / `resource`。
    非文本块**只报类型与体积，不把 base64 灌进上下文**。
    """
    blocks = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        # 非文本块：image / audio / resource（含 base64 data）
        kind = getattr(block, "type", None) or type(block).__name__
        raw = getattr(block, "data", None) or ""
        size = len(raw) if isinstance(raw, str) else 0
        parts.append(f"（{kind} 内容已省略，约 {size} 字符；如需读取请改用内置 read_file）")
    rendered = "\n".join(p for p in parts if p).strip()
    return rendered or "(该 MCP 工具没有返回文本内容)"


def clip_output(text: str, limit: int | None = None) -> str:
    """按配置上限截断，并明确告知被截断了多少（模型才知道可以换更窄的调用）"""
    cap = limit if limit is not None else config.MCP_MAX_OUTPUT_CHARS
    if len(text) <= cap:
        return text
    return f"{text[:cap]}\n…（已截断 {len(text) - cap} 字符，超出单次 MCP 输出上限 {cap}）"


class MCPToolAdapter(BaseTool):
    """把一个 MCP tool 暴露成本项目的 `BaseTool`

    调用走 `group.call_tool(全名, args)` —— 路由由官方 SDK 负责
    （它内部维护 `工具全名 → 会话` 的映射），所以适配层不需要自己管连接。
    """

    def __init__(self, full_name: str, tool: Any, group: Any):
        self._full_name = full_name
        self._tool = tool
        self._group = group

    # ------------------------------------------------------------------
    # BaseTool 接口
    # ------------------------------------------------------------------
    def name(self) -> str:
        # 已带命名空间（SDK hook 生成），直接透传
        return self._full_name

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._full_name,
            description=(getattr(self._tool, "description", "") or "").strip(),
            # 零转换：MCP inputSchema 就是 JSON Schema
            input_schema=tool_input_schema(self._tool),
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        try:
            # 双保险：session 侧有 read_timeout_seconds，这里再兜一层整体超时
            async with asyncio.timeout(config.MCP_CALL_TIMEOUT):
                result = await self._group.call_tool(self._full_name, arguments)
        except TimeoutError:
            return format_error(
                ErrorCode.MCP_TIMEOUT,
                f"{self._full_name} 调用超时（>{config.MCP_CALL_TIMEOUT}s）",
            )
        except Exception as e:  # noqa: BLE001 —— 第三方能力，失败即降级为错误码
            logger.warning("[MCP] 工具 %s 调用失败: %s", self._full_name, e)
            return format_error(
                ErrorCode.MCP_TOOL_FAILED, f"{self._full_name}: {e}"
            )

        # MCP 层的业务错误（server 正常响应，但工具自身失败）
        if getattr(result, "isError", False):
            return format_error(
                ErrorCode.MCP_TOOL_ERROR, clip_output(render_content(result))
            )

        return clip_output(render_content(result))
