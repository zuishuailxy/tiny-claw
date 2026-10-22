"""Anthropic Provider — Anthropic Messages 协议实现

与 OpenAI Chat Completion 协议的六处结构性差异（本文件的存在意义）：

1. System Prompt 是**顶层 `system` 字段**，不能作为 role="system" 放进 messages
2. 工具调用是 content block 里的 `tool_use`，不是独立的 `tool_calls` 字段
3. 工具结果是 **user 消息里的 `tool_result` block**，不是 role="tool" 的独立消息
4. 多个连续工具结果**必须合并进同一条 user 消息**（严格 user/assistant 交替）
5. 停止信号是 `stop_reason="tool_use"`，不是 `finish_reason="tool_calls"`
6. 推理链是 **`thinking` block**，不是 DeepSeek 私有的 `reasoning_content`
   两者统一映射到 Message.reasoning，上层对厂商无感知

DeepSeek 在 https://api.deepseek.com/anthropic 提供了 Anthropic 协议端点，
因此本实现可用同一个 DEEPSEEK_API_KEY 走 Anthropic 协议，
与 DeepSeekProvider 构成「同模型、同厂商、仅协议不同」的对照实验。
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from tiny_claw import config
from tiny_claw.provider.base import LLMProvider
from tiny_claw.schema import Message, Role, ToolCall, ToolDefinition, Usage

logger = logging.getLogger("tiny-claw.provider.anthropic")

# 端点 / 模型名 / 输出上限统一来自 tiny_claw.config（默认 DeepSeek 的 Anthropic
# 协议端点 https://api.deepseek.com/anthropic，可用 .env 覆盖）。


# ═══════════════════════════════════════════════════════════════
# 出站翻译：内部 Message → Anthropic Messages 格式
# ═══════════════════════════════════════════════════════════════


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """把 SYSTEM 消息从历史中剥离，合并为顶层 system 字符串。

    Anthropic 不接受 messages 里出现 role="system"。
    """
    system_parts = [m.content for m in messages if m.role == Role.SYSTEM and m.content]
    rest = [m for m in messages if m.role != Role.SYSTEM]
    return "\n\n".join(system_parts), rest


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """翻译消息列表。

    关键处理：
    - 携带 tool_call_id 的 USER 消息视为工具结果，累积到 pending 后合并成一条
    - assistant 消息 content 为空且无 tool_use 时整体丢弃（协议不允许空 content）
    """
    out: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for msg in messages:
        if msg.role == Role.USER:
            if msg.tool_call_id:
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
                continue
            flush_results()
            out.append(
                {"role": "user", "content": [{"type": "text", "text": msg.content}]}
            )

        elif msg.role == Role.ASSISTANT:
            flush_results()
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})

    flush_results()
    return normalize_alternation(out)


def normalize_alternation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """强制严格 user/assistant 交替，且首条必须是 user。

    Anthropic 协议不接受连续两条同角色消息，也不接受 assistant 开头。
    内部 Schema 里工具结果是「一条消息一个」，合并是本协议特有的必要步骤。
    """
    if not messages:
        return messages

    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append({"role": msg["role"], "content": list(msg["content"])})

    if merged[0]["role"] == "assistant":
        merged.insert(
            0, {"role": "user", "content": [{"type": "text", "text": "(继续)"}]}
        )
    return merged


def to_anthropic_tools(
    tools: list[ToolDefinition] | None,
) -> list[dict[str, Any]] | None:
    """翻译工具定义。注意 Anthropic 用 input_schema，不是 parameters。"""
    if not tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": (
                t.input_schema
                if t.input_schema
                else {"type": "object", "properties": {}}
            ),
        }
        for t in tools
    ]


# ═══════════════════════════════════════════════════════════════
# 入站翻译：Anthropic 响应 → 内部 Message
# ═══════════════════════════════════════════════════════════════


def from_anthropic_response(response: Any) -> Message:
    """把 Anthropic 响应反解为内部 Message。

    content 是 block 数组，按 type 分流：thinking → reasoning、text → content、
    tool_use → ToolCall。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in response.content:
        block_type = getattr(block, "type", None)

        if block_type == "thinking":
            text = getattr(block, "thinking", "") or ""
            if text:
                reasoning_parts.append(text)

        elif block_type == "text":
            text = getattr(block, "text", "") or ""
            if text:
                content_parts.append(text)

        elif block_type == "tool_use":
            raw_input = getattr(block, "input", None)
            arguments = raw_input if isinstance(raw_input, dict) else {}
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, arguments=arguments)
            )

    result = Message(
        role=Role.ASSISTANT,
        content="".join(content_parts),
        tool_calls=tool_calls,
    )

    # 统一字段：Anthropic 的 thinking block 与 DeepSeek 的 reasoning_content
    # 在此收敛为同一个 reasoning，上层引擎对厂商无感知
    if reasoning_parts:
        result.reasoning = "\n\n".join(reasoning_parts)

    if response.usage:
        result.usage = Usage(
            prompt_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "output_tokens", 0) or 0,
        )

    return result


# ═══════════════════════════════════════════════════════════════
# Provider
# ═══════════════════════════════════════════════════════════════


class AnthropicProvider(LLMProvider):
    """Anthropic Messages 协议实现。

    凭据解析顺序：显式参数 > ANTHROPIC_API_KEY > DEEPSEEK_API_KEY
    端点解析顺序：显式参数 > ANTHROPIC_BASE_URL > DeepSeek Anthropic 端点
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        resolved_key = api_key or config.anthropic_api_key()
        if not resolved_key:
            raise ValueError(
                "未找到模型凭据：请在项目根 .env 中设置 ANTHROPIC_API_KEY "
                "或 DEEPSEEK_API_KEY（后者可用 DeepSeek 的 Anthropic 协议端点）"
            )

        self.client = AsyncAnthropic(
            api_key=resolved_key,
            base_url=base_url or config.anthropic_base_url(),
        )
        self.model = model or config.MODEL
        self.max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        logger.info(
            "Anthropic provider 初始化完成，模型: %s, base_url: %s",
            model,
            self.client.base_url,
        )

    def _build_params(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None,
    ) -> dict[str, Any]:
        system_text, rest = split_system(messages)

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,  # Anthropic 协议下必填
            "messages": to_anthropic_messages(rest),
        }
        if system_text:
            params["system"] = system_text

        if available_tools is not None:
            params["tools"] = to_anthropic_tools(available_tools)
        else:
            # 与 OpenAI 侧保持同一契约：available_tools=None 即进入慢思考。
            # 原生 Anthropic 需要 budget_tokens；DeepSeek 端点会忽略它。
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(
                    self.max_tokens * config.THINKING_BUDGET_RATIO
                ),
            }

        return params

    async def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> Message:
        params = self._build_params(messages, available_tools)

        try:
            response = await self.client.messages.create(**params)
        except Exception as e:
            raise RuntimeError(f"Anthropic API 请求失败: {e}") from e

        return from_anthropic_response(response)

    async def generate_stream(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str]:
        params = self._build_params(messages, available_tools)
        params["stream"] = True

        try:
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise RuntimeError(f"Anthropic 流式请求失败: {e}") from e
