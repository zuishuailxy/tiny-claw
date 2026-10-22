"""DeepSeek Provider

基于 OpenAI 兼容接口的 DeepSeek 实现。
"""

from collections.abc import AsyncIterator
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from langsmith.wrappers import wrap_openai

from tiny_claw import config
from tiny_claw.provider.base import LLMProvider
from tiny_claw.schema import Message, Role, ToolCall, ToolDefinition, Usage

logger = logging.getLogger("tiny-claw.provider.deepseek")

# 端点 / 模型名 / 凭据统一来自 tiny_claw.config（可用 .env 覆盖），此处不硬编码。


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """将内部 Message 翻译为 OpenAI 格式"""
    openai_msgs: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            openai_msgs.append({"role": "system", "content": msg.content})

        elif msg.role == Role.USER:
            if msg.tool_call_id:
                openai_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                openai_msgs.append({"role": "user", "content": msg.content})

        elif msg.role == Role.ASSISTANT:
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.reasoning:
                entry["reasoning_content"] = msg.reasoning
            if msg.content:
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            openai_msgs.append(entry)

    return openai_msgs


def _to_openai_tools(
    tools: list[ToolDefinition] | None,
) -> list[dict[str, Any]] | None:
    """将内部 ToolDefinition 翻译为 OpenAI function tools 格式"""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": (
                    t.input_schema
                    if t.input_schema
                    else {
                        "type": "object",
                        "properties": {},
                    }
                ),
            },
        }
        for t in tools
    ]


def _from_openai_response(response: Any) -> Message:
    """将 OpenAI response 反向翻译为内部 Message"""
    choice = response.choices[0].message

    result = Message(role=Role.ASSISTANT, content=choice.content or "")

    # 提取 DeepSeek 慢思考的推理链（不含 fake tool calls 的纯思考文本）
    # 归一化到统一的 reasoning 字段，与 Anthropic 的 thinking block 对齐
    if hasattr(choice, "reasoning_content") and choice.reasoning_content:
        result.reasoning = choice.reasoning_content

    if choice.tool_calls:
        for tc in choice.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result.tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
            )

    # 提取 Token 消耗
    if response.usage:
        result.usage = Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    return result


class DeepSeekProvider(LLMProvider):
    """DeepSeek Provider — 基于 OpenAI 兼容接口"""

    def __init__(self, model: str | None = None):
        api_key = config.deepseek_api_key()
        if not api_key:
            raise ValueError(
                "未找到模型凭据：请在项目根 .env 中设置 DEEPSEEK_API_KEY，"
                "或设置环境变量 DEEPSEEK_API_KEY"
            )

        self.client = wrap_openai(
            AsyncOpenAI(
                api_key=api_key,
                base_url=config.deepseek_base_url(),
            )
        )
        self.model = model or config.MODEL
        logger.info("DeepSeek provider 初始化完成，模型: %s", self.model)

    async def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> Message:
        """发起推理请求"""
        openai_msgs = _to_openai_messages(messages)
        openai_tools = _to_openai_tools(available_tools)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": openai_msgs,
        }
        # 当 available_tools 为 None 时，不传 tools 字段（实现慢思考）
        if openai_tools is not None:
            params["tools"] = openai_tools

        try:
            response = await self.client.chat.completions.create(**params)
        except Exception as e:
            raise RuntimeError(f"DeepSeek API 请求失败: {e}") from e

        if not response.choices:
            raise RuntimeError("DeepSeek API 返回了空的 Choices")

        return _from_openai_response(response)

    async def generate_stream(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str]:
        """流式推理 — 逐 token 产出文本"""
        openai_msgs = _to_openai_messages(messages)
        openai_tools = _to_openai_tools(available_tools)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": openai_msgs,
            "stream": True,
        }
        if openai_tools is not None:
            params["tools"] = openai_tools

        try:
            stream = await self.client.chat.completions.create(**params)
        except Exception as e:
            raise RuntimeError(f"DeepSeek 流式请求失败: {e}") from e

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
