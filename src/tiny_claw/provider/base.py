"""Provider 基类"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from tiny_claw.schema import Message, ToolDefinition


class LLMProvider(ABC):
    """与大模型通信的统一契约"""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> Message:
        """接收上下文历史 + 可用工具列表，发起一次大模型推理。"""
        ...

    async def generate_stream(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str]:
        """流式推理，逐 token 产出文本。

        默认回退：调用 generate() 后按字符分块模拟流式。
        子类可覆盖实现真正的流式。
        """
        response = await self.generate(messages, available_tools)
        # 模拟逐字符流式
        chunk_size = 4
        for i in range(0, len(response.content), chunk_size):
            yield response.content[i:i + chunk_size]
