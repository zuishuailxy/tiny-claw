"""Reporter 接口

定义 Agent 引擎向外界输出信息的规范。
支持无缝切换 CLI、飞书、钉钉、WebUI 等不同展现层。
"""

from abc import ABC, abstractmethod
from typing import Any


class Reporter(ABC):
    """Agent 引擎输出接口"""

    @abstractmethod
    async def on_thinking(self, content: str) -> None:
        """模型开始慢思考 (Reasoning) 时调用"""
        ...

    @abstractmethod
    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """模型决定调用工具时调用"""
        ...

    @abstractmethod
    async def on_tool_result(self, tool_name: str, output: str, is_error: bool) -> None:
        """工具执行完毕返回结果时调用"""
        ...

    @abstractmethod
    async def on_message(self, content: str) -> None:
        """模型输出最终纯文本回答时调用"""
        ...


class CliReporter(Reporter):
    """CLI 终端输出实现"""

    async def on_thinking(self, content: str) -> None:
        print(f"🧠 [内部思考 Trace]: {content}")

    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        print(f"🛠️  调用工具: {tool_name}({args})")

    async def on_tool_result(self, tool_name: str, output: str, is_error: bool) -> None:
        prefix = "❌" if is_error else "✅"
        print(f"{prefix} {tool_name}: {output[:200]}")

    async def on_message(self, content: str) -> None:
        print(f"🤖 [对外回复]: {content}")
