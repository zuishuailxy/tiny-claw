"""工具基类"""

from abc import ABC, abstractmethod
from typing import Any

from tiny_claw.schema import ToolDefinition


class BaseTool(ABC):
    """所有具体工具必须实现的通用接口"""

    @abstractmethod
    def name(self) -> str:
        """返回工具的全局唯一名称（大模型通过这个名字调用它）"""
        ...

    @abstractmethod
    def definition(self) -> ToolDefinition:
        """返回提交给大模型的工具元信息和参数 JSON Schema"""
        ...

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> str:
        """接收大模型吐出的参数，执行具体业务逻辑。

        Args:
            arguments: 已反序列化的参数 dict

        Returns:
            工具执行的输出字符串
        """
        ...
