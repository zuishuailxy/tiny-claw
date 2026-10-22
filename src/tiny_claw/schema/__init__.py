"""公共数据结构"""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---- Role 角色定义 ----


class Role(StrEnum):
    """消息角色，与大模型沟通的基石"""

    SYSTEM = "system"  # 系统提示词：确立 Agent 的性格与红线
    USER = "user"  # 用户输入 / 工具执行的返回结果 (Observation)
    ASSISTANT = "assistant"  # 模型的输出：推理(Reasoning) 或工具调用(ToolCall)


# 引擎运行时注入的提醒统一前缀。
# 放在 schema 而不是某个具体模块：reminder（打断）与 reminders（压力/轮数预警）
# 都要用它，而它们分属 engine 与 context 两层——常量放任何一方都会造成
# 跨层依赖（context 反向 import engine）。
SYSTEM_REMINDER_PREFIX = "[SYSTEM REMINDER"


# ---- 核心消息结构 ----


@dataclass
class ToolCall:
    """模型请求调用某个具体的工具"""

    id: str  # 工具调用的唯一 ID
    name: str  # 工具名称 (例如 "bash")
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, id: str, name: str, arguments: str) -> ToolCall:
        """从 JSON 字符串参数构建（延迟解析：避免参数格式错误在反序列化阶段就炸）"""
        return cls(id=id, name=name, arguments=json.loads(arguments))


@dataclass
class Usage:
    """单次大模型 API 调用的 Token 消耗"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Message:
    """上下文中传递的单条消息"""

    role: str  # system / user / assistant / tool
    content: str = ""  # 纯文本内容
    tool_calls: list[ToolCall] = field(default_factory=list)  # 模型决定调用工具时填充
    tool_call_id: str = ""  # 工具调用响应的关联 ID
    usage: Usage | None = None  # 如果是 Assistant 回复，存放本次调用的 Token 消耗
    # 统一推理链字段（厂商中立）。DeepSeek 侧映射自私有的 reasoning_content，
    # Anthropic 侧映射自 thinking block。上层引擎只认本字段，对厂商无感知。
    reasoning: str = ""
    # 系统提醒标记（System Reminder）。引擎在运行时注入的提醒（死循环打断、
    # 上下文压力、轮数预警）走的是 user 通道——协议不允许在对话中途插 system 消息——
    # 但它们在语义上**不是**用户指令。压缩器靠这个标记把它们与「用户原始指令」
    # 区分开：后者永不淘汰，前者一旦跑出工作记忆窗口就可以清理。
    is_system_reminder: bool = False


@dataclass
class ToolResult:
    """工具在本地执行完毕后返回的物理结果"""

    tool_call_id: str
    output: str  # 控制台输出或报错堆栈
    is_error: bool = False  # 是否失败，供后续错误自愈


@dataclass
class ToolDefinition:
    """工具元信息，供模型理解工具有什么用"""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)  # JSON Schema


__all__ = [
    "Role",
    "Message",
    "ToolCall",
    "ToolResult",
    "ToolDefinition",
    "Usage",
]
