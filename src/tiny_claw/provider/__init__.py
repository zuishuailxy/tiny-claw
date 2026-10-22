"""LLM Provider 层 — 大模型协议抽象与具体实现。

两个实现对应两套**协议**，不是两个模型厂商：

- `DeepSeekProvider`  → OpenAI Chat Completion 协议
- `AnthropicProvider` → Anthropic Messages 协议

DeepSeek 官方在 https://api.deepseek.com/anthropic 提供了 Anthropic 协议端点，
因此同一个 DEEPSEEK_API_KEY 可以同时驱动两个实现。这构成了
「同厂商、同模型、仅协议不同」的对照实验：
跑出的任何行为差异只能归因于协议翻译层，用于验证抽象层确实隔离了协议差异。
"""

from tiny_claw.provider.anthropic import AnthropicProvider
from tiny_claw.provider.base import LLMProvider
from tiny_claw.provider.openai import DeepSeekProvider


def create_provider(name: str) -> LLMProvider:
    """按协议名称创建 Provider。

    两套协议、同一套引擎：切换协议不改动 engine / context / tools 的任何一行代码，
    这正是 Provider 抽象层要交付的价值。
    """
    if name == "openai":
        return DeepSeekProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"未知的协议实现: {name}，可选 openai / anthropic")


__all__ = ["LLMProvider", "DeepSeekProvider", "AnthropicProvider", "create_provider"]
