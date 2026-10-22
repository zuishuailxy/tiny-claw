"""LangSmith 追踪集成 — 为 Agent 调用链添加可观测性。

使用 LangSmith 的 @traceable 装饰器，不依赖 LangChain。
"""

import os
from functools import wraps

from langsmith import traceable

from tiny_claw import config

# ---- LangSmith 初始化 ----
# LangSmith 自身只认环境变量，所以这里把配置层的值回写进 os.environ。
# 注意用 setdefault：不覆盖调用方显式 export 的值。
os.environ.setdefault(
    "LANGSMITH_TRACING", "true" if config.LANGSMITH_TRACING else "false"
)
os.environ.setdefault("LANGSMITH_PROJECT", config.LANGSMITH_PROJECT)
os.environ.setdefault("LANGSMITH_API_KEY", config.langsmith_api_key())


def trace(name: str | None = None, **kwargs):
    """轻量封装：为函数添加 LangSmith 追踪。

    用法：
        @trace("agent-think")
        async def generate(...): ...

    等同于 langsmith.traceable()，但提供项目级默认值。
    """
    return traceable(
        name=name,
        project_name=config.LANGSMITH_PROJECT,
        **kwargs,
    )


def trace_llm(func):
    """为 LLM 调用添加 LangSmith 追踪标记。"""
    return traceable(
        run_type="llm",
        name="llm-generate",
        project_name=config.LANGSMITH_PROJECT,
    )(func)


def trace_tool(func):
    """为工具执行添加 LangSmith 追踪标记。"""
    return traceable(
        run_type="tool",
        name="tool-execute",
        project_name=config.LANGSMITH_PROJECT,
    )(func)
