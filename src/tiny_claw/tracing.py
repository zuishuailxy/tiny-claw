"""可观测性

包含两个子系统：
  - CostTracker：LLMProvider 装饰器，透明追踪 Token 消耗和费用
  - Span/Trace ：本地链路追踪，替代 LangSmith，保存为 JSON 文件
"""

import asyncio
import contextvars
import json
import logging
import time
from collections.abc import AsyncIterator
from functools import wraps
from pathlib import Path
from typing import Any

from tiny_claw import config
from tiny_claw.provider.base import LLMProvider
from tiny_claw.schema import Message, ToolDefinition

logger = logging.getLogger("tiny-claw.tracker")

# ═══════════════════════════════════════════════════════════════
# 定价模型（人民币/百万 Token）
# ═══════════════════════════════════════════════════════════════

# 口径说明（务必读完再用这个数字做任何结论）：
#   1. 取「空闲时段 + 缓存未命中」价 —— 这是**偏高的上界**。DeepSeek 按前缀
#      自动缓存，Agent 循环里 system prompt 与历史前缀不变，命中率通常很高，
#      命中价仅 0.05 元/百万，比未命中便宜 30 倍。所以真实扣费一般低于此。
#   2. 高峰时段（北京时间 9:00-12:00、14:00-18:00）为下表价格的 **2 倍**。
#   3. 来源：api-docs.deepseek.com/zh-cn/quick_start/pricing（2026-09-09 核对）。
#      ⚠️ deepseek-v4-flash 自 2026-09-10 12:00 起降价，届时未命中输入
#      1.5→1.0、输出 4.5→4.0（元/百万）。若在此之后跑分请同步更新本表，
#      否则成本会被系统性高估约 33%。
PRICING: dict[str, dict[str, float]] = {
    # 旧模型名（官方公告 2026/07/24 起逐步弃用，等价于 v4-flash 的非思考/思考模式）
    "deepseek-chat": {"input": 1.5, "output": 4.5},
    "deepseek-reasoner": {"input": 1.5, "output": 4.5},
    "deepseek-v4-flash": {"input": 1.5, "output": 4.5},
    "deepseek-v4-flash-vision-exp": {"input": 1.5, "output": 4.5},
    # Pro 约为 Flash 的 3 倍
    "deepseek-v4-pro": {"input": 4.5, "output": 13.5},
}

# 已告警过的未知模型，避免每个 chunk 刷屏
_warned_unknown_models: set[str] = set()


def _resolve_price(model: str) -> dict[str, float] | None:
    """查表，未命中则回退到 .env 里的自定义单价。

    注意：PRICING 只覆盖已核实公开价格的模型。新模型（如 v4-flash）未收录时
    绝不猜价格——猜出来的成本会直接污染 benchmark 结论。
    """
    price = PRICING.get(model)
    if price:
        return price

    custom = config.custom_price()
    if custom:
        return {"input": custom[0], "output": custom[1]}

    if model not in _warned_unknown_models:
        _warned_unknown_models.add(model)
        logger.warning(
            "[Tracker] ⚠️ 模型 %s 未在 PRICING 表中，成本将统计为 0。"
            "如需计费请在 .env 设置 TINY_CLAW_PRICE_INPUT / TINY_CLAW_PRICE_OUTPUT"
            "（单位：元/百万 token）",
            model,
        )
    return None


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = _resolve_price(model)
    if price is None:
        return 0.0
    return (
        prompt_tokens / 1_000_000 * price["input"]
        + completion_tokens / 1_000_000 * price["output"]
    )


# ═══════════════════════════════════════════════════════════════
# Span — 本地链路追踪
# ═══════════════════════════════════════════════════════════════

# ContextVar 用于在 asyncio 中传递当前 Span（线程安全）
_current_span: contextvars.ContextVar["Span | None"] = contextvars.ContextVar(
    "_current_span", default=None
)


class Span:
    """链路追踪中的一个时间跨度和操作节点。"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.end_time: float = 0.0
        self.duration_ms: int = 0
        self.attributes: dict[str, Any] = {}
        self.children: list[Span] = []
        self._lock = asyncio.Lock()

    def end(self) -> None:
        """结束跨度，计算耗时"""
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)

    def _end_if_needed(self) -> None:
        """幂等结束：如果尚未 end，自动补调。"""
        if self.end_time == 0.0:
            self.end()

    def add_attribute(self, key: str, value: Any) -> None:
        """记录关键元数据（如 token 消耗、执行的命令）"""
        self.attributes[key] = value

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children],
        }


def start_span(name: str) -> Span:
    """开启一个新的追踪跨度，自动挂到当前父 Span 下。

    使用 ContextVar 在 asyncio 协程间传递父子关系。
    """
    span = Span(name)
    parent = _current_span.get()
    if parent:
        # 并发安全：使用 append 到列表（GIL 保护基本操作）
        parent.children.append(span)

    # 将新 Span 设为当前，后续 start_span 都会挂到它下面
    _current_span.set(span)
    return span


def end_span(span: Span) -> None:
    """结束跨度，恢复父 Span 为当前。"""
    span.end()

    # 恢复父节点
    parent = _find_parent(span)
    _current_span.set(parent)


def _find_parent(child: Span) -> Span | None:
    """在 _current_span 的子树中查找 child 的父节点。"""
    root = _current_span.get()
    if root is None or root is child:
        return None

    # 广度优先搜索
    queue = [root]
    for node in queue:
        if child in node.children:
            return node
        queue.extend(node.children)
    return None


def export_trace(work_dir: str, session_id: str) -> str | None:
    """将当前根 Span 序列化保存为 JSON 文件。

    import_trace 可能在根 Span.end() 之前被调用（例如在 @trace 装饰器的
    finally 之前），因此自动补调 root.end()。
    """
    root = _current_span.get()
    if root is None:
        return None

    # 向上查找真正的根节点
    parent = _find_parent(root)
    while parent:
        root = parent
        parent = _find_parent(root)

    # 根 Span 可能尚未结束（export_trace 在 wrapper finally 之前调用），自动补调
    root._end_if_needed()

    trace_dir = Path(work_dir) / ".claw" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    filename = trace_dir / f"trace_{session_id}_{int(time.time())}.json"
    data = json.dumps(root.to_dict(), indent=2, ensure_ascii=False, default=str)
    filename.write_text(data, encoding="utf-8")

    logger.info("[Trace] 链路追踪已保存: %s", filename)
    return str(filename)


# ---- 便捷装饰器：替代 LangSmith @trace ----
def trace(name: str | None = None, **kwargs):
    """本地 trace 装饰器，替代 LangSmith 的 @traceable。

    Args:
        name: Span 名称，默认取函数名。
        process_inputs: 可选 callable，接收 {"arg_name": value, ...} 返回属性 dict。
        **kwargs: 其他 LangSmith 兼容参数（忽略）。
    """
    process_inputs = kwargs.pop("process_inputs", None)

    def decorator(func):
        import inspect

        @wraps(func)
        async def wrapper(*args, **kw):
            span_name = name or func.__name__
            span = start_span(span_name)

            # 提取函数参数名 → 值映射，供 process_inputs 使用
            if process_inputs:
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kw)
                bound.apply_defaults()
                attrs = process_inputs(bound.arguments)
                for k, v in (attrs or {}).items():
                    span.add_attribute(k, v)

            try:
                return await func(*args, **kw)
            finally:
                end_span(span)

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════
# CostTracker — LLMProvider 装饰器
# ═══════════════════════════════════════════════════════════════


class CostTracker(LLMProvider):
    """包装真实 Provider，透明监控每次 API 调用的消耗。"""

    def __init__(self, provider: LLMProvider, model: str):
        self._next = provider
        self._model = model
        self._session: Any = None

    def bind_session(self, session: Any) -> None:
        self._session = session

    async def generate(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> Message:
        start = time.monotonic()
        resp = await self._next.generate(messages, available_tools)
        latency = time.monotonic() - start

        if resp.usage:
            prompt_tk = resp.usage.prompt_tokens
            completion_tk = resp.usage.completion_tokens
            cost = _calculate_cost(self._model, prompt_tk, completion_tk)

            logger.info(
                "[Tracker] 📊 API 调用完成 | 耗时: %.2fs | "
                "输入: %d tk | 输出: %d tk | 花费: ¥%.6f",
                latency,
                prompt_tk,
                completion_tk,
                cost,
            )

            if self._session:
                await self._session.record_usage(prompt_tk, completion_tk, cost)
        else:
            logger.info("[Tracker] ⚠️ API 无 Usage 数据 | 耗时: %.2fs", latency)

        return resp

    async def generate_stream(
        self,
        messages: list[Message],
        available_tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._next.generate_stream(messages, available_tools):
            yield chunk
