"""工具注册表"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from tiny_claw.context.recovery import ErrorCode, format_error
from tiny_claw.schema import ToolCall, ToolDefinition, ToolResult
from tiny_claw.tools.base import BaseTool

logger = logging.getLogger("tiny-claw.tools")

# --- 中间件类型 ---

MiddlewareFunc = Callable[[ToolCall], Awaitable[tuple[bool, str]]]
"""中间件签名：接收 ToolCall，返回 (allowed, reject_reason)。

- allowed=True   → 放行，继续执行工具
- allowed=False  → 拦截，reject_reason 作为 ToolResult(output, is_error=True) 返回给 LLM
"""


class Registry(ABC):
    """工具的注册与分发执行接口"""

    @abstractmethod
    def register(self, tool: BaseTool) -> None:
        """挂载一个新的工具到系统中"""
        ...

    @abstractmethod
    def use(self, mw: MiddlewareFunc) -> None:
        """挂载全局中间件。中间件按注册顺序依次执行，任一拦截即终止。"""
        ...

    @abstractmethod
    def get_available_tools(self) -> list[ToolDefinition]:
        """返回当前系统挂载的所有可用工具的 Schema"""
        ...

    @abstractmethod
    async def execute(self, call: ToolCall) -> ToolResult:
        """实际路由并执行模型请求的工具调用"""
        ...


class RegistryImpl(Registry):
    """Registry 的默认实现 — 基于 dict 的 O(1) 工具路由 + 中间件链"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._middlewares: list[MiddlewareFunc] = []

    # ------------------------------------------------------------------
    # 工具注册
    # ------------------------------------------------------------------
    def register(self, tool: BaseTool) -> None:
        name = tool.name()
        if name in self._tools:
            logger.warning("工具 '%s' 已被注册，将被覆盖。", name)
        self._tools[name] = tool
        logger.info("成功挂载工具: %s", name)

    # ------------------------------------------------------------------
    # 中间件
    # ------------------------------------------------------------------
    def use(self, mw: MiddlewareFunc) -> None:
        """按注册顺序追加中间件"""
        self._middlewares.append(mw)

    # ------------------------------------------------------------------
    # 工具列表
    # ------------------------------------------------------------------
    def get_available_tools(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # 工具执行（含中间件链）
    # ------------------------------------------------------------------
    async def execute(self, call: ToolCall) -> ToolResult:
        # 1. 路由查找：找不到工具说明模型产生了幻觉
        tool = self._tools.get(call.name)
        if tool is None:
            # ⚠️ 必须走 format_error：引擎层的 RecoveryManager 只认 `[ERR:CODE]` 前缀，
            # 裸文案会让这条错误拿不到救援提示——而"调错工具名"正是最需要提示的场景。
            # （本文件下面那段注释描述的就是同一个坑，但此前只修了异常分支）
            return ToolResult(
                tool_call_id=call.id,
                output=format_error(
                    ErrorCode.TOOL_NOT_FOUND,
                    f"系统中不存在名为 '{call.name}' 的工具。",
                ),
                is_error=True,
            )

        # 2. 中间件链：任一拦截即终止，返回 is_error=True
        for mw in self._middlewares:
            allowed, reason = await mw(call)
            if not allowed:
                logger.info(
                    "[Registry] ⚠️ 工具 %s 被 Middleware 拦截: %s",
                    call.name,
                    reason,
                )
                # 同样必须带错误码：拦截多来自人工审批拒绝/超时（approve.py），
                # 模型拿到裸文案会以为"再试一次就好"，于是反复撞同一道墙。
                return ToolResult(
                    tool_call_id=call.id,
                    output=format_error(
                        ErrorCode.TOOL_BLOCKED, f"执行被系统拦截。原因: {reason}"
                    ),
                    is_error=True,
                )

        # 3. 执行工具逻辑
        try:
            output = await tool.execute(call.arguments)
        except Exception as e:
            # 兜底必须带 [ERR:CODE]：RecoveryManager 靠这个前缀提取错误码并注入救援
            # 提示；旧写法输出裸的 "Error executing X: ..."，提取不到代码，模型只能
            # 看到一句英文异常、拿不到任何「下一步该怎么做」的指引。
            # 注意：引擎层还有一层兜底，但本函数先 catch 了异常，那层平时不会触发。
            return ToolResult(
                tool_call_id=call.id,
                output=format_error(
                    ErrorCode.UNKNOWN, f"工具 {call.name} 执行异常: {e}"
                ),
                is_error=True,
            )

        # 4. 封装结果：以 [ERR: 开头的输出自动标记为错误
        is_error = output.startswith("[ERR:")
        return ToolResult(tool_call_id=call.id, output=output, is_error=is_error)
