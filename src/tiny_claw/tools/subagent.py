"""子智能体工具

派出专门用于深度探索（Exploration）的子智能体。
当你需要阅读大量代码、跨文件查找逻辑时调用此工具。
"""

import logging
from typing import Any, Protocol, runtime_checkable

from tiny_claw.tools.base import BaseTool
from tiny_claw.tools.registry import Registry
from tiny_claw.engine.reporter import Reporter
from tiny_claw.schema import ToolDefinition

logger = logging.getLogger("tiny-claw.tools.subagent")


# ═══════════════════════════════════════════════════════════════
# AgentRunner — 打破 tools ↔ engine 循环依赖的抽象接口
# ═══════════════════════════════════════════════════════════════


@runtime_checkable
class AgentRunner(Protocol):
    """工具层 → 引擎层的反向调用契约。

    因为 SubagentTool 位于 tools 包，而完整的 AgentEngine 位于 engine 包，
    我们通过此 Protocol 让 Tool 能拉起 Engine，而无需直接 import engine。
    """

    async def run_sub(
        self,
        task_prompt: str,
        read_only_registry: Registry,
        reporter: Reporter | None = None,
    ) -> str:
        """启动一个匿名的、一次性的子智能体任务。

        Args:
            task_prompt: 给子智能体下达的明确指令。
            read_only_registry: 为子智能体准备的专属、受限的工具注册表。
            reporter: 可选的输出上报器。

        Returns:
            子智能体探索完毕后返回的极度精炼的摘要报告。
        """
        ...


# ═══════════════════════════════════════════════════════════════
# SubagentTool — 派出子智能体
# ═══════════════════════════════════════════════════════════════


class SubagentTool(BaseTool):
    """派出一个专门用于深度探索的子智能体。"""

    def __init__(
        self,
        runner: AgentRunner,
        read_only_registry: Registry,
        reporter: Reporter | None = None,
    ):
        self._runner = runner
        self._read_only_registry = read_only_registry
        self._reporter = reporter

    # ------------------------------------------------------------------
    # BaseTool 接口
    # ------------------------------------------------------------------
    def name(self) -> str:
        return "spawn_subagent"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description=(
                "派出一个专门用于深度探索（Exploration）的子智能体。"
                "当你需要阅读大量代码、跨文件查找逻辑时请调用此工具。"
                "它在探索完毕后，会给你返回一份极度精炼的摘要报告。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task_prompt": {
                        "type": "string",
                        "description": "给子智能体下达的明确指令。例如：'查找项目中所有使用了 UserModel 的文件并总结用法'。",
                    },
                },
                "required": ["task_prompt"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        task_prompt = arguments.get("task_prompt", "")
        if not task_prompt:
            return "[ERR:MISSING_PARAM] 缺少 task_prompt 参数"

        logger.info(
            "[Subagent] 🚀 主 Agent 发起委派！正在拉起探路者: [%s]...",
            task_prompt[:100],
        )

        try:
            summary = await self._runner.run_sub(
                task_prompt=task_prompt,
                read_only_registry=self._read_only_registry,
                reporter=self._reporter,
            )

            logger.info("[Subagent] ✅ 子智能体任务结束。报告返回给主干...")

            # 几万字的代码探索，化作一段轻量级 Summary，像普通 API 调用一样返回给主 Agent
            return f"【子智能体探索报告】:\n{summary}"
        except Exception as e:
            logger.error("[Subagent] 子智能体执行失败: %s", e)
            return f"[ERR:UNKNOWN] 子智能体执行失败: {e}"
