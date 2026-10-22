"""SkillTool — 按需加载技能正文到上下文"""

from typing import Any

from tiny_claw.context.skill import SkillLoader
from tiny_claw.tools.base import BaseTool
from tiny_claw.schema import ToolDefinition


class SkillTool(BaseTool):
    """按需加载技能正文。模型判断需要某个技能时调用此工具。"""

    def __init__(self, work_dir: str):
        self._loader = SkillLoader(work_dir)

    def name(self) -> str:
        return "load_skill"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description=(
                "加载一个技能的完整指南到上下文中。"
                "当 System Prompt 中列出的某个技能的触发条件满足时，调用此工具获取该技能的详细执行步骤。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要加载的技能名称（与 System Prompt 中列出的一致）",
                    },
                },
                "required": ["skill_name"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        skill_name = arguments.get("skill_name", "")
        body = self._loader.get_skill_body(skill_name)
        if body is None:
            available = [s.name for s in self._loader.load_all()]
            return (
                f"Error: 技能 '{skill_name}' 不存在。可用技能: {', '.join(available)}"
            )

        return f"## 技能: {skill_name}\n\n{body}\n\n---\n请严格按照以上指南执行。"
