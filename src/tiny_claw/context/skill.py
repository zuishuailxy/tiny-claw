"""技能加载器

扫描 .claw/skills/ 目录，解析 SKILL.md 的 YAML Frontmatter。
支持按需加载：System Prompt 只注入元数据，技能正文由 SkillTool 触发加载。
"""

import logging
from pathlib import Path

logger = logging.getLogger("tiny-claw.context.skill")


class Skill:
    """从 SKILL.md 解析出的标准化技能"""

    def __init__(self, name: str = "Unknown Skill", description: str = "", body: str = ""):
        self.name = name
        self.description = description
        self.body = body


class SkillLoader:
    """负责从本地文件系统加载技能模板"""

    def __init__(self, work_dir: str):
        self._work_dir = Path(work_dir).resolve()

    def load_all(self) -> list[Skill]:
        """扫描 .claw/skills/，返回所有解析后的 Skill 对象"""
        skills_dir = self._work_dir / ".claw" / "skills"
        if not skills_dir.is_dir():
            return []

        skills: list[Skill] = []
        for skill_file in sorted(skills_dir.rglob("SKILL.md")):
            try:
                content = skill_file.read_text(encoding="utf-8")
                skills.append(_parse_skill_md(content))
            except Exception as e:
                logger.warning("解析技能文件失败 %s: %s", skill_file, e)
        return skills

    def load_metadata(self) -> str:
        """只返回技能的元数据（名称+描述），用于 System Prompt"""
        skills = self.load_all()
        if not skills:
            return ""

        lines = [
            "\n### 可用专业技能 (Agent Skills)\n",
            "以下技能可按需加载。当你判断当前任务需要某个技能时，调用 `load_skill` 工具获取完整指南：\n\n",
        ]
        for s in skills:
            lines.append(f"- **{s.name}**: {s.description}\n")
        return "".join(lines)

    def get_skill_body(self, name: str) -> str | None:
        """按名称获取技能正文"""
        for skill in self.load_all():
            if skill.name == name:
                return skill.body
        return None

    def load_all_text(self) -> str:
        """一次性加载全部技能正文（旧接口，保留兼容）"""
        skills = self.load_all()
        if not skills:
            return ""
        parts = [
            "\n### 可用专业技能 (Agent Skills)\n",
            "以下是你拥有的标准化外挂技能，请在符合 description 描述的场景下严格遵循其正文指令：\n\n",
        ]
        for s in skills:
            parts.append(f"#### 技能名称: {s.name}\n")
            parts.append(f"**触发条件**: {s.description}\n\n")
            parts.append("**执行指南**:\n")
            parts.append(s.body)
            parts.append("\n\n---\n")
        return "".join(parts)


def _parse_skill_md(content: str) -> Skill:
    """解析带 YAML Frontmatter (--- 包裹) 的 Markdown"""
    skill = Skill(body=content)

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            skill.body = parts[2].strip()

            for line in frontmatter.split("\n"):
                line = line.strip()
                if line.startswith("name:"):
                    skill.name = line.removeprefix("name:").strip()
                elif line.startswith("description:"):
                    skill.description = line.removeprefix("description:").strip()

    return skill
