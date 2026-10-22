"""技能系统与 Prompt 组装。

覆盖项目材料里「上下文成本工程：受众拆分 + 按需加载」这条的两个落点：

- `SkillLoader`：扫描 `.claw/skills/**/SKILL.md`、解析 YAML frontmatter
- `PromptComposer`：AGENTS.md 注入 + **只注入技能元数据**，正文按需拉取

⚠️ 本文件最关键的断言是「元数据注入了、正文没注入」。
如果哪天有人图省事把 `load_metadata()` 换成 `load_all_text()`，
System Prompt 就会把所有技能正文一次性塞进去——那就是
「常驻体积 × 技能数」的退化（技能越多，每轮越贵）。这条用例就是拦住它的守卫。
"""

from pathlib import Path

from tiny_claw.context.composer import PromptComposer
from tiny_claw.context.skill import SkillLoader, _parse_skill_md
from tiny_claw.schema import Role
from tiny_claw.tools.builtin.skill import SkillTool

# ─────────────────────────────────────────────────────────────
# 夹具辅助
# ─────────────────────────────────────────────────────────────

BODY = "# 执行步骤\n\n1. 先用 read_file 读取目标文件\n2. 再逐条核对规范\n"


def _write_skill(
    work_dir: Path, rel: str, name: str, description: str, body: str = BODY
) -> Path:
    """在 work_dir/<rel> 写一个带 frontmatter 的 SKILL.md"""
    p = work_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return p


# ─────────────────────────────────────────────────────────────
# 1. 解析：frontmatter 拆成「元数据 + 正文」
# ─────────────────────────────────────────────────────────────


def test_parse_splits_frontmatter_and_body():
    raw = f"---\nname: 代码审查\ndescription: 用户要求审查代码质量时使用\n---\n\n{BODY}\n"
    s = _parse_skill_md(raw)
    assert s.name == "代码审查"
    assert s.description == "用户要求审查代码质量时使用"
    # 正文必须去掉 frontmatter 与首尾空白
    assert s.body == BODY.strip()
    assert "description:" not in s.body
    assert "---" not in s.body


def test_parse_without_frontmatter_keeps_defaults():
    """没有 frontmatter 的文件不能崩，正文即全文、名字退化为默认值"""
    a = _parse_skill_md("就是一段普通的 Markdown")
    assert a.name == "Unknown Skill"
    assert a.description == ""
    assert a.body == "就是一段普通的 Markdown"


def test_parse_with_only_opening_delimiter_is_safe():
    """只有开头 `---` 而没有闭合——不能把正文吃掉或抛异常"""
    a = _parse_skill_md("---\nname: 半个")
    assert a.name == "Unknown Skill"
    assert "name: 半个" in a.body


# ─────────────────────────────────────────────────────────────
# 2. 扫描：目录不存在 / 嵌套 / 多个技能
# ─────────────────────────────────────────────────────────────


def test_no_skills_dir_returns_empty(tmp_path):
    """工作区里没有 .claw/skills 是常态（大多数项目都没有），必须静默返回空"""
    loader = SkillLoader(str(tmp_path))
    assert loader.load_all() == []
    assert loader.load_metadata() == ""


def test_scans_nested_skill_dirs(tmp_path):
    """SKILL.md 藏在子目录里也要找得到（rglob 而非 glob）"""
    _write_skill(tmp_path, ".claw/skills/review/SKILL.md", "代码审查", "审查代码时使用")
    _write_skill(tmp_path, ".claw/skills/deep/write/SKILL.md", "写作", "写文档时使用")

    loader = SkillLoader(str(tmp_path))
    names = [s.name for s in loader.load_all()]

    assert set(names) == {"代码审查", "写作"}
    # 顺序按**文件路径**排序（不是技能名），所以同一份目录每次结果一致 → 可复现
    assert names == ["写作", "代码审查"]
    assert [s.name for s in loader.load_all()] == names


def test_bad_skill_file_does_not_break_others(tmp_path):
    """一个技能文件坏掉不能拖垮整个技能库（外部输入，必须降级）"""
    _write_skill(tmp_path, ".claw/skills/good/SKILL.md", "好技能", "正常")
    bad = tmp_path / ".claw/skills/bad/SKILL.md"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"\xff\xfe\x00\x00\xff")  # 非法 UTF-8

    skills = SkillLoader(str(tmp_path)).load_all()
    assert [s.name for s in skills] == ["好技能"]


# ─────────────────────────────────────────────────────────────
# 3. 渐进注入：元数据进 System Prompt，正文不进 ← 核心
# ─────────────────────────────────────────────────────────────


def test_metadata_injects_description_but_not_body(tmp_path):
    """🔴 核心断言：注入的是「目录」，不是「全书」"""
    _write_skill(tmp_path, ".claw/skills/review/SKILL.md", "代码审查", "审查代码质量时使用")

    meta = SkillLoader(str(tmp_path)).load_metadata()

    assert "代码审查" in meta
    assert "审查代码质量时使用" in meta
    # 正文一个字都不能进 System Prompt
    assert "read_file" not in meta
    assert "执行步骤" not in meta
    # 还要明确告诉模型「怎么取正文」
    assert "load_skill" in meta


def test_metadata_scales_with_count_not_body_size(tmp_path):
    """技能正文再长，元数据也不变长——这正是「按需加载」的意义"""
    _write_skill(tmp_path, ".claw/skills/a/SKILL.md", "技能A", "描述A", body="x" * 50)
    short = SkillLoader(str(tmp_path)).load_metadata()

    long_body = "# 步骤\n" + ("y" * 20000) + "\n"
    _write_skill(tmp_path, ".claw/skills/a/SKILL.md", "技能A", "描述A", body=long_body)
    long = SkillLoader(str(tmp_path)).load_metadata()

    assert len(short) == len(long)  # 正文从 50 字变成 20000 字，元数据长度不变


# ─────────────────────────────────────────────────────────────
# 4. 按需取回正文
# ─────────────────────────────────────────────────────────────


def test_get_skill_body_by_name(tmp_path):
    _write_skill(tmp_path, ".claw/skills/review/SKILL.md", "代码审查", "描述")
    loader = SkillLoader(str(tmp_path))
    assert loader.get_skill_body("代码审查") == BODY.strip()
    assert loader.get_skill_body("不存在的技能") is None


async def test_skill_tool_returns_body_when_found(tmp_path):
    _write_skill(tmp_path, ".claw/skills/review/SKILL.md", "代码审查", "描述")
    out = await SkillTool(str(tmp_path)).execute({"skill_name": "代码审查"})
    assert "代码审查" in out
    assert BODY.strip() in out


async def test_skill_tool_unknown_name_lists_available(tmp_path):
    """名字写错时要告诉模型「有哪些可用」，而不是只说失败——
    否则模型只能猜，会反复试错（与错误自愈的取向一致）"""
    _write_skill(tmp_path, ".claw/skills/review/SKILL.md", "代码审查", "描述")
    out = await SkillTool(str(tmp_path)).execute({"skill_name": "拼错了"})
    assert out.startswith("Error:")
    assert "代码审查" in out  # 可用清单


def test_skill_tool_definition_requires_skill_name():
    d = SkillTool(".").definition()
    assert d.name == "load_skill"
    assert d.input_schema["required"] == ["skill_name"]


# ─────────────────────────────────────────────────────────────
# 5. PromptComposer：受众拆分的落点
# ─────────────────────────────────────────────────────────────


def test_composer_returns_single_system_message(tmp_path):
    msg = PromptComposer(str(tmp_path)).build()
    assert msg.role == Role.SYSTEM
    assert "tiny-claw" in msg.content


def test_composer_injects_agents_md_when_present(tmp_path):
    (tmp_path / "AGENTS.md").write_text("本项目禁止直接改 SSOT", encoding="utf-8")
    content = PromptComposer(str(tmp_path)).build().content
    assert "本项目禁止直接改 SSOT" in content
    assert "AGENTS.md" in content


def test_composer_omits_agents_section_when_absent(tmp_path):
    assert "来自 AGENTS.md" not in PromptComposer(str(tmp_path)).build().content


def test_composer_plan_mode_is_opt_in(tmp_path):
    """plan_mode 是开关，不能默认打开——否则每次调用都多付一大段提示词"""
    off = PromptComposer(str(tmp_path), plan_mode=False).build().content
    on = PromptComposer(str(tmp_path), plan_mode=True).build().content
    assert "PLAN.md" not in off
    assert "PLAN.md" in on
    assert len(on) > len(off)


def test_composer_injects_skill_metadata_only(tmp_path):
    """🔴 System Prompt 里应有技能目录 + 加载指引，但不应有技能正文"""
    _write_skill(tmp_path, ".claw/skills/review/SKILL.md", "代码审查", "审查代码质量时使用")
    content = PromptComposer(str(tmp_path)).build().content

    assert "代码审查" in content          # 元数据在
    assert "load_skill" in content         # 加载指引在
    assert "执行步骤" not in content        # 正文不在
    assert "read_file 读取目标文件" not in content
