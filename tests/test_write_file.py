"""WriteFileTool。

三个内置写类工具里唯一没有测试的那个（`edit_file` 21 例、`read_file` 20 例都在，
只有 `write_file` 是空白）。补上主要是守住两条底线：

1. **路径穿越防护**——这是写类工具唯一的安全边界，一旦失效，
   模型可以把文件写到工作区之外。必须有用例盯着。
2. **失败返回错误码而不是抛异常**——项目约定「失败是返回值」，
   异常会打断整个 Agent 循环。写失败（无权限 / 目标是目录）必须降级。
"""

from pathlib import Path

from tiny_claw.tools.builtin.write_file import WriteFileTool


def _tool(tmp_path: Path) -> WriteFileTool:
    return WriteFileTool(str(tmp_path))


# ─────────────────────────────────────────────────────────────
# 正常路径
# ─────────────────────────────────────────────────────────────


async def test_writes_file_and_creates_parent_dirs(tmp_path):
    """父目录不存在要自动创建——模型给路径时不会先 mkdir"""
    out = await _tool(tmp_path).execute(
        {"path": "a/b/c/note.md", "content": "hello"}
    )
    p = tmp_path / "a/b/c/note.md"
    assert p.is_file()
    assert p.read_text(encoding="utf-8") == "hello"
    assert "note.md" in out


async def test_overwrites_existing_file(tmp_path):
    """是「创建或覆盖」语义，不是追加"""
    p = tmp_path / "x.txt"
    p.write_text("旧内容", encoding="utf-8")

    await _tool(tmp_path).execute({"path": "x.txt", "content": "新内容"})
    assert p.read_text(encoding="utf-8") == "新内容"


async def test_writes_empty_content(tmp_path):
    """内容为空字符串是合法输入，不能因为 falsy 就判成缺参数"""
    out = await _tool(tmp_path).execute({"path": "empty.txt", "content": ""})
    assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""
    assert not out.startswith("[ERR:")


async def test_unicode_content_roundtrip(tmp_path):
    """中文与 emoji 必须原样落盘（UTF-8），不能走字节口径"""
    text = "中文内容 · 缩进\t制表符 🚀"
    await _tool(tmp_path).execute({"path": "cn.txt", "content": text})
    assert (tmp_path / "cn.txt").read_text(encoding="utf-8") == text


# ─────────────────────────────────────────────────────────────
# 错误路径：必须返回错误码，不抛异常
# ─────────────────────────────────────────────────────────────


async def test_missing_path_returns_error_not_exception(tmp_path):
    out = await _tool(tmp_path).execute({"content": "x"})
    assert out.startswith("[ERR:")
    assert "MISSING_PARAM" in out


async def test_path_traversal_is_blocked(tmp_path):
    """🔴 唯一的安全边界：不能写到工作区之外。

    `../` 与绝对路径两种写法都要挡住。
    """
    tool = _tool(tmp_path)

    out = await tool.execute({"path": "../../escape.txt", "content": "x"})
    assert out.startswith("[ERR:")
    assert "PATH_TRAVERSAL" in out
    assert not (tmp_path.parent.parent / "escape.txt").exists()

    # 工作区内做一个子目录，再尝试用它当跳板跳出（这是最容易被漏掉的一种）
    (tmp_path / "sub").mkdir()
    out2 = await tool.execute({"path": "sub/../../escape2.txt", "content": "x"})
    assert "PATH_TRAVERSAL" in out2


async def test_writing_to_a_directory_returns_error(tmp_path):
    """目标是目录时要降级成错误码，不能崩掉整轮"""
    (tmp_path / "adir").mkdir()
    out = await _tool(tmp_path).execute({"path": "adir", "content": "x"})
    assert out.startswith("[ERR:")
    assert "IS_DIRECTORY" in out


# ─────────────────────────────────────────────────────────────
# 契约
# ─────────────────────────────────────────────────────────────


async def test_success_output_has_no_error_prefix(tmp_path):
    """成功消息不能带 [ERR: 前缀——registry 靠这个前缀判定失败"""
    out = await _tool(tmp_path).execute({"path": "ok.txt", "content": "x"})
    assert not out.startswith("[ERR:")


def test_definition_shape(tmp_path):
    d = _tool(tmp_path).definition()
    assert d.name == "write_file"
    assert d.input_schema["required"] == ["path", "content"]
    assert set(d.input_schema["properties"]) == {"path", "content"}
