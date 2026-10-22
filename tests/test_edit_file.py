"""edit_file 的四级模糊匹配与错误码用例

补空白的理由：仓库此前**没有任何 edit_file 用例**（`pytest --collect-only |
grep edit` 为空），而「四级模糊匹配」是被 README 与手册反复引用的能力。
这类工具最危险的失败模式是「改错地方」和「失败却动了文件」，所以这里两条都守。
"""

import pytest

from tiny_claw.tools.builtin.edit_file import (
    EditFileTool,
    _fuzzy_replace,
    _line_by_line_replace,
)

# ══════════════════════════════════════════════════════════════
# 一级：精确匹配
# ══════════════════════════════════════════════════════════════


def test_l1_exact_match():
    assert _fuzzy_replace("a\nb\nc\n", "b", "B") == "a\nB\nc\n"


def test_l1_rejects_ambiguous_match():
    """多处命中必须报错 —— 猜一处改，是 edit 类工具最危险的失败模式"""
    with pytest.raises(ValueError, match="2 处"):
        _fuzzy_replace("x\nx\n", "x", "y")


# ══════════════════════════════════════════════════════════════
# 二级：换行符归一化（\r\n ↔ \n）
# ══════════════════════════════════════════════════════════════


def test_l2_crlf_file_lf_query():
    """文件是 CRLF、模型传的是 LF —— 归一化后应能命中"""
    assert _fuzzy_replace("a\r\nb\r\nc", "b\nc", "B\nC") == "a\nB\nC"


def test_l2_crlf_query_lf_file():
    assert _fuzzy_replace("a\nb\nc", "b\r\nc", "B\nC") == "a\nB\nC"


# ══════════════════════════════════════════════════════════════
# 三级：首尾空白差异
# ══════════════════════════════════════════════════════════════


def test_l3_trims_outer_whitespace():
    assert _fuzzy_replace("foo\nbar\nbaz\n", "  bar  ", "BAR") == "foo\nBAR\nbaz\n"


# ══════════════════════════════════════════════════════════════
# 四级：逐行去缩进
# ══════════════════════════════════════════════════════════════


def test_l4_ignores_indentation():
    """模型给的片段没有缩进、文件里有 —— 按逐行去缩进比较

    注意 new_text 由调用方自带缩进：L4 只负责**定位**，不负责重排缩进。
    """
    content = "def f():\n    x = 1\n    return x\n"
    out = _fuzzy_replace(content, "x = 1\nreturn x", "    y = 2\n    return y")
    assert out == "def f():\n    y = 2\n    return y\n"


def test_l4_new_text_is_inserted_verbatim():
    """如实记录的边界：L4 把 new_text **原样插入**，不会自动补缩进。

    所以模型在把 old_text 去掉缩进提交时，new_text 也必须自带缩进，
    否则写回去的代码缩进会丢。这不是「待修的 bug」—— 自动补缩进要猜层级，
    猜错比不补更糟（会静默生成层级错误的代码）。
    """
    content = "def f():\n    x = 1\n    return x\n"
    out = _fuzzy_replace(content, "x = 1\nreturn x", "y = 2\nreturn y")
    assert out == "def f():\ny = 2\nreturn y\n"


def test_l4_sliding_window_locates_block():
    assert _fuzzy_replace("head\nmid\ntail\n", "mid\ntail", "MID\nTAIL") == (
        "head\nMID\nTAIL\n"
    )


def test_l4_missing_raises():
    with pytest.raises(ValueError, match="未找到|找不到"):
        _line_by_line_replace("a\nb\n", "no\nsuch\nline", "x")


def test_l4_ambiguous_raises():
    with pytest.raises(ValueError, match="模糊匹配到了 2 处"):
        _line_by_line_replace("a\na\n", "a", "b")


def test_l4_old_text_longer_than_content_raises():
    with pytest.raises(ValueError, match="找不到"):
        _line_by_line_replace("a\n", "a\nb\nc", "x")


# ══════════════════════════════════════════════════════════════
# EditFileTool：错误码映射与「失败不落盘」
# ══════════════════════════════════════════════════════════════


def _tool(work_dir):
    return EditFileTool(str(work_dir))


async def test_tool_success(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")
    out = await _tool(tmp_path).execute(
        {"path": "a.txt", "old_text": "world", "new_text": "there"}
    )
    assert out.startswith("✅")
    assert p.read_text(encoding="utf-8") == "hello there\n"


async def test_tool_missing_path(tmp_path):
    out = await _tool(tmp_path).execute({"old_text": "a", "new_text": "b"})
    assert out.startswith("[ERR:MISSING_PARAM]")


async def test_tool_missing_old_text(tmp_path):
    out = await _tool(tmp_path).execute({"path": "a.txt", "new_text": "b"})
    assert out.startswith("[ERR:MISSING_PARAM]")


async def test_tool_path_traversal(tmp_path):
    out = await _tool(tmp_path).execute(
        {"path": "../escape.txt", "old_text": "a", "new_text": "b"}
    )
    assert out.startswith("[ERR:PATH_TRAVERSAL]")


async def test_tool_file_not_found(tmp_path):
    out = await _tool(tmp_path).execute(
        {"path": "nope.txt", "old_text": "a", "new_text": "b"}
    )
    assert out.startswith("[ERR:FILE_NOT_FOUND]")


async def test_tool_is_directory(tmp_path):
    (tmp_path / "d").mkdir()
    out = await _tool(tmp_path).execute(
        {"path": "d", "old_text": "a", "new_text": "b"}
    )
    assert out.startswith("[ERR:IS_DIRECTORY]")


async def test_tool_old_text_not_found(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    out = await _tool(tmp_path).execute(
        {"path": "a.txt", "old_text": "zzz", "new_text": "y"}
    )
    assert out.startswith("[ERR:EDIT_OLD_TEXT_NOT_FOUND]")


async def test_tool_multiple_match(tmp_path):
    (tmp_path / "a.txt").write_text("x\nx\n", encoding="utf-8")
    out = await _tool(tmp_path).execute(
        {"path": "a.txt", "old_text": "x", "new_text": "y"}
    )
    assert out.startswith("[ERR:EDIT_MULTIPLE_MATCH]")


async def test_tool_failure_never_touches_file(tmp_path):
    """失败必须原样保留文件 —— 这是 edit 类工具的底线，也是错误自愈能工作的前提"""
    p = tmp_path / "a.txt"
    original = "x\nx\n"
    p.write_text(original, encoding="utf-8")
    await _tool(tmp_path).execute(
        {"path": "a.txt", "old_text": "x", "new_text": "y"}
    )
    assert p.read_text(encoding="utf-8") == original


async def test_tool_fuzzy_edit_writes_through(tmp_path):
    """四级匹配真的落到磁盘（不是只返回成功消息）"""
    p = tmp_path / "code.py"
    p.write_text("def f():\n    x = 1\n    return x\n", encoding="utf-8")
    await _tool(tmp_path).execute(
        {
            "path": "code.py",
            "old_text": "x = 1\nreturn x",
            "new_text": "    z = 3\n    return z",
        }
    )
    text = p.read_text(encoding="utf-8")
    assert "z = 3" in text and "x = 1" not in text
