"""ReadFileTool 测试：截断口径 / 输出卸载 / 局部读取

重点覆盖两类回归风险：
1. 中文文件截断失效（旧实现按字节判断、按字符切分，实际返回 3 倍阈值）
2. 卸载后模型必须能读回内容（否则制造「读全文→卸载→再读全文」死循环）
"""

import shutil
from pathlib import Path

import pytest

from tiny_claw.tools.builtin.read_file import (
    MAX_CHARS,
    OFFLOAD_DIR,
    PREVIEW_LINES,
    ReadFileTool,
)


@pytest.fixture(autouse=True)
def _cleanup_real_offload():
    """真实日志用例会在仓库的 workspace/.claw/offload 落盘，测试后清掉

    tmp_path 用例的卸载文件随临时目录自动回收，无需处理。
    """
    yield
    real = Path(__file__).resolve().parent.parent / "workspace" / ".claw" / "offload"
    if real.exists():
        shutil.rmtree(real)


# ═══════════════════════════════════════════════════════════════
# 1. 截断口径回归
# ═══════════════════════════════════════════════════════════════


async def test_chinese_file_does_not_blow_char_budget(tmp_path):
    """回归：旧实现判断用字节、切分用字符，中文文件实际返回 8000 字符 = 24000 字节。

    现在统一为字符口径并触发卸载 + 字符预览，返回体必须远小于原文件，
    且提示语不再谎称「字节」。
    """
    (tmp_path / "cn.txt").write_text("中" * 10000, encoding="utf-8")
    out = await ReadFileTool(str(tmp_path)).execute({"path": "cn.txt"})

    assert len(out) < 6000, f"返回 {len(out)} 字符，未受控"
    assert "字节" not in out, "提示语口径必须统一为字符"
    assert "卸载" in out


async def test_small_file_returned_untouched(tmp_path):
    """小文件原样返回，不带行号（避免干扰 edit_file 的字符串匹配）"""
    (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
    out = await ReadFileTool(str(tmp_path)).execute({"path": "a.txt"})

    assert out == "hello\nworld\n"
    assert "卸载" not in out


async def test_hard_truncation_stays_within_char_budget(tmp_path):
    """兜底硬截断：输出不超过 MAX_CHARS + 提示语长度"""
    (tmp_path / "big.txt").write_text("a" * 30000, encoding="utf-8")
    out = await ReadFileTool(str(tmp_path)).execute({"path": "big.txt"})

    # 30000 字符单行 → 走字符预览卸载，返回体应受控
    assert len(out) < MAX_CHARS


# ═══════════════════════════════════════════════════════════════
# 2. 输出卸载
# ═══════════════════════════════════════════════════════════════


def _make_multiline_file(path, n_lines: int = 500, width: int = 80) -> str:
    content = "\n".join(f"line {i:04d} " + "x" * width for i in range(1, n_lines + 1))
    path.write_text(content, encoding="utf-8")
    return content


async def test_offload_writes_complete_content_to_disk(tmp_path):
    """卸载文件必须包含完整原文，否则模型拿到的就是残缺数据"""
    target = tmp_path / "big.log"
    content = _make_multiline_file(target, n_lines=500)
    out = await ReadFileTool(str(tmp_path)).execute({"path": "big.log"})

    offload_file = tmp_path / OFFLOAD_DIR
    assert offload_file.is_dir(), "未创建卸载目录"

    written = list(offload_file.glob("*.txt"))
    assert len(written) == 1
    assert written[0].read_text(encoding="utf-8") == content, "卸载内容不完整"
    assert "卸载" in out


async def test_offload_message_contains_guidance(tmp_path):
    """卸载消息必须告诉模型：总共多少行、文件在哪、下一步怎么读"""
    _make_multiline_file(tmp_path / "big.log", n_lines=500)
    out = await ReadFileTool(str(tmp_path)).execute({"path": "big.log"})

    assert "共 500 行" in out
    assert OFFLOAD_DIR in out
    assert "offset" in out, "必须指引模型用 offset/limit 分段"
    # 头尾预览都要出现
    assert "line 0001" in out
    assert "line 0500" in out


async def test_offload_is_not_an_error(tmp_path):
    """卸载是正常结果，不是失败 —— 否则会被 Reminder 的指纹检测误判为重试"""
    _make_multiline_file(tmp_path / "big.log", n_lines=500)
    out = await ReadFileTool(str(tmp_path)).execute({"path": "big.log"})

    assert not out.startswith("[ERR:"), "卸载不应标记为错误"


async def test_offloaded_file_is_readable_back(tmp_path):
    """关键：卸载出去的内容必须能被同一个工具读回来

    这是「卸载不制造死循环」的前提。子 agent 的 read_file 共享同一 work_dir，
    且 .claw/offload 在工作区内，可通过路径穿越检查。
    """
    target = tmp_path / "src" / "big.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    _make_multiline_file(target, n_lines=500)

    tool = ReadFileTool(str(tmp_path))
    out = await tool.execute({"path": "src/big.log"})

    off_rel = f"{OFFLOAD_DIR}/big.log__"
    assert off_rel.split("__")[0] in out

    offload_name = [p.name for p in (tmp_path / OFFLOAD_DIR).glob("*.txt")][0]
    back = await tool.execute(
        {"path": f"{OFFLOAD_DIR}/{offload_name}", "offset": 250, "limit": 3}
    )

    assert "第 250-252 行" in back
    assert "line 0250" in back


async def test_repeated_read_reuses_same_offload_file(tmp_path):
    """同一文件重复读取应覆盖同一卸载文件，而不是堆积垃圾"""
    _make_multiline_file(tmp_path / "big.log", n_lines=500)
    tool = ReadFileTool(str(tmp_path))

    await tool.execute({"path": "big.log"})
    await tool.execute({"path": "big.log"})

    assert len(list((tmp_path / OFFLOAD_DIR).glob("*.txt"))) == 1


async def test_single_long_line_falls_back_to_char_preview(tmp_path):
    """单行超长文件：行号分段失效，退化为字符头尾预览"""
    (tmp_path / "min.json").write_text('{"k":"' + "y" * 20000 + '"}', encoding="utf-8")
    out = await ReadFileTool(str(tmp_path)).execute({"path": "min.json"})

    assert "行数很少但单行极长" in out
    assert len(out) < MAX_CHARS


# ═══════════════════════════════════════════════════════════════
# 3. 局部读取 offset / limit
# ═══════════════════════════════════════════════════════════════


async def test_offset_limit_returns_requested_slice(tmp_path):
    (tmp_path / "f.txt").write_text(
        "\n".join(f"L{i}" for i in range(1, 101)), encoding="utf-8"
    )
    out = await ReadFileTool(str(tmp_path)).execute(
        {"path": "f.txt", "offset": 11, "limit": 3}
    )

    assert "第 11-13 行 / 共 100 行" in out
    assert "L11" in out and "L12" in out and "L13" in out
    assert "L10" not in out and "L14" not in out


async def test_line_numbers_are_prefixed(tmp_path):
    (tmp_path / "f.txt").write_text(
        "\n".join(f"L{i}" for i in range(1, 101)), encoding="utf-8"
    )
    out = await ReadFileTool(str(tmp_path)).execute(
        {"path": "f.txt", "offset": 91, "limit": 5}
    )

    assert " 91| L91" in out, "分段读取应带行号，便于模型定位与后续分段"
    assert " 95| L95" in out


async def test_offset_beyond_eof_is_clamped(tmp_path):
    """offset 越界不报错，钳到末行 —— 静默钳制比让模型重试一次更划算"""
    (tmp_path / "f.txt").write_text(
        "\n".join(f"L{i}" for i in range(1, 11)), encoding="utf-8"
    )
    out = await ReadFileTool(str(tmp_path)).execute(
        {"path": "f.txt", "offset": 9999, "limit": 2}
    )

    assert not out.startswith("[ERR:")
    assert "L10" in out


async def test_limit_omitted_reads_to_end(tmp_path):
    (tmp_path / "f.txt").write_text(
        "\n".join(f"L{i}" for i in range(1, 51)), encoding="utf-8"
    )
    out = await ReadFileTool(str(tmp_path)).execute({"path": "f.txt", "offset": 45})

    assert "第 45-50 行" in out


async def test_string_params_are_tolerated(tmp_path):
    """模型偶尔把整数写成字符串，容错比报错划算"""
    (tmp_path / "f.txt").write_text(
        "\n".join(f"L{i}" for i in range(1, 51)), encoding="utf-8"
    )
    out = await ReadFileTool(str(tmp_path)).execute(
        {"path": "f.txt", "offset": "5", "limit": "2"}
    )

    assert "第 5-6 行" in out


async def test_slice_still_truncated_if_oversized(tmp_path):
    """分段后仍过大时二次截断，并提示缩小 limit"""
    (tmp_path / "f.txt").write_text(
        "\n".join("z" * 500 for _ in range(200)), encoding="utf-8"
    )
    out = await ReadFileTool(str(tmp_path)).execute(
        {"path": "f.txt", "offset": 1, "limit": 200}
    )

    assert "请缩小 limit" in out
    assert len(out) < MAX_CHARS + 200


# ═══════════════════════════════════════════════════════════════
# 4. 安全与错误路径
# ═══════════════════════════════════════════════════════════════


async def test_path_traversal_blocked(tmp_path):
    out = await ReadFileTool(str(tmp_path)).execute({"path": "../../etc/passwd"})
    assert out.startswith("[ERR:")


async def test_missing_file_reports_error(tmp_path):
    out = await ReadFileTool(str(tmp_path)).execute({"path": "nope.txt"})
    assert out.startswith("[ERR:")


async def test_missing_path_param(tmp_path):
    out = await ReadFileTool(str(tmp_path)).execute({})
    assert out.startswith("[ERR:")


async def test_directory_reports_error(tmp_path):
    (tmp_path / "d").mkdir()
    out = await ReadFileTool(str(tmp_path)).execute({"path": "d"})
    assert out.startswith("[ERR:")


# ═══════════════════════════════════════════════════════════════
# 5. 真实大文件（workspace/error.log，约 140KB）
# ═══════════════════════════════════════════════════════════════


async def test_real_workspace_error_log():
    """用仓库里真实的 140KB 日志文件验证，不是造数据"""
    work_dir = Path(__file__).resolve().parent.parent / "workspace"
    log = work_dir / "error.log"
    if not log.exists():
        pytest.skip("workspace/error.log 不存在")

    tool = ReadFileTool(str(work_dir))
    out = await tool.execute({"path": "error.log"})

    assert "卸载" in out, "140KB 日志文件应触发卸载"
    assert len(out) < MAX_CHARS, f"卸载消息 {len(out)} 字符，超出预算"

    # 读回第 100 行附近，确认卸载内容可用
    name = [p.name for p in (work_dir / OFFLOAD_DIR).glob("error.log__*.txt")][0]
    back = await tool.execute(
        {"path": f"{OFFLOAD_DIR}/{name}", "offset": 100, "limit": 2}
    )
    assert "第 100-101 行" in back
