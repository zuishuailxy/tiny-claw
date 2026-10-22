"""EditFileTool

对文件进行局部字符串替换，比 write_file 更安全快速。
支持四级模糊匹配：精确 → 换行归一化 → 去空格 → 逐行去缩进。
"""

import logging
from pathlib import Path
from typing import Any

from tiny_claw.tools.base import BaseTool
from tiny_claw.schema import ToolDefinition
from tiny_claw.context.recovery import ErrorCode, format_error

logger = logging.getLogger("tiny-claw.tools.edit_file")


def _fuzzy_replace(original: str, old_text: str, new_text: str) -> str:
    """四级模糊匹配替换。

    Raises:
        ValueError: 匹配失败时抛出（调用方转为 ErrorCode 后发给模型）
    """
    # L1: 精确匹配
    count = original.count(old_text)
    if count == 1:
        return original.replace(old_text, new_text, 1)
    if count > 1:
        raise ValueError(
            f"old_text 匹配到了 {count} 处，请提供更多的上下文代码以确保唯一性"
        )

    # L2: 换行符归一化（\r\n → \n）
    norm_content = original.replace("\r\n", "\n")
    norm_old = old_text.replace("\r\n", "\n")
    count = norm_content.count(norm_old)
    if count == 1:
        return norm_content.replace(norm_old, new_text, 1)

    # L3: Trim 空格匹配
    trimmed_old = norm_old.strip()
    if trimmed_old:
        count = norm_content.count(trimmed_old)
        if count == 1:
            return norm_content.replace(trimmed_old, new_text, 1)

    # L4: 逐行去缩进匹配
    return _line_by_line_replace(norm_content, norm_old, new_text)


def _line_by_line_replace(content: str, old_text: str, new_text: str) -> str:
    """逐行去缩进后匹配替换"""
    content_lines = content.split("\n")
    old_lines = [line.strip() for line in old_text.strip().split("\n")]

    if not old_lines or len(content_lines) < len(old_lines):
        raise ValueError("找不到该代码片段")

    # 滑动窗口匹配（去缩进后比较）
    match_count = 0
    match_start = -1
    match_end = -1

    for i in range(len(content_lines) - len(old_lines) + 1):
        if all(
            content_lines[i + j].strip() == old_lines[j] for j in range(len(old_lines))
        ):
            match_count += 1
            match_start = i
            match_end = i + len(old_lines)

    if match_count == 0:
        raise ValueError("在文件中未找到 old_text，请检查内容和缩进")
    if match_count > 1:
        raise ValueError(f"模糊匹配到了 {match_count} 处代码，请提供更多上下文以定位")

    # 拼接：旧内容之前 + 新文本 + 旧内容之后
    return "\n".join(
        content_lines[:match_start] + [new_text] + content_lines[match_end:]
    )


class EditFileTool(BaseTool):
    """对现有文件进行局部字符串替换"""

    def __init__(self, work_dir: str):
        self._work_dir = Path(work_dir).resolve()

    def name(self) -> str:
        return "edit_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description=(
                "对现有文件进行局部的字符串替换。这比重写整个文件更安全、更快速。"
                "请提供足够的 old_text 上下文以确保匹配的唯一性。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要修改的文件路径",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "文件中原有的文本，需包含足够上下文确保唯一性",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "要替换成的新文本",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        rel_path = arguments.get("path", "")
        old_text = arguments.get("old_text", "")
        new_text = arguments.get("new_text", "")

        if not rel_path:
            return format_error(ErrorCode.MISSING_PARAM, "缺少 path 参数")
        if not old_text:
            return format_error(ErrorCode.MISSING_PARAM, "缺少 old_text 参数")

        full_path = (self._work_dir / rel_path).resolve()
        if not str(full_path).startswith(str(self._work_dir)):
            return format_error(
                ErrorCode.PATH_TRAVERSAL,
                f"'{rel_path}' 超出了工作区范围",
            )

        # 读取原文件
        try:
            original = full_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return format_error(ErrorCode.FILE_NOT_FOUND, f"文件不存在: {rel_path}")
        except PermissionError:
            return format_error(ErrorCode.PERMISSION_DENIED, f"无权限读取: {rel_path}")
        except IsADirectoryError:
            return format_error(ErrorCode.IS_DIRECTORY, f"'{rel_path}' 是一个目录")
        except Exception as e:
            return format_error(ErrorCode.FILE_READ_FAILED, f"读取文件失败: {e}")

        # 执行替换
        try:
            result = _fuzzy_replace(original, old_text, new_text)
        except ValueError as e:
            msg = str(e)
            if "未找到" in msg or "找不到" in msg:
                return format_error(ErrorCode.EDIT_OLD_TEXT_NOT_FOUND, msg)
            if "匹配到了" in msg or "多处" in msg:
                return format_error(ErrorCode.EDIT_MULTIPLE_MATCH, msg)
            return format_error(ErrorCode.UNKNOWN, msg)

        # 写入
        try:
            full_path.write_text(result, encoding="utf-8")
        except PermissionError:
            return format_error(ErrorCode.PERMISSION_DENIED, f"无权限写入: {rel_path}")
        except Exception as e:
            return format_error(ErrorCode.FILE_WRITE_FAILED, f"写入文件失败: {e}")

        return f"✅ 成功修改文件: {rel_path}"
