"""WriteFileTool

创建或覆盖写入文件，带路径穿越防护和自动创建父目录。
"""

from pathlib import Path
from typing import Any

from tiny_claw.tools.base import BaseTool
from tiny_claw.schema import ToolDefinition
from tiny_claw.context.recovery import ErrorCode, format_error


class WriteFileTool(BaseTool):
    """在工作区内创建或覆盖写入文件"""

    def __init__(self, work_dir: str):
        self._work_dir = Path(work_dir).resolve()

    def name(self) -> str:
        return "write_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description="创建或覆盖写入一个文件。如果目录不存在会自动创建。请提供相对路径。",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要写入的文件路径，如 src/main.py",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文件内容",
                    },
                },
                "required": ["path", "content"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        rel_path = arguments.get("path", "")
        content = arguments.get("content", "")

        if not rel_path:
            return format_error(ErrorCode.MISSING_PARAM, "缺少 path 参数")

        # 路径穿越防护
        full_path = (self._work_dir / rel_path).resolve()
        if not str(full_path).startswith(str(self._work_dir)):
            return format_error(
                ErrorCode.PATH_TRAVERSAL,
                f"'{rel_path}' 超出了工作区范围",
            )

        try:
            # 自动创建父目录
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
        except PermissionError:
            return format_error(ErrorCode.PERMISSION_DENIED, f"无权限写入: {rel_path}")
        except IsADirectoryError:
            return format_error(ErrorCode.IS_DIRECTORY, f"'{rel_path}' 是一个目录，无法写入")
        except Exception as e:
            return format_error(ErrorCode.FILE_WRITE_FAILED, f"写入文件失败: {e}")

        return f"成功将内容写入到文件: {rel_path}"
