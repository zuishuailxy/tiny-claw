"""ReadFileTool

读取本地文件内容，带路径穿越防护、输出卸载与局部读取能力。

长文件的三层防御（由近及远）：

1. **局部读取**：`offset` / `limit` 按行号取片段 —— 这是模型"按需翻阅"长文件的前提。
   没有它，卸载就会制造死循环：读全文 → 超阈值 → 卸载 → 再读全文 → 又卸载。
2. **输出卸载（Offloading）**：单次读取超过阈值时，完整内容落盘到 `.claw/offload/`，
   只回传「行数统计 + 头尾预览 + 卸载路径 + 用法说明」。
   卸载是**正常结果不是错误**（is_error=False），模型据此自行决定如何分段。
3. **硬截断**：兜底，保证任何情况下单次工具输出都不会撑爆上下文。

⚠️ 历史 bug（已修）：旧实现按**字节**判断（`len(content.encode("utf-8"))`）、
按**字符**切分（`content[:MAX_BYTES]`）。UTF-8 下一个汉字占 3 字节，
导致中文文件实际返回 24000 字节 = 设计阈值的 3 倍，且提示语宣称"截断至前 8000 字节"
与事实不符。现统一为**字符**口径。
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

from tiny_claw import config
from tiny_claw.tools.base import BaseTool
from tiny_claw.schema import ToolDefinition
from tiny_claw.context.recovery import ErrorCode, format_error

logger = logging.getLogger("tiny-claw.tools.read_file")

# 以下常量保留原名（单测直接 import 它们），但真实来源已迁至 tiny_claw.config
MAX_CHARS = config.READ_MAX_CHARS  # 单次返回的最大字符数（⚠️ 字符口径，不是字节）
PREVIEW_LINES = config.PREVIEW_LINES  # 卸载时头尾各预览的行数
PREVIEW_CHARS = 2000  # 行数过少（如单行超长文件）时，改用字符预览的头尾长度
OFFLOAD_DIR = config.OFFLOAD_DIR  # 卸载目录，位于工作区内（子 agent 同 work_dir 可读到）


def _to_positive_int(value: Any) -> int | None:
    """把模型传入的参数转成正整数。

    模型偶尔会把 100 写成 "100"，直接 int() 容错比报错更划算；
    无法解析时返回 None（视为未传该参数）。
    """
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


class ReadFileTool(BaseTool):
    """读取工作区内指定路径的文件内容"""

    def __init__(self, work_dir: str):
        self._work_dir = Path(work_dir).resolve()

    def name(self) -> str:
        return "read_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description=(
                "读取指定路径的文件内容。请提供相对工作区的路径。"
                "文件较长时可用 offset/limit 按行号分段读取。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读取的文件路径，如 src/tiny_claw/main.py",
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "起始行号（从 1 开始）。省略则从头读取。"
                            "文件被卸载时用它指定要查看的片段起点。"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "读取的行数。省略则读至文件末尾（仍可能触发卸载）。"
                            "建议单次不超过 200 行。"
                        ),
                    },
                },
                "required": ["path"],
            },
        )

    # ------------------------------------------------------------------
    # 卸载
    # ------------------------------------------------------------------
    def _offload_target(self, rel_path: str) -> tuple[Path, str]:
        """计算卸载文件的落盘路径。

        同一源文件稳定映射到同一卸载文件（重复读取覆盖而非堆积垃圾），
        文件名带 8 位 hash 防止不同目录下的同名文件互相覆盖。
        """
        digest = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]
        stem = Path(rel_path).name or "file"
        if len(stem) > 60:  # 防止文件名过长导致 OS 报错
            stem = stem[:60]
        filename = f"{stem}__{digest}.txt"
        return self._work_dir / OFFLOAD_DIR / filename, f"{OFFLOAD_DIR}/{filename}"

    def _build_offload_message(
        self, rel_path: str, off_rel: str, lines: list[str]
    ) -> str:
        """构造「头尾预览 + 路径引用 + 用法说明」的卸载消息。"""
        total = len(lines)

        # 行数过少（典型：单行压缩 JSON、超长单行日志）时行号分段没有意义，
        # 退化为按字符预览 —— 至少保证模型能看到首尾，而不是只看到开头。
        if total <= PREVIEW_LINES * 2:
            content = "\n".join(lines)
            head_txt = content[:PREVIEW_CHARS]
            tail_txt = content[-PREVIEW_CHARS:]
            parts = [
                f"[read_file] 文件过长：{rel_path} 共 {len(content)} 字符（{total} 行），"
                f"完整内容已卸载至 {off_rel}",
                "",
                "⚠️ 该文件行数很少但单行极长，按行号分段无效。",
                f"如需查看中间部分，请用 bash 工具（如 sed / cut）对 {off_rel} 做切片。",
                "",
                f"─── 头部 {PREVIEW_CHARS} 字符 ───",
                head_txt,
                "",
                f"─── 尾部 {PREVIEW_CHARS} 字符 ───",
                tail_txt,
            ]
            return "\n".join(parts)

        head = lines[:PREVIEW_LINES]
        tail = lines[-PREVIEW_LINES:] if total > PREVIEW_LINES else []
        tail_start = total - len(tail) + 1
        width = len(str(total))

        parts = [
            f"[read_file] 文件过长：{rel_path} 共 {total} 行，"
            f"完整内容已卸载至 {off_rel}",
            "",
            f"如需查看其余部分，请分段读取，例如：",
            f'  read_file(path="{rel_path}", offset={PREVIEW_LINES + 1}, limit=100)',
            f'  read_file(path="{rel_path}", offset={max(1, total - 99)}, limit=100)',
            f"也可直接读卸载文件：read_file(path=\"{off_rel}\", offset=1, limit=100)",
            "",
            f"─── 头部 1-{len(head)} 行 ───",
        ]

        for i, line in enumerate(head, start=1):
            parts.append(f"{i:>{width}}| {line}")

        if tail:
            parts.append("")
            parts.append(f"─── 尾部 {tail_start}-{total} 行 ───")
            for i, line in enumerate(tail, start=tail_start):
                parts.append(f"{i:>{width}}| {line}")

        parts.append("")
        parts.append(f"（中间 {max(0, total - len(head) - len(tail))} 行未显示）")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    async def execute(self, arguments: dict[str, Any]) -> str:
        # 1. 参数校验
        rel_path = arguments.get("path", "")
        if not rel_path:
            return format_error(ErrorCode.MISSING_PARAM, "缺少 path 参数")

        offset = _to_positive_int(arguments.get("offset"))
        limit = _to_positive_int(arguments.get("limit"))

        # 2. 拼接并解析绝对路径，防路径穿越 (../../etc/passwd)
        full_path = (self._work_dir / rel_path).resolve()

        if not str(full_path).startswith(str(self._work_dir)):
            return format_error(
                ErrorCode.PATH_TRAVERSAL,
                f"'{rel_path}' 超出了工作区范围",
            )

        # 3. 读取文件
        try:
            content = full_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return format_error(ErrorCode.FILE_NOT_FOUND, f"文件不存在: {rel_path}")
        except PermissionError:
            return format_error(ErrorCode.PERMISSION_DENIED, f"无权限读取: {rel_path}")
        except IsADirectoryError:
            return format_error(ErrorCode.IS_DIRECTORY, f"'{rel_path}' 是一个目录，不是文件")
        except UnicodeDecodeError:
            return format_error(ErrorCode.NOT_TEXT_FILE, f"'{rel_path}' 不是文本文件，无法读取")
        except Exception as e:
            return format_error(ErrorCode.FILE_READ_FAILED, f"读取文件失败: {e}")

        lines = content.splitlines()
        total_lines = len(lines)

        # 4. 局部读取模式：按行号取片段，输出带行号便于模型定位
        if offset is not None or limit is not None:
            start = (offset - 1) if offset is not None else 0
            start = min(start, max(0, total_lines - 1)) if total_lines else 0
            end = total_lines if limit is None else start + limit

            selected = lines[start:end]
            width = len(str(max(total_lines, start + len(selected))))
            body = "\n".join(
                f"{start + i + 1:>{width}}| {line}" for i, line in enumerate(selected)
            )

            header = f"[{rel_path}] 第 {start + 1}-{min(end, total_lines)} 行 / 共 {total_lines} 行\n"

            # 兜底：即使分段也可能过大（例如单行超长），强制截断并提示缩小 limit
            if len(body) > MAX_CHARS:
                body = (
                    body[:MAX_CHARS]
                    + f"\n...[超出单次返回上限 {MAX_CHARS} 字符，请缩小 limit 参数]..."
                )

            if not selected:
                return header + "（该范围内没有内容）"

            return header + body

        # 5. 完整读取：超过阈值则卸载（预览方式由 _build_offload_message 按行数决定）
        if len(content) > MAX_CHARS:
            target, off_rel = self._offload_target(rel_path)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                logger.info(
                    "[read_file] 文件过长已卸载: %s (%d 行) -> %s",
                    rel_path,
                    total_lines,
                    off_rel,
                )
                return self._build_offload_message(rel_path, off_rel, lines)
            except Exception as e:
                # 卸载失败不能让工具整体失败，降级为硬截断
                logger.error("[read_file] 卸载失败，降级为截断: %s", e)

        # 6. 兜底硬截断（字符口径，与判断条件一致）
        if len(content) > MAX_CHARS:
            content = (
                content[:MAX_CHARS]
                + f"\n\n...[内容过长，已截断至前 {MAX_CHARS} 字符]..."
            )

        return content
