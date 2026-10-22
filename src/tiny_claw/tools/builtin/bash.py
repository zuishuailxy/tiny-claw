"""BashTool

在工作区执行 bash 命令，带超时控制、自愈机制、输出截断。
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from tiny_claw import config
from tiny_claw.tools.base import BaseTool
from tiny_claw.schema import ToolDefinition
from tiny_claw.context.recovery import ErrorCode, format_error

logger = logging.getLogger("tiny-claw.tools.bash")

# 常量保留原名（模块内引用），真实来源已迁至 tiny_claw.config
TIMEOUT = config.BASH_TIMEOUT  # 命令最大执行秒数
MAX_OUTPUT = config.BASH_MAX_OUTPUT  # 最大输出字符数（⚠️ 字符口径，不是字节）


class BashTool(BaseTool):
    """在当前工作区执行 bash 命令"""

    def __init__(self, work_dir: str):
        self._work_dir = str(Path(work_dir).resolve())

    def name(self) -> str:
        return "bash"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name(),
            description=(
                "在当前工作区执行任意的 bash 命令。支持链式命令(如 &&)。"
                "返回标准输出(stdout)和标准错误(stderr)。"
                "对于常驻服务（如 npm run dev），请设置 background=true 避免阻塞。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 bash 命令，例如: ls -la 或 python test.py",
                    },
                    "background": {
                        "type": "boolean",
                        "description": "是否为后台常驻任务（如 npm run dev）。true 时立即返回 PID 不等待。",
                    },
                },
                "required": ["command"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        command = arguments.get("command", "")
        background = arguments.get("background", False)
        if not command:
            return format_error(ErrorCode.MISSING_PARAM, "缺少 command 参数")

        try:
            # 通过 bash -c 执行，支持管道、&& 等 Shell 语法
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._work_dir,
            )

            # 后台模式：不等待，直接返回 PID
            if background:
                return (
                    f"后台任务已启动 (PID: {proc.pid})。使用 `kill {proc.pid}` 可终止。"
                )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            return format_error(
                ErrorCode.BASH_TIMEOUT,
                f"命令执行超时({TIMEOUT}s)，已被系统强制终止。如果是启动常驻服务，请尝试将其转入后台",
            )
        except FileNotFoundError:
            return format_error(ErrorCode.BASH_COMMAND_NOT_FOUND, "bash 不可用")
        except Exception as e:
            return format_error(ErrorCode.UNKNOWN, f"执行报错: {e}")

        output = stdout.decode("utf-8", errors="replace")
        if stderr:
            output += stderr.decode("utf-8", errors="replace")

        # 自愈机制：bash 报错不抛异常，把错误原样返回让模型自己分析
        if proc.returncode != 0:
            detail = output.strip() if output else f"exit code {proc.returncode}"
            return format_error(ErrorCode.BASH_NONZERO_EXIT, detail)

        if not output:
            return "命令执行成功，无终端输出。"

        # 长度截断保护（防 OOM）
        # ⚠️ 判定与切分必须同为字符口径。旧实现是「字节判断 + 字符切分」，
        # 中文（UTF-8 一字 3 字节）实际会截出 3 倍于阈值的内容，且提示语宣称的
        # 字节数与事实不符——与 read_file 曾踩的坑完全相同。
        if len(output) > MAX_OUTPUT:
            return (
                output[:MAX_OUTPUT]
                + f"\n\n...[终端输出过长，已截断至前 {MAX_OUTPUT} 字符]..."
            )

        return output
