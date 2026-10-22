"""错误恢复管理器

基于标准化 Error Code 的错误分析与自愈提示。
工具层统一以 [ERR:CODE] 前缀输出错误，RecoveryManager 解析后按 switch-case 分发。
"""

import re
from enum import StrEnum

# --- 标准化错误码 ---

ERR_PREFIX = "[ERR:"
ERR_PATTERN = re.compile(r"^\[ERR:(\w+)\]\s*")


class ErrorCode(StrEnum):
    """工具层统一错误码。以 [ERR:CODE] 前缀格式嵌入错误消息。"""

    # ---- edit_file ----
    EDIT_OLD_TEXT_NOT_FOUND = "EDIT_OLD_TEXT_NOT_FOUND"
    EDIT_MULTIPLE_MATCH = "EDIT_MULTIPLE_MATCH"

    # ---- read_file / write_file ----
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    IS_DIRECTORY = "IS_DIRECTORY"
    NOT_TEXT_FILE = "NOT_TEXT_FILE"
    FILE_WRITE_FAILED = "FILE_WRITE_FAILED"
    FILE_READ_FAILED = "FILE_READ_FAILED"

    # ---- bash ----
    BASH_TIMEOUT = "BASH_TIMEOUT"
    BASH_COMMAND_NOT_FOUND = "BASH_COMMAND_NOT_FOUND"
    BASH_SYNTAX_ERROR = "BASH_SYNTAX_ERROR"
    BASH_NONZERO_EXIT = "BASH_NONZERO_EXIT"

    # ---- MCP（第三方能力，错误必须自带前缀，不能靠 registry 兜底）----
    MCP_SERVER_UNAVAILABLE = "MCP_SERVER_UNAVAILABLE"
    MCP_TOOL_FAILED = "MCP_TOOL_FAILED"
    MCP_TOOL_ERROR = "MCP_TOOL_ERROR"
    MCP_TIMEOUT = "MCP_TIMEOUT"

    # ---- 通用 ----
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_BLOCKED = "TOOL_BLOCKED"
    MISSING_PARAM = "MISSING_PARAM"
    UNKNOWN = "UNKNOWN"


def format_error(code: ErrorCode, message: str) -> str:
    """以统一格式输出错误：`[ERR:CODE] message`"""
    return f"{ERR_PREFIX}{code.value}] {message}"


def extract_code(error_message: str) -> ErrorCode | None:
    """从错误消息中提取 ErrorCode，失败返回 None"""
    m = ERR_PATTERN.match(error_message)
    if not m:
        return None
    try:
        return ErrorCode(m.group(1))
    except ValueError:
        return None


# --- 恢复管理器 ---

# 每个 ErrorCode → 救援提示的映射表
_RECOVERY_HINTS: dict[ErrorCode, str] = {
    ErrorCode.EDIT_OLD_TEXT_NOT_FOUND: (
        "你提供的 old_text 与文件当前内容不一致，或者缺少必要的缩进。"
        "请先使用 `read_file` 工具重新读取该文件，获取最新、准确的内容后，再重新发起编辑。"
    ),
    ErrorCode.EDIT_MULTIPLE_MATCH: (
        "你的 old_text 不够具体，命中了多个相同代码块。"
        "请在 old_text 中增加上下相邻的几行代码，以确保替换的唯一性。"
    ),
    ErrorCode.FILE_NOT_FOUND: (
        "路径似乎不正确。请不要凭空猜测，先使用 `bash` 执行 `ls -la` "
        "或 `find . -name` 命令查找正确的目录结构和文件名。"
    ),
    ErrorCode.PERMISSION_DENIED: (
        "你没有权限操作该文件。请检查工作区限制，或者思考是否需要修改其他文件。"
    ),
    ErrorCode.PATH_TRAVERSAL: (
        "你提供的路径试图访问工作区之外的文件，已被系统拦截。"
        "请使用工作区内的相对路径，不要使用 ../ 跳出工作区。"
    ),
    ErrorCode.IS_DIRECTORY: (
        "你指定的是一个目录，不是文件。请检查路径，或使用 `bash ls` 查看目录内容。"
    ),
    ErrorCode.NOT_TEXT_FILE: ("该文件不是文本文件，无法读取。请确认文件类型是否正确。"),
    ErrorCode.BASH_TIMEOUT: (
        "该命令执行被超时强杀。如果它是一个常驻服务（如 server 或 watch），"
        "请将其转入后台执行（例如使用 `nohup ... &`），不要阻塞主线程。"
    ),
    ErrorCode.BASH_COMMAND_NOT_FOUND: (
        "系统中未安装该命令。请先思考：是否有替代命令？"
        "或者你需要先编写脚本进行安装？"
    ),
    ErrorCode.BASH_SYNTAX_ERROR: (
        "Bash 语法错误。请检查引号转义或特殊字符，确保命令在终端中可直接运行。"
    ),
    ErrorCode.BASH_NONZERO_EXIT: (
        "命令执行返回了非零退出码。请仔细阅读 stderr 输出，"
        "分析失败原因并尝试修正命令后重试。"
    ),
    ErrorCode.TOOL_NOT_FOUND: (
        "你尝试调用了一个不存在的工具。请检查工具名是否正确，"
        "或使用系统提供的其他工具完成任务。"
    ),
    ErrorCode.TOOL_BLOCKED: (
        "这次调用被执行策略拦截了（人工审批被拒绝，或审批超时自动拒绝）。"
        "**不要用相同参数重试**——重试不会通过。请改用不需要审批的等价操作"
        "（例如换个更安全的命令、或改用内置读写工具），"
        "或者向用户说明你为什么必须执行这一步。"
    ),
    ErrorCode.MISSING_PARAM: (
        "工具调用缺少必填参数。请检查参数列表，确保所有 required 参数都已提供。"
    ),
    ErrorCode.FILE_READ_FAILED: (
        "读取文件时发生未知错误。请检查文件是否被其他进程锁定，"
        "或尝试使用 `bash ls -la` 确认文件状态。"
    ),
    ErrorCode.FILE_WRITE_FAILED: (
        "写入文件时发生未知错误。请检查磁盘空间是否充足，"
        "或尝试使用 `bash df -h` 确认磁盘状态。"
    ),
    # ---- MCP ----
    # 共同的要点：明确告诉模型「换内置工具」，而不是反复重试同一个失效的第三方工具
    ErrorCode.MCP_SERVER_UNAVAILABLE: (
        "该 MCP Server 当前不可用（进程未启动或已退出）。"
        "请改用内置工具（read_file / bash 等）完成当前步骤，"
        "不要用相同参数反复重试同一个 MCP 工具。"
    ),
    ErrorCode.MCP_TOOL_FAILED: (
        "MCP 工具调用失败，可能是连接中断或对端进程异常。"
        "请先确认参数无误；若仍失败，改用内置工具完成当前步骤，不要反复重试。"
    ),
    ErrorCode.MCP_TOOL_ERROR: (
        "MCP Server 报告该工具执行失败。请仔细阅读上面的错误内容，"
        "调整参数或换一个工具再试；若确认参数无误，改用内置工具完成当前步骤。"
    ),
    ErrorCode.MCP_TIMEOUT: (
        "MCP 工具调用超时（对端可能卡住）。"
        "请缩小操作范围后重试一次；仍超时就改用内置工具，不要用相同参数反复重试。"
    ),
    # UNKNOWN 不映射 — 让 LLM 自行分析
}


class RecoveryManager:
    """基于 ErrorCode 的表驱动错误救援系统"""

    def analyze_and_inject(self, raw_error: str) -> str:
        """解析错误码，匹配救援提示，返回增强后的报错信息"""
        code = extract_code(raw_error)

        if code is None:
            return raw_error

        hint = _RECOVERY_HINTS.get(code)
        if hint is None:
            return raw_error

        return f"{raw_error}\n\n[系统救援指南]: {hint}"
