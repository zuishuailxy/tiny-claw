"""飞书审批管理器

高危操作在执行前挂起，通过飞书发送审批请求，等待人工确认后放行或拒绝。
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from tiny_claw.feishu.bot import FeishuReporter

# Reporter 可以是一个直接引用，也可以是一个延迟解析的 callable
ReporterLike = Union["FeishuReporter", Callable[[], "FeishuReporter | None"], None]

logger = logging.getLogger("tiny-claw.feishu.approve")

# =========================================================================
# 数据结构
# =========================================================================


@dataclass
class ApprovalResult:
    """审批结果"""

    allowed: bool
    reason: str = ""


# =========================================================================
# 高危命令检测
# =========================================================================

# Bash 高危模式正则
_DANGEROUS_BASH_PATTERNS: list[re.Pattern] = [
    re.compile(r"rm\s+-r", re.IGNORECASE),  # 级联删除
    re.compile(r"sudo\s+", re.IGNORECASE),  # 提权操作
    re.compile(r"drop\s+", re.IGNORECASE),  # 数据库危险命令
    re.compile(r">.*\.py", re.IGNORECASE),  # 恶意覆盖 Python 源码
    re.compile(r"nginx\s+-s", re.IGNORECASE),  # Nginx 服务重启或停止
    re.compile(r"systemctl\s+", re.IGNORECASE),  # 系统级服务管理
    re.compile(r"kill\s+", re.IGNORECASE),  # 杀进程操作
]


def _mcp_needs_approval(tool_name: str, args: str) -> bool:
    """MCP（第三方）工具的审批策略。

    ⚠️ 为什么必须有这个分支：下面按**裸工具名**匹配，而 MCP 工具名带 `mcp__` 前缀，
    一条规则都不会命中 → 落到兜底 `return False` → **一个能写文件、能移动文件的
    第三方工具默认放行**（实测见 docs/MCP_INTEGRATION_PLAN.md §12.4）。

    策略取自 `mcp.json` 里每个 server 的 `approval` 字段：
    `always`（默认，最保守）/ `never` / `dangerous`（退回参数关键字匹配）。
    **查不到策略时按 always 处理**——查不到不等于可信。
    """
    policy = None
    try:
        from tiny_claw.mcp import MCPManager

        mgr = MCPManager.current()  # 不建新实例，没跑过 MCP 就返回 None
        if mgr is not None:
            policy = mgr.approval_for(tool_name)
    except Exception:  # noqa: BLE001 —— 审批判定绝不能因为查策略失败而放行
        policy = None

    policy = policy or "always"

    if policy == "never":
        return False
    if policy == "always":
        return True
    # dangerous：退回到与 bash 相同的参数关键字匹配
    return any(p.search(args) for p in _DANGEROUS_BASH_PATTERNS)


def is_dangerous_command(tool_name: str, args: str) -> bool:
    """判断该工具调用是否需要触发审批。

    - **MCP 工具（`mcp__` 前缀）**：按 server 配置的审批策略，默认最保守（挂起）
    - 纯读取工具（read_file）：白名单放行，默认 YOLO 模式
    - write_file / edit_file：修改任何文件都是高危操作，一律挂起审批
    - bash：按高危特征库匹配，命中即拦截
    """
    # 第三方工具优先判定：加前缀后不会命中下面任何裸名规则
    if tool_name.startswith("mcp__"):
        return _mcp_needs_approval(tool_name, args)

    # 白名单放行：纯读取工具
    if tool_name == "read_file":
        return False

    # write_file / edit_file — 始终需要审批
    if tool_name in ("write_file", "edit_file"):
        return True

    # bash — 按高危模式匹配
    if tool_name == "bash":
        for pattern in _DANGEROUS_BASH_PATTERNS:
            if pattern.search(args):
                return True

    # 未命中高危特征，默认放行（如 ls -la, tail -n 50 等探测命令）
    return False


# =========================================================================
# 审批管理器
# =========================================================================


class ApprovalManager:
    """统一管理当前正在等待人工审批的任务。

    作为全局单例，在 Registry Middleware 和飞书消息回调之间共享状态：
    - Middleware 调用 wait_for_approval() 挂起协程
    - 飞书 Webhook 收到 "approve/reject <task_id>" 时调用 resolve_approval() 唤醒
    """

    def __init__(self):
        # task_id → asyncio.Future[ApprovalResult]
        self._pending: dict[str, asyncio.Future[ApprovalResult]] = {}

    # ------------------------------------------------------------------
    # 挂起等待
    # ------------------------------------------------------------------
    async def wait_for_approval(
        self,
        tool_name: str,
        args: str,
        reporter: ReporterLike = None,
        timeout: float = 300.0,
    ) -> ApprovalResult:
        """发送审批通知，挂起协程等待人工批复。

        Args:
            tool_name: 工具名
            args: 工具参数（用于展示）
            reporter: 飞书消息发送器（None 时回退到终端打印）
            timeout: 超时秒数，超时后自动拒绝

        Returns:
            ApprovalResult(allowed, reason)
        """
        task_id = uuid.uuid4().hex[:8]  # 短 ID 方便人类输入

        # 创建 Future 用于阻塞
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalResult] = loop.create_future()
        self._pending[task_id] = future

        # 构建通知消息
        notice = (
            f"⚠️ **高危操作审批请求**\n"
            f"Agent 试图执行以下动作:\n"
            f"- 工具: `{tool_name}`\n"
            f"- 参数: `{args}`\n"
            f"\n"
            f"任务 ID: **{task_id}**\n"
            f"\n"
            f"👉 请回复 `approve {task_id}` 或 `reject {task_id}` 来决定是否放行。\n"
            f"⏰ 超时 {timeout:.0f}s 后将自动拒绝。"
        )

        # 延迟解析 reporter（支持 callable 模式）
        r = reporter() if callable(reporter) else reporter

        # 发送通知
        if r:
            await r.send_msg(notice)
        else:
            # 回退到终端
            print(f"\n\033[31m[需要审批 TaskID: {task_id}]\033[0m")
            print(notice)

        logger.info("[Approval] 已发送审批请求 (TaskID: %s)，协程挂起等待...", task_id)

        try:
            # 挂起等待（带超时）
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[Approval] 审批超时 (TaskID: %s)，自动拒绝", task_id)
            result = ApprovalResult(allowed=False, reason="审批超时，自动拒绝")

        # 清理
        self._pending.pop(task_id, None)
        return result

    # ------------------------------------------------------------------
    # 回调唤醒
    # ------------------------------------------------------------------
    def resolve_approval(self, task_id: str, allowed: bool, reason: str = "") -> bool:
        """由飞书消息回调触发，向挂起的协程发送审批结果。

        Returns:
            bool: 是否成功唤醒（task_id 不存在时返回 False）
        """
        future = self._pending.get(task_id)
        if future is None:
            logger.info(
                "[Approval] 找不到对应的 TaskID: %s，可能已超时或处理完毕", task_id
            )
            return False

        if future.done():
            logger.info("[Approval] TaskID: %s 已完成，忽略重复回调", task_id)
            return False

        logger.info(
            "[Approval] 收到审批结果 (TaskID: %s, Allowed: %s)",
            task_id,
            allowed,
        )
        future.set_result(ApprovalResult(allowed=allowed, reason=reason))
        return True


# =========================================================================
# 全局单例
# =========================================================================

global_approval_mgr = ApprovalManager()


# =========================================================================
# 便捷工厂：创建审批中间件
# =========================================================================


def create_approval_middleware(
    reporter: ReporterLike = None,
    timeout: float = 300.0,
):
    """生成一个可与 Registry.use() 配合的审批中间件。

    Usage:
        from tiny_claw.feishu.approve import create_approval_middleware

        # 直接引用（CLI 模式）
        registry.use(create_approval_middleware(reporter=None))

        # 延迟解析（飞书模式，reporter 在 handle_agent 时才绑定）
        registry.use(create_approval_middleware(reporter=bot.reporter))

    Args:
        reporter: 飞书消息发送器，或返回 Reporter 的可调用对象。
                  None 时回退到终端打印。
        timeout: 审批超时秒数。
    """
    import json

    async def _mw(call):
        # 延迟解析 reporter（支持 bot.reporter 属性延迟绑定）
        r = reporter() if callable(reporter) else reporter

        # 提取参数为可读字符串
        try:
            args_str = json.dumps(call.arguments, ensure_ascii=False)
        except Exception:
            args_str = str(call.arguments)

        # 判断是否需要审批
        if not is_dangerous_command(call.name, args_str):
            return True, ""

        # 挂起等待审批
        result = await global_approval_mgr.wait_for_approval(
            tool_name=call.name,
            args=args_str,
            reporter=r,
            timeout=timeout,
        )

        return result.allowed, result.reason

    return _mw
