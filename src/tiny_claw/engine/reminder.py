"""死循环打断器

监控工具调用指纹，当模型连续 3 次以相同参数调用同一工具且均失败时，
注入强力打断消息，强制 LLM 跳出局部执念。
"""

import hashlib
import json
import logging

from tiny_claw.schema import (
    SYSTEM_REMINDER_PREFIX,
    Message,
    Role,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger("tiny-claw.engine.reminder")

# 触发打断的连续失败阈值
NAG_THRESHOLD = 3

# 打断消息模板
NUDGE_TEMPLATE = SYSTEM_REMINDER_PREFIX + """ 警告] 
你似乎陷入了死循环。你刚刚连续 {fail_count} 次使用相同的参数调用了 '{tool_name}' 工具，并且都失败了。
请立即停止这种无效的重试！你的注意力被当前的报错过度吸引了。
你需要：
1. 停止猜测参数。跳出当前的局部思维。
2. 彻底改变你的策略。
3. 如果你确实无法通过系统工具解决当前问题，请直接结束任务并向用户说明你需要什么人工帮助，而不是继续盲目消耗 API 资源尝试。"""


class ReminderInjector:
    """基于调用指纹的死循环检测与打断"""

    def __init__(self):
        # 指纹 → 连续失败次数
        self._consecutive_failures: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 指纹生成
    # ------------------------------------------------------------------
    @staticmethod
    def _fingerprint(tool_name: str, arguments: dict) -> str:
        """对 (工具名 + 参数) 做 MD5，生成唯一调用指纹"""
        args_json = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        raw = f"{tool_name}:{args_json}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # 核心检测逻辑
    # ------------------------------------------------------------------
    def check_and_inject(
        self, tool_call: ToolCall, result: ToolResult
    ) -> Message | None:
        """分析工具执行结果，必要时返回打断消息。

        Returns:
            Message | None: 需要打断时返回一条 Role.USER 消息，否则 None
        """
        fingerprint = self._fingerprint(tool_call.name, tool_call.arguments)

        # 成功 → 清空所有计数器，Agent 走出了死胡同
        if not result.is_error:
            self._consecutive_failures.clear()
            return None

        # 失败 → 累加该指纹的失败计数
        self._consecutive_failures[fingerprint] = (
            self._consecutive_failures.get(fingerprint, 0) + 1
        )
        fail_count = self._consecutive_failures[fingerprint]

        logger.info(
            "[Reminder] 监控到工具 %s 执行失败，该参数特征连续失败次数: %d",
            tool_call.name,
            fail_count,
        )

        # 未达阈值 → 暂不干预
        if fail_count < NAG_THRESHOLD:
            return None

        # 触发打断！
        logger.warning("[Reminder] ⚠️ 触发死循环干预！注入强力修正指令。")

        nudge_msg = NUDGE_TEMPLATE.format(
            fail_count=fail_count,
            tool_name=tool_call.name,
        )
        return Message(
            role=Role.USER,
            content=nudge_msg,
            is_system_reminder=True,  # 不是用户指令，压缩时可清理
        )
