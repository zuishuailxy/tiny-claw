"""上下文压缩器

监控和压缩上下文内存，防止大模型 OOM。

淘汰策略按「信息价值」分级，而不是单纯按时间远近一刀切：

    成功的大段工具输出 → 信息已被模型消化，且可用 offset/limit 重新取回 → 先丢
    模型的推理文本    → 体积小但承载决策链                          → 次之
    失败记录          → 含 [ERR:CODE] 与救援提示，丢掉会让人重复踩坑  → 最后才压
    System / 用户原始指令 / tool_calls           → 永不淘汰

且淘汰是「渐进式」的：从最低价值档开始，一降到阈值以下就立刻停手，
避免旧实现那种「一超阈值就把全部远期历史一次性掩码」的过度压缩。
"""

import logging

from tiny_claw.context.tokens import estimate_messages_tokens
from tiny_claw.schema import Message, Role

logger = logging.getLogger("tiny-claw.context.compactor")

# 工具层用该前缀统一标记失败（见 registry 与 recovery），压缩时必须识别
_ERR_PREFIX = "[ERR:"

# ── 淘汰档位：数字越大，越先被淘汰 ──
_TIER_IMMORTAL = 0  # System / 用户原始指令 / 失败记录 —— 永不主动淘汰
_TIER_REASONING_RECENT = 1  # 近期模型推理
_TIER_REASONING_OLD = 2  # 远期模型推理
_TIER_TOOL_RECENT = 3  # 近期成功的工具输出
_TIER_TOOL_OLD = 4  # 远期成功的工具输出
_TIER_REMINDER_OLD = 5  # 已跑出工作记忆窗口的系统提醒 —— 最先淘汰

# ⚠️ 为什么提醒消息需要单独一档：它们走 user 通道（协议不允许中途插 system 消息），
# 但**不是**用户指令。若混进「用户原始指令」那一档就会永不淘汰 ——
# 而打断指令恰恰是在上下文吃紧时才产生的，于是形成正反馈：
# 上下文越满 → 越容易失败 → 越多打断 → 越多永生消息 → 上下文越满。
# 近期提醒（还在工作记忆窗口内）仍需保留，模型得先看见它才谈得上响应。

_HEAD_TAIL_KEEP = 500  # 掐头去尾时头尾各保留的字符数
_ERR_HEAD_TAIL_KEEP = 1500  # 失败记录头尾保留量：错误码在头、救援提示在尾，两边都不能丢
_FOLD_THRESHOLD = 200  # 短于该长度的内容压缩没有意义，原样保留


def _is_failure(content: str) -> bool:
    """工具层以 [ERR:CODE] 前缀统一标记失败（见 registry / recovery）"""
    return content.startswith(_ERR_PREFIX)


class Compactor:
    """上下文压缩器"""

    def __init__(self, max_tokens: int = 8000, retain_last: int = 6):
        self.max_tokens = max_tokens
        self.retain_last = retain_last

    def compact(self, messages: list[Message]) -> list[Message]:
        """压缩消息列表，返回压缩后的副本（不修改原数组）。"""
        current_tokens = self._estimate_tokens(messages)

        if current_tokens < self.max_tokens:
            return messages

        logger.info(
            "⚠️ 内存告警：上下文约 %d token 超过阈值 (%d)，按信息价值分级淘汰...",
            current_tokens,
            self.max_tokens,
        )

        protect_start = max(0, len(messages) - self.retain_last)
        result = [self._copy(m) for m in messages]

        # 从最低价值档开始逐档淘汰；一降到阈值以下就停手，避免过度压缩
        for tier, action in (
            (_TIER_REMINDER_OLD, self._mask_reminder),
            (_TIER_TOOL_OLD, self._mask),
            (_TIER_TOOL_RECENT, self._truncate),
            (_TIER_REASONING_OLD, self._fold),
            (_TIER_REASONING_RECENT, self._fold),
        ):
            for i, msg in enumerate(result):
                if self._tier(i, msg, protect_start) != tier:
                    continue
                action(msg)
                if self._estimate_tokens(result) < self.max_tokens:
                    break
            if self._estimate_tokens(result) < self.max_tokens:
                break

        # 兜底：上述全部淘汰完仍超阈值，才动失败记录 —— 只截断，绝不掩码
        for msg in result:
            if self._estimate_tokens(result) < self.max_tokens:
                break
            if msg.role != Role.SYSTEM and _is_failure(msg.content):
                self._truncate_error(msg)

        new_tokens = self._estimate_tokens(result)
        logger.info(
            "✅ 压缩完成。上下文从 %d 降至 %d token。", current_tokens, new_tokens
        )
        return result

    # ── 淘汰动作 ──

    @staticmethod
    def _mask(msg: Message) -> None:
        """远期成功工具输出：全量掩码，保留长度提示让模型知道信息丢了。"""
        if len(msg.content) > _FOLD_THRESHOLD:
            msg.content = (
                f"...[为了节省内存，早期的工具输出已被系统强制清理。"
                f"原始长度: {len(msg.content)} 字符]..."
            )

    @staticmethod
    def _mask_reminder(msg: Message) -> None:
        """已生效的系统提醒：整条抹掉，只留痕迹。

        提醒是一次性的「现在发生了什么」，模型响应过之后就不再有价值；
        留着只会挤占配额。用带说明的占位而不是空串，避免模型以为消息丢了。
        """
        if msg.content:
            msg.content = "...[早前的系统提醒已清理]..."

    @staticmethod
    def _truncate(msg: Message) -> None:
        """近期成功工具输出：掐头去尾。命令回显的头部有价値，报错堆栈在尾部。"""
        max_keep = _HEAD_TAIL_KEEP * 2
        if len(msg.content) > max_keep:
            head = msg.content[:_HEAD_TAIL_KEEP]
            tail = msg.content[-_HEAD_TAIL_KEEP:]
            dropped = len(msg.content) - max_keep
            msg.content = (
                f"{head}\n\n...[内容过长，中间 {dropped} 字符已被系统截断]...\n\n{tail}"
            )

    @staticmethod
    def _fold(msg: Message) -> None:
        """模型推理文本：折叠成一行。"""
        if len(msg.content) > _FOLD_THRESHOLD:
            msg.content = "...[推理思考过程已折叠]..."

    @staticmethod
    def _truncate_error(msg: Message) -> None:
        """失败记录兜底截断：错误码在头部、系统救援提示在尾部，两边都必须保住。"""
        max_keep = _ERR_HEAD_TAIL_KEEP * 2
        if len(msg.content) > max_keep:
            head = msg.content[:_ERR_HEAD_TAIL_KEEP]
            tail = msg.content[-_ERR_HEAD_TAIL_KEEP:]
            dropped = len(msg.content) - max_keep
            msg.content = (
                f"{head}\n\n...[失败记录过长，中间 {dropped} 字符已被系统截断]...\n\n{tail}"
            )

    # ── 分级判定 ──

    @staticmethod
    def _tier(index: int, msg: Message, protect_start: int) -> int:
        """判定消息的信息价值档位：数字越大越先被淘汰。"""
        if msg.role == Role.SYSTEM:
            return _TIER_IMMORTAL
        # 系统提醒不是用户指令：还在工作记忆窗口内才保留，跑出去就可以清理
        if msg.is_system_reminder:
            return (
                _TIER_IMMORTAL
                if index >= protect_start
                else _TIER_REMINDER_OLD
            )
        # 无 tool_call_id 的 USER 消息是用户原始指令，承载任务目标
        if msg.role == Role.USER and not msg.tool_call_id:
            return _TIER_IMMORTAL
        # 失败记录含错误码与救援提示，丢掉会让模型重复踩同一个坑
        if _is_failure(msg.content):
            return _TIER_IMMORTAL

        in_working_memory = index >= protect_start
        if msg.tool_call_id:
            return _TIER_TOOL_RECENT if in_working_memory else _TIER_TOOL_OLD
        if msg.role == Role.ASSISTANT:
            return _TIER_REASONING_RECENT if in_working_memory else _TIER_REASONING_OLD
        return _TIER_IMMORTAL

    @staticmethod
    def _copy(msg: Message) -> Message:
        """拷贝消息。usage 与 reasoning 必须一并带上，否则压缩会静默吞掉计费与推理链。"""
        return Message(
            role=msg.role,
            content=msg.content,
            tool_calls=list(msg.tool_calls),
            tool_call_id=msg.tool_call_id,
            usage=msg.usage,
            reasoning=msg.reasoning,
            is_system_reminder=msg.is_system_reminder,
        )

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """估算上下文总 token 数（阈值判定用）

        ⚠️ 与提示语的分工别搞混：
        - **阈值判定用 token** —— 它是安全边界，模型窗口按 token 计费和截断
        - **掩码/截断提示语报字符数** —— 模型要靠它判断"值不值得重读"，
          而字符数比 token 数直观得多（模型对 token 没有体感）
        """
        # 口径统一收敛到 context.tokens，benchmark 插桩复用同一函数做校准
        return estimate_messages_tokens(messages)
