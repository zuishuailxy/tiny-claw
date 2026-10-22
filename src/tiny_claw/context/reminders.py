"""System Reminders — 运行时状态回灌通道

与 System Prompt 的分工：

    System Prompt   静态，启动时组装，回答「你是谁、有哪些纪律」
    System Reminder 动态，运行时触发，回答「现在发生了什么」

没有后者的 agent 只能靠 System Prompt 里的静态纪律硬扛——
但「还剩几轮」「上下文快满了」这些状态启动时根本不存在，
模型看不见，就只能一直撞到南墙（跑满轮数被强制掐断、历史被静默压缩）。

三条硬约束，缺一条提醒就会退化成噪声：

1. **冷却**：同一类提醒在 N 轮内不重复发。否则每轮都喊「上下文快满了」，
   模型的反应不是收敛，而是麻木（并且每轮白烧一次 token）。
2. **限流**：单轮最多发 N 条。多条提醒同时出现时，模型往往一条都不听。
3. **不写 Session**：提醒每轮重算，只进当轮 context。
   写进 Session 会累积成历史垃圾，还会污染成本统计。

注入位置固定在工作记忆之后、压缩之前：提醒是「本轮决策前最后看到的话」，
放在末尾注意力最好；先压缩再追加则可能被一起清掉。
"""

import logging
from dataclasses import dataclass

from tiny_claw import config
from tiny_claw.schema import SYSTEM_REMINDER_PREFIX, Message, Role

logger = logging.getLogger("tiny-claw.context.reminders")


@dataclass(frozen=True)
class _Reminder:
    kind: str  # 冷却与去重的键
    content: str


def _build_context_pressure(ctx_tokens: int, ctx_limit: int) -> _Reminder:
    pct = int(ctx_tokens / ctx_limit * 100) if ctx_limit else 0
    return _Reminder(
        kind="context_pressure",
        content=(
            f"{SYSTEM_REMINDER_PREFIX} 上下文压力] "
            f"当前上下文约 {ctx_tokens:,} token，已达压缩阈值 {ctx_limit:,} 的 {pct}%。"
            "\n请立刻把中间结论与待办写入文件（write_file / TODO.md），"
            "不要依赖短期记忆——再往后早期历史会被压缩清理，届时就取不回来了。"
            "\n需要引用大文件时用 read_file 的 offset/limit 分段取，不要整篇读。"
        ),
    )


def _build_turn_limit(turn: int, max_turns: int) -> _Reminder:
    remain = max_turns - turn
    return _Reminder(
        kind="turn_limit",
        content=(
            f"{SYSTEM_REMINDER_PREFIX} 轮数预警] "
            f"你已运行 {turn}/{max_turns} 轮，还剩 {max(remain, 0)} 轮就会被强制终止。"
            "\n请立即收敛：优先完成当前子任务并给出阶段性总结，"
            "不要开启新的探索或重构。"
        ),
    )


class ReminderBus:
    """收集本轮该发的系统提醒，并施加冷却与限流。

    用法：每轮构建完 context 后调 `collect()`，把返回值 extend 到 context 末尾。
    """

    def __init__(
        self,
        cooldown_turns: int | None = None,
        max_per_turn: int | None = None,
    ):
        self._cooldown = (
            config.REMINDER_COOLDOWN_TURNS if cooldown_turns is None else cooldown_turns
        )
        self._max_per_turn = (
            config.REMINDER_MAX_PER_TURN if max_per_turn is None else max_per_turn
        )
        # kind -> 上次触发的轮次
        self._last_fired: dict[str, int] = {}

    # ------------------------------------------------------------------
    # 冷却
    # ------------------------------------------------------------------
    def _in_cooldown(self, kind: str, turn: int) -> bool:
        last = self._last_fired.get(kind)
        if last is None:
            return False
        return (turn - last) < self._cooldown

    # ------------------------------------------------------------------
    # 收集
    # ------------------------------------------------------------------
    def collect(
        self,
        *,
        turn: int,
        max_turns: int = 0,
        ctx_tokens: int = 0,
        ctx_limit: int = 0,
    ) -> list[Message]:
        """按当前运行状态生成提醒；已施加冷却与限流。

        Args:
            turn: 当前轮次（从 1 开始）
            max_turns: 本轮次上限，0 表示不限制
            ctx_tokens: 当前上下文的 token 估算
            ctx_limit: 压缩阈值，0 表示未知（此时不做压力判定）
        """
        candidates: list[_Reminder] = []

        if ctx_limit > 0 and ctx_tokens > 0:
            ratio = ctx_tokens / ctx_limit
            if ratio >= config.REMINDER_CONTEXT_PRESSURE:
                candidates.append(_build_context_pressure(ctx_tokens, ctx_limit))

        if max_turns > 0 and (max_turns - turn) <= config.REMINDER_TURN_WARN_REMAINING:
            candidates.append(_build_turn_limit(turn, max_turns))

        out: list[Message] = []
        for item in candidates:
            if self._in_cooldown(item.kind, turn):
                continue
            if len(out) >= self._max_per_turn:
                break
            self._last_fired[item.kind] = turn
            logger.info("[ReminderBus] 注入提醒: %s (turn=%d)", item.kind, turn)
            out.append(
                Message(
                    role=Role.USER,
                    content=item.content,
                    is_system_reminder=True,
                )
            )
        return out
