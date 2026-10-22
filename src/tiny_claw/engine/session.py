"""会话管理"""

import asyncio
from datetime import datetime, timezone

from tiny_claw.schema import Message


class Session:
    """一次持续的人机交互过程"""

    def __init__(self, session_id: str, work_dir: str):
        self.id = session_id
        self.work_dir = work_dir
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at

        # 累计 Token 消耗 & 费用
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_cny = 0.0

        # 容错/顺滑度指标
        self.error_turns = 0  # 至少一次工具调用失败的 Turn 数
        self.first_error_token: int | None = None  # 首次错误时的累计 Token 数
        self.total_turns = 0  # 总 Turn 数

        self._history: list[Message] = []
        self._lock = asyncio.Lock()

    @property
    def total_tokens(self) -> int:
        """当前累计 Token 总数"""
        return self.total_prompt_tokens + self.total_completion_tokens

    # ------------------------------------------------------------------
    # 容错追踪
    # ------------------------------------------------------------------
    async def mark_error_turn(self) -> None:
        """标记当前 Turn 有工具调用失败，并记录首次出错的 Token 位置"""
        async with self._lock:
            self.error_turns += 1
            if self.first_error_token is None:
                self.first_error_token = self.total_tokens

    # ------------------------------------------------------------------
    # 计费
    # ------------------------------------------------------------------
    async def record_usage(
        self, prompt: int, completion: int, cost: float = 0.0
    ) -> None:
        """累加本次调用的 Token 消耗和费用。

        由外部 Tracker 或 engine 在每次 LLM 调用后调用。
        """
        async with self._lock:
            self.total_prompt_tokens += prompt
            self.total_completion_tokens += completion
            self.total_cost_cny += cost

    # ------------------------------------------------------------------
    # 消息管理
    # ------------------------------------------------------------------
    async def append(self, *msgs: Message) -> None:
        """线程安全地追加消息"""
        async with self._lock:
            self._history.extend(msgs)
            self.updated_at = datetime.now(timezone.utc)

    async def get_working_memory(self, limit: int = 0) -> list[Message]:
        """获取短期工作记忆 — 截取最近 N 条。

        自动剔除截断产生的孤儿工具响应。
        """
        async with self._lock:
            total = len(self._history)
            if limit <= 0 or total <= limit:
                return list(self._history)

            recent = self._history[total - limit :]
            while recent and recent[0].tool_call_id:
                recent.pop(0)
            return recent

    def __len__(self) -> int:
        return len(self._history)


class SessionManager:
    """全局会话管理器 — 多用户/多终端隔离"""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str, work_dir: str) -> Session:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id, work_dir)
            return self._sessions[session_id]


global_session_mgr = SessionManager()
