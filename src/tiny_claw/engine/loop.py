"""Agent 核心引擎"""

import asyncio
import logging

from tiny_claw import config
from tiny_claw.provider import LLMProvider
from tiny_claw.schema import Message, Role, ToolCall, ToolResult
from tiny_claw.tools import Registry
from tiny_claw.engine.reporter import Reporter
from tiny_claw.engine.terminal_reporter import TerminalReporter
from tiny_claw.engine.session import Session
from tiny_claw.context.composer import PromptComposer
from tiny_claw.context.compactor import Compactor
from tiny_claw.context.recovery import RecoveryManager
from tiny_claw.context.reminders import ReminderBus
from tiny_claw.context.tokens import estimate_messages_tokens
from tiny_claw.engine.reminder import ReminderInjector
from tiny_claw.tracing_with_langsmith import trace

logger = logging.getLogger("tiny-claw.engine")

# 以下常量保留原名（模块内多处引用），但真实来源已迁至 tiny_claw.config
MAX_TURNS = config.MAX_TURNS
WORKING_MEMORY_LIMIT = config.WORKING_MEMORY_LIMIT  # 给压缩器充足的判断空间

_ERR_PREFIX = "[ERR:"


def _guard_missing_error_code(result: ToolResult) -> None:
    """不变量守卫：失败结果必须自带 `[ERR:CODE]`，否则救援提示注入不进去。

    为什么需要这道守卫：救援提示只在**引擎层**注入（见下方 `analyze_and_inject`），
    而 `RecoveryManager` 靠 `[ERR:` 前缀识别错误。于是「工具层某条错误出口漏了这个
    前缀」的后果是**静默的**——`is_error=True` 照样成立、模型照样收到文案，
    只是拿不到任何"下一步该做什么"。这类缺口靠人读代码发现（`registry.py` 就漏过
    两条），所以在消费端加一道告警，下次谁再新增裸错误路径，日志里立刻可见。

    只告警、不改写结果：改写会掩盖调用方的 bug，也让 A/B 对照组失真。
    """
    if result.is_error and not result.output.startswith(_ERR_PREFIX):
        logger.warning(
            "[Recovery] ⚠️ 失败结果缺少 [ERR:CODE] 前缀，救援提示无法注入: %s",
            result.output[:80],
        )


class AgentEngine:
    """微型 OS 的核心驱动"""

    def __init__(
        self,
        provider: LLMProvider,
        registry: Registry,
        *,
        plan_mode: bool = False,
        enable_thinking: bool = True,
        max_parallel_tools: int = config.MAX_PARALLEL_TOOLS,
        reporter: Reporter | None = None,
        compactor: Compactor | None = None,
        max_turns: int | None = None,
        enable_recovery: bool = True,
    ):
        self.provider = provider
        self.registry = registry
        self.plan_mode = plan_mode
        self.enable_thinking = enable_thinking
        # 允许调用方（如 benchmark）收紧轮数上限控成本；不传则沿用全局配置。
        # 此前 TestCase.max_turns 字段定义了却无处生效，等于死代码。
        self.max_turns = max_turns or MAX_TURNS
        self.reporter = reporter or TerminalReporter()
        # 慢思考模式下推理文本占用更多上下文，提高压缩阀值避免工具结果被挤出
        if compactor:
            self.compactor = compactor
        elif enable_thinking:
            # token 口径（见 context/tokens.py）。1:1 保守换算：中文项目下与改造前的
            # 「8000 字符」行为一致；英文项目获得约 4 倍余量（英文约 4 字符才 1 token）。
            # 阈值数字本身的上调（对齐 1M 窗口）另需 benchmark 支撑，暂不动。
            self.compactor = Compactor(
                max_tokens=config.COMPACT_TOKENS_THINKING,
                retain_last=config.RETAIN_LAST_THINKING,
            )
        else:
            self.compactor = Compactor(
                max_tokens=config.COMPACT_TOKENS_NORMAL,
                retain_last=config.RETAIN_LAST_NORMAL,
            )
        self.recovery = RecoveryManager()
        self.reminder = ReminderInjector()
        self.reminders = ReminderBus()
        # 错误自愈总开关（救援提示 + 死循环打断）。默认开；
        # 关掉用于 A/B 实验的**基线组** —— 要证明自愈有效，就得有"没有自愈"的对照。
        self._enable_recovery = enable_recovery
        self._tool_semaphore = asyncio.Semaphore(max_parallel_tools)

    @trace(
        "agent-run",
        process_inputs=lambda inputs: {
            "session_id": inputs["session"].id,
            "user_input": (
                inputs["session"]._history[-1].content[:200]
                if inputs["session"]._history
                else ""
            ),
        },
    )
    async def run(self, session: Session) -> None:
        """启动 Agent，处理 Session 中最后一条用户消息"""
        logger.info("唤醒会话 [%s]，锁定工作区: %s", session.id, session.work_dir)

        composer = PromptComposer(session.work_dir, plan_mode=self.plan_mode)
        system_msg = composer.build()

        turn_count = 0

        while turn_count < self.max_turns:
            turn_count += 1
            logger.info(
                "========== [Turn %d/%d] 开始 ==========",
                turn_count,
                self.max_turns,
            )

            # 上下文：工作记忆 + System Prompt → 运行时提醒 → 压缩 → 发给 LLM
            working_memory = await session.get_working_memory(
                limit=WORKING_MEMORY_LIMIT
            )
            context = [system_msg] + working_memory

            # 提醒必须在压缩**之前**注入：先压缩再追加，提醒可能被一起清掉。
            # 且不写 Session —— 每轮重算，既不累积历史也不污染成本统计。
            context.extend(
                self.reminders.collect(
                    turn=turn_count,
                    max_turns=self.max_turns,
                    ctx_tokens=estimate_messages_tokens(context),
                    ctx_limit=self.compactor.max_tokens,
                )
            )
            context = self.compactor.compact(context)

            available_tools = self.registry.get_available_tools()

            # 2. Phase 1: 慢思考（仅注入推理链到上下文，不写入 Session）
            thinking_content = ""
            if self.enable_thinking:
                logger.info("[Phase 1] 慢思考...")
                think_resp = await self.provider.generate(context, None)
                # 使用 reasoning（纯推理）而非 content（含假 XML 工具调用）
                # 该字段由各 Provider 归一化而来，引擎不感知具体厂商
                if think_resp.reasoning:
                    thinking_content = think_resp.reasoning
                    await self.reporter.on_thinking(thinking_content)
                    # 只将推理链注入上下文，不传递 content（避免污染 Phase 2）
                    context.append(
                        Message(
                            role=Role.ASSISTANT,
                            content=thinking_content,
                            reasoning=thinking_content,
                        )
                    )

            # 3. Phase 2: 行动
            logger.info("[Phase 2] 行动...")
            response = await self.provider.generate(context, available_tools)

            # 推理仅注入本轮上下文（提升 Phase 2 决策质量），
            # 不写入 Session（避免占用 Compactor 配额，导致工具结果被挤出）
            final_msg = Message(
                role=Role.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
                reasoning=response.reasoning,
            )
            await session.append(final_msg)

            if response.content:
                await self.reporter.on_message(response.content)

            if not response.tool_calls:
                logger.info("任务完成，挂起等待下一条指令。")
                break

            # 4. 并发执行工具
            logger.info("并发调用 %d 个工具...", len(response.tool_calls))
            results: list[tuple[ToolCall, ToolResult | None]] = [
                (tc, None) for tc in response.tool_calls
            ]

            async def _execute_one(idx: int, tc: ToolCall) -> None:
                async with self._tool_semaphore:
                    # 协议完整性优先于一切：每个 tool_call 都必须回填一条结果。
                    # 一旦缺失，下一轮请求里 assistant.tool_calls 就会少一条配对的
                    # tool 消息（孤儿 tool_call），LLM 侧直接返回 400 打死整个会话。
                    # 因此：reporter 异常只记日志不外抛，工具异常降级为 error 结果。
                    try:
                        await self.reporter.on_tool_call(tc.name, tc.arguments)
                    except Exception as e:
                        logger.warning("上报工具调用失败 %s: %s", tc.name, e)

                    try:
                        result = await trace(f"tool-{tc.name}")(self.registry.execute)(
                            tc
                        )
                    except Exception as e:
                        logger.error("工具 %s 执行抛出异常: %s", tc.name, e)
                        result = ToolResult(
                            tool_call_id=tc.id,
                            output=f"[ERR:UNKNOWN] 工具 {tc.name} 执行异常: {e}",
                            is_error=True,
                        )

                    display = (
                        result.output[:200] + "..."
                        if len(result.output) > 200
                        else result.output
                    )
                    try:
                        await self.reporter.on_tool_result(
                            tc.name, display, result.is_error
                        )
                    except Exception as e:
                        logger.warning("上报工具结果失败 %s: %s", tc.name, e)

                    results[idx] = (tc, result)

            await asyncio.gather(
                *[_execute_one(i, tc) for i, tc in enumerate(response.tool_calls)],
                return_exceptions=True,  # 兜底：绝不让异常穿透把整轮打死
            )
            logger.info("所有并发工具执行完毕。")

            # 5. 后处理：恢复建议 → 死循环检测 → 写入 Session
            observations: list[Message] = []
            nudges: list[Message] = []  # 打断消息放在工具结果之后，保持 tool_call 配对

            # 协议兜底：_execute_one 已保证回填，这里再补一道防线。
            # 任何一条缺失都会让 tool_call 失去配对结果，宁可给一条错误结果
            # 也不能少一条消息。
            for i, (tc, result) in enumerate(results):
                if result is None:
                    logger.error(
                        "工具 %s 未回填结果，补占位以避免孤儿 tool_call", tc.name
                    )
                    results[i] = (
                        tc,
                        ToolResult(
                            tool_call_id=tc.id,
                            output=f"[ERR:UNKNOWN] 工具 {tc.name} 未返回执行结果",
                            is_error=True,
                        ),
                    )

            # 检测本轮是否有工具失败（用于容错追踪）
            has_error = False
            for _, result in results:
                if result and result.is_error:
                    has_error = True
                    break

            if has_error:
                await session.mark_error_turn()

            for tc, result in results:
                if result is None:
                    continue

                # 5a. RecoveryManager 注入救援指南。
                #    基线组（enable_recovery=False）拿到的是裸错误 —— 这就是对照组。
                _guard_missing_error_code(result)
                output = (
                    self.recovery.analyze_and_inject(result.output)
                    if (self._enable_recovery and result.is_error)
                    else result.output
                )

                observations.append(
                    Message(role=Role.USER, content=output, tool_call_id=tc.id)
                )

                # 5b. ReminderInjector 死循环打断检测（延迟到工具结果之后）
                nudge = (
                    self.reminder.check_and_inject(tc, result)
                    if self._enable_recovery
                    else None
                )
                if nudge:
                    nudges.append(nudge)

            await session.append(*(observations + nudges))

        else:
            logger.warning("达到最大轮次 %d，强制退出", self.max_turns)

        # 记录本次会话的总 Turn 数（供评测系统收集顺滑度指标）
        session.total_turns = turn_count

    # ------------------------------------------------------------------
    # AgentRunner 实现 — 供 SubagentTool 反向调用
    # ------------------------------------------------------------------
    async def run_sub(
        self,
        task_prompt: str,
        read_only_registry: Registry,
        reporter: Reporter | None = None,
    ) -> str:
        """启动一个匿名的、只读的子智能体任务。

        不依赖外部 Session，使用精简的独立循环，打完就打。
        每个子智能体拥有独立的 Semaphore、RecoveryManager、ReminderInjector，互不干扰。

        自愈路径与主循环**同构**（救援提示注入 + 死循环打断），并同样受
        `self._enable_recovery` 约束 —— 否则关掉自愈的A/B 基线组里，
        子智能体仍会带着救援指南跑，对照组就不成立了。
        """
        import uuid

        # ---- 隔离资源：防止并行子智能体互相污染 ----
        sub_id = uuid.uuid4().hex[:6]  # 唯一 ID，用于日志和 Reporter 区分
        sub_semaphore = asyncio.Semaphore(
            config.SUBAGENT_PARALLEL_TOOLS
        )  # 独立信号量（子智能体只有 2 个工具，默认 3 足够）
        sub_recovery = RecoveryManager()  # 独立恢复管理器（不污染主引擎的失败计数器）
        # 独立死循环打断器，理由同上：ReminderInjector 内部按调用指纹累加失败次数，
        # 共用会让并行子智能体互相把对方拉过阈值，产生假的打断。
        sub_reminder = ReminderInjector()
        sub_prefix = f"[Sub:{sub_id}]"

        # 子智能体专用 System Prompt
        context: list[Message] = [
            Message(
                role=Role.SYSTEM,
                content=(
                    "你是一个专门负责深度探索的探路者 (Explorer Subagent)。\n"
                    "你的任务是根据主架构师的指令，在当前工作区内仔细阅读代码、"
                    "查阅日志，搜集足够的信息。\n"
                    "\n"
                    "【核心纪律】\n"
                    "1. 你必须、且只能依靠内置工具（如 bash 的 find/grep，"
                    "或 read_file）去寻找答案。绝对不允许凭空捏造或猜测！\n"
                    "2. 如果你没有找到确切的答案，你必须继续使用工具深入搜索。\n"
                    "3. 当且仅当你找到了确切的线索后，停止调用工具，"
                    "直接输出一段纯文本作为你的终极汇报。"
                    "主架构师会根据你的汇报来做下一步决策。"
                ),
            ),
            Message(role=Role.USER, content=task_prompt),
        ]

        r = reporter or self.reporter
        max_sub_turns = 10
        turn_count = 0

        logger.info("[Subagent:%s] 🚀 探路者出发...", sub_id)

        while True:
            turn_count += 1
            if turn_count > max_sub_turns:
                logger.warning(
                    "[Subagent:%s] 超过 %d 轮被强制召回", sub_id, max_sub_turns
                )
                return (
                    f"子智能体探索过于深入，超过 {max_sub_turns} 轮被强制召回，"
                    "请主 Agent 给它更明确的指令"
                )

            available_tools = read_only_registry.get_available_tools()
            compacted = self.compactor.compact(context)

            # 子任务要求急速响应，不启用慢思考，直接预测行动
            response = await self.provider.generate(compacted, available_tools)
            context.append(response)

            # 子智能体一旦不调用工具了，说明它做好了总结汇报
            if not response.tool_calls:
                logger.info("[Subagent:%s] ✅ 探索完成", sub_id)
                return response.content or "(子智能体未产生输出)"

            # ---- 并发执行只读工具 ----
            # 三元组 (tc, result, output)：`output` 是**注入模型的那份文本**，
            # `result.output` 是**工具返回的原文**。两者在失败时不同——
            # output 多了 `[系统救援指南]`。曾经只回填 (tc, result)，
            # 救援提示算出来后只用于终端上报，模型拿到的始终是裸错误。
            results: list[tuple[ToolCall, ToolResult | None, str | None]] = [
                (tc, None, None) for tc in response.tool_calls
            ]

            async def _execute_sub(idx: int, tc: ToolCall) -> None:
                async with sub_semaphore:
                    # 与主循环同款防御：工具异常降级为 error 结果，
                    # 缺失结果会破坏 tool_call 配对，导致下一轮请求被拒。
                    try:
                        await r.on_tool_call(f"{sub_prefix} {tc.name}", tc.arguments)
                    except Exception as e:
                        logger.warning("[Subagent:%s] 上报工具调用失败: %s", sub_id, e)

                    try:
                        result = await trace(f"sub-{tc.name}")(
                            read_only_registry.execute
                        )(tc)
                    except Exception as e:
                        logger.error(
                            "[Subagent:%s] 工具 %s 执行抛出异常: %s",
                            sub_id,
                            tc.name,
                            e,
                        )
                        result = ToolResult(
                            tool_call_id=tc.id,
                            output=f"[ERR:UNKNOWN] 工具 {tc.name} 执行异常: {e}",
                            is_error=True,
                        )

                    _guard_missing_error_code(result)
                    output = (
                        sub_recovery.analyze_and_inject(result.output)  # ← 独立恢复
                        if (self._enable_recovery and result.is_error)
                        else result.output
                    )

                    display = output[:200] + "..." if len(output) > 200 else output
                    try:
                        await r.on_tool_result(
                            f"{sub_prefix} {tc.name}", display, result.is_error
                        )
                    except Exception as e:
                        logger.warning("[Subagent:%s] 上报工具结果失败: %s", sub_id, e)

                    results[idx] = (tc, result, output)

            await asyncio.gather(
                *[_execute_sub(i, tc) for i, tc in enumerate(response.tool_calls)],
                return_exceptions=True,
            )

            # 将工具观察注入上下文（用 output —— 含救援指南的那份）
            nudges: list[Message] = []
            for tc, result, output in results:
                if result is None:
                    logger.error(
                        "[Subagent:%s] 工具 %s 未回填结果，补占位", sub_id, tc.name
                    )
                    result = ToolResult(
                        tool_call_id=tc.id,
                        output=f"[ERR:UNKNOWN] 工具 {tc.name} 未返回执行结果",
                        is_error=True,
                    )
                    output = (
                        sub_recovery.analyze_and_inject(result.output)
                        if self._enable_recovery
                        else result.output
                    )
                context.append(
                    Message(
                        role=Role.USER,
                        content=output,  # ← 增强文本，不是 result.output
                        tool_call_id=tc.id,
                    )
                )

                # 死循环打断：与主循环同款，同指纹连续失败 NAG_THRESHOLD 次触发。
                # 缺了它，子智能体撞上"重试也没用"的错误时会一路烧到
                # max_sub_turns 上限，中途没有任何强制转向。
                if self._enable_recovery:
                    nudge = sub_reminder.check_and_inject(tc, result)
                    if nudge:
                        nudges.append(nudge)

            # 打断消息统一在工具结果之后追加，且带 is_system_reminder=True，
            # 否则会被压缩器判成"用户原始指令"而永不淘汰（踩过的坑）。
            context.extend(nudges)
