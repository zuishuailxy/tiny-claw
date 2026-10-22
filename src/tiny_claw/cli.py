"""tiny-claw 入口

支持两种运行模式：
  - CLI 模式：  python main.py -p "任务描述" [-d /path/to/workdir] [-s session_id]
  - 飞书模式：  python main.py --mode feishu
"""

import argparse
import asyncio
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

# 配置由 tiny_claw.config 统一加载（项目根 .env），此处不再单独 load_dotenv，
# 避免两处加载路径不一致（旧的 load_dotenv() 按 cwd 搜索，切换目录运行时会失效）。
from tiny_claw import config
from tiny_claw.engine import AgentEngine
from tiny_claw.engine.session import Session, global_session_mgr
from tiny_claw.engine.terminal_reporter import TerminalReporter
from tiny_claw.mcp import MCPManager
from tiny_claw.provider import create_provider
from tiny_claw.schema import Message, Role
from tiny_claw.tools import RegistryImpl
from tiny_claw.tools.subagent import SubagentTool
from tiny_claw.tools.builtin import (
    BashTool,
    EditFileTool,
    ReadFileTool,
    SkillTool,
    WriteFileTool,
)
from tiny_claw.tracing import CostTracker, start_span, end_span, export_trace

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("tiny-claw")


class _DropMcpCapabilityNoise(logging.Filter):
    """压掉 MCP SDK 的 capabilities 探测噪音。

    官方 `ClientSessionGroup._aggregate_components` 会对每个 server 依次探测
    prompts / resources / tools，server 没实现前两者就返回 "Method not found"，
    而 SDK 用**裸 `logging.warning()`** 上报（不带 logger 名），于是落到 root logger，
    在 `[%(name)s]` 格式下显示成 `[root] Could not fetch prompts: Method not found`。

    本项目只用 tools 这一类能力，所以 prompts / resources 报错是**必然**的；
    它们会随 server 数量线性增长，最后把真正该看的 WARNING 淹没掉。
    `Could not fetch tools` **不在此列**——那条意味着工具发现失败，必须放行。

    之所以挂在 handler 上而不是 logger 上：记录来自 root logger，按名字过滤 `mcp`
    logger 没有用；而挂 root logger 的 filter 又会影响所有子 logger 的传播，
    范围太大。挂 handler 只影响输出，不改变日志系统的判定。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name == "root" and "Could not fetch prompts" in record.getMessage()
        ) and not (
            record.name == "root" and "Could not fetch resources" in record.getMessage()
        )


for _handler in logging.getLogger().handlers:
    _handler.addFilter(_DropMcpCapabilityNoise())


def setup_file_logging(work_dir: str) -> Path | None:
    """把 agent 自己的日志也写一份到工作区，终端照旧输出。

    为什么必须落盘：在此之前**全仓库没有任何 FileHandler** —— 日志只走 stderr，
    终端一翻走就没了，"跑完复盘"这件事根本做不到；出问题时只能靠重跑复现。

    为什么放在 `main()` 里而不是模块顶部：`logging.basicConfig` 在 import 时执行，
    那时还不知道 `-d` 指向哪。日志落点必须跟随工作区，否则 `-d` 形同虚设。

    为什么用轮转：agent 长任务会打几千行，不封顶的日志文件迟早变成几十 MB。
    5MB × 3 份对排障足够，也不会把工作区撑爆。

    返回日志路径（失败返回 None）——调用方会把它播报出来，
    不然用户又得猜日志在哪（这就是这次要修的问题本身）。
    """
    try:
        log_dir = Path(work_dir) / ".claw" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "agent.log"
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s"))
        # 与终端同一条口径：能力探测噪音在文件里也不该出现
        handler.addFilter(_DropMcpCapabilityNoise())
        logging.getLogger().addHandler(handler)
        return log_path
    except OSError as e:  # 只读工作区、磁盘满、权限不足都不该让 agent 起不来
        logger.warning("日志文件不可写，只输出到终端: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# 引擎工厂
# ═══════════════════════════════════════════════════════════════


def build_engine(
    work_dir: str,
    *,
    plan_mode: bool = False,
    enable_thinking: bool = True,
    #  anthropic 也可以运行
    provider: str = "openai",
) -> AgentEngine:
    """组装引擎：挂载全部工具 + 子智能体 + 费用追踪"""
    raw_llm = create_provider(provider)
    llm = CostTracker(raw_llm, model=raw_llm.model)

    main_registry = RegistryImpl()
    main_registry.register(ReadFileTool(work_dir))
    main_registry.register(WriteFileTool(work_dir))
    main_registry.register(BashTool(work_dir))
    main_registry.register(EditFileTool(work_dir))
    main_registry.register(SkillTool(work_dir))

    engine = AgentEngine(
        llm,
        main_registry,
        plan_mode=plan_mode,
        enable_thinking=enable_thinking,
    )

    # 只读子智能体注册表（防御沙箱）
    read_only_registry = RegistryImpl()
    read_only_registry.register(ReadFileTool(work_dir))
    read_only_registry.register(BashTool(work_dir))

    main_registry.register(
        SubagentTool(
            runner=engine,
            read_only_registry=read_only_registry,
            reporter=engine.reporter,
        )
    )

    return engine


# ═══════════════════════════════════════════════════════════════
# CLI 模式
# ═══════════════════════════════════════════════════════════════


async def run_cli(
    prompt: str,
    session_id: str,
    work_dir: str,
    *,
    plan_mode: bool = False,
    enable_thinking: bool = True,
    provider: str = "openai",
) -> None:
    """CLI 模式：按给定 prompt 驱动 Agent"""
    print("=" * 50)
    print("🚀 启动 tiny-claw CLI 引擎...")
    print(f"📁 锁定工作区: {work_dir}")
    print(
        f"🧠 慢思考: {'开启' if enable_thinking else '关闭'} | 📋 计划模式: {'开启' if plan_mode else '关闭'} | 🔌 协议: {provider}"
    )
    print("=" * 50)

    engine = build_engine(
        work_dir,
        plan_mode=plan_mode,
        enable_thinking=enable_thinking,
        provider=provider,
    )

    # 获取持久化 Session + 绑定 CostTracker
    session = await global_session_mgr.get_or_create(session_id, work_dir)
    if isinstance(engine.provider, CostTracker):
        engine.provider.bind_session(session)

    # 全息追踪：开启根 Span
    root_span = start_span("CLI.TaskRun")
    root_span.add_attribute("prompt", prompt[:200])

    # 初始化彩色终端输出器
    engine.reporter = TerminalReporter()

    print(f"\n🎯 收到任务: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")

    await session.append(Message(role=Role.USER, content=prompt))

    try:
        # MCP 生命周期必须与 engine.run() 在**同一个 task** 内成对出现：
        # SDK 基于 anyio task group，跨 task 关闭会报
        # "Attempted to exit cancel scope in a different task"。
        # preload 只跑一次（进程级单例），register_into 是纯同步注册。
        async with MCPManager.instance(work_dir=work_dir).lifespan() as mcp:
            registered = mcp.register_into(engine.registry)
            if mcp.servers:
                if registered:
                    print(
                        f"🔌 MCP: {len(mcp.servers)} 个 server / "
                        f"{registered} 个工具已接入\n"
                    )
                else:
                    # 配了 MCP 但一个工具都没注册上 —— 必须让人看见，而且要带**具体原因**。
                    # 否则会静默退化成"agent 照常启动、只是少了这批工具"，
                    # 用户还得自己翻日志才知道是超时、路径写错、还是 server 崩了。
                    detail = (
                        "；".join(f"{n}（{r}）" for n, r in mcp.failures)
                        or "没有任何 server 被启用"
                    )
                    print(
                        f"⚠️ MCP: 配置了 {len(mcp.servers)} 个 server 但**没有工具可用**\n"
                        f"   原因: {detail}\n"
                        f"   排查: ① TINY_CLAW_MCP_EXTRA_PATH 是否补了 node/npx 所在目录；\n"
                        f"         ② npx 链路实测不稳，可把 mcp.json 的 command 直接换成\n"
                        f"            node + 本地脚本路径（见 mcp.json.example 的备选写法）\n"
                    )

            await engine.run(session)
    except Exception as e:
        logger.error("引擎运行崩溃: %s", e)
        raise
    finally:
        end_span(root_span)
        export_trace(work_dir, session.id)

    # 打印汇总
    print("\n" + "=" * 50)
    print("✨ 任务圆满结束。")
    if session.total_prompt_tokens:
        print(
            f"💰 Session 累计消耗: ¥{session.total_cost_cny:.6f} | "
            f"Token: 输入 {session.total_prompt_tokens}, "
            f"输出 {session.total_completion_tokens}"
        )
        print(f"📊 容错: {session.error_turns} 次失败轮 / 共 {session.total_turns} 轮")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════
# 飞书模式
# ═══════════════════════════════════════════════════════════════


def run_feishu(work_dir: str, provider: str = "openai") -> None:
    """飞书模式（AgentOps）：启动 FeishuBot + 审批中间件 + 生产级配置"""
    from tiny_claw.feishu import FeishuBot, create_approval_middleware

    work_dir = os.path.abspath(work_dir)
    logger.info("🚀 正在启动 tiny-claw AgentOps 飞书服务端...")
    logger.info("📁 工作区: %s", work_dir)

    # 引擎工厂：每次消息创建独立引擎（计费隔离 + 安全防御）
    def engine_factory(session: Session) -> AgentEngine:
        # AgentOps 生产环境：关闭慢思考（省 Token）、关闭计划模式（运维不需要）
        engine = build_engine(
            work_dir,
            plan_mode=False,
            enable_thinking=False,
            provider=provider,
        )
        if isinstance(engine.provider, CostTracker):
            engine.provider.bind_session(session)
        # 注入审批中间件：高危操作挂起等待飞书人工确认
        engine.registry.use(create_approval_middleware(reporter=lambda: bot.reporter))
        return engine

    bot = FeishuBot(engine_factory, work_dir)

    logger.info("🛡️ 安全防御 Middleware 已挂载。")
    logger.info("📡 飞书长连接已启动，等待消息...")
    bot.start()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="tiny-claw — 驾驭工程驱动研发助手")
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="要交给 Agent 执行的任务描述（CLI 模式必填）",
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Agent 运行的工作区目录路径（默认为当前目录）",
    )
    parser.add_argument(
        "-s",
        "--session-id",
        default="cli_default_session",
        help="会话 ID（多次运行相同 ID 可实现断点续传）",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["cli", "feishu"],
        default="cli",
        help="运行模式：cli（命令行）或 feishu（飞书机器人）",
    )
    parser.add_argument(
        "--plan-mode",
        action="store_true",
        default=False,
        help="开启计划模式（自动创建 PLAN.md / TODO.md，支持断点续传）",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help="关闭慢思考（Phase 1 Thinking），减少 Token 消耗",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help=(
            "协议实现：openai（OpenAI Chat Completion 协议）或 "
            "anthropic（Anthropic Messages 协议）。两者可指向同一模型，用于验证抽象层"
        ),
    )
    args = parser.parse_args()

    # 环境变量检查
    if args.mode == "feishu":
        if not (config.feishu_app_id() and config.feishu_app_secret()):
            logger.fatal(
                "飞书模式需要凭据：请在项目根 .env 中设置 "
                "FEISHU_APP_ID 和 FEISHU_APP_SECRET（参考 .env.example）"
            )
            sys.exit(1)
    else:
        if not args.prompt:
            parser.error("CLI 模式必须指定 -p/--prompt")

    # 两套协议共用一个 key 体系：OpenAI 协议要 DEEPSEEK_API_KEY，
    # Anthropic 协议可用 ANTHROPIC_API_KEY 或复用 DEEPSEEK_API_KEY
    if not config.has_llm_credentials():
        logger.fatal(
            "未找到模型凭据：请在项目根 .env 中设置 DEEPSEEK_API_KEY"
            "（或 ANTHROPIC_API_KEY），参考 .env.example"
        )
        sys.exit(1)

    work_dir = os.path.abspath(args.dir)

    # 日志落盘必须在 work_dir 确定之后（见 setup_file_logging 的说明）
    log_path = setup_file_logging(work_dir)
    if log_path:
        logger.info("📝 日志文件: %s", log_path)

    if args.mode == "feishu":
        run_feishu(work_dir, provider=args.provider)
    else:
        asyncio.run(
            run_cli(
                args.prompt,
                args.session_id,
                work_dir,
                plan_mode=args.plan_mode,
                enable_thinking=not args.no_thinking,
                provider=args.provider,
            )
        )


if __name__ == "__main__":
    main()
