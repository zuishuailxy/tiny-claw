"""统一配置层 — 所有可调常量与凭据的唯一来源（SSOT）。

之所以要有这一层：此前常量散落在 8 个文件里（模型名、端点、压缩阈值、
工具预算、并发度……），改一个阈值要翻遍 engine / tools / provider。
而凭据与调参常量混在一起，也让"哪些该进 .env、哪些不该"没有准绳。

设计要点
--------
1. **加载顺序**：项目根 `.env` → 进程环境变量（后者优先）。
   即 `load_dotenv(override=False)`——CI 或 shell 里 `export` 的值能覆盖 `.env`，
   避免本地配置意外污染 CI。

2. **命名约定**：
   - 业界标准名（不加点号前缀）：`DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY` /
     `FEISHU_APP_ID` … 沿用社区惯例，方便与外部工具链对接。
   - 项目自有常量：统一 `TINY_CLAW_` 前缀，避免与系统环境变量撞名。

3. **读取时机**（这是本层唯一需要留意的设计）：
   - **凭据类**用函数 `deepseek_api_key()` 等，运行时读取。
     原因：支持 key 轮换，也保证单测 `monkeypatch.setenv` 能生效
     （若在 import 时快照缓存，monkeypatch 就会失效）。
   - **调参常量**用模块级常量，import 时快照。
     原因：它们是"本次运行的配置快照"，运行中被改会导致同一轮对话里
     前后行为不一致——反而更难排查。

4. **测试隔离**：设置 `TINY_CLAW_DISABLE_DOTENV=1` 可完全跳过 `.env` 加载，
   保证单测结果可复现（见 tests/conftest.py）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("tiny-claw.config")

# 项目根：src/tiny_claw/config.py → 上两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

_TRUTHY = {"1", "true", "yes", "on"}


def _load_dotenv() -> None:
    """加载项目根 .env。

    三重防御：开关禁用 / 文件不存在 / 未装 python-dotenv，任何一种都不报错，
    只降级为"纯环境变量模式"。配置层不该成为启动失败的原因。
    """
    if os.getenv("TINY_CLAW_DISABLE_DOTENV", "").strip().lower() in _TRUTHY:
        logger.debug("已跳过 .env 加载（TINY_CLAW_DISABLE_DOTENV 已开启）")
        return

    if not ENV_FILE.exists():
        logger.debug("未找到 %s，使用纯环境变量模式", ENV_FILE)
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("未安装 python-dotenv，跳过 %s 加载", ENV_FILE)
        return

    # override=False：进程里已有的环境变量优先，.env 只填补空缺
    load_dotenv(ENV_FILE, override=False)
    logger.debug("已加载配置文件 %s", ENV_FILE)


_load_dotenv()


# ═══════════════════════════════════════════════════════════════
# 类型化读取
# ═══════════════════════════════════════════════════════════════


def get_str(key: str, default: str = "") -> str:
    """读取字符串。空字符串视为未设置，回退默认值。"""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def get_int(key: str, default: int) -> int:
    """读取整数。非法值回退默认值并告警，不让一个拼错的数字搞崩启动。"""
    raw = get_str(key)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "配置项 %s=%r 不是合法整数，已回退默认值 %d", key, raw, default
        )
        return default


def get_float(key: str, default: float) -> float:
    """读取浮点数，非法值回退默认值。"""
    raw = get_str(key)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "配置项 %s=%r 不是合法数字，已回退默认值 %s", key, raw, default
        )
        return default


def get_bool(key: str, default: bool) -> bool:
    """读取布尔值，接受 1/true/yes/on（不分大小写）。"""
    raw = get_str(key)
    if raw == "":
        return default
    return raw.strip().lower() in _TRUTHY


# ═══════════════════════════════════════════════════════════════
# 凭据 — 运行时读取（支持轮换与 monkeypatch）
# ═══════════════════════════════════════════════════════════════


def deepseek_api_key() -> str:
    return get_str("DEEPSEEK_API_KEY")


def anthropic_api_key() -> str:
    """Anthropic 凭据：优先专用 key，回退 DeepSeek key。

    回退有意义——DeepSeek 官方提供 Anthropic 协议端点，同一个 key 可驱动
    两套协议（这是本项目「同模型双协议对照」验证的基础）。
    """
    return get_str("ANTHROPIC_API_KEY") or get_str("DEEPSEEK_API_KEY")


def deepseek_base_url() -> str:
    return get_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


def anthropic_base_url() -> str:
    return get_str("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")


def feishu_app_id() -> str:
    return get_str("FEISHU_APP_ID")


def feishu_app_secret() -> str:
    return get_str("FEISHU_APP_SECRET")


def langsmith_api_key() -> str:
    return get_str("LANGSMITH_API_KEY")


def has_llm_credentials() -> bool:
    """是否存在任一可用模型凭据。供入口做启动前自检。"""
    return bool(deepseek_api_key() or anthropic_api_key())


# ═══════════════════════════════════════════════════════════════
# 模型参数 — import 时快照
# ═══════════════════════════════════════════════════════════════

# 两套协议共用同一个模型，是「同模型双协议」对照实验的前提
MODEL = get_str("TINY_CLAW_MODEL", "deepseek-v4-flash")

# Anthropic 协议下 max_tokens 为必填字段（OpenAI 侧可选）
MAX_OUTPUT_TOKENS = get_int("TINY_CLAW_MAX_TOKENS", 4096)

# 慢思考预算占 max_tokens 的比例。
# 原生 Anthropic 要求 budget_tokens < max_tokens；DeepSeek 端点会忽略该字段，
# 只认 thinking.type="enabled"，因此这里只影响原生 Anthropic。
THINKING_BUDGET_RATIO = get_float("TINY_CLAW_THINKING_BUDGET_RATIO", 0.5)

# ═══════════════════════════════════════════════════════════════
# 引擎参数
# ═══════════════════════════════════════════════════════════════

MAX_TURNS = get_int("TINY_CLAW_MAX_TURNS", 30)
WORKING_MEMORY_LIMIT = get_int("TINY_CLAW_WORKING_MEMORY_LIMIT", 20)
MAX_PARALLEL_TOOLS = get_int("TINY_CLAW_MAX_PARALLEL_TOOLS", 5)

# ── System Reminders：运行时把「引擎知道、模型不知道」的状态回灌进上下文 ──
# 约束见 context/reminders.py：去重 + 冷却 + 限流，否则提醒本身会变成噪声。
REMINDER_COOLDOWN_TURNS = get_int("TINY_CLAW_REMINDER_COOLDOWN_TURNS", 3)
REMINDER_MAX_PER_TURN = get_int("TINY_CLAW_REMINDER_MAX_PER_TURN", 2)
# 上下文用量达到压缩阈值的该比例时提醒「该把进度外部化到文件了」
REMINDER_CONTEXT_PRESSURE = get_float("TINY_CLAW_REMINDER_CONTEXT_PRESSURE", 0.8)
# 剩余轮数少于该值时提醒收敛
REMINDER_TURN_WARN_REMAINING = get_int("TINY_CLAW_REMINDER_TURN_WARN_REMAINING", 5)

# ── MCP 客户端（见 docs/MCP_INTEGRATION_PLAN.md）──
# 依赖装在 optional extra 里：pip install -e ".[mcp]"
MCP_ENABLED = get_bool("TINY_CLAW_MCP_ENABLED", True)
# 社区 mcpServers 格式的配置文件，默认落项目根。真实文件不入库（含机器路径），
# 仓库里只有 mcp.json.example
MCP_CONFIG_PATH = get_str("TINY_CLAW_MCP_CONFIG", "mcp.json")
# 连接 + initialize 的超时。给得宽是因为 stdio server 常用 npx 拉起，而 npx 的
# 启动耗时**极不稳定**：实测同一套代码冷启动 48.8s、热启动 4.7~14s，环境异常时
# 还遇到过 >240s 不返回（此时 node -v 正常、包也在缓存里）。
# 取 120s：宁可慢一点，也不要误判成「server 挂了」而静默少掉这批工具。
MCP_CONNECT_TIMEOUT = get_int("TINY_CLAW_MCP_CONNECT_TIMEOUT", 120)
# 单次工具调用超时（交给 SDK 的 read_timeout_seconds）
MCP_CALL_TIMEOUT = get_int("TINY_CLAW_MCP_CALL_TIMEOUT", 60)
# 额外注入子进程 PATH 的目录（冒号分隔）。
# 为什么需要：node/npx 常装在非默认位置（如托管环境的 binaries 目录），
# 不补 PATH 会直接 connect 失败，而报错信息只有 "No such file or directory"
MCP_EXTRA_PATH = get_str("TINY_CLAW_MCP_EXTRA_PATH", "")
# 单个 MCP 工具返回内容的字符上限。MCP 可以返回 base64 图片/音频块，
# 不截断会直接把上下文撑爆（Phase 0 实测 read_media_file 会返回 base64）
MCP_MAX_OUTPUT_CHARS = get_int("TINY_CLAW_MCP_MAX_OUTPUT_CHARS", 8000)

# 子智能体只有 2 个工具（read + bash），并发度默认低于主引擎
SUBAGENT_PARALLEL_TOOLS = get_int("TINY_CLAW_SUBAGENT_PARALLEL_TOOLS", 3)

# 压缩阈值（token 口径，见 context/tokens.py）
# 2026-09-09 重标定：先用 run_benchmark.py 录制语料（35 次 API 调用），
# 再用 tools/replay_thresholds.py 离线重放网格搜索，零 API 成本。依据：
#   1. 校准模型 real ≈ 309 + 1.86×est + 55.8×消息数 —— 旧 0.25 系数低估
#      1.86 倍，tokens.py 已修正；阈值数字按新口径等比重标定（×4），
#      实际触发行为与旧版近似等价（差异 <3%，见 threshold_report.json）。
#   2. 观察负载（4 用例 35 份快照）上下文峰值 est=1797，旧阈值从未触发。
#      上调解决的是已知层级冲突：READ_MAX_CHARS=8000 字符的中文最坏情况
#      ≈ 8001 token est，旧 NORMAL=4000 意味着读一次文件就顶爆全局预算。
#   3. 安全水位：新阈值对应真实 ~30k/60k token，距 1M 窗口 25% 水位
#      （250k）仍有 4~8 倍余量。
# 局限（如实记录）：语料不含「超长多轮任务」，更大幅度的上调仍需压测支撑。
COMPACT_TOKENS_THINKING = get_int("TINY_CLAW_COMPACT_TOKENS_THINKING", 32000)
COMPACT_TOKENS_NORMAL = get_int("TINY_CLAW_COMPACT_TOKENS_NORMAL", 16000)
RETAIN_LAST_THINKING = get_int("TINY_CLAW_RETAIN_LAST_THINKING", 10)
RETAIN_LAST_NORMAL = get_int("TINY_CLAW_RETAIN_LAST_NORMAL", 6)

# ═══════════════════════════════════════════════════════════════
# 工具参数
# ═══════════════════════════════════════════════════════════════

# 单次工具输出的字符上限（read_file 用）。
# ⚠️ 刻意维持字符口径：这是给模型的"体感预算"，与引擎层的 token 安全边界
# 是两个层级。它应显著小于全局压缩阈值，否则"读一个文件就触发全局压缩"。
READ_MAX_CHARS = get_int("TINY_CLAW_READ_MAX_CHARS", 8000)
PREVIEW_LINES = get_int("TINY_CLAW_PREVIEW_LINES", 30)
OFFLOAD_DIR = get_str("TINY_CLAW_OFFLOAD_DIR", ".claw/offload")

# bash 工具：单次输出字符上限与超时秒数。
# ⚠️ 已知缺陷：bash 输出超阈值时是硬丢弃，不像 read_file 那样卸载落盘，
# 超出部分永久丢失（且 bash 没有行号分段能力，取回更难）。详见 docs/PROJECT_GUIDE.md §7「已知边界」。
BASH_MAX_OUTPUT = get_int("TINY_CLAW_BASH_MAX_OUTPUT", 8000)
BASH_TIMEOUT = get_int("TINY_CLAW_BASH_TIMEOUT", 30)

# ═══════════════════════════════════════════════════════════════
# 可观测性
# ═══════════════════════════════════════════════════════════════

# ⚠️ 默认为 True 是沿用改造前的行为（旧代码里写死 setdefault("true")）。
# 若未配置 LANGSMITH_API_KEY，想彻底关掉外部追踪请在 .env 设 false。
LANGSMITH_TRACING = get_bool("LANGSMITH_TRACING", True)
LANGSMITH_PROJECT = get_str("LANGSMITH_PROJECT", "tiny-claw")


def custom_price() -> tuple[float, float] | None:
    """用户在 .env 里自定义的模型单价（元/百万 token）。

    用途：PRICING 表里没有的模型（尤其是新上的 v4 系列）会算不出成本。
    与其静默算 0，不如让使用者自己填单价。两个值都要给才生效。
    """
    raw_in = get_str("TINY_CLAW_PRICE_INPUT")
    raw_out = get_str("TINY_CLAW_PRICE_OUTPUT")
    if raw_in == "" or raw_out == "":
        return None
    return (
        get_float("TINY_CLAW_PRICE_INPUT", 0.0),
        get_float("TINY_CLAW_PRICE_OUTPUT", 0.0),
    )
