"""配置层单测。

重点验证两类契约：
1. 类型化读取：非法值回退默认值而不是抛异常（配置不该成为启动失败的原因）
2. 凭据运行时读取：保证 monkeypatch 与 key 轮换生效（若在 import 时快照就会失效）

注意：本文件跑在 conftest 设置的 TINY_CLAW_DISABLE_DOTENV=1 之下，
因此读取的都是代码默认值，不受开发者本地 .env 影响。
"""

import logging

import pytest

from tiny_claw import config


# ═══════════════════════════════════════════════════════════════
# 类型化读取
# ═══════════════════════════════════════════════════════════════


def test_get_str_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("TINY_CLAW_TEST_STR", raising=False)
    assert config.get_str("TINY_CLAW_TEST_STR", "fallback") == "fallback"


def test_get_str_treats_blank_as_unset(monkeypatch):
    """空白字符串视为未设置——避免 .env 里留空项被当成有效值"""
    monkeypatch.setenv("TINY_CLAW_TEST_STR", "   ")
    assert config.get_str("TINY_CLAW_TEST_STR", "fallback") == "fallback"


def test_get_int_parses_and_falls_back(monkeypatch):
    monkeypatch.setenv("TINY_CLAW_TEST_INT", "42")
    assert config.get_int("TINY_CLAW_TEST_INT", 7) == 42

    monkeypatch.setenv("TINY_CLAW_TEST_INT", "not-a-number")
    assert config.get_int("TINY_CLAW_TEST_INT", 7) == 7


def test_get_int_warns_on_invalid_value(monkeypatch, caplog):
    """非法值要留下痕迹，否则配置拼错了没人知道"""
    monkeypatch.setenv("TINY_CLAW_TEST_INT", "oops")
    with caplog.at_level(logging.WARNING, logger="tiny-claw.config"):
        config.get_int("TINY_CLAW_TEST_INT", 7)
    assert any("不是合法整数" in r.message for r in caplog.records)


def test_get_float_parses_and_falls_back(monkeypatch):
    monkeypatch.setenv("TINY_CLAW_TEST_FLOAT", "0.25")
    assert config.get_float("TINY_CLAW_TEST_FLOAT", 1.0) == 0.25

    monkeypatch.setenv("TINY_CLAW_TEST_FLOAT", "abc")
    assert config.get_float("TINY_CLAW_TEST_FLOAT", 1.0) == 1.0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("", False),
    ],
)
def test_get_bool_accepts_common_truthy_forms(monkeypatch, raw, expected):
    monkeypatch.setenv("TINY_CLAW_TEST_BOOL", raw)
    assert config.get_bool("TINY_CLAW_TEST_BOOL", False) is expected


# ═══════════════════════════════════════════════════════════════
# 凭据：运行时读取（这是本层最关键的设计）
# ═══════════════════════════════════════════════════════════════


def test_deepseek_api_key_reads_at_runtime(monkeypatch):
    """凭据必须运行时读取，否则 monkeypatch 与 key 轮换都会失效"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert config.deepseek_api_key() == ""

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-runtime")
    assert config.deepseek_api_key() == "sk-runtime"


def test_anthropic_api_key_falls_back_to_deepseek(monkeypatch):
    """同一个 key 驱动两套协议，是双协议对照验证的基础"""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-shared")
    assert config.anthropic_api_key() == "sk-shared"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dedicated")
    assert config.anthropic_api_key() == "sk-dedicated"


def test_has_llm_credentials(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert config.has_llm_credentials() is False

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    assert config.has_llm_credentials() is True


def test_provider_picks_up_monkeypatched_key(monkeypatch):
    """端到端验证：monkeypatch 的 key 能一路传到 Provider 构造"""
    from tiny_claw.provider import DeepSeekProvider

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-for-provider")
    provider = DeepSeekProvider()
    assert provider.model  # 来自 config.MODEL，不应为空


def test_provider_raises_without_credentials(monkeypatch):
    from tiny_claw.provider import DeepSeekProvider

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="未找到模型凭据"):
        DeepSeekProvider()


# ═══════════════════════════════════════════════════════════════
# 自定义单价
# ═══════════════════════════════════════════════════════════════


def test_custom_price_none_when_unset(monkeypatch):
    monkeypatch.delenv("TINY_CLAW_PRICE_INPUT", raising=False)
    monkeypatch.delenv("TINY_CLAW_PRICE_OUTPUT", raising=False)
    assert config.custom_price() is None


def test_custom_price_requires_both_values(monkeypatch):
    """只填一个不算数——半套单价比没有更危险"""
    monkeypatch.setenv("TINY_CLAW_PRICE_INPUT", "1.0")
    monkeypatch.delenv("TINY_CLAW_PRICE_OUTPUT", raising=False)
    assert config.custom_price() is None

    monkeypatch.setenv("TINY_CLAW_PRICE_OUTPUT", "4.0")
    assert config.custom_price() == (1.0, 4.0)


# ═══════════════════════════════════════════════════════════════
# 成本统计：未知模型不能静默归零
# ═══════════════════════════════════════════════════════════════


def test_known_model_returns_real_cost():
    from tiny_claw.tracing import _calculate_cost

    # deepseek-v4-pro：闲时未命中 输入 4.5 元/百万、输出 13.5 元/百万
    # （来源 api-docs.deepseek.com/zh-cn/quick_start/pricing，2026-09-09 核对）
    cost = _calculate_cost("deepseek-v4-pro", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)


def test_unknown_model_warns_instead_of_silent_zero(monkeypatch, caplog):
    """默认模型 v4-flash 未收录时若静默算 0，benchmark 成本数据会全错"""
    from tiny_claw import tracing

    monkeypatch.delenv("TINY_CLAW_PRICE_INPUT", raising=False)
    monkeypatch.delenv("TINY_CLAW_PRICE_OUTPUT", raising=False)
    tracing._warned_unknown_models.clear()

    with caplog.at_level(logging.WARNING, logger="tiny-claw.tracker"):
        cost = tracing._calculate_cost("some-unknown-model", 1_000_000, 1_000_000)

    assert cost == 0.0
    assert any("未在 PRICING 表中" in r.message for r in caplog.records)


def test_custom_price_is_used_for_unknown_model(monkeypatch):
    monkeypatch.setenv("TINY_CLAW_PRICE_INPUT", "2.0")
    monkeypatch.setenv("TINY_CLAW_PRICE_OUTPUT", "8.0")
    from tiny_claw import tracing

    cost = tracing._calculate_cost("some-unknown-model", 1_000_000, 1_000_000)
    assert cost == pytest.approx(10.0)


# ═══════════════════════════════════════════════════════════════
# 默认配置健全性
# ═══════════════════════════════════════════════════════════════


def test_defaults_are_sane():
    """默认值不该是空的或 0——这类错误只会在运行时才炸"""
    assert config.MODEL
    assert config.MAX_TURNS > 0
    assert config.MAX_PARALLEL_TOOLS > 0
    assert config.COMPACT_TOKENS_THINKING > 0
    assert config.COMPACT_TOKENS_NORMAL > 0
    assert config.READ_MAX_CHARS > 0
    assert config.PREVIEW_LINES > 0
    assert config.OFFLOAD_DIR


def test_tool_output_budget_not_above_global_budget():
    """中文最坏情况下，单次工具输出不应超过全局压缩阈值。

    这是「把常量抽到一处」才暴露出来的问题：此前两个值分散在
    read_file.py 与 loop.py、且单位不同，没人能一眼看出它们冲突。
    曾以 xfail(strict=True) 记录该冲突；2026-09-09 benchmark 基线
    （语料重放，tools/replay_thresholds.py）支撑阈值重标定
    （NORMAL 4000→16000）后解除。
    """
    from tiny_claw.context.tokens import estimate_tokens

    worst_case = estimate_tokens("中" * config.READ_MAX_CHARS)
    assert worst_case <= config.COMPACT_TOKENS_NORMAL


def test_tool_output_budget_ok_for_ascii():
    """英文场景下同样的预算是安全的——说明冲突只在中日韩内容下发作。

    保留这条是为了把问题范围钉死：不是预算整体偏小，而是字符口径对
    中文严重低估导致的。
    """
    from tiny_claw.context.tokens import estimate_tokens

    best_case = estimate_tokens("a" * config.READ_MAX_CHARS)
    assert best_case <= config.COMPACT_TOKENS_NORMAL


# ═══════════════════════════════════════════════════════════════
# 工具输出口径回归（字符 vs 字节）
# ═══════════════════════════════════════════════════════════════


async def test_bash_truncates_chinese_output_by_character():
    """bash 曾与 read_file 一样「字节判断 + 字符切分」。

    UTF-8 汉字 3 字节，中文输出会截出 3 倍于阈值的内容（24000 而非 8000），
    且提示语宣称的字节数与事实不符。英文输出完全正常，所以只在中文下发作。
    """
    import tempfile

    from tiny_claw.tools.builtin.bash import BashTool

    with tempfile.TemporaryDirectory() as work_dir:
        tool = BashTool(work_dir)
        out = await tool.execute({"command": "python3 -c \"print('中' * 10000)\""})

    assert "...[终端输出过长" in out, f"预期触发截断，实际输出前 200 字：{out[:200]}"
    # rstrip 掉 print 自带的换行与拼接空行，只比较正文长度
    body = out.split("...[终端输出过长")[0].rstrip()
    assert len(body) <= config.BASH_MAX_OUTPUT
    assert "字节" not in out, "提示语应报字符数，报字节数是口径错误"
