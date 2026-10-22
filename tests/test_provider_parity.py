"""双协议翻译对照测试 — 验证 Provider 抽象层确实隔离了协议差异

这些用例**不需要网络与 API Key**，跑的是纯函数：
验证「同一组内部 Message 经 OpenAI / Anthropic 两条翻译路径后语义等价」。

背景：DeepSeek 在 https://api.deepseek.com/anthropic 提供 Anthropic 协议端点，
因此两个 Provider 可指向同一模型（deepseek-v4-pro），
构成「同厂商、同模型、仅协议不同」的对照实验 ——
跑出的任何差异只能归因于协议翻译层本身。
"""

import json
from types import SimpleNamespace as NS

import pytest

from tiny_claw.provider import create_provider
from tiny_claw.provider.anthropic import (
    AnthropicProvider,
    from_anthropic_response,
    normalize_alternation,
    split_system,
    to_anthropic_messages,
    to_anthropic_tools,
)
from tiny_claw.provider.openai import (
    DeepSeekProvider,
    _from_openai_response,
    _to_openai_messages,
    _to_openai_tools,
)
from tiny_claw.schema import Message, Role, ToolCall, ToolDefinition

# ═══════════════════════════════════════════════════════════════
# 1. System Prompt 的位置
# ═══════════════════════════════════════════════════════════════


def test_system_prompt_isolated_in_both_protocols():
    """OpenAI 放进 messages[0]，Anthropic 提为顶层字段 —— 位置不同但都被隔离"""
    msgs = [
        Message(role=Role.SYSTEM, content="你是 tiny-claw"),
        Message(role=Role.USER, content="hi"),
    ]

    oa = _to_openai_messages(msgs)
    assert oa[0]["role"] == "system"
    assert oa[0]["content"] == "你是 tiny-claw"

    sys_text, rest = split_system(msgs)
    assert sys_text == "你是 tiny-claw"
    assert all(m.role != Role.SYSTEM for m in rest)
    assert all(m["role"] != "system" for m in to_anthropic_messages(rest))


def test_multiple_system_messages_are_merged_for_anthropic():
    """多条 system 在 Anthropic 侧必须合并成一个字符串（顶层只有一个 system）"""
    msgs = [
        Message(role=Role.SYSTEM, content="第一段"),
        Message(role=Role.USER, content="hi"),
        Message(role=Role.SYSTEM, content="第二段"),
    ]
    sys_text, rest = split_system(msgs)
    assert sys_text == "第一段\n\n第二段"
    assert len(rest) == 1


# ═══════════════════════════════════════════════════════════════
# 2. 工具调用与工具结果的关联 ID
# ═══════════════════════════════════════════════════════════════


def test_tool_call_arguments_parity():
    """同一 ToolCall，两家的参数承载字段不同但值等价"""
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls -la"})
    msgs = [
        Message(role=Role.USER, content="列一下目录"),
        Message(role=Role.ASSISTANT, content="", tool_calls=[call]),
    ]

    oa = _to_openai_messages(msgs)
    fn = oa[-1]["tool_calls"][0]["function"]
    assert fn["name"] == "bash"
    assert json.loads(fn["arguments"]) == {"command": "ls -la"}

    an = to_anthropic_messages(msgs)
    block = an[-1]["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "bash"
    assert block["input"] == {"command": "ls -la"}
    assert block["id"] == "c1"


def test_tool_result_correlation_id_parity():
    """工具结果的关联 ID：OpenAI 用 tool_call_id，Anthropic 用 tool_use_id"""
    msgs = [
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="bash", arguments={"command": "ls"})
            ],
        ),
        Message(role=Role.USER, content="file.txt", tool_call_id="call_1"),
    ]

    oa = _to_openai_messages(msgs)
    assert oa[-1]["role"] == "tool"
    assert oa[-1]["tool_call_id"] == "call_1"

    _, rest = split_system(msgs)
    an = to_anthropic_messages(rest)
    block = an[-1]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "file.txt"


# ═══════════════════════════════════════════════════════════════
# 3. 协议差异：工具结果是否合并（这是最容易被抽象层漏掉的一条）
# ═══════════════════════════════════════════════════════════════


def test_anthropic_merges_consecutive_tool_results_openai_does_not():
    """内部 Schema 是「一条消息一个工具结果」：
    OpenAI 原样透传为多条 role=tool 消息；
    Anthropic 必须合并进同一条 user 消息的 content 数组。"""
    msgs = [
        Message(role=Role.USER, content="帮我看看这两个东西"),
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(id="c1", name="bash", arguments={"command": "ls"}),
                ToolCall(id="c2", name="read_file", arguments={"path": "a.txt"}),
            ],
        ),
        Message(role=Role.USER, content="out1", tool_call_id="c1"),
        Message(role=Role.USER, content="out2", tool_call_id="c2"),
    ]

    oa = _to_openai_messages(msgs)
    assert len([m for m in oa if m["role"] == "tool"]) == 2

    _, rest = split_system(msgs)
    an = to_anthropic_messages(rest)

    # 携带 tool_result 的 user 消息必须只有一条（两个 block 合并）
    result_msgs = [
        m
        for m in an
        if m["role"] == "user"
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(result_msgs) == 1
    assert [b["tool_use_id"] for b in result_msgs[0]["content"]] == ["c1", "c2"]


def test_normalize_alternation_merges_same_role_neighbors():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ]
    merged = normalize_alternation(msgs)
    assert len(merged) == 1
    assert len(merged[0]["content"]) == 2


def test_anthropic_enforces_strict_alternation_and_user_first():
    """Anthropic 不接受 assistant 开头，也不接受连续同角色"""
    msgs = [
        Message(role=Role.ASSISTANT, content="我先想想"),
        Message(role=Role.ASSISTANT, content="再想想"),
    ]
    an = to_anthropic_messages(msgs)
    assert an[0]["role"] == "user"
    for a, b in zip(an, an[1:]):
        assert a["role"] != b["role"]


def test_anthropic_drops_empty_assistant_message():
    """Anthropic 不允许空 content 的 assistant 消息"""
    msgs = [
        Message(role=Role.USER, content="hi"),
        Message(role=Role.ASSISTANT, content=""),
    ]
    an = to_anthropic_messages(msgs)
    assert all(m["role"] != "assistant" for m in an)


# ═══════════════════════════════════════════════════════════════
# 4. 工具定义：parameters vs input_schema
# ═══════════════════════════════════════════════════════════════


def test_tool_definition_schema_key_differs_but_value_matches():
    schema = {"type": "object", "properties": {"command": {"type": "string"}}}
    td = ToolDefinition(name="bash", description="执行命令", input_schema=schema)

    oa = _to_openai_tools([td])
    assert oa[0]["function"]["parameters"] == schema

    an = to_anthropic_tools([td])
    assert an[0]["input_schema"] == schema


# ═══════════════════════════════════════════════════════════════
# 5. 推理链归一化（本次重构的核心：两家的私有字段收敛到 reasoning）
# ═══════════════════════════════════════════════════════════════


def test_reasoning_is_normalized_across_protocols():
    """DeepSeek 的 reasoning_content 与 Anthropic 的 thinking block
    必须收敛到同一个 Message.reasoning 字段，上层引擎才可能对厂商无感知。"""
    oa_resp = NS(
        choices=[
            NS(
                message=NS(
                    content="答案是 42",
                    tool_calls=[],
                    reasoning_content="我是这样想的",
                )
            )
        ],
        usage=None,
    )
    an_resp = NS(
        content=[
            NS(type="thinking", thinking="我是这样想的"),
            NS(type="text", text="答案是 42"),
        ],
        usage=None,
    )

    m_openai = _from_openai_response(oa_resp)
    m_anthropic = from_anthropic_response(an_resp)

    assert m_openai.reasoning == "我是这样想的"
    assert m_anthropic.reasoning == "我是这样想的"
    assert m_openai.reasoning == m_anthropic.reasoning
    assert m_openai.content == m_anthropic.content == "答案是 42"


def test_reasoning_absent_when_model_does_not_think():
    oa_resp = NS(
        choices=[NS(message=NS(content="直接回答", tool_calls=[]))], usage=None
    )
    assert _from_openai_response(oa_resp).reasoning == ""

    an_resp = NS(content=[NS(type="text", text="直接回答")], usage=None)
    assert from_anthropic_response(an_resp).reasoning == ""


def test_reasoning_is_sent_back_to_openai_thinking_api():
    msgs = [
        Message(
            role=Role.ASSISTANT,
            reasoning="先分析用户需求",
            content="",
        )
    ]

    assert _to_openai_messages(msgs) == [
        {
            "role": "assistant",
            "reasoning_content": "先分析用户需求",
        }
    ]


# ═══════════════════════════════════════════════════════════════
# 6. 响应反解与 Usage 映射
# ═══════════════════════════════════════════════════════════════


def test_anthropic_response_roundtrip_preserves_tool_call():
    resp = NS(
        content=[NS(type="tool_use", id="c9", name="bash", input={"command": "pwd"})],
        stop_reason="tool_use",
        usage=NS(input_tokens=11, output_tokens=7),
    )
    msg = from_anthropic_response(resp)

    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "bash"
    assert msg.tool_calls[0].arguments == {"command": "pwd"}
    assert msg.usage.prompt_tokens == 11
    assert msg.usage.completion_tokens == 7

    # 再翻译回 Anthropic 格式，参数不失真
    again = to_anthropic_messages([Message(role=Role.USER, content="执行一下"), msg])
    tool_use = [
        b
        for m in again
        if m["role"] == "assistant"
        for b in m["content"]
        if b["type"] == "tool_use"
    ]
    assert tool_use[0]["input"] == {"command": "pwd"}
    assert tool_use[0]["id"] == "c9"


def test_anthropic_non_dict_tool_input_degrades_safely():
    """工具参数不是 dict 时降级为空 dict，而不是抛异常"""
    resp = NS(
        content=[NS(type="tool_use", id="c1", name="bash", input="not-a-dict")],
        usage=None,
    )
    msg = from_anthropic_response(resp)
    assert msg.tool_calls[0].arguments == {}


def test_openai_malformed_tool_arguments_degrade_safely():
    resp = NS(
        choices=[
            NS(
                message=NS(
                    content="",
                    tool_calls=[
                        NS(id="c1", function=NS(name="bash", arguments="{bad json"))
                    ],
                )
            )
        ],
        usage=None,
    )
    msg = _from_openai_response(resp)
    assert msg.tool_calls[0].arguments == {}


# ═══════════════════════════════════════════════════════════════
# 7. 慢思考开关契约：两家必须一致（tools=None 即进入思考）
# ═══════════════════════════════════════════════════════════════


def test_slow_thinking_contract_is_identical(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-unit-test")
    provider = AnthropicProvider()
    msgs = [Message(role=Role.USER, content="hi")]

    thinking_params = provider._build_params(msgs, None)
    assert thinking_params["thinking"]["type"] == "enabled"
    assert "tools" not in thinking_params

    acting_params = provider._build_params(
        msgs, [ToolDefinition(name="bash", description="", input_schema={})]
    )
    assert "tools" in acting_params
    assert "thinking" not in acting_params


def test_max_tokens_always_present_for_anthropic(monkeypatch):
    """max_tokens 在 Anthropic 协议下是必填项，OpenAI 侧则可选"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-unit-test")
    provider = AnthropicProvider()
    params = provider._build_params([Message(role=Role.USER, content="hi")], None)
    assert params["max_tokens"] > 0


def test_thinking_budget_smaller_than_max_tokens(monkeypatch):
    """原生 Anthropic 要求 budget_tokens < max_tokens"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-unit-test")
    provider = AnthropicProvider()
    params = provider._build_params([Message(role=Role.USER, content="hi")], None)
    assert params["thinking"]["budget_tokens"] < params["max_tokens"]


def test_provider_factory_returns_both_protocols(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-for-unit-test")
    assert isinstance(create_provider("openai"), DeepSeekProvider)
    assert isinstance(create_provider("anthropic"), AnthropicProvider)


def test_provider_factory_rejects_unknown_name():
    with pytest.raises(ValueError):
        create_provider("gemini")
