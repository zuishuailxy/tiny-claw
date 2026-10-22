"""Token 估算 — 上下文计量的统一刻度

为什么不能继续用字符数：中文约 1 token/字，英文约 4 字符 1 token，
同一份「8000」的阈值在中英文项目下实际占用的上下文能差 3~4 倍——
等于给同一个 agent 配了两套行为，而你还以为只有一套。

为什么不引入 tiktoken：这里只需要一个上界估计来决定「要不要压缩」，
精确分词是过度设计。原则是**宁可高估（提前压缩）也不低估（撑爆窗口）**。

⚠️ 系数不是拍脑袋：benchmark 语料（35 次 API 调用）最小二乘拟合出
   real ≈ 309 + 1.86×est + 55.8×消息数（tools/replay_thresholds.py 可复算）。
   旧系数 0.25 系统性低估约 1.86 倍——代码/JSON 场景每字符实际
   接近 0.4~0.5 token（引号、括号、转义都按独立 token 计）。
   2026-09-09 已按实测把非 CJK 系数修到 0.47，并在消息级补上
   协议结构开销。阈值数字（config.COMPACT_TOKENS_*）随之重标定，
   触发行为与旧版基本等价，但数字第一次和真实扣费对得上。
"""

import re

# CJK 统一汉字 + CJK 标点符号 + 全角字符，均按 1 token 计
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")

# 非 CJK 按 ~2.1 字符 ≈ 1 token 折算（旧值 0.25 被实测证明低估 1.86 倍）
_NON_CJK_RATIO = 0.47

# 每条消息的协议结构开销：role 包裹、分隔符、JSON 模板等。
# 来自校准模型的 c≈55.8，取整留余量。旧版完全没算这笔账。
_PER_MESSAGE_OVERHEAD = 60


def estimate_tokens(text: str) -> int:
    """保守估算文本的 token 数。

    CJK 按 1 token/字，其余按 0.47 token/字符（benchmark 校准值）。
    对纯中文仍略微高估（实际约 0.6~1 token/字），方向安全。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    return int(cjk + (len(text) - cjk) * _NON_CJK_RATIO) + 1


def estimate_messages_tokens(messages) -> int:
    """估算一整份上下文（含 system / tool_calls 参数）的 token 数。

    之所以放在这里而不是 Compactor 私有方法里：压缩阈值判定和 benchmark
    插桩必须用**同一个口径**。若两边各算各的，校准出来的
    「本地估算 vs 真实 prompt token」系数就对不上压缩逻辑，
    校准结果也就失去意义。
    """
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
        for tc in msg.tool_calls:
            total += estimate_tokens(tc.name) + estimate_tokens(str(tc.arguments))
    # 消息数在压缩判定里不可忽略：20 条消息的结构开销 ≈ 1200 token
    return total + len(messages) * _PER_MESSAGE_OVERHEAD
