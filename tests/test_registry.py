"""Registry 的中间件链与执行路由用例

补空白的理由：中间件链此前只有 1 个用例（并发配对），链本身的语义没有守卫 ——
放行 / 拦截 / 顺序 / 短路 / 与路由的先后，都没测。而审批回路正是挂在中间件上的
（`feishu/approve.py` 的 `is_dangerous_command` 通过 `registry.use(...)` 接入）。
"""

from tiny_claw.context.recovery import ErrorCode, extract_code
from tiny_claw.schema import ToolCall, ToolDefinition
from tiny_claw.tools.base import BaseTool
from tiny_claw.tools.registry import RegistryImpl


class _Tool(BaseTool):
    def __init__(self, name="t", output="ok", boom=None):
        self._name, self._output, self._boom = name, output, boom
        self.seen: list[dict] = []

    def name(self):
        return self._name

    def definition(self):
        return ToolDefinition(
            name=self._name, description="测试用", input_schema={"type": "object"}
        )

    async def execute(self, arguments):
        self.seen.append(arguments)
        if self._boom:
            raise self._boom
        return self._output


def _call(name="t", cid="c1", **args):
    return ToolCall(id=cid, name=name, arguments=args)


async def _allow(_call):
    return True, ""


async def _deny(reason):
    return False, reason


# ══════════════════════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════════════════════


async def test_execute_routes_and_passes_arguments_through():
    reg, tool = RegistryImpl(), _Tool(output="hello")
    reg.register(tool)
    res = await reg.execute(_call(x=1))
    assert res.output == "hello"
    assert res.is_error is False
    assert tool.seen == [{"x": 1}]


async def test_unknown_tool_is_error_result_not_exception():
    """模型幻觉出工具名时返回错误结果，不要抛异常打断循环"""
    reg = RegistryImpl()
    res = await reg.execute(_call(name="nope"))
    assert res.is_error is True
    assert "不存在名为 'nope' 的工具" in res.output
    # ⚠️ 必须带错误码：否则引擎层的救援提示注入不进去（extract_code 解析不到）
    assert res.output.startswith("[ERR:TOOL_NOT_FOUND]")


async def test_err_prefix_output_is_flagged_as_error():
    """工具用 [ERR:CODE] 表达失败 → 自动标记 is_error，供自愈体系识别"""
    reg = RegistryImpl()
    reg.register(_Tool(output="[ERR:FILE_NOT_FOUND] 文件不存在"))
    res = await reg.execute(_call())
    assert res.is_error is True


async def test_tool_exception_becomes_error_result():
    """工具抛异常不能穿透 —— 必须降级为**带错误码**的结果。

    两层理由：① 不降级会打死整轮循环；② 不带 [ERR:CODE] 则 RecoveryManager
    提取不到错误码，救援提示注入不进去，模型只看到一句英文异常。
    （旧实现输出裸的 "Error executing X: ..."，就是这个毛病。）
    """
    reg = RegistryImpl()
    reg.register(_Tool(boom=RuntimeError("kaboom")))
    res = await reg.execute(_call())
    assert res.is_error is True
    assert res.output.startswith("[ERR:UNKNOWN]")
    assert "kaboom" in res.output
    assert extract_code(res.output) is ErrorCode.UNKNOWN  # 自愈体系能识别


# ══════════════════════════════════════════════════════════════
# 中间件链
# ══════════════════════════════════════════════════════════════


async def test_middleware_allow_passes_through():
    reg, tool = RegistryImpl(), _Tool(output="ok")
    reg.register(tool)
    order: list[str] = []

    async def mw(call):
        order.append("mw")
        return True, ""

    reg.use(mw)
    res = await reg.execute(_call())
    assert res.output == "ok"
    assert order == ["mw"]


async def test_middleware_reject_short_circuits_tool():
    """拦截时工具**绝不能**被执行"""
    reg, tool = RegistryImpl(), _Tool()
    reg.register(tool)
    reg.use(lambda call: _deny("危险命令"))

    res = await reg.execute(_call())
    assert res.is_error is True
    assert "执行被系统拦截" in res.output
    assert "危险命令" in res.output
    assert tool.seen == []


async def test_middleware_reject_carries_error_code():
    """拦截结果必须带错误码，且该码要有救援提示

    真实来源是审批回路（人工拒绝 / 超时拒绝）：模型拿到一句"被拦截"却不知道
    下一步该干什么时，最自然的反应是**用相同参数重试**，然后一次次撞同一道墙。
    """
    from tiny_claw.context.recovery import RecoveryManager, _RECOVERY_HINTS

    reg = RegistryImpl()
    reg.register(_Tool())
    reg.use(lambda call: _deny("审批超时，自动拒绝"))

    res = await reg.execute(_call())
    assert res.output.startswith("[ERR:TOOL_BLOCKED]")
    assert extract_code(res.output) is ErrorCode.TOOL_BLOCKED
    # 该错误码必须配有救援提示，否则加了这个码等于白加
    assert ErrorCode.TOOL_BLOCKED in _RECOVERY_HINTS
    assert "系统救援指南" in RecoveryManager().analyze_and_inject(res.output)


async def test_middlewares_run_in_registration_order():
    reg = RegistryImpl()
    reg.register(_Tool())
    order: list[int] = []

    async def m1(call):
        order.append(1)
        return True, ""

    async def m2(call):
        order.append(2)
        return True, ""

    reg.use(m1)
    reg.use(m2)
    await reg.execute(_call())
    assert order == [1, 2]


async def test_first_rejection_stops_later_middlewares():
    """任一拦截即终止：后面的中间件不该再被调用"""
    reg = RegistryImpl()
    reg.register(_Tool())
    order: list[str] = []

    async def deny(call):
        order.append("deny")
        return False, "no"

    async def never(call):
        order.append("never")
        return True, ""

    reg.use(deny)
    reg.use(never)
    res = await reg.execute(_call())
    assert order == ["deny"]
    assert res.is_error is True


async def test_middleware_not_reached_for_unknown_tool():
    """工具不存在时在**路由阶段**就返回了，不该走中间件"""
    reg = RegistryImpl()
    called: list[int] = []

    async def mw(call):
        called.append(1)
        return True, ""

    reg.use(mw)
    await reg.execute(_call(name="nope"))
    assert called == []


# ══════════════════════════════════════════════════════════════
# 注册与工具清单
# ══════════════════════════════════════════════════════════════


def test_duplicate_registration_overwrites():
    """重名是 warning + 覆盖 —— MCP 工具加命名空间前缀正是为了防这个"""
    reg = RegistryImpl()
    reg.register(_Tool(output="first"))
    reg.register(_Tool(output="second"))
    assert len(reg.get_available_tools()) == 1


def test_get_available_tools_returns_definitions():
    reg = RegistryImpl()
    reg.register(_Tool(name="alpha"))
    defs = reg.get_available_tools()
    assert [d.name for d in defs] == ["alpha"]
    assert defs[0].input_schema == {"type": "object"}
