"""终端 Reporter

在终端直观打印 Agent 状态，比 CliReporter 有更好的视觉层次。
"""

from typing import Any

from tiny_claw.engine.reporter import Reporter


class TerminalReporter(Reporter):
    """终端输出实现"""

    async def on_thinking(self, content: str) -> None:
        print(f"\n[🤔 思考中] 模型正在推理...")

    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        print(f"[🛠️  调用工具] {tool_name}")
        args_str = str(args).replace("\n", "\\n").replace("\r", "\\r")
        if len(args_str) > 150:
            args_str = args_str[:150] + "... (已截断)"
        print(f"   参数: {args_str}")

    async def on_tool_result(self, tool_name: str, output: str, is_error: bool) -> None:
        if is_error:
            print(f"[❌ 执行失败] {tool_name}")
            if output:
                print(f"   错误: {output[:200]}")
        else:
            print(f"[✅ 执行成功] {tool_name}")

    async def on_message(self, content: str) -> None:
        if not content:
            return
        print(f"\n🤖 Agent 回复:\n{content}\n")
