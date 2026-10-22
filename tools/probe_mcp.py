"""MCP 连接探针 —— Phase 0 一次性脚本（对应 docs/MCP_INTEGRATION_PLAN.md §8）

在写任何 adapter 之前，先证明三件事成立：

  1. 本机能起一个标准 MCP server 子进程（stdio 传输）
  2. `initialize` 握手能成功
  3. `list_tools` 返回的 schema 是**标准 JSON Schema**，可直接塞进
     现有的 `ToolDefinition`（plan 的「决策 2：schema 直通」）

地基不成立，后面两天的计划就得重排 —— 所以这一步必须在写代码之前做完。

跑法：

    .venv/bin/python tools/probe_mcp.py
    .venv/bin/python tools/probe_mcp.py --dir ./workspace
    .venv/bin/python tools/probe_mcp.py --command /abs/path/to/npx --timeout 300
"""

import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# WorkBuddy 托管的 node 不在默认 PATH 里，子进程要显式指路
_NODE_BIN_CANDIDATES = [
    Path.home() / ".workbuddy/binaries/node/versions/22.22.2-3/bin",
]

DEFAULT_PACKAGE = "@modelcontextprotocol/server-filesystem"


def _unwrap_and_print(exc: BaseException, depth: int = 0) -> None:
    """递归展开 ExceptionGroup，打印真正的叶子异常。

    MCP SDK 基于 anyio task group，任何失败都会被包成
    `ExceptionGroup: unhandled errors in a TaskGroup`——直接打这行等于没打，
    必须挖到叶子才知道是连接断了、server 崩了、还是协议帧有问题。
    """
    indent = "  " * (depth + 1)
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            _unwrap_and_print(sub, depth + 1)
    else:
        print(f"{indent}↳ {type(exc).__name__}: {exc}")


def _build_env() -> dict[str, str]:
    """给 npx 子进程准备环境：补 PATH，确保 node/npx 能被找到。"""
    env = dict(os.environ)
    extra = [str(p) for p in _NODE_BIN_CANDIDATES if p.is_dir()]
    if extra:
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _resolve_npx(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    found = shutil.which("npx")
    if found:
        return found
    for p in _NODE_BIN_CANDIDATES:
        candidate = p / "npx"
        if candidate.is_file():
            return str(candidate)
    return None


def _input_schema(tool) -> dict:
    """取工具的入参 schema。

    ⚠️ 字段名坑：MCP **线协议**上叫 `inputSchema`，但 Python SDK v2 的模型属性
    是 `input_schema`（`inputSchema` 只是 alias）。照协议名写 `tool.inputSchema`
    会直接 AttributeError —— Phase 0 就是靠这个把 bug 提前抓出来的。
    两个都试，兼容 SDK 改名。
    """
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return schema or {}


def _describe_tool(tool) -> None:
    schema = _input_schema(tool)
    props = list((schema.get("properties") or {}).keys())
    required = schema.get("required") or []
    desc = (tool.description or "").strip().replace("\n", " ")
    print(f"    • {tool.name}")
    print(f"        描述: {desc[:96]}{'…' if len(desc) > 96 else ''}")
    print(f"        参数: {props or '（无）'}  必填: {required or '（无）'}")


async def probe(command: str, args: list[str], overall_timeout: float) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from tiny_claw.schema import ToolDefinition  # 用来验证 schema 同构

    params = StdioServerParameters(command=command, args=args, env=_build_env())

    print(f"  server 命令: {command} {' '.join(args)}")
    print("  （首次运行 npx 会下载包，可能要几十秒）\n")

    async with asyncio.timeout(overall_timeout):
        t0 = time.monotonic()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                # ---- 验证点 1：initialize 握手 ----
                await session.initialize()
                server = getattr(session, "server_info", None)
                print(
                    f"[1/3] ✅ initialize 成功  {time.monotonic() - t0:.1f}s"
                    f"  server={getattr(server, 'name', '?')} "
                    f"{getattr(server, 'version', '')}"
                )

                # ---- 验证点 2：工具发现（含分页）----
                # list_tools 支持游标分页：只取第一页会**静默漏掉工具**，
                # 必须顺着 next_cursor 收完。
                t1 = time.monotonic()
                tools, cursor, pages = [], None, 0
                while True:
                    listed = await session.list_tools(cursor) if cursor else await session.list_tools()
                    tools.extend(listed.tools)
                    pages += 1
                    cursor = getattr(listed, "next_cursor", None)
                    if not cursor:
                        break
                print(
                    f"[2/3] ✅ list_tools 成功  {time.monotonic() - t1:.2f}s"
                    f"  共 {len(tools)} 个工具（{pages} 页）\n"
                )
                for tool in tools:
                    _describe_tool(tool)

                # ---- 验证点 3：schema 能否零转换塞进 ToolDefinition ----
                if not tools:
                    print("\n[3/3] ⚠️ 该 server 没有暴露任何工具")
                    return 1

                first = tools[0]
                schema = _input_schema(first) or {"type": "object", "properties": {}}
                definition = ToolDefinition(
                    name=f"mcp__probe__{first.name}",
                    description=first.description or "",
                    input_schema=schema,
                )
                is_object = definition.input_schema.get("type") == "object"
                print(
                    f"\n[3/3] ✅ schema 直通验证  "
                    f"ToolDefinition(name={definition.name!r}) 构造成功"
                )
                print(
                    f"      schema.type={definition.input_schema.get('type')!r}"
                    f"  properties={list((definition.input_schema.get('properties') or {}).keys())}"
                    f"  → {'是标准 JSON Schema，可零转换透传' if is_object else '⚠️ 不是 object 类型，需要转换'}"
                )
    return 0


def main() -> int:
    # 关掉块缓冲：管道下（| tail）异常退出会丢掉未 flush 的输出，
    # 而探针的价值恰恰在于「崩之前打到了哪一步」
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="MCP 连接探针（Phase 0）")
    parser.add_argument(
        "--dir",
        default=str(PROJECT_ROOT / "workspace"),
        help="暴露给 filesystem server 的目录（默认 ./workspace）",
    )
    parser.add_argument("--command", default=None, help="npx 路径，默认自动探测")
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="MCP server npm 包名")
    parser.add_argument("--timeout", type=float, default=240.0, help="整体超时秒数")
    args = parser.parse_args()

    npx = _resolve_npx(args.command)
    if not npx:
        print("❌ 找不到 npx：请确认已安装 Node.js，或用 --command 指定绝对路径")
        return 1
    if not Path(args.dir).is_dir():
        print(f"❌ 目录不存在: {args.dir}")
        return 1

    print("=" * 72)
    print("MCP 探针启动（Phase 0：环境打通）")
    print("=" * 72)

    try:
        return asyncio.run(
            probe(npx, ["-y", args.package, args.dir], args.timeout)
        )
    except TimeoutError:
        print(f"\n❌ 整体超时（{args.timeout:.0f}s）：可能是 npx 下载包太慢或网络不可达")
        return 1
    except Exception as e:  # noqa: BLE001 —— 探针就是要如实报告任何一种失败
        print(f"\n❌ 连接失败: {type(e).__name__}: {e}")
        print("   叶子异常（挖开 TaskGroup）：")
        _unwrap_and_print(e)
        print("\n排查方向：")
        print("  1. 网络能否访问 npm registry（npx 首次要下载包）")
        print("  2. node/npx 是否可用：npx --version")
        print("  3. 手动试一次：npx -y " + args.package + " " + args.dir)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
