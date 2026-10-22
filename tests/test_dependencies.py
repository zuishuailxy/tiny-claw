"""依赖声明与真实 import 的一致性守门

**为什么需要它**：本项目栽过一次「声明与代码脱节」，而且**三处口径互相矛盾**：

| 位置 |当时的声称 |
|---|---|
| `pyproject.toml` | `mcp` 是 optional extra |
| `mcp/__init__.py` | 「依赖是 optional extra」 |
| `mcp/client.py` | 「`mcp` 在 pyproject 里是必装依赖」← 与事实不符 |

而 `cli.py` 顶层 `from tiny_claw.mcp import MCPManager`、`_StderrRoutingGroup` 又
继承 SDK 的真类 —— 这条链**没法延迟加载**。结果是 `pip install tiny-claw`
（不装 extra）连 `--help` 都跑不起来，用户拿到一个装完即废的包。

这类 bug 的特征和「文档数字过期」一样：**没有任何测试会因此变红**。
所以这里把「代码顶层 import 的第三方包」与「pyproject 声明的依赖」对账。

**两条判据，缺一不可**：

1. **顶层 import 必须在 dependencies 里**（或标准库）——漏声明 = 装完起不来
2. **dependencies 里的包必须真被用到**——多声明 = 白让用户装东西

⚠️ 只扫**顶层** import：函数内的延迟 import（如 `from mcp import
StdioServerParameters`）不构成硬依赖，那正是「可选依赖」该有的写法。
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
SRC = ROOT / "src" / "tiny_claw"

# 标准库顶层模块名交给解释器给，别手维护 —— 手写的名单必然漏
# （第一版就把 lark_oapi 误划进标准库，导致「白声明」判据失真）。
STDLIB = set(sys.stdlib_module_names) | {"tiny_claw"}

# import 名 → PyPI 包名（不一致的才列）
IMPORT_TO_PYPI = {
    "dotenv": "python-dotenv",
    "lark_oapi": "lark-oapi",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}


def _core_deps() -> set[str]:
    """**只**取 `dependencies`，不含 optional extra。

    为什么 extra 不算：extra 本来就要用户显式选装，它压根不是
    「装完就有」的保证。把 extra 算进来，这条判据就会对
    「顶层 import 了某个 extra 包」完全沉默 —— 而那正是本项目栽过的坑。
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return {_pypi_name(s) for s in data["project"]["dependencies"]}


def _all_deps() -> set[str]:
    """`dependencies` + 所有 extra —— 用于「白声明」那条判据"""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    raw = list(data["project"]["dependencies"])
    for extra in (data["project"].get("optional-dependencies") or {}).values():
        raw.extend(extra)
    return {_pypi_name(s) for s in raw}


def _pypi_name(spec: str) -> str:
    """`"pyjwt[crypto]>=2.10"` / `"uvicorn ; sys_platform != 'x'"` → `"pyjwt"` / `"uvicorn"`"""
    name = spec.split("[")[0].split(">")[0].split("<")[0].split("=")[0]
    return name.split(";")[0].strip().lower()


def _imports_in(src_root: Path, *, module_level_only: bool) -> set[str]:
    """扫出第三方顶层包名。

    `module_level_only=True` 时只取**模块顶层**的 import（不含函数体/类体）——
    那才是硬依赖；函数内的延迟 import 正是「可选依赖」该有的写法，不该算进去。
    """
    found: set[str] = set()
    for path in sorted(src_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        # 模块顶层的行号：不在任何函数/类体范围内
        nested: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.body[0].lineno
                nested.update(range(start, (node.end_lineno or start) + 1))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if module_level_only and node.lineno in nested:
                continue

            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue  # 相对 import
                if node.module:
                    names = [node.module.split(".")[0]]

            for n in names:
                if n and n not in STDLIB:
                    found.add(n)
    return found


def _top_level_third_party_imports() -> set[str]:
    """含函数体/类体内的一切 import（延迟加载也算真在用）"""
    return _imports_in(SRC, module_level_only=False)


def _module_level_imports() -> set[str]:
    """只算模块顶层 —— 这才是装不装得上直接相关的硬依赖"""
    return _imports_in(SRC, module_level_only=True)


# ═══════════════════════════════════════════════════════════════
# 判据 1：顶层 import 必须被声明
# ═══════════════════════════════════════════════════════════════


def test_cli_imports_without_optional_extras():
    """CLI 的 import 链不许依赖任何没进 `dependencies` 的包。

    这是本文件存在的**主要原因**：曾经的 `mcp` 就是「声明成可选 extra、
    实际模块顶层 import」—— 裸`pip install tiny-claw` 装完连 `--help` 都跑不起来。

    只认 `dependencies`、**不认 extra**：extra 要用户显式选装，
    它不是「装完就有」的承诺，所以顶层 import 一个 extra 包同样算脱节。
    """
    core = _core_deps()
    missing = set()
    for name in _module_level_imports():
        pypi = IMPORT_TO_PYPI.get(name, name).lower()
        if pypi not in core:
            missing.add(f"{name}（PyPI 名 {pypi}）")
    assert not missing, (
        "这些包在模块顶层被 import，但没进 pyproject 的 dependencies——"
        f"裸 `pip install tiny-claw` 会直接 ModuleNotFoundError：{sorted(missing)}"
    )


def test_mcp_sdk_is_a_hard_dependency():
    """`mcp` 必须留在 dependencies 里（曾经被误降级成 optional extra）。

    定向守这条，因为它是唯一「继承 SDK 真类、无法延迟加载」的依赖：
    `_StderrRoutingGroup(ClientSessionGroup)` 在模块顶层就得成立。
    """
    core = _core_deps()
    assert "mcp" in core, (
        "`mcp` 被移出 dependencies 了——但 `cli.py` 顶层 import 它、"
        "且 `_StderrRoutingGroup` 继承它的真类，无法延迟加载"
    )


# ═══════════════════════════════════════════════════════════════
# 判据 2：声明的依赖必须真被用到（防白装）
# ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("pypi_name", ["mcp", "openai", "anthropic", "lark-oapi"])
def test_declared_core_dep_is_actually_imported(pypi_name):
    """核心依赖不许是白声明的 —— 否则用户白装一堆东西。

    用 `_top_level_third_party_imports`（含函数体）而不是
    `_module_level_imports`：只要代码里**任何地方**用到就算数，
    延迟 import 也是真用。
    """
    imported = {
        IMPORT_TO_PYPI.get(n, n).lower() for n in _top_level_third_party_imports()
    }
    assert pypi_name.lower() in imported, (
        f"`{pypi_name}` 在 dependencies 里，但 src/ 里找不到任何 import——"
        "要么它已不再需要，要么 import 被改成了动态形式"
    )