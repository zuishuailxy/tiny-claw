"""兼容入口 —— 真正的实现已移入包内 `tiny_claw.cli`

保留这个文件是为了让两种用法都成立：

    python main.py -p "任务描述"     # 直接跑，无需安装（把 src 挂上 sys.path）
    tiny-claw -p "任务描述"          # pip install -e . 之后由 entry point 提供

之所以不把入口留在根目录：`pyproject.toml` 的包发现范围是 `src/`，
根目录的模块不会被安装，entry point 也就找不到它（曾指向不存在的
`tiny_claw.main`，导致 `pip install -e .` 装完没有可用命令）。
"""

import sys
from pathlib import Path

# 未安装时（直接 python main.py）把 src 加入搜索路径，省掉一步安装
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tiny_claw.cli import main  # noqa: E402  （必须在 sys.path 调整之后导入）

if __name__ == "__main__":
    main()
