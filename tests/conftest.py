"""pytest 全局引导。

做两件事：
1. 路径引导：让 pytest 能直接 import src/tiny_claw（无需先 pip install -e .）
2. 配置隔离：在任何 tiny_claw 模块被 import 之前关掉 .env 加载

第 2 点很关键。tiny_claw.config 在 import 时会读取项目根 .env 并对
调参常量做快照。若不加隔离，同一个测试在「开发者填了 .env」和
「CI 没有 .env」两台机器上会跑出不同结果——这正是最难排查的一类问题。
隔离后所有用例都跑在代码默认配置上，结果可复现。

⚠️ 因此这段必须在 sys.path 设置之后、任何 tiny_claw import 之前执行。
"""

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 关掉 .env：单测一律使用代码默认值，不受开发者本地配置影响。
# 需要测「读取 env」的用例请用 monkeypatch.setenv 显式构造。
os.environ["TINY_CLAW_DISABLE_DOTENV"] = "1"
