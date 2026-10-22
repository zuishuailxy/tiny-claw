# CLI
# uv pip install -e . 2>&1 && uv run main.py -p "用 chrome-devtools 工具打开 https://example.com，告诉我页面标题" -d workspace --no-thinking -s chrome-test-2  2>&1

uv pip install -e . 2>&1 && uv run main.py -p "读取 a b log文件告诉我里面有什么" -d workspace --no-thinking -s test  2>&1


# 飞书 mode
# uv pip install -e . 2>&1 && uv run main.py -d "./workspace" --m feishu 2>&1

# run Bunchmark
# uv pip install -e . 2>&1 && uv run run_benchmark.py 2>&1