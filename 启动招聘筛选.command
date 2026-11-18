#!/bin/zsh
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "未找到 .venv，请先运行：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  read -k 1 "?按任意键关闭窗口..."
  exit 1
fi

# 从本机 .env 加载模型配置；不打印密钥内容。没有配置密钥时仍可使用模拟模式。
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

exec .venv/bin/streamlit run screening_app.py
