#!/bin/zsh
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "没有找到 Python 3。请先安装 Python 3.10 或更新版本。"
  read -k 1 "?按任意键关闭窗口..."
  exit 1
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv || exit 1
fi
source .venv/bin/activate
python -m pip install -r requirements.txt || {
  echo "依赖安装失败。请检查网络后重新运行。"
  read -k 1 "?按任意键关闭窗口..."
  exit 1
}
streamlit run app.py
