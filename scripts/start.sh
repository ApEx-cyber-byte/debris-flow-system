#!/usr/bin/env bash
# 一键启动 / One-shot launcher (Linux / macOS)
#
#   ./scripts/start.sh
#
# 创建 Python 虚拟环境、安装前后端依赖，然后同时拉起 FastAPI (:8000)
# 与 Vite (:5173)。Ctrl+C 会一并停止两个服务。

set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="$root/backend"
frontend="$root/frontend"
venv_python="$backend/.venv/bin/python"

step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }

if ! command -v python3 >/dev/null 2>&1; then
  echo "找不到 python3，请安装 Python 3.11–3.13。" >&2
  exit 1
fi

step "泥石流坡度滚落物理动态分析系统 — 启动中"
echo "项目根目录: $root"

# ---------------------------------------------------------------- backend
if [[ ! -x "$venv_python" ]]; then
  step "创建 Python 虚拟环境"
  python3 -m venv "$backend/.venv"
fi

step "安装后端依赖"
"$venv_python" -m pip install --upgrade pip --quiet
"$venv_python" -m pip install -r "$backend/requirements.txt" --quiet

step "启动后端 http://127.0.0.1:$BACKEND_PORT"
( cd "$backend" && "$venv_python" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" ) &
backend_pid=$!

# ---------------------------------------------------------------- frontend
if ! command -v pnpm >/dev/null 2>&1 && [[ ! -d "$frontend/node_modules" ]]; then
  echo "找不到 pnpm。请先安装：npm install -g pnpm" >&2
  kill "$backend_pid" 2>/dev/null || true
  exit 1
fi

if [[ ! -d "$frontend/node_modules" ]]; then
  step "安装前端依赖"
  ( cd "$frontend" && pnpm install )
fi

step "启动前端 http://127.0.0.1:$FRONTEND_PORT"
( cd "$frontend" && pnpm dev --port "$FRONTEND_PORT" ) &
frontend_pid=$!

cleanup() {
  step "停止服务"
  kill "$backend_pid" "$frontend_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cat <<EOF

就绪：
  前端      http://127.0.0.1:$FRONTEND_PORT
  后端 API  http://127.0.0.1:$BACKEND_PORT
  接口文档  http://127.0.0.1:$BACKEND_PORT/docs

按 Ctrl+C 停止。
EOF

wait
