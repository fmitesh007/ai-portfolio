#!/usr/bin/env bash
# One entry point for the ai-portfolio services. Shell-agnostic: run with
#   bash scripts/run.sh <command>   (works in fish, zsh, bash)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

ensure_venv() {
  if [ ! -x "$PY" ]; then
    echo "Creating virtualenv at $ROOT/.venv"
    python3 -m venv "$ROOT/.venv"
  fi
  if ! "$PY" -c "import fastapi, pytest, chromadb, langgraph" >/dev/null 2>&1; then
    echo "Installing requirements into .venv"
    "$PY" -m pip install -r "$ROOT/requirements.txt"
  fi
}

serve() {
  echo "Starting $1 on http://localhost:$2  (CTRL+C to stop)"
  cd "$ROOT"
  exec "$PY" -m uvicorn "$1" --reload --port "$2"
}

usage() {
  cat <<EOF
Usage: bash scripts/run.sh <command>

  setup      create .venv and install requirements
  test       run pytest (17 tests, no API key needed)
  docquery   RAG service       -> http://localhost:8001/docs
  llmmeter   telemetry + dash  -> http://localhost:8002
  agentflow  LangGraph agent   -> http://localhost:8003/docs
  all        start all three services together (no reload)
  help       show this message

Examples (fish):
  bash scripts/run.sh test
  bash scripts/run.sh docquery
  bash scripts/run.sh all
EOF
}

case "${1:-help}" in
  setup) ensure_venv ;;
  test)
    ensure_venv
    cd "$ROOT"
    "$PY" -m pytest -q
    ;;
  docquery) ensure_venv; serve docquery.app.main:app 8001 ;;
  llmmeter) ensure_venv; serve llmmeter.app.main:app 8002 ;;
  agentflow) ensure_venv; serve agentflow.app.main:app 8003 ;;
  all)
    ensure_venv
    pids=()
    start_bg() {
      (cd "$ROOT" && "$PY" -m uvicorn "$1" --port "$2") &
      pids+=("$!")
      echo "$1 -> http://localhost:$2 (pid $!)"
    }
    start_bg docquery.app.main:app 8001
    start_bg llmmeter.app.main:app 8002
    start_bg agentflow.app.main:app 8003
    trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT INT TERM
    echo "All three running. CTRL+C stops them."
    wait || true
    ;;
  help | -h | --help | *) usage ;;
esac
