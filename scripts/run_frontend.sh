#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${ISL_PYTHON:-}" ]]; then
  PY="$ISL_PYTHON"
elif [[ -f "$REPO_ROOT/.python-version" ]]; then
  pyenv_ver="$(tr -d ' \t\n\r' < "$REPO_ROOT/.python-version" || true)"
  if [[ -n "$pyenv_ver" && -x "$HOME/.pyenv/versions/$pyenv_ver/bin/python" ]]; then
    PY="$HOME/.pyenv/versions/$pyenv_ver/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
  else
    PY="python"
  fi
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

PORT="${ISL_FRONTEND_PORT:-5173}"
for _ in {1..10}; do
  if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    break
  fi
  PORT=$((PORT + 1))
done

cd "$REPO_ROOT/frontend"
echo "Frontend serving on http://localhost:$PORT"
exec "$PY" -m http.server "$PORT"
