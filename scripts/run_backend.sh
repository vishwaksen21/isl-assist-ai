#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pick_python() {
  local -a candidates=()

  if [[ -n "${ISL_PYTHON:-}" ]]; then
    candidates+=("$ISL_PYTHON")
  fi

  if [[ -f "$REPO_ROOT/.python-version" ]]; then
    local pyenv_ver
    pyenv_ver="$(tr -d ' \t\n\r' < "$REPO_ROOT/.python-version" || true)"
    if [[ -n "$pyenv_ver" && -x "$HOME/.pyenv/versions/$pyenv_ver/bin/python" ]]; then
      candidates+=("$HOME/.pyenv/versions/$pyenv_ver/bin/python")
    fi
  fi

  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    candidates+=("$REPO_ROOT/.venv/bin/python")
  fi

  if command -v python3.11 >/dev/null 2>&1; then
    candidates+=("python3.11")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("python3")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("python")
  fi

  for c in "${candidates[@]}"; do
    if "$c" -c 'import uvicorn' >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done

  return 1
}

if ! PY="$(pick_python)"; then
  echo "Couldn't find a Python with uvicorn installed." >&2
  echo "Tip: set ISL_PYTHON to your pyenv python, e.g.:" >&2
  echo "  export ISL_PYTHON=\"$HOME/.pyenv/versions/3.11.9/bin/python\"" >&2
  exit 1
fi

cd "$REPO_ROOT/backend"
exec "$PY" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
