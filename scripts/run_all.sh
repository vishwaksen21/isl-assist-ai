#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND_PID=""
FRONTEND_PID=""

if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "Backend already running: http://localhost:8000"
else
  echo "Starting backend: http://localhost:8000"
  bash "$REPO_ROOT/scripts/run_backend.sh" &
  BACKEND_PID=$!
  sleep 1
fi

if lsof -nP -iTCP:5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "Note: port 5173 already in use; frontend will choose another port."
fi

echo "Starting frontend (http.server)"
bash "$REPO_ROOT/scripts/run_frontend.sh" &
FRONTEND_PID=$!

cleanup() {
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

PIDS=()
[[ -n "$BACKEND_PID" ]] && PIDS+=("$BACKEND_PID")
[[ -n "$FRONTEND_PID" ]] && PIDS+=("$FRONTEND_PID")

wait "${PIDS[@]}"
