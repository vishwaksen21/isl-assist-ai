#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLIST_SRC="$REPO_ROOT/scripts/macos/com.isl-assist-ai.backend.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.isl-assist-ai.backend.plist"

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

mkdir -p "$HOME/Library/LaunchAgents"

# Replace __REPO_ROOT__ placeholder with the real path
sed -e "s|__REPO_ROOT__|$REPO_ROOT|g" -e "s|__PYTHON__|$PY|g" "$PLIST_SRC" > "$PLIST_DST"

# (Re)load
launchctl bootout "gui/$UID" "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$PLIST_DST"
launchctl enable "gui/$UID/com.isl-assist-ai.backend" || true
launchctl kickstart -k "gui/$UID/com.isl-assist-ai.backend" || true

echo "Installed. Backend should be auto-started on login and now running on http://localhost:8000"
echo "Logs: tail -f /tmp/isl-assist-ai-backend.log"
