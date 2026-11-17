#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
    exec ".venv/bin/python" launch_dashboard.py "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 launch_dashboard.py "$@"
elif command -v python >/dev/null 2>&1; then
    exec python launch_dashboard.py "$@"
else
    echo "[ERRO] Python 3.11+ não encontrado no PATH." >&2
    exit 1
fi
