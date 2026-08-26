#!/usr/bin/env bash
# JARVIS Knowledge Core launcher
# Usage: ./run.sh [port]   (default 8630)
cd "$(dirname "$0")"
PORT="${1:-8630}"
mkdir -p data
exec .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port "$PORT" "${@:2}"
