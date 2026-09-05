#!/bin/bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${1:-}"
PORT="${PORT:-5001}"

if [ -z "$CODE_DIR" ]; then
  read -r -p "Full path to the logic-grading-facilitator code folder: " CODE_DIR
fi
if [ ! -f "$CODE_DIR/app/server.py" ]; then
  echo "That folder does not contain app/server.py."
  exit 1
fi

export DATA_DIR="$BUNDLE_ROOT/data"
export DATA_ARCHIVE_DIR="$BUNDLE_ROOT/archive"
export ENV_MODE="PRODUCTION"
export TF_GRADER_MODE="1"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_COMMAND="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_COMMAND="python"
else
  echo "Python was not found. Install Python 3, reopen Terminal, and try again."
  exit 1
fi

cd "$CODE_DIR"
echo "TF grader: http://127.0.0.1:$PORT/problem/Q1/strips"
echo "Keep this window open while grading. Press Control-C to stop."
"$PYTHON_COMMAND" -m uvicorn app.server:app --port "$PORT"
