#!/usr/bin/env bash
# ------------------------------------------------------------------
# YAND-MVSK Tail-Risk Studio — one-command launcher.
# Creates a virtualenv, installs dependencies, and starts the studio.
# Open http://127.0.0.1:8000 when it says "Application startup complete".
# ------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PY="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "→ Creating virtual environment ($VENV) ..."
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "→ Installing dependencies (first run only) ..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
echo ""
echo "  ┌────────────────────────────────────────────────┐"
echo "  │  YAND-MVSK Tail-Risk Studio                     │"
echo "  │  → http://127.0.0.1:${PORT}                        │"
echo "  └────────────────────────────────────────────────┘"
echo ""
exec python -m uvicorn yand_mvsk.api.app:app --host 127.0.0.1 --port "$PORT"
