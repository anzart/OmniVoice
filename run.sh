#!/usr/bin/env bash
# Launch the OmniVoice (Gradio) TTS server using the local virtualenv.
# Run from anywhere: the script cd's into its own directory so that the
# `omnivoice` package is importable and relative model caches resolve.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x "env/bin/python" ]]; then
  echo "[omnivoice] No virtualenv found at omnivoice-server/env." >&2
  echo "[omnivoice] Create it with:" >&2
  echo "  /opt/homebrew/opt/python@3.10/bin/python3.10 -m venv env" >&2
  echo "  env/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

# Gradio host/port (override via env if needed).
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

exec env/bin/python app.py
