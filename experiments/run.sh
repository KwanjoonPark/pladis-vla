#!/bin/bash
# Env wrapper for all harness runs: `bash experiments/run.sh <script.py> [args]`.
# Inline VAR=... prefixes are unreliable across launch paths (backgrounded
# shells dropped them, 2026-07-14) — exports live here instead, in one place.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(dirname "$REPO")"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MAGICK_HOME="$WS/LIBERO-plus/.magick"
export LD_LIBRARY_PATH="$WS/LIBERO-plus/.magick/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_TOKEN="$(cat /home/reallab/.hf_user_token)"
export TOKENIZERS_PARALLELISM=false
export RLINF_PATH="$WS/RLinf"

PY="$WS/RLinf/gr00t_n1d7/bin/python"
exec "$PY" "$@"
