#!/bin/bash
# Environment wrapper — every Python entry point runs through this script.
#
#   bash experiments/run.sh [--venv gr00t|openpi|lerobot] <script.py> [args...]
#
# Selects the model track's venv interpreter (default: gr00t — all existing
# invocations unchanged), sets EGL rendering, the ImageMagick library path,
# PYTHONPATH, and the HF token. Inline VAR=... prefixes are unreliable across
# launch paths (backgrounded shells dropped them, 2026-07-14) — exports live
# here, in one place. Machine-specific paths come from
# experiments/load_machine_env.sh (+ optional machine.env override).
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_machine_env.sh"

VENV_NAME="gr00t"
if [ "${1:-}" = "--venv" ]; then
  VENV_NAME="${2:?--venv needs a value (gr00t|openpi|lerobot)}"
  shift 2
fi
case "$VENV_NAME" in
  gr00t)   VENV_PATH="$PLADIS_VENV_GR00T" ;;
  openpi)  VENV_PATH="$PLADIS_VENV_OPENPI" ;;
  lerobot) VENV_PATH="$PLADIS_VENV_LEROBOT" ;;
  *) echo "[run.sh] unknown --venv '$VENV_NAME' (expected gr00t|openpi|lerobot)" >&2; exit 2 ;;
esac
if [ -z "$VENV_PATH" ] || [ ! -x "$VENV_PATH/bin/python" ]; then
  echo "[run.sh] venv '$VENV_NAME' not usable at '${VENV_PATH:-<unset>}'" >&2
  echo "[run.sh] set PLADIS_VENV_* in experiments/machine.env (build recipe: docs/SETUP.md)" >&2
  exit 2
fi

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MAGICK_HOME="$PLADIS_MAGICK_HOME"
export LD_LIBRARY_PATH="$PLADIS_MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
_PYPATH="$REPO"
# the openpi track reuses RLinf's serving reference (toolkits./rlinf. imports)
[ "$VENV_NAME" = "openpi" ] && _PYPATH="$_PYPATH:$PLADIS_RLINF_PATH"
export PYTHONPATH="$_PYPATH${PYTHONPATH:+:$PYTHONPATH}"
export HF_TOKEN="$(cat "$PLADIS_HF_TOKEN_FILE")"
export TOKENIZERS_PARALLELISM=false
export RLINF_PATH="$PLADIS_RLINF_PATH"

exec "$VENV_PATH/bin/python" "$@"
