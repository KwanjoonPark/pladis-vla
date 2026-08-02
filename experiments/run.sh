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
if [ "$VENV_NAME" = "openpi" ]; then
  # RLinf is NOT on the pi0.5 eval path (harness/model_pi05.py builds the official
  # openpi Policy from openpi.* only — the GR00T lesson at model_gr00t.py:4-11).
  # It stays importable solely for the serving-route bisect reference,
  # toolkits/standalone_eval_scripts/openpi/libero_eval.py.
  _PYPATH="$_PYPATH:$PLADIS_RLINF_PATH"
  # jax is a HARD import dependency of openpi even on the PyTorch path
  # (openpi.transforms / openpi.policies.policy / openpi.shared.download all import
  # it). Left on GPU it preallocates ~75% of a device at import time, which OOMs the
  # rollout or, on this shared box, steals a neighbouring project's GPU.
  export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
  # pi0_pytorch.py:112 unconditionally torch.compile(sample_actions, "max-autotune").
  # A compiled graph bakes in whichever eager_attention_forward global it traced, which
  # would silently un-install the PLADIS monkeypatch AND hide the flow-noise RNG.
  # harness/model_pi05.py also un-compiles explicitly and asserts; this is the belt.
  export TORCH_COMPILE_DISABLE=1
fi
export PYTHONPATH="$_PYPATH${PYTHONPATH:+:$PYTHONPATH}"
export HF_TOKEN="$(cat "$PLADIS_HF_TOKEN_FILE")"
export TOKENIZERS_PARALLELISM=false
export RLINF_PATH="$PLADIS_RLINF_PATH"

exec "$VENV_PATH/bin/python" "$@"
