# Shared machine-config loader — source this, do not execute.
#
# Precedence: experiments/machine.env (gitignored, per-machine) is sourced
# first; anything it leaves unset falls back to workspace-derived defaults
# ($WS = parent directory of this repo). Copy machine.env.example to
# machine.env on a new machine and edit only what differs (docs/SETUP.md).
_PLADIS_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$_PLADIS_ENV_DIR")"
WS="$(dirname "$REPO")"

if [ -f "$_PLADIS_ENV_DIR/machine.env" ]; then
  . "$_PLADIS_ENV_DIR/machine.env"
fi

# venv per model track (bin/python inside each)
: "${PLADIS_VENV_GR00T:=$WS/RLinf/gr00t_n1d7}"
: "${PLADIS_VENV_OPENPI:=$WS/RLinf/openpi}"
: "${PLADIS_VENV_LEROBOT:=$WS/RLinf/gr00t_n1d7}"  # lerobot 0.4.4 lives in the gr00t venv on this machine
# runtime dependencies
: "${PLADIS_HF_TOKEN_FILE:=$HOME/.hf_user_token}"
: "${PLADIS_MAGICK_HOME:=$WS/LIBERO-plus/.magick}"
: "${PLADIS_RLINF_PATH:=$WS/RLinf}"
# checkpoint roots per model (downloaded per machine, never in git)
: "${MODEL_ROOT_GR00T_N17:=$WS/models/GR00T-N1.7-LIBERO}"
: "${MODEL_ROOT_PI0:=$WS/models/RLinf-Pi0-LIBERO-Long-SFT}"
: "${MODEL_ROOT_PI05:=$WS/models/pi05_libero}"

export REPO WS \
  PLADIS_VENV_GR00T PLADIS_VENV_OPENPI PLADIS_VENV_LEROBOT \
  PLADIS_HF_TOKEN_FILE PLADIS_MAGICK_HOME PLADIS_RLINF_PATH \
  MODEL_ROOT_GR00T_N17 MODEL_ROOT_PI0 MODEL_ROOT_PI05

# Sweeps must run from committed code so every eplog is attributable to a
# commit (the run's `git describe` lands in the .arm sidecar). Call this at
# the top of every sweep driver; PLADIS_ALLOW_DIRTY=1 overrides deliberately.
pladis_require_clean_tree() {
  if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] \
      && [ "${PLADIS_ALLOW_DIRTY:-0}" != "1" ]; then
    echo "[sweep] ABORT: repo working tree is DIRTY — sweep results would not be" >&2
    echo "[sweep] attributable to a commit. Commit first, or PLADIS_ALLOW_DIRTY=1." >&2
    exit 1
  fi
}
