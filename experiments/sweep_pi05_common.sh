# Shared driver body for the pi0.5 sweeps — source this, do not execute.
#
# Structure is SUITE-OUTER / ARM-INNER, the inverse of the n17 drivers. That is what
# makes the 4-GPU layout legitimate under docs/SETUP.md §0: the analysis pairs episodes
# by (suite, episode) (analysis/analyze.py), so a McNemar pair never crosses suites, and
# pinning one suite to one device keeps every pair on one numeric stack. Pinning ARMS to
# devices instead would put a numeric-path difference INSIDE a pair — that is the thing
# §0 forbids. The suite->GPU map must therefore stay fixed across every arm of a
# campaign, and each run records its device in the .arm sidecar.
#
#   CUDA_VISIBLE_DEVICES=4 bash experiments/sweep_pi05_language.sh libero_10
#   CUDA_VISIBLE_DEVICES=5 bash experiments/sweep_pi05_language.sh libero_goal
#   ...
#
# Resume-safe at episode granularity (eval_arm.py skips episodes already logged), so
# re-running a driver is the resume mechanism and completed arms cost seconds.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
. experiments/load_machine_env.sh

SUITE="${1:?usage: $0 <libero_10|libero_goal|libero_object|libero_spatial>}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES — GPUs 0-3 belong to another project}"

# openpi's own LIBERO protocol: per-suite step caps (standalone libero_eval.py:88-99,
# "longest training demo has N steps"), 10-step decoded chunk with the first 5 executed.
# NOT the GR00T 720/8-of-16 protocol — the 92.4 libero_10 anchor was produced under this
# one, and reproducing a published number is the whole point of the anchor gate.
case "$SUITE" in
  libero_spatial) MAX_STEPS=220 ;;
  libero_object)  MAX_STEPS=280 ;;
  libero_goal)    MAX_STEPS=300 ;;
  libero_10)      MAX_STEPS=520 ;;
  *) echo "[sweep] unknown suite '$SUITE'" >&2; exit 2 ;;
esac
EXEC_HORIZON=5

MODEL_ROOT="$MODEL_ROOT_PI05"   # one checkpoint for all four suites (unlike n17)
if [ ! -f "$MODEL_ROOT/model.safetensors" ]; then
  echo "[sweep] ABORT: no checkpoint at $MODEL_ROOT (docs/SETUP.md §4)" >&2
  exit 1
fi
pladis_require_clean_tree
mkdir -p results/sweep

echo "[sweep] $PREFIX / $SUITE on GPU $CUDA_VISIBLE_DEVICES"
echo "[sweep] protocol: max_steps=$MAX_STEPS exec_horizon=$EXEC_HORIZON chunk=10"

run() {  # $1 = arm tag, rest = pladis flags
  local tag="$1"; shift
  local out="results/sweep/${PREFIX}_${tag}_${SUITE}_eplog.tsv"
  echo "[sweep] === $tag / $SUITE ($(date +%H:%M:%S)) ==="
  bash experiments/run.sh --venv openpi experiments/eval_arm.py \
    --model pi05 --suite "$SUITE" --axis "$AXIS" --episodes "$EPISODES" --seed 0 \
    --max-steps "$MAX_STEPS" --exec-horizon "$EXEC_HORIZON" \
    --model-path "$MODEL_ROOT" --out "$out" "$@" \
    > "results/sweep/${PREFIX}_${tag}_${SUITE}.out" 2>&1
  local rc=$?
  tail -1 "results/sweep/${PREFIX}_${tag}_${SUITE}.out"
  # device provenance, appended as a sidecar line (NOT line 1 — eplog.py:73-80 only
  # identity-checks line 1, so this cannot break resume after a reboot or a GPU swap)
  [ -f "$out.arm" ] && echo "dev ${CUDA_VISIBLE_DEVICES} $(nvidia-smi --id=0 --query-gpu=name,uuid --format=csv,noheader 2>/dev/null)" >> "$out.arm"
  return $rc
}
