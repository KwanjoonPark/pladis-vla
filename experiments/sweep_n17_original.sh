#!/bin/bash
# GR00T N1.7 x LIBERO original (unperturbed) reference arm for the language
# sweep: axis=none enumerates each suite's 10 base tasks with their ORIGINAL
# instructions; 100 eps/suite = 10 visits/task over init states 0-9 (seed-0
# schedule, paired across arms). Gives per-task in-dist baselines (n=10) for
# robustness-drop reporting. Runs AFTER sweep_n17_language.sh releases the GPU.
# Requires the _moved-aware _VARIANT_MARKER (env.py) — without it, libero_goal
# axis=none enumerates 20 "bases" incl. 10 layout-perturbed *_moved scenes.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
MODEL_ROOT=/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO
SUITES="libero_10 libero_goal libero_object libero_spatial"
LANG_PID="${1:-2689008}"  # sweep_n17_language.sh driver to wait on

# block while the language sweep is alive (match script name, not just pid)
while ps -p "$LANG_PID" -o args= 2>/dev/null | grep -q sweep_n17_language; do
  sleep 300
done
# only proceed on clean completion; a crashed language sweep needs resuming first
if ! grep -q "\[sweep\] ALL DONE" results/sweep/driver.out 2>/dev/null; then
  echo "[orig] ABORT: language sweep exited without ALL DONE - resume it first" >&2
  exit 1
fi

wait_ckpt() {
  until [ -f "$MODEL_ROOT/$1/config.json" ] && ! ls "$MODEL_ROOT/$1"/*.incomplete >/dev/null 2>&1; do
    echo "[orig] waiting for checkpoint $1 ..."; sleep 60
  done
}

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/n17_orig_${tag}_${S}_eplog.tsv"
    wait_ckpt "$S"
    echo "[orig] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis none --episodes 100 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_orig_${tag}_${S}" "$@" \
      > "results/sweep/n17_orig_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_orig_${tag}_${S}.out"
  done
}

run vanilla
run base0        --pladis-install --pladis-scale 0
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

echo "[orig] ALL DONE $(date +%H:%M:%S)"
