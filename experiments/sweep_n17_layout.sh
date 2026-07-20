#!/bin/bash
# GR00T N1.7 x LIBERO-plus LAYOUT axis, full curated set: 1,525 episodes per
# arm (10/goal/object/spatial = 312/425/403/385), each variant exactly once,
# seed-0 schedule, paired across arms. Scene-altering axis: episodes come from
# BDDL placement sampling under per-episode np.random reseeding, NOT base-task
# init states (harness/env.py SCENE_ALTERING_AXES; gates:
# experiments/verify_layout_axis.py, all passed 2026-07-16).
# Arms: vanilla, base0 (hook @ l=0), {state,action}x{text,image} + allxall @ l=1.
# Resume-safe at episode granularity. Runs AFTER sweep_n17_original.sh.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
MODEL_ROOT=/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO
SUITES="libero_10 libero_goal libero_object libero_spatial"
ORIG_PID="${1:-1260591}"  # sweep_n17_original.sh driver to wait on

# block while the original sweep is alive (match script name, not just pid)
while ps -p "$ORIG_PID" -o args= 2>/dev/null | grep -q sweep_n17_original; do
  sleep 300
done
# only proceed on clean completion; a crashed original sweep needs resuming first
if ! grep -q "\[orig\] ALL DONE" results/sweep/driver_orig.out 2>/dev/null; then
  echo "[layout] ABORT: original sweep exited without ALL DONE - resume it first" >&2
  exit 1
fi

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/n17_layout_${tag}_${S}_eplog.tsv"
    echo "[layout] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis layout --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_layout_${tag}_${S}" "$@" \
      > "results/sweep/n17_layout_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_layout_${tag}_${S}.out"
  done
}

run vanilla
run base0        --pladis-install --pladis-scale 0
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

echo "[layout] ALL DONE $(date +%H:%M:%S)"
