#!/bin/bash
# GR00T N1.7 x LIBERO-plus ROBOT-init axis, full curated set: 1,550 episodes
# per arm (10/goal/object/spatial = 393/409/398/350), each variant exactly
# once (init_state_id 0 = official num_trials=1 protocol), seed-0 schedule,
# paired across arms. Runtime axis: `_initstate_<k>` swaps in a Panda{k}
# robot class with perturbed init_qpos (levels 0.1-0.5); delivery = partial
# pose offset + persistent OSC nullspace bias (gates:
# experiments/verify_robot_axis.py, all passed 2026-07-20).
# Arms: vanilla + {state,action}x{text,image} + allxall @ l=1. NO base0 arm:
# since the 2026-07-20 lambda=0 SDPA delegation base0 is BIT-identical to
# vanilla (verify_base0_parity.py) and would duplicate it episode-for-episode.
# Resume-safe at episode granularity.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
MODEL_ROOT=/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO
SUITES="libero_10 libero_goal libero_object libero_spatial"

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/n17_robot_${tag}_${S}_eplog.tsv"
    echo "[robot] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis robot --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_robot_${tag}_${S}" "$@" \
      > "results/sweep/n17_robot_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_robot_${tag}_${S}.out"
  done
}

run vanilla
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

echo "[robot] ALL DONE $(date +%H:%M:%S)"
