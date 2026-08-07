#!/bin/bash
# GR00T N1.7 x LIBERO-plus CAMERA axis (Camera Viewpoints), FULL curated set:
# four suites, 1,599 episodes per arm (10/goal/object/spatial =
# 419/408/396/376), seed-0 paired schedule. Scene-side runtime perturbation of
# the AGENTVIEW camera pose only, from the curated name's
# `_view_<h>_<v>_<scale>_<rot>_<vert>` tail: orbit 443 (yaw +-75 deg) /
# orbit_up 549 (the same yaw after a 15 deg elevation) / zoom 313 (pivot-ray
# push-out 115-200%) / reaim 294 (bearing-only, +-10 deg). The pose is a
# closed-form function of the tail -> no RNG on this axis (NOT in
# RUNTIME_RNG_AXES), and it lands in sim.model.cam_pos/cam_quat, outside
# sim.get_state(), so set_init_state cannot revert it. Wrist camera, sim
# state, fixture body_pos and the instruction are all bit-identical to the
# paired base episode (verify_camera_axis.py, ALL GATES PASSED 2026-08-07).
# Arms (operator grid 2026-08-07): vanilla + the TEXT-LOCUS DOSE LADDER at
# both query groups — a-x-t and all-x-t at lambda {1, 1.5, 2.0}. This axis
# skips the standard lambda=1 modality grid on purpose: on language and robot
# the lambda=1 cells were flat and every signal that appeared came from the
# text-locus dose row, so the seven episodes-budget goes into dose depth at
# the two text loci rather than breadth across cells that already read null.
# base0 is omitted (bit-identical to vanilla since the 07-20 lambda=0 SDPA
# delegation, verify_base0_parity.py); with no image arm, the surviving locus
# contrast is the QUERY-GROUP one (all-x-t vs a-x-t at matched dose), which is
# what analysis/analyze.py uses as this axis's locus pair.
# Cost: ~15-20 s/ep projected from the robot axis's success/failure wall-time
# split, i.e. ~7-9 h per arm and ~46-62 h for the seven arms on one A5000.
# Resume-safe at episode granularity.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
. experiments/load_machine_env.sh
MODEL_ROOT="$MODEL_ROOT_GR00T_N17"
pladis_require_clean_tree
SUITES="libero_10 libero_goal libero_object libero_spatial"

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/n17_camera_${tag}_${S}_eplog.tsv"
    echo "[camera] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis camera --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_camera_${tag}_${S}" "$@" \
      > "results/sweep/n17_camera_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_camera_${tag}_${S}.out"
  done
}

run vanilla

# action-row text locus, lambda 1.0 -> 1.5 -> 2.0
run actionxtext   --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run actionxtext15 --pladis-install --pladis-scale 1.5 --pladis-qgroup action --pladis-kind text
run actionxtext20 --pladis-install --pladis-scale 2.0 --pladis-qgroup action --pladis-kind text

# all-row (state+action) text locus, same ladder — pairs with the row above at
# matched dose, so the difference isolates the query group
run allxtext      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind text
run allxtext15    --pladis-install --pladis-scale 1.5 --pladis-qgroup all    --pladis-kind text
run allxtext20    --pladis-install --pladis-scale 2.0 --pladis-qgroup all    --pladis-kind text

echo "[camera] ALL DONE $(date +%H:%M:%S)"
