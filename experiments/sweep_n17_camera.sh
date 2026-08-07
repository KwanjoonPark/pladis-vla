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
# Arms: the standard lambda=1 locus grid, same as the noise axis (base0
# omitted — bit-identical to vanilla since the 07-20 lambda=0 SDPA
# delegation, verify_base0_parity.py).
# Cost: ~15-20 s/ep projected from the robot axis's success/failure wall-time
# split, i.e. ~7-9 h per arm and ~40-53 h for the six arms on one A5000.
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
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

echo "[camera] ALL DONE $(date +%H:%M:%S)"
