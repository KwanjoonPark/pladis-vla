#!/bin/bash
# GR00T N1.7 x LIBERO-plus CAMERA axis — TEMPERATURE-SOFTMAX row, run
# CONCURRENTLY with sweep_n17_camera.sh on the same machine.
#
# Arms: all-x-text with the sparse branch swapped for softmax(beta*l), beta=2
# (the ent15-matched strength calibration used on language and robot), at
# lambda {1.0, 1.5, 2.0} — the matched-dose counterparts of that driver's
# entmax arms allxtext / allxtext15 / allxtext20. Together the two drivers ask
# the zeros-vs-sharpening question on this axis: does the camera-axis response
# need entmax's exact zeros, or is generic sharpening enough?
#
# WHY A SEPARATE FILE, not three more `run` lines in sweep_n17_camera.sh:
# bash reads a script incrementally by byte offset, so editing one that is
# mid-execution makes the running shell resume at a stale offset and execute
# garbage. This row was requested while the entmax driver was already live
# (2026-08-08), so appending was not an option. The two arm lists are DISJOINT,
# so no eplog, `.out` log or video directory can collide. Once both drivers
# have finished, fold these three `run` lines into sweep_n17_camera.sh and
# delete this file — the axis should go back to having one driver as its arm
# vocabulary's source of truth (CLAUDE.md, "Adding an arm").
#
# CONCURRENCY is safe for the science, and was measured before launching
# (2026-08-08, one A5000 + 16 cores, entmax driver live): GPU 6.9/24.5 GB and
# ~23% util, 1.4 of 16 cores busy, so both processes fit with headroom. Arms
# stay paired regardless of interleaving because every source of randomness is
# pinned per episode, not per process: the schedule permutation (seed), the env
# reseed (seed*1_000_003 + episode) and the flow-matching noise pin
# (episode_seed*100_003 + step). The repo's "one campaign, one machine, one
# stack" invariant is about splitting a campaign ACROSS machines; this is one
# machine, one venv, one commit.
# WATCH RAM, not the GPU: each eval process holds ~10.3 GB RSS of 31 GB total,
# so two fit and a third would not (this box has an OOM-kill history).
#
# 3 arms x 1,599 episodes. Resume-safe at episode granularity.
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
    echo "[camera-temp] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis camera --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_camera_${tag}_${S}" "$@" \
      > "results/sweep/n17_camera_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_camera_${tag}_${S}.out"
  done
}

run allxt-temp20    --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20l15 --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20l20 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text

echo "[camera-temp] ALL DONE $(date +%H:%M:%S)"
