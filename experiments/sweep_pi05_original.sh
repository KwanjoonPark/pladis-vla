#!/bin/bash
# π0.5 x LIBERO original (unperturbed) — the ANCHOR gate (README §5 gate 1) and the
# severity baseline analysis/analyze.py needs.
#
#   CUDA_VISIBLE_DEVICES=4 bash experiments/sweep_pi05_original.sh libero_10
#
# axis=none enumerates the suite's 10 base tasks with their ORIGINAL instructions;
# 100 eps/suite = 10 visits/task over init states 0-9 (seed-0 schedule, paired with every
# other arm of the same suite).
#
# ACCEPTANCE: this must reproduce openpi's published pi05_libero table
# (openpi/examples/libero/README.md, "π0.5 @ 30k (finetuned)"):
#     libero_spatial 98.8 | libero_object 98.2 | libero_goal 98.0 | libero_10 92.4
# At n=100 the libero_10 binomial SE is 2.7pp, so the accept band is ~[87, 98].
# A miss means the checkpoint, the venv, or the harness is wrong — NOT that the numbers
# moved. Do not start a sweep until this passes.
PREFIX=pi05_orig
AXIS=none
EPISODES=100
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sweep_pi05_common.sh"

run vanilla

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
