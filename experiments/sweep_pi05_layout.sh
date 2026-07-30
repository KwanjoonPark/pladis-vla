#!/bin/bash
# π0.5 x LIBERO-plus LAYOUT axis, full curated set: 1,525 episodes per arm across the
# four suites (10/goal/object/spatial = 312/425/403/385), each curated variant exactly
# once, seed-0 schedule, paired across arms.
#
#   CUDA_VISIBLE_DEVICES=0 bash experiments/sweep_pi05_layout.sh libero_10
#   CUDA_VISIBLE_DEVICES=1 bash experiments/sweep_pi05_layout.sh libero_goal
#   CUDA_VISIBLE_DEVICES=2 bash experiments/sweep_pi05_layout.sh libero_object
#   CUDA_VISIBLE_DEVICES=3 bash experiments/sweep_pi05_layout.sh libero_spatial
#
# LAYOUT IS SCENE-ALTERING (harness/env.py SCENE_ALTERING_AXES). The perturbation lives
# in the BDDL's own placement regions, so episodes come from BDDL placement sampling
# under per-episode np.random reseeding, NOT from the base task's init states — applying
# those would silently restore the original layout. That path is model-independent and is
# gated by experiments/verify_layout_axis.py; it needs no π0.5-specific work.
#
# ---------------------------------------------------------------------------------
# ARMS (3), λ=1. Deliberately the language axis's PHASE 1 shape, not its λ ladder.
#
# The language-axis ladder (2026-07-30) found the locus contrast largest at λ=1.5
# (+3.15pp) rather than λ=2.0, but also that the `text` arm barely moves with λ at all
# (80.20 / 80.33 / 80.33) — what changes is `image` getting worse (78.81 / 77.18 / 77.68).
# Since λ>1 is extrapolation and puts 66% of the image block below zero against text's
# 6.5% (diag_pi05_support.py), a widening λ>1 contrast is partly the negative lobe rather
# than locus. λ=1 is the regime where both arms carry ZERO negative weight, so it is the
# clean place to ask the locus question on a new axis. Extend to a ladder only if λ=1
# shows a locus effect here.
#
#   text   -> language keys [768:968]. Direct port of the FLUX intervention.
#   image  -> image keys [0:768]. The contrast that makes "locus matters" testable.
#
# NOT run here: base0 (bit-identical to vanilla, verify_pi05_parity.py check (b));
# base_dense (vanilla is the reporting reference, operator decision 2026-07-29 — its λ=1
# size is already measured on the language axis at -0.91pp); prefix/all (phase 2).
#
# NOTE the layout axis asks a DIFFERENT question than language. Language perturbs the
# instruction, so sharpening language keys plausibly helps. Layout perturbs the SCENE
# with the instruction untouched, so the prior runs the other way: `image` is the arm
# with a story here, and `text` is closer to a control. That asymmetry is the point —
# if the locus effect flips sign with the perturbed modality, that is far stronger
# evidence for locus than either axis alone.
# ---------------------------------------------------------------------------------
PREFIX=pi05_layout
AXIS=layout
EPISODES=0        # 0 = every curated variant exactly once
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sweep_pi05_common.sh"

run vanilla --video-dir "results/sweep/videos/${PREFIX}_vanilla_${SUITE}"
run text  --video-dir "results/sweep/videos/${PREFIX}_text_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-kind text
run image --video-dir "results/sweep/videos/${PREFIX}_image_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-kind image

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
