#!/bin/bash
# π0.5 x LIBERO-plus ROBOT-init axis, full curated set: 1,550 episodes per arm across
# the four suites (10/goal/object/spatial = 393/409/398/350), each curated variant
# exactly once (init_state_id 0 = official num_trials=1 protocol), seed-0 schedule,
# paired across arms.
#
#   CUDA_VISIBLE_DEVICES=0 bash experiments/sweep_pi05_robot.sh libero_10
#   CUDA_VISIBLE_DEVICES=1 bash experiments/sweep_pi05_robot.sh libero_goal
#   CUDA_VISIBLE_DEVICES=2 bash experiments/sweep_pi05_robot.sh libero_object
#   CUDA_VISIBLE_DEVICES=3 bash experiments/sweep_pi05_robot.sh libero_spatial
#
# RUNTIME AXIS: `_initstate_<k>` swaps in a Panda{k} robot class with perturbed
# init_qpos (levels ||d|| = 0.1..0.5 by k-century). Delivery under the official
# protocol is INDIRECT — partial pose offset after settle + persistent OSC nullspace
# bias toward the perturbed config (harness/env.py:45-56; gates:
# experiments/verify_robot_axis.py). Scene and instruction stay paired with the base
# task.
#
# WHY THIS AXIS IS DIFFERENT ON π0.5 (and worth running): pi05_libero is STATE-BLIND —
# proprioception enters through neither the prompt (config.py:736
# discrete_state_input=False) nor the suffix (pi0_pytorch.py:237-261). GR00T reads an
# 8-D state vector, so a displaced arm reaches it directly; π0.5 can perceive the
# displacement ONLY through the cameras (a shifted wrist view, a displaced arm in the
# agent view). The perturbation is therefore delivered EXCLUSIVELY through the image
# modality, which makes the locus prior the sharpest of the three axes:
#
#   language axis: instruction perturbed  -> text has the story   (measured: locus * )
#   layout axis:   scene perturbed        -> image has the story  (measured: locus * ,
#                                            sign flipped as predicted)
#   robot axis:    pose perturbed, visible-only -> image is the ONLY channel the
#                  perturbation can use; text is a pure control
#
# If the sign-flip pattern is real, `text - image` here should look like layout
# (negative), and `text` should be inert-to-harmful while `image` decides the outcome.
#
# ---------------------------------------------------------------------------------
# ARMS (3), λ=1, same shape as the layout axis and for the same reasons: λ=1 is the
# only rung where both arms carry zero negative weight (diag_pi05_support.py), and the
# language ladder showed λ>1 gains are confounded with the image block's negative lobe.
# base0/base_dense: not run (verify_pi05_parity.py check (b); operator decision
# 2026-07-29 — vanilla is the reporting reference, base_dense's λ=1 size is known).
# ---------------------------------------------------------------------------------
PREFIX=pi05_robot
AXIS=robot
EPISODES=0        # 0 = every curated variant exactly once
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sweep_pi05_common.sh"

# Triton entmax backend — support-identical to entmax 1.3 (verify_pi05_hook.py gate G),
# ~5x faster; new axis, shares no eplog with the entmax-backend language campaign.
BACKEND="--pladis-sparse-backend adasplash"

run vanilla --video-dir "results/sweep/videos/${PREFIX}_vanilla_${SUITE}"
run text  --video-dir "results/sweep/videos/${PREFIX}_text_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-kind text $BACKEND
run image --video-dir "results/sweep/videos/${PREFIX}_image_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-kind image $BACKEND

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
