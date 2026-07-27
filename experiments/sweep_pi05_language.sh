#!/bin/bash
# π0.5 x LIBERO-plus LANGUAGE axis, full curated set: 1,537 episodes per arm across the
# four suites (10/goal/object/spatial = 383/410/354/390), each curated variant exactly
# once, seed-0 schedule, paired across arms.
#
#   CUDA_VISIBLE_DEVICES=4 bash experiments/sweep_pi05_language.sh libero_10
#   CUDA_VISIBLE_DEVICES=5 bash experiments/sweep_pi05_language.sh libero_goal
#   CUDA_VISIBLE_DEVICES=6 bash experiments/sweep_pi05_language.sh libero_object
#   CUDA_VISIBLE_DEVICES=7 bash experiments/sweep_pi05_language.sh libero_spatial
#
# ---------------------------------------------------------------------------------
# PHASE 1 ARMS (3). The design axis for π0.5 is the KEY SUB-BLOCK, not a query group:
# its suffix is action-only (pi05_libero sets discrete_state_input=False, and
# pi0_pytorch.py:243-261 only builds a state token when NOT pi05), so every arm is
# implicitly "action-row x <keys>". Key axis at a suffix step, verified on the real
# checkpoint by verify_pi05_delivery.py to be (q,k)=(10,978):
#
#     [ image 0:768 | language 768:968 | suffix 968:978 ]
#
#   text   -> columns 768:968. The DIRECT PORT of the official FLUX intervention:
#             PLADIS/pipeline/pipeline_flux.py:104-113 sharpens exactly one
#             (generative-query x conditioning-key) block, mass-preserving. Motivated
#             by LIBERO-plus Finding 3/7/8 — VLAs largely ignore language.
#   image  -> columns 0:768. No upstream precedent (in FLUX the image tokens are the
#             QUERIES, never the keys); this is the contrast that makes "locus matters"
#             testable, and the reason this repo exists.
#
# NOT run here, deliberately:
#   base0 (λ=0)  bit-identical to vanilla (verify_pi05_hook.py gate A), so it would burn
#                1,537 episodes re-proving a 10-episode assertion. Covered by
#                verify_pi05_parity.py — the same call the robot axis made for n17
#                (sweep_n17_robot.sh:9-12).
#   eager-dense  method=softmax,β=1 makes `m*p == dense[sub]` an identity, so the blend
#                collapses to dense for ANY λ. In GR00T this arm absorbed a real
#                fused-SDPA-vs-eager KERNEL difference; π0.5 is on the eager path
#                already (pi0_pytorch.py:447), so vanilla IS the numeric control.
#                Quantified instead by verify_pi05_parity.py check (c).
#   prefix/all   phase 2, gated on the support-size diagnostic
#                (experiments/diag_pi05_support.py) — see that file for why.
# ---------------------------------------------------------------------------------
PREFIX=pi05_lang
AXIS=language
EPISODES=0        # 0 = every curated variant exactly once
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sweep_pi05_common.sh"

# Videos are recorded (as the n17 sweeps do): for a LOCUS study the qualitative question
# "HOW does text-sparsification fail" is worth the disk, and harness/video.py is verified
# not to perturb the RNG path. ~3% wall-clock, a few GB per arm.
run vanilla --video-dir "results/sweep/videos/${PREFIX}_vanilla_${SUITE}"
run text  --video-dir "results/sweep/videos/${PREFIX}_text_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-kind text
run image --video-dir "results/sweep/videos/${PREFIX}_image_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-kind image

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
