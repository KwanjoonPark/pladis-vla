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
# PHASE 1 ARMS (4). The design axis for π0.5 is the KEY SUB-BLOCK, not a query group:
# its suffix is action-only, so every arm is implicitly "action-row x <keys>".
#
# WHY there is no state row (2026-07-28, checked against the π0.5 paper §IV-A/App. E and
# the checkpoint's own config): the paper's π0.5 discretizes proprioceptive state into
# TEXT tokens and puts it in the prefix, and openpi implements exactly that as the
# DEFAULT (pi0_config.py:38-39 sets discrete_state_input = pi05). `pi05_libero` is the
# one config that overrides it to False (config.py:736, authored by the paper's first
# author), and embed_suffix (pi0_pytorch.py:237-261) only builds a state token when NOT
# pi05. So on THIS checkpoint the state reaches the model through neither path — it is
# simply not an input. Two consequences: the qgroup axis genuinely collapses, and the
# language key block is PURE INSTRUCTION (no state digits mixed in), which is what makes
# "locus = language" a clean claim rather than "language + proprioception".
#
# Key axis at a suffix step, verified on the real checkpoint by
# verify_pi05_delivery.py to be (q,k)=(10,978):
#
#     [ image 0:768 | language 768:968 | suffix 968:978 ]
#
# Those widths are the ALLOCATED widths; the ATTENDABLE widths are smaller and that is
# what the dose should be read against (2026-07-28):
#   * language 200 -> ~9-20 real tokens. PaligemmaTokenizer pads to max_token_len=200
#     with mask=False and embed_prefix (pi0_pytorch.py) feeds that mask into pad_masks,
#     so ~185 of the 200 columns are masked out. Measured on real LIBERO instructions.
#   * image 768 -> 512. LIBERO has two cameras, so libero_policy.py:62-70 fills
#     `right_wrist_0_rgb` with zeros and sets its image_mask False (True only for
#     PI0_FAST), masking those 256 columns.
# The slicing is unaffected (masked columns carry dense~0, so the mass-preserving
# `m = dense[..., lo:hi].sum()` already integrates only the attendable mass) — but
# experiments/diag_pi05_support.py reports support as a fraction of the ALLOCATED width,
# so its "0.5% of 200" is ~7% of the attendable ~15, and "1.0% of 768" is 1.6% of 512.
#
#   text   -> columns 768:968. The DIRECT PORT of the official FLUX intervention:
#             PLADIS/pipeline/pipeline_flux.py:104-113 sharpens exactly one
#             (generative-query x conditioning-key) block, mass-preserving. Motivated
#             by LIBERO-plus Finding 3/7/8 — VLAs largely ignore language.
#   image  -> columns 0:768. No upstream precedent (in FLUX the image tokens are the
#             QUERIES, never the keys); this is the contrast that makes "locus matters"
#             testable, and the reason this repo exists.
#
#   base_dense -> the EAGER-DENSE CONTROL (method=softmax, β=1). Reinstated 2026-07-28 on
#             the measurement of verify_pi05_parity.py check (c), which was written to
#             retire this arm and instead argued for keeping it. The algebra says
#             `m*p == dense[sub]` is an identity, and at MODULE level it holds: check (c)
#             finds 0.0000% of bf16 attention elements differing, max|dw| = 0. But
#             END-TO-END over 10 libero_10 episodes, dense diverged from vanilla in every
#             single one (n_steps 197->186, 246->241, 270->259, ..., SR 0.900 -> 1.000).
#             Float32 reassociation inside the λ>0 branch is below bf16 resolution per
#             call, and 18 layers x 10 denoise steps x ~250 chunks of closed-loop rollout
#             amplifies it into a different trajectory. So π0.5 does have a numeric term
#             to control for — not GR00T's fused-vs-eager KERNEL difference, but the
#             reassociation the λ>0 code path itself introduces. Without this arm,
#             `text - vanilla` confounds locus with that term; `text - base_dense` does
#             not. λ=0 is exempt: it returns dense unmodified and IS bit-identical (check
#             (b): eplogs equal over 10 episodes).
#
# NOT run here, deliberately:
#   base0 (λ=0)  bit-identical to vanilla (verify_pi05_hook.py gate A, re-confirmed
#                end-to-end by verify_pi05_parity.py check (b)), so it would burn 1,537
#                episodes re-proving a 10-episode assertion. Same call the robot axis
#                made for n17 (sweep_n17_robot.sh:9-12). NOTE this is exactly why base0
#                does NOT substitute for base_dense: base0 never enters the λ>0 branch.
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
# kind=text is arbitrary for base_dense — with method=softmax,β=1 the blend is the dense
# identity on every block, so the arm is the λ>0 NUMERIC PATH with no sparsification. It
# must carry --pladis-scale 1.0 (not 0) or it would silently become base0 and stop
# controlling for anything.
run base_dense --video-dir "results/sweep/videos/${PREFIX}_base_dense_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 1.0 \
          --pladis-kind text
run text  --video-dir "results/sweep/videos/${PREFIX}_text_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-kind text
run image --video-dir "results/sweep/videos/${PREFIX}_image_${SUITE}" \
          --pladis-install --pladis-scale 1.0 --pladis-kind image

# ---------------------------------------------------------------------------------
# PHASE 2, 2026-07-29 — DOSE LADDER λ = 1.0 -> 1.5 -> 2.0 over the locus pair.
#
# Motivated by the λ=1 result: the locus contrast `text - image` = +1.63pp
# (z=+2.65, p_bonf=0.048) was the only contrast of six to survive Bonferroni, but the
# effect is small and `text - vanilla` was +0.33pp, i.e. the intervention barely beats
# doing nothing. A dose ladder tests whether the locus effect is monotone in λ or
# whether λ=1 already sits at the useful maximum.
#
# λ>1 IS EXTRAPOLATION, not interpolation. PLADIS is parameterized
# `λ*sparse + (1-λ)*dense` (pipeline_flux.py:109), so at λ>1 the dense coefficient is
# negative and every column entmax dropped comes out at `(1-λ)*dense < 0` — a NEGATIVE
# attention weight (λ=1.5: -0.5*dense; λ=2.0: -1.0*dense). The row still sums to the
# block's original mass, because the survivors are pushed above dense by exactly the
# deficit. λ=1.5 is the regime the PLADIS authors recommend and the n17 track already
# runs it, but π0.5 has no precedent here, so diag_pi05_support.py measured the negative
# lobe first (2026-07-29, 3 eps/cell). Result — the extrapolation is well behaved, with
# ONE asymmetry that matters for interpretation:
#
#   λ    kind   block cols negative   min w (med)   negative mass   row sum
#   1.0  both                  0.0%       +0.0000          0.0000   1.000000
#   1.5  text                  6.5%       -0.0046          0.0206   1.000000
#   1.5  image                65.9%       -0.0021          0.0710   1.000000
#   2.0  text                  6.5%       -0.0092          0.0415   1.000000
#   2.0  image                66.0%       -0.0043          0.1460   1.000000
#
#   * Normalization holds exactly: row sums are 1.000000 with max deviation 2.7e-06.
#   * Magnitudes stay small (>= -0.05 even at λ=2) and negative mass is LINEAR in λ-1,
#     which is the closed-form prediction — a cross-check that the implementation
#     follows PLADIS's parameterization.
#   * text vs image differ 10x in negative EXPOSURE, and it is the padding structure:
#     image has 512 attendable columns of which ~8 survive, so most attendable columns
#     go negative; text has ~15 attendable of 200, and the ~185 masked columns carry
#     dense~0 so their negative image is ~0 (6.5% x 200 = 13 cols = the attendable
#     width). CONSEQUENCE: at λ>1 the `text - image` contrast carries an extra term
#     that λ=1 did not have — image absorbs a much larger negative perturbation. The
#     per-λ base_dense does NOT absorb it (base_dense never sparsifies, so it has no
#     negative lobe at all). Interpret the λ>1 locus contrasts with that in mind.
#
# base_dense IS REPEATED PER λ, and that is not redundancy. Its blend is
# `w = dense + λ*(m*p - dense[sub])`; the bracket is algebraically zero but numerically
# ε, so the residual scales as λ*ε — the very term this arm exists to absorb GROWS with
# λ. At λ=1 it was already worth -0.91pp (base_dense - vanilla). Contrasting a λ=2 arm
# against the λ=1 control would leave half of it uncontrolled.
# ---------------------------------------------------------------------------------
run base_dense15 --pladis-install --pladis-scale 1.5 --pladis-method softmax \
          --pladis-beta 1.0 --pladis-kind text
run text15  --pladis-install --pladis-scale 1.5 --pladis-kind text
run image15 --pladis-install --pladis-scale 1.5 --pladis-kind image

run base_dense20 --pladis-install --pladis-scale 2.0 --pladis-method softmax \
          --pladis-beta 1.0 --pladis-kind text
run text20  --pladis-install --pladis-scale 2.0 --pladis-kind text
run image20 --pladis-install --pladis-scale 2.0 --pladis-kind image

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
