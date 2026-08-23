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
# Cost: MEASURED 2026-08-23 over the 12 completed arms (1,599 eps each):
# 13.1-14.8 s/ep = 5.8-6.6 h per arm, mean 6.1 h on one A5000. The pre-run
# projection was ~15-20 s/ep from the robot axis's success/failure wall-time
# split, so the axis came in at the fast end of it. This file carries TEN arms
# (~61 h); the three sharp-softmax arms in sweep_n17_camera_temp.sh add ~18 h.
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


# 2026-08-18 denoising-step schedule row (operator request; the language axis's
# 08-16/17 finding carried onto the other perturbation axes). On language the
# benefit lives in the LATE denoising steps: at lambda=2 on all-x-text, late
# [0,0,1,1] scored +2.86pp vs vanilla (z=3.64 — the axis's only Bonferroni-
# surviving arm-vs-vanilla result) and inc [0,0.5,1,1.5] +2.80 (z=3.37), while
# early [1,1,0,0] was -1.30 and late-minus-early +4.16 (z=5.00, Bonf*): the SAME
# total dose helps or hurts depending on WHERE in the flow's time axis it is spent.
# On THIS axis the flat parent allxtext20 read -1.69pp vs vanilla (z=-1.71, n.s.)
# with the whole ladder drifting mildly negative, so the question is whether that
# is a null or a CANCELLATION: early-step harm plus late-step benefit summing to
# zero. Dropping the early half is the direct test.
# Two arms, not the four-shape row, because this axis already carries the iso-dose
# flat controls the other two shapes would have supplied. Writing the
# time-integrated dose as sum_i lambda_i over the N=4 Euler steps (lambda_i =
# scale * w_i, so both arms peak at lambda=3 on their last step):
#     late [0,0,2,2] = 4 = flat lambda=1   (allxtext,   collected)
#     inc  [0,1,2,3] = 6 = flat lambda=1.5 (allxtext15, collected)
# so each new arm is read three ways: vs vanilla, vs its flat parent allxtext20
# ([2,2,2,2], sum 8 — same peak lambda, twice the total dose), and vs the
# collected flat arm at the SAME total dose — the contrast that separates WHEN
# the intervention acts from HOW MUCH of it there is. The [1,1,1,1] row needs no
# arm: it is bit-identical to allxtext20 (verify_step_schedule.py gate F).
run allxt-late-l2 --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1
run allxt-inc-l2  --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0.5,1,1.5

# 2026-08-21 sharp-softmax mirror of the LATE schedule arm (operator request).
# The 08-16/17 language rows put the benefit in the LATE denoising steps on BOTH
# sparse branches: entmax late [0,0,1,1] at lambda=2 all-x-text +2.86pp vs vanilla
# (z=3.64, Bonf*) and its softmax(2*l) twin +1.69 (z=2.17), the two statistically
# indistinguishable from each other (temp-late - late -1.17pp, z=-1.59) — so ON
# LANGUAGE the time structure belongs to the sharpening, not to entmax's exact
# zeros. The 08-18 row then carried the ENTMAX late arm to every other axis.
# On THIS axis the row moved, and it moved DOWN: late -2.81pp vs vanilla (z=-2.82,
# nominal) and inc -3.94 (z=-3.83, Bonf* HARMFUL), while the flat softmax parent
# allxt-temp20l20 was -1.31 (n.s.). Camera is therefore the one place off language
# where the swap is asked of a SIGNAL rather than of a null — is the late-step HARM
# entmax-specific, or does sharpening late hurt on this axis either way?
# This arm is the branch swap of that one here: same all-x-text locus, same
# lambda=2 base, same [0,0,1,1] weights, sparse branch entmax-1.5 -> softmax(2*l)
# at beta=2 (the ent15-strength-matched setting of supp G.1; beta=1 would collapse
# the sparse branch onto the dense one and void every "did blend" assertion).
# Effective lambda per step = 2 * w = [0,0,2,2].
# Read four ways: vs vanilla; vs its flat parent allxt-temp20l20 ([2,2,2,2], same
# peak lambda, twice the total dose, collected); vs the iso-dose flat arm
# allxt-temp20 (sum_i lambda_i = 4 for both, collected) — WHEN vs HOW MUCH; and vs
# allxt-late-l2, the SAME shape on the entmax branch — the zeros-not-special
# question on the TIME axis, asked off the language axis for the first time.
# One arm, not the four-shape row: the flat [1,1,1,1] mirror is bit-identical to
# allxt-temp20l20 (verify_step_schedule.py gate F) and the iso-dose control is
# already collected, so only the shape itself is missing.
run allxt-temp20-late-l2 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1

echo "[camera] ALL DONE $(date +%H:%M:%S)"
