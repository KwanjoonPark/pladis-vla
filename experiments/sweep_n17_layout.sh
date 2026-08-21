#!/bin/bash
# GR00T N1.7 x LIBERO-plus LAYOUT axis, full curated set: 1,525 episodes per
# arm (10/goal/object/spatial = 312/425/403/385), each variant exactly once,
# seed-0 schedule, paired across arms. Scene-altering axis: episodes come from
# BDDL placement sampling under per-episode np.random reseeding, NOT base-task
# init states (harness/env.py SCENE_ALTERING_AXES; gates:
# experiments/verify_layout_axis.py, all passed 2026-07-16).
# Arms: vanilla, base0 (hook @ l=0), {state,action}x{text,image} + allxall @ l=1.
# Resume-safe at episode granularity. Runs AFTER sweep_n17_original.sh.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
. experiments/load_machine_env.sh
MODEL_ROOT="$MODEL_ROOT_GR00T_N17"
pladis_require_clean_tree
SUITES="libero_10 libero_goal libero_object libero_spatial"
ORIG_PID="${1:-1260591}"  # sweep_n17_original.sh driver to wait on

# block while the original sweep is alive (match script name, not just pid)
while ps -p "$ORIG_PID" -o args= 2>/dev/null | grep -q sweep_n17_original; do
  sleep 300
done
# only proceed on clean completion; a crashed original sweep needs resuming first
if ! grep -q "\[orig\] ALL DONE" results/sweep/driver_orig.out 2>/dev/null; then
  echo "[layout] ABORT: original sweep exited without ALL DONE - resume it first" >&2
  exit 1
fi

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/n17_layout_${tag}_${S}_eplog.tsv"
    echo "[layout] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis layout --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_layout_${tag}_${S}" "$@" \
      > "results/sweep/n17_layout_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_layout_${tag}_${S}.out"
  done
}

run vanilla
run base0        --pladis-install --pladis-scale 0
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

# 2026-07-28 text-locus dose row (entmax), mirroring the language axis:
# a-x-t lambda {1.5, 2.0} (lambda=1 exists above) + all-x-t lambda {1, 1.5, 2}.
run actionxtext15 --pladis-install --pladis-scale 1.5 --pladis-qgroup action --pladis-kind text
run actionxtext20 --pladis-install --pladis-scale 2.0 --pladis-qgroup action --pladis-kind text
run allxtext      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind text
run allxtext15    --pladis-install --pladis-scale 1.5 --pladis-qgroup all    --pladis-kind text
run allxtext20    --pladis-install --pladis-scale 2.0 --pladis-qgroup all    --pladis-kind text

# 2026-08-03 s-x-i / composite dose row (operator request): stateximage lambda
# {1.5, 2.0} (lambda=1 exists above) + axt-sxi (actionxtext+stateximage cells,
# first appearance on this axis) lambda {1.5, 2.0} — layout was the NULL axis
# at lambda=1; tests whether the state-x-image locus or its composite wakes up
# under extrapolation.
run stateximage15 --pladis-install --pladis-scale 1.5 --pladis-qgroup state --pladis-kind image
run stateximage20 --pladis-install --pladis-scale 2.0 --pladis-qgroup state --pladis-kind image
run axt-sxi15     --pladis-install --pladis-scale 1.5 --pladis-cells actionxtext,stateximage
run axt-sxi20     --pladis-install --pladis-scale 2.0 --pladis-cells actionxtext,stateximage


# 2026-08-18 denoising-step schedule row (operator request; the language axis's
# 08-16/17 finding carried onto the other perturbation axes). On language the
# benefit lives in the LATE denoising steps: at lambda=2 on all-x-text, late
# [0,0,1,1] scored +2.86pp vs vanilla (z=3.64 — the axis's only Bonferroni-
# surviving arm-vs-vanilla result) and inc [0,0.5,1,1.5] +2.80 (z=3.37), while
# early [1,1,0,0] was -1.30 and late-minus-early +4.16 (z=5.00, Bonf*): the SAME
# total dose helps or hurts depending on WHERE in the flow's time axis it is spent.
# On THIS axis the flat parent allxtext20 read -1.70pp vs vanilla (z=-1.91, n.s.,
# the axis's most negative text-locus rung), so the question is whether that is a
# null or a CANCELLATION: early-step harm plus late-step benefit summing to zero.
# Dropping the early half is the direct test.
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
# On THIS axis it did not carry: late read -1.11pp vs vanilla (z=-1.26, n.s.)
# and its inc sibling -3.67pp (z=-3.88, Bonf* HARMFUL), against a flat parent
# allxtext20 that was itself -1.70 (n.s.).
# This arm is the branch swap of that one here: same all-x-text locus, same
# lambda=2 base, same [0,0,1,1] weights, sparse branch entmax-1.5 -> softmax(2*l)
# at beta=2 (the ent15-strength-matched setting of supp G.1; beta=1 would collapse
# the sparse branch onto the dense one and void every "did blend" assertion).
# Effective lambda per step = 2 * w = [0,0,2,2].
# Read TWO ways here, not four: this axis carries NO sharp-softmax arm at all
# (allxt-temp20{,l15,l20} were never run on layout), so only vs vanilla and the
# branch swap vs allxt-late-l2 (same shape, entmax) resolve. The flat-parent and
# iso-dose readings the other axes get would need allxt-temp20l20 and
# allxt-temp20 collected here first (~12 h more); analyze.py carries only the two
# contrasts that exist rather than pretending to the other two.
run allxt-temp20-late-l2 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1

echo "[layout] ALL DONE $(date +%H:%M:%S)"
