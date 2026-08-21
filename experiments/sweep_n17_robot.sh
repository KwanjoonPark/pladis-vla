#!/bin/bash
# GR00T N1.7 x LIBERO-plus ROBOT-init axis, full curated set: 1,550 episodes
# per arm (10/goal/object/spatial = 393/409/398/350), each variant exactly
# once (init_state_id 0 = official num_trials=1 protocol), seed-0 schedule,
# paired across arms. Runtime axis: `_initstate_<k>` swaps in a Panda{k}
# robot class with perturbed init_qpos (levels 0.1-0.5); delivery = partial
# pose offset + persistent OSC nullspace bias (gates:
# experiments/verify_robot_axis.py, all passed 2026-07-20).
# Arms: vanilla + {state,action}x{text,image} + allxall @ l=1, then an
# OLD-BASIS base0 arm. lambda=0-as-base0 stays omitted (bit-identical to
# vanilla since the 2026-07-20 lambda=0 SDPA delegation, verify_base0_parity.py);
# the old-basis arm instead reproduces the pre-07-20 eager-dense path
# bit-for-bit via --pladis-scale 1.0 --pladis-method softmax (sparse branch ==
# dense branch -> blend collapses to eager dense), giving this axis the same
# [fused vanilla | eager base0 | eager lambda=1] ladder as language/layout.
# It is gated on (1) attn_gr00t_n17.py still being the weight-space hook and
# (2) a 2-episode language parity check against the stored pre-07-20 base0
# eplog (requires sweep_n17_language.sh results).
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
    local out="results/sweep/n17_robot_${tag}_${S}_eplog.tsv"
    echo "[robot] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis robot --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_robot_${tag}_${S}" "$@" \
      > "results/sweep/n17_robot_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_robot_${tag}_${S}.out"
  done
}

run vanilla
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

# 2026-08-05 far-extrapolation rungs: the 07-28 row was NULL through lambda=2.0
# on this axis with a pooled-n.s. upward drift at L5, and language's all-x-t
# kept rising to its lambda=2.0 peak — extend both text loci to {2.5, 3.0} to
# test whether the robot null holds where the blend pushes weights negative.
run actionxtext25 --pladis-install --pladis-scale 2.5 --pladis-qgroup action --pladis-kind text
run actionxtext30 --pladis-install --pladis-scale 3.0 --pladis-qgroup action --pladis-kind text
run allxtext25    --pladis-install --pladis-scale 2.5 --pladis-qgroup all    --pladis-kind text
run allxtext30    --pladis-install --pladis-scale 3.0 --pladis-qgroup all    --pladis-kind text

# 2026-08-06 temperature-softmax row (supp G.1; beta=2 = ent15-matched strength,
# same calibration as language's allxt-temp20* row), all-x-text locus only
# (operator grid 2026-08-06): the entmax ladder above was NULL through
# lambda=2.0 with far-extrapolation rungs at {2.5, 3.0} — swap the sparse
# branch for softmax(2*l) at lambda {1, 1.5, 2, 2.5} to test whether the
# robot-axis response is entmax-specific (exact zeros) or generic sharpening.
# lambda=2.5 pairs the far-extrapolation rung at matched blend weight.
run allxt-temp20    --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20l15 --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20l20 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20l25 --pladis-install --pladis-scale 2.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text


# 2026-08-18 denoising-step schedule row (operator request; the language axis's
# 08-16/17 finding carried onto the other perturbation axes). On language the
# benefit lives in the LATE denoising steps: at lambda=2 on all-x-text, late
# [0,0,1,1] scored +2.86pp vs vanilla (z=3.64 — the axis's only Bonferroni-
# surviving arm-vs-vanilla result) and inc [0,0.5,1,1.5] +2.80 (z=3.37), while
# early [1,1,0,0] was -1.30 and late-minus-early +4.16 (z=5.00, Bonf*): the SAME
# total dose helps or hurts depending on WHERE in the flow's time axis it is spent.
# On THIS axis the flat parent allxtext20 read +1.03pp vs vanilla (z=+0.95, n.s.)
# and the ladder stayed flat out to the lambda=3.0 far-extrapolation rung, so the
# question is whether that null is a null or a CANCELLATION: early-step harm plus
# late-step benefit summing to zero. Dropping the early half is the direct test.
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
# On THIS axis late read +1.23pp vs vanilla (z=+1.17, n.s.) — the largest positive
# point estimate the shape has outside language, but well inside noise — with inc
# at -1.16 (n.s.). The softmax flat parent allxt-temp20l20 is +1.48 (z=1.49, n.s.),
# so both branches sit in the same n.s. band and the swap starts level.
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

# ---- OLD-BASIS base0 (eager-dense) arm, gated ----
REF=results/sweep/n17_lang_base0_libero_10_eplog.tsv
if [ ! -f "$REF" ]; then
  echo "[robot] ABORT base0: parity reference $REF missing (run sweep_n17_language.sh first)"
  exit 1
fi
# Parity gate: episodes 0-1 of the language axis must reproduce the stored
# pre-07-20 base0 eplog exactly (proves method-softmax == old eager-dense).
PAR=results/sweep/robot_base0_parity_eplog.tsv
rm -f "$PAR"
echo "[robot] base0 parity gate: 2 language eps vs stored pre-07-20 base0 ..."
bash experiments/run.sh experiments/eval_arm.py \
  --suite libero_10 --axis language --episodes 2 --seed 0 \
  --model-path "$MODEL_ROOT/libero_10" --out "$PAR" \
  --pladis-install --pladis-scale 1.0 --pladis-method softmax \
  > results/sweep/robot_base0_parity.out 2>&1
python3 - <<'EOF'
import csv, sys
def load(p):
    return {r["episode"]: r for r in csv.DictReader(open(p), delimiter="\t")}
new = load("results/sweep/robot_base0_parity_eplog.tsv")
old = load("results/sweep/n17_lang_base0_libero_10_eplog.tsv")
bad = [e for e, r in new.items()
       if old.get(e) is None
       or any(r[k] != old[e][k] for k in ("task_name", "success_once", "n_steps"))]
print(f"[robot] base0 parity rows={len(new)} mismatches={len(bad)}", flush=True)
sys.exit(0 if new and not bad else 1)
EOF
if [ $? -ne 0 ]; then
  echo "[robot] ABORT base0: parity gate FAILED - method-softmax path is NOT the old base0 basis"
  exit 1
fi
echo "[robot] base0 parity gate PASSED"
run base0 --pladis-install --pladis-scale 1.0 --pladis-method softmax

echo "[robot] ALL DONE $(date +%H:%M:%S)"
