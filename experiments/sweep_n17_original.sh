#!/bin/bash
# GR00T N1.7 x LIBERO original (unperturbed): axis=none enumerates each suite's
# 10 base tasks with their ORIGINAL instructions and no perturbation at all —
# the campaign's IN-DISTRIBUTION control. Two uses:
#   (a) the per-base-task in-dist baseline every axis's "perturbation severity"
#       table is measured against (analyze.py reads n17_orig_vanilla_*),
#   (b) since 2026-08-16, an arm grid: does an intervention that helps on
#       instruction-OOD do anything when nothing is OOD?
# Seed-0 schedule, paired across arms. Requires the _moved-aware
# _VARIANT_MARKER (env.py) — without it, libero_goal axis=none enumerates 20
# "bases" incl. 10 layout-perturbed *_moved scenes.
#
# EPISODES (08-16): `.pruned_init` carries 50 init states per base task
# (env.py:11), but the original 2026-07-16 run used only 100 eps/suite = init
# 0-9. ORIG_EPISODES raises that; 500 = init 0-49 exhausted, 2,000 eps/arm,
# which is the n the perturbation axes run at (1,525-1,601) and 5x the paired
# precision: McNemar SE 1.2-1.4pp at n=400 -> ~0.6pp at n=2,000. That matters
# because the effect this axis has to bound is the language axis's +2.5pp; at
# n=400 the test had ~40% power against it, i.e. its null was "not seen", not
# "not there".
#   The schedule is a strict PREFIX-SUPERSET — schedule(500,0)[:100] ==
#   schedule(100,0), verified 08-16 (task_name + init_state_id + episode index
#   all equal) — and the episode count is NOT in the arm signature, so raising
#   it resumes: already-logged episodes are skipped and only the tail runs.
#   ⚠️SIDE EFFECT: extending vanilla makes every axis's severity table quote a
#   5x more precise in-dist baseline, so those numbers will shift by ~1pp from
#   what was reported before 08-16. The old figures stay recoverable exactly,
#   because the first 100 rows per suite are the same episodes.
ORIG_EPISODES="${ORIG_EPISODES:-100}"
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
. experiments/load_machine_env.sh
MODEL_ROOT="$MODEL_ROOT_GR00T_N17"
pladis_require_clean_tree
SUITES="libero_10 libero_goal libero_object libero_spatial"

# Historical: this driver was chained behind sweep_n17_language.sh (2026-07-16),
# waiting on its pid and refusing to start unless it had logged ALL DONE. That
# sweep finished 07-16; the guard is kept because re-running this driver is the
# resume mechanism and the completed-language precondition still holds, but the
# pid now comes from the environment so that "$@" is free for arm selection.
LANG_PID="${LANG_PID:-2689008}"
while ps -p "$LANG_PID" -o args= 2>/dev/null | grep -q sweep_n17_language; do
  sleep 300
done
if ! grep -q "\[sweep\] ALL DONE" results/sweep/driver.out 2>/dev/null; then
  echo "[orig] ABORT: language sweep exited without ALL DONE - resume it first" >&2
  exit 1
fi

wait_ckpt() {
  until [ -f "$MODEL_ROOT/$1/config.json" ] && ! ls "$MODEL_ROOT/$1"/*.incomplete >/dev/null 2>&1; do
    echo "[orig] waiting for checkpoint $1 ..."; sleep 60
  done
}

# The axis's arm vocabulary, in run order. A command-line selection only FILTERS
# this list — it can never introduce an arm — so an unknown name is an abort
# rather than a driver that quietly runs nothing.
ARMS="vanilla base0 actionximage actionxtext statextext stateximage allxall \
allxt-temp20l20 allxt-temp20l15 allxt-temp20 \
allxtext20 allxt-late-l2 allxt-inc-l2 allxt-temp20-late-l2 \
allxt-temp20l30 allxt-temp20-nagn-l10 allxt-temp20-nagn-l15 \
allxt-temp20-nagn-l20 allxt-temp20-nagn-l30"
if [ "$#" -gt 0 ]; then SELECT="$*"; else SELECT="$ARMS"; fi
for a in $SELECT; do
  case " $ARMS " in
    *" $a "*) ;;
    *) echo "[orig] ABORT: unknown arm '$a'; this axis carries: $ARMS" >&2; exit 2 ;;
  esac
done
echo "[orig] arms: $SELECT (episodes/suite=$ORIG_EPISODES)"

run() { # $1=tag, rest = pladis args; a tag outside $SELECT is skipped
  local tag="$1"; shift
  case " $SELECT " in *" $tag "*) ;; *) return 0 ;; esac
  for S in $SUITES; do
    local out="results/sweep/n17_orig_${tag}_${S}_eplog.tsv"
    wait_ckpt "$S"
    echo "[orig] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis none --episodes "$ORIG_EPISODES" --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_orig_${tag}_${S}" "$@" \
      > "results/sweep/n17_orig_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_orig_${tag}_${S}.out"
  done
}

run vanilla
run base0        --pladis-install --pladis-scale 0
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

# 08-16 IN-DISTRIBUTION CONTROL — all-x-text sharp-softmax (softmax(2*l), beta=2)
# at lambda {2.0, 1.5, 1.0}, highest dose first. These are the exact arms that
# produced the campaign's only Bonferroni-surviving positive on the language
# axis (allxt-temp20l20 vs vanilla +2.54pp z=3.36 p_bonf=.047), so running them
# where NOTHING is out of distribution is the direct test of the
# grounding-specificity claim: the gain should vanish. The pre-existing arms
# above are the lambda=1 modality grid and never covered this dose region — on
# language, lambda=1 a-x-t was itself n.s. (+0.59), so they could not have.
run allxt-temp20l20 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20l15 --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20    --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text


# 08-18 IN-DISTRIBUTION CONTROL for the denoising-step schedule row (operator
# request). The 08-16 block above ran the in-dist control for the campaign's
# sharp-softmax winner; the 08-16/17 language rows then produced a BETTER one on
# the entmax branch — late [0,0,1,1] at lambda=2 on all-x-text, +2.86pp vs
# vanilla (z=3.64, Bonf*), with inc [0,0.5,1,1.5] at +2.80 (z=3.37). The
# grounding-specificity claim predicts both do nothing where nothing is OOD.
# Three arms, because this axis has no entmax lambda=2 rung at all: allxtext20 is
# the flat parent ([2,2,2,2]) the two shapes are read against, and it doubles as
# the entmax-vs-softmax branch swap in-dist (its softmax twin allxt-temp20l20 is
# already collected here at -1.25pp vs vanilla, n.s.). The iso-dose flat controls
# the perturbation axes use (allxtext at lambda=1, allxtext15) are deliberately
# NOT run: in-dist the question is "does the winning arm do anything at all",
# which is the vs-vanilla contrast, not the when-vs-how-much decomposition.
# POWER: at ORIG_EPISODES=100 (n=400) the paired SE is ~1.3pp — measured, not
# assumed: the existing allxt-temp20l20-vs-vanilla pair discordant 11:16 gives
# sqrt(27)/400 = 1.30pp — so a language-sized +2.86pp would land at z~2.2 and a
# null here bounds the in-dist effect only to about +/-2.5pp. ORIG_EPISODES=500
# (n=2,000, init 0-49) takes that to ~0.6pp; the episode count is not in the arm
# signature, so running at 100 now and extending later RESUMES rather than
# re-runs (see the ORIG_EPISODES note at the head of this file), and vanilla
# must be extended with them.
run allxtext20    --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text
run allxt-late-l2 --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1
run allxt-inc-l2  --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0.5,1,1.5

# 2026-08-21 sharp-softmax mirror of the LATE schedule arm (operator request).
# The 08-16/17 language rows put the benefit in the LATE denoising steps on BOTH
# sparse branches: entmax late [0,0,1,1] at lambda=2 all-x-text +2.86pp vs vanilla
# (z=3.64, Bonf*) and its softmax(2*l) twin +1.69 (z=2.17), the two statistically
# indistinguishable from each other (temp-late - late -1.17pp, z=-1.59) — so ON
# LANGUAGE the time structure belongs to the sharpening, not to entmax's exact
# zeros. The 08-18 row then carried the ENTMAX late arm to every other axis.
# IN-DISTRIBUTION CONTROL. Here late read -1.25pp vs vanilla (z=-0.85, n.s.) and
# inc -3.50 (z=-2.21, nominal): the language gain vanished where nothing is OOD,
# which is what the grounding-specificity claim predicts. This arm completes the
# 2x2 (branch x shape) in-dist — allxtext20 / allxt-temp20l20 are the flat pair,
# allxt-late-l2 / this are the late pair — so the entmax-vs-softmax swap is read
# in-dist at the same shape as on the perturbation axes.
# POWER: at n=400 the paired SE is ~1.3pp, so this bounds an in-dist effect to
# about +/-2.5pp; it cannot resolve a language-sized +2.9 as significant on its own.
# This arm is the branch swap of that one here: same all-x-text locus, same
# lambda=2 base, same [0,0,1,1] weights, sparse branch entmax-1.5 -> softmax(2*l)
# at beta=2 (the ent15-strength-matched setting of supp G.1; beta=1 would collapse
# the sparse branch onto the dense one and void every "did blend" assertion).
# Effective lambda per step = 2 * w = [0,0,2,2].
# Read three ways: vs vanilla (does the winning composition do anything in-dist),
# vs its flat parent allxt-temp20l20, and vs allxt-late-l2 (the branch swap at
# matched shape). The iso-dose flat control allxt-temp20 is collected here too, so
# that reading comes for free.
run allxt-temp20-late-l2 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1

# 2026-08-26 NAG normalization row (docs/nag.md; operator's 260821 deck §2 "NAG
# Normalization in Ours"). PROBLEM: the all-x-text beta=2 family is the strongest
# intervention on four axes, but its best lambda is not the same one — original
# peaks at 1.0, noise at 1.5, language and robot at 2.0, and one dose rung swings
# 4.2pp across axes with the sign flipping. NAG caps each (head, query row)'s
# attention output at tau x the DENSE branch's L1 magnitude, direction preserved:
#     R      = ||Z_d + lambda(Z_s - Z_d)||_1 / ||Z_d||_1        per query row
#     Z_NPL  = min(R, tau)/R * Z_PL                             (rho=1: no refinement)
# The question is not "is it better here" but whether the cap WIDENS THE PLATEAU,
# so that ONE setting (lambda=2, tau=2.5 — shared across axes, never tuned per
# axis, or the claim is empty) is near-optimal everywhere.
# tau=2.5 comes from experiments/diag_nag.py, not from the paper: measured on this
# checkpoint it clips 0.8% of query rows at lambda=1, 4.6% at 1.5, 11.9% at 2 and
# 31.8% at 3 — inert where each axis's ladder is healthy, active where it turns
# over (the docs/nag.md §6 rule; it is also the paper's own default, by coincidence).
# rho<1 is deliberately NOT run: with the cap inactive it is algebraically the plain
# arm at scale=rho*lambda (docs/nag.md §2b), a rung this ladder already has.
# TIER 2, the plateau CURVE. This axis is 400 episodes (~1.3 h/arm) and falls
# fastest from its lambda=1 optimum (-2.25pp by lambda=2, z=-1.80), so the whole
# shape story fits here for the price of one language arm. Four capped rungs plus
# the one plain rung the ladder is missing (lambda=3), giving both curves on the
# same episodes:
#   plain  lambda 1 / 1.5 / 2   collected        + lambda 3 below
#   capped lambda 1 / 1.5 / 2 / 3               all below
# lambda=1 doubles as the INERTNESS CONTROL: a selective tau must leave this axis's
# own optimum essentially unchanged (it clips 0.8% of rows there).
# POWER: n=400 gives a paired SE of ~1.5pp, so this row shows whether the cap does
# anything at all — it cannot settle a 2pp question by itself, which is why the
# 20 h/arm `noise` axis is only funded after the diag's cross-axis ranking holds.
run allxt-temp20l30       --pladis-install --pladis-scale 3.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20-nagn-l10 --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-tau 2.5
run allxt-temp20-nagn-l15 --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-tau 2.5
run allxt-temp20-nagn-l20 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-tau 2.5
run allxt-temp20-nagn-l30 --pladis-install --pladis-scale 3.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-tau 2.5

echo "[orig] ALL DONE $(date +%H:%M:%S)"
