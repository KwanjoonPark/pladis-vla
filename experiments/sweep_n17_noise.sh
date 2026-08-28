#!/bin/bash
# GR00T N1.7 x LIBERO-plus NOISE axis (Sensor Noise), FULL curated set: four
# suites, 1,601 episodes per arm (10/goal/object/spatial = 449/379/422/351),
# seed-0 paired schedule. Obs-side corruption of the AGENTVIEW camera only:
# N 1-10 motion blur / 11-20 gaussian / 21-30 zoom / 31-40 fog / 41-50 glass
# (5 families x 10 severities; seed-0 schedule composition, measured 2026-08-10:
# motion 336 / gauss 341 / zoom 288 / fog 272 / glass 364). Corruption draws and
# fixture placement are pinned per episode (RUNTIME_RNG_AXES;
# verify_noise_axis.py ALL GATES PASSED, re-confirmed at 2a69b42 on 2026-08-10).
#
# ARMS (operator grid 2026-08-10): vanilla + the TEXT-LOCUS DOSE LADDER at both
# query groups — a-x-t and all-x-t at lambda {1, 1.5, 2.0}, i.e. the camera
# axis's grid. Like camera it skips the track's default lambda=1 modality grid:
# on language and robot those cells read flat and every signal that appeared
# came from the text-locus dose row, so the budget goes into dose depth at the
# two text loci. base0 is omitted (bit-identical to vanilla since the 07-20
# lambda=0 SDPA delegation, verify_base0_parity.py). With no image arm the
# modality locus pair does not exist here, so this axis's locus contrast is the
# QUERY-GROUP one — all-x-t vs a-x-t at matched dose — which is what
# analysis/analyze.py uses (its `model_overrides` entry for this axis).
#
# COST — this axis is 3-4x more expensive per arm than any other, and one family
# is the entire reason. Per-frame corruption cost measured at HEAD on the 256x256
# agentview the harness actually renders (2026-08-10, mean over ten severities):
#   motion 448 ms (sev1 140 -> sev10 893)   <- ImageMagick, external C
#   zoom 141 ms    gauss 6.3 ms    glass 5.8 ms    fog 3.0 ms
# glass is already the bit-exact numba rebind (1fbc3c5); motion has no bit-exact
# speedup (MAGICK_THREAD_LIMIT has no effect on this build), so 21% of the
# episodes carry 77% of the wall time: corruption alone costs 196 s per arm per
# step-of-mean-episode-length. The uncorrupted base is 40 ms/step, measured at
# sweep scale on the camera axis (n=1,599 each: vanilla 39.6, actionxtext 41.2,
# allxtext20 38.0 — the lambda>0 eager path is NOT measurably slower than fused
# SDPA at this scale, so every arm shares one cost model). Per arm:
#     mean steps   350     450     550     720 (every episode at the cap)
#     hours       25.3    32.5    39.8    52.1
# camera's own mean was 334 steps; noise runs lower SR, so budget the middle of
# that band. The model is validated on the one clean measured point available:
# motion sev1 at 720 steps -> predicted 133.6 s vs 140.25 s logged
# (results/noise_try5_eplog.tsv, ep0), and it was right about the ORDER: this is
# ~20x the camera axis per episode, not the ~55 h a camera-sized reading suggests.
# MEASURED 2026-08-23 over the 13 completed arms (1,601 eps each): 42.8-46.0 s/ep
# = 19.0-20.5 h per arm, mean 19.9. The projection over-predicted because the mean
# episode landed near 350 steps, the bottom of the band above. Budget ~20 h/arm:
# this file's ELEVEN arms are ~220 h, and the three sharp-softmax arms in
# sweep_n17_noise_temp.sh add ~60 h.
#
# CONCURRENCY: the bottleneck here is single-threaded CPU (the corruption call),
# not the GPU, so two drivers over DISJOINT arms nearly halve the wall clock.
# Arms are selectable as arguments; with none, this runs all eleven in order:
#   nohup bash experiments/sweep_n17_noise.sh vanilla actionxtext actionxtext15 actionxtext20 \
#        > results/sweep/driver_noise.out 2>&1 &
#   nohup bash experiments/sweep_n17_noise.sh allxtext allxtext15 allxtext20 \
#        > results/sweep/driver_noise_b.out 2>&1 &
# Two processes is the ceiling: each holds ~10.3 GB RSS of this box's 31 GB
# (measured 2026-08-08 on the camera axis) and the box has an OOM-kill history.
# Selection is why this axis does NOT need the second-file workaround the camera
# temperature row used — appending a row to a file bash is mid-execution makes the
# running shell resume at a stale byte offset, but selecting arms from a file that
# is never edited mid-run does not. One file stays the arm vocabulary's source of
# truth (CLAUDE.md, "Adding an arm").
# Interleaving cannot break pairing: every RNG source is pinned per episode, not
# per process — the schedule permutation (seed), the env reseed
# (seed*1_000_003 + episode), the flow-noise pin (episode_seed*100_003 + step),
# and on this axis the per-episode np reseed that pins the corruption draws.
# Resume-safe at episode granularity.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
. experiments/load_machine_env.sh
MODEL_ROOT="$MODEL_ROOT_GR00T_N17"
pladis_require_clean_tree
SUITES="libero_10 libero_goal libero_object libero_spatial"

# The axis's arm vocabulary, in run order. A command-line selection only FILTERS
# this list — it can never introduce an arm — so an unknown name is an abort
# rather than a driver that quietly runs nothing for four days.
ARMS="vanilla actionxtext actionxtext15 actionxtext20 allxtext allxtext15 allxtext20 \
allxt-late-l2 allxt-inc-l2 allxt-temp20-late-l2 allxt-temp20-late-l15 allxt-late-l15"
if [ "$#" -gt 0 ]; then SELECT="$*"; else SELECT="$ARMS"; fi
for a in $SELECT; do
  case " $ARMS " in
    *" $a "*) ;;
    *) echo "[noise] ABORT: unknown arm '$a'; this axis carries: $ARMS" >&2; exit 2 ;;
  esac
done
echo "[noise] arms: $SELECT"

run() { # $1=tag, rest = pladis args; a tag outside $SELECT is skipped
  local tag="$1"; shift
  # A run line whose tag is not in ARMS can never be selected (no-arg SELECT=ARMS,
  # explicit args validated against ARMS above), so it would no-op silently on
  # every invocation while looking wired. This bit sweep_n17_noise_temp.sh on 2026-08-27.
  case " $ARMS " in *" $tag "*) ;; *) echo "[noise] ABORT: run line for '$tag' but it is missing from ARMS — this arm would silently never run" >&2; exit 2 ;; esac
  case " $SELECT " in *" $tag "*) ;; *) return 0 ;; esac
  for S in $SUITES; do
    local out="results/sweep/n17_noise_${tag}_${S}_eplog.tsv"
    echo "[noise] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis noise --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_noise_${tag}_${S}" "$@" \
      > "results/sweep/n17_noise_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_noise_${tag}_${S}.out"
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
# On THIS axis the flat parent allxtext20 read -0.25pp vs vanilla (z=-0.30) and
# every one of the six entmax ladder cells came in flat, so the question is
# whether that null is a null or a CANCELLATION: early-step harm plus late-step
# benefit summing to zero. Dropping the early half is the direct test.
# COST: ~20 h per arm at this axis's measured 45 s/ep x 1,601 eps (the corruption
# call, not the GPU, sets it), i.e. ~40 h for the pair sequentially. They are
# disjoint arms, so the two-driver split this file already documents halves it:
#   nohup bash experiments/sweep_n17_noise.sh allxt-late-l2 > results/sweep/driver_noise_late.out 2>&1 &
#   nohup bash experiments/sweep_n17_noise.sh allxt-inc-l2  > results/sweep/driver_noise_inc.out  2>&1 &
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
# On THIS axis late read +0.12pp vs vanilla (z=+0.15) and inc -0.75 (z=-0.85):
# flat, like every other arm here — the axis reads at ceiling on the text locus.
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
# COST: ~15-20 h for this one arm at 1,601 eps (the per-frame corruption, not the
# GPU, sets the rate), so it is a one-arm driver of its own:
#   nohup bash experiments/sweep_n17_noise.sh allxt-temp20-late-l2 > results/sweep/driver_noise_late_temp.out 2>&1 &
run allxt-temp20-late-l2 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1

# 2026-08-21 dose rung UNDER the late shape (operator request), softmax branch:
# the same [0,0,1,1] weights and the same beta=2 at a lambda=1.5 base instead of 2
# (lambda_i = [0,0,1.5,1.5], sum 3). The 08-16/17 language row measured the shape at
# ONE dose only, so "does the late shape need lambda=2, or does it survive the rung
# below" is untested everywhere; this asks it here.
# Read THREE ways, and it is worth being explicit about which two do NOT exist:
#   vs vanilla;
#   vs its flat parent allxt-temp20l15 ([1.5,1.5,1.5,1.5], sum 6 — same peak lambda,
#     twice the total dose), collected on this axis;
#   vs allxt-temp20-late-l2, the SAME shape and branch one dose rung up — the
#     within-row dose step, which is the contrast this arm is FOR.
#   NOT available: the iso-dose flat control (sum lambda = 3 would need a flat
#     lambda=0.75 arm, a rung no axis of the campaign carries) and the entmax branch
#     swap (allxt-late-l15 was never run — the entmax schedule row is lambda=2 only).
#   So a null here is readable as "the shape does not survive the dose cut", but the
#   when-vs-how-much decomposition the lambda=2 rung gets is not reproduced at 1.5.
# COST: another ~15-20 h at 1,601 eps, and this axis reads at ceiling on the text
# locus (every arm flat vs vanilla), so budget it as a dose-ladder completion rather
# than as a place a signal is expected. Its own one-arm driver:
#   nohup bash experiments/sweep_n17_noise.sh allxt-temp20-late-l15 > results/sweep/driver_noise_late_temp_l15.out 2>&1 &
run allxt-temp20-late-l15 --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1

# 2026-08-29 (operator request) the ENTMAX counterpart of the 08-23 lambda=1.5
# late rung above. This axis's plain optimum is lambda=1.5, so the late shape
# [0,0,1.5,1.5] here sits at the dose the axis actually wants; the sharp-softmax
# twin allxt-temp20-late-l15 is collected, which makes this the branch swap at
# matched shape AND matched dose (zeros-not-special on the time axis, at the
# axis optimum rather than at lambda=2). Also read vs its flat parent allxtext15
# (same peak lambda, twice the time-integrated dose) and vs allxt-late-l2 (the
# dose step within the entmax late shape). COST: ~20 h at 45 s/ep x 1,601.
run allxt-late-l15 --pladis-install --pladis-scale 1.5 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1

echo "[noise] ALL DONE $(date +%H:%M:%S)"
