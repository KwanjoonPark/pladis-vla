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
# SDPA at this scale, so all seven arms share one cost model). Per arm:
#     mean steps   350     450     550     720 (every episode at the cap)
#     hours       25.3    32.5    39.8    52.1
# camera's own mean was 334 steps; noise runs lower SR, so budget the middle of
# that band. The model is validated on the one clean measured point available:
# motion sev1 at 720 steps -> predicted 133.6 s vs 140.25 s logged
# (results/noise_try5_eplog.tsv, ep0). Seven arms is therefore ~180-280 h of
# machine time, not the ~55 h a camera-sized reading of "seven arms" suggests.
#
# CONCURRENCY: the bottleneck here is single-threaded CPU (the corruption call),
# not the GPU, so two drivers over DISJOINT arms nearly halve the wall clock.
# Arms are selectable as arguments; with none, this runs all seven in order:
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
ARMS="vanilla actionxtext actionxtext15 actionxtext20 allxtext allxtext15 allxtext20"
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

echo "[noise] ALL DONE $(date +%H:%M:%S)"
