#!/bin/bash
# GR00T N1.7 x LIBERO-plus NOISE axis — TEMPERATURE-SOFTMAX row, run
# CONCURRENTLY with sweep_n17_noise.sh on the same machine.
#
# Arms: all-x-text with the sparse branch swapped for softmax(beta*l), beta=2
# (the ent15-matched strength calibration used on language, robot and camera),
# at lambda {1.0, 1.5, 2.0} — the matched-dose counterparts of the entmax row
# allxtext / allxtext15 / allxtext20 that driver B finished on 2026-08-13.
# Together the two rows ask the zeros-vs-sharpening question on this axis.
#
# WHAT THIS AXIS'S ENTMAX ROW ALREADY SAID (2026-08-13, all six cells complete
# except a-x-t lambda=2, paired McNemar vs vanilla on 1,601 matched episodes,
# success_at_end): actionxtext +0.25 z=+0.33 / actionxtext15 -0.19 z=-0.23 /
# allxtext +0.12 z=+0.16 / allxtext15 +0.50 z=+0.62 / allxtext20 -0.25 z=-0.30.
# FLAT — noise is the first axis where the text-locus dose ladder shows nothing
# at either query group. So this row is not chasing an effect: it is the
# SYMMETRIC-NULL control for the zeros-not-special claim. Language and camera
# established temp ~= entmax where entmax HELPS; a null axis where both are
# equally inert is the other half of that equivalence, and the arm to spend
# first is the one matching the cross-axis headline cell (lambda=2.0).
#
# WHY A SEPARATE FILE, not three more `run` lines in sweep_n17_noise.sh: bash
# reads a script incrementally by byte offset, so editing one that is
# mid-execution makes the running shell resume at a stale offset and execute
# garbage. sweep_n17_noise.sh is live as this is written (2026-08-13, driver A
# on actionxtext20), exactly the situation that forced the same split on camera
# (sweep_n17_camera_temp.sh, 2026-08-08). Once both drivers have finished, fold
# these `run` lines into sweep_n17_noise.sh and delete this file — the axis
# should go back to one driver as its arm vocabulary's source of truth
# (CLAUDE.md, "Adding an arm").
#
# COST — MEASURED, not modelled. sweep_n17_noise.sh's header budgets 25-52 h per
# arm from a 350-720 mean-step band; the seven completed arms came in at 240-248
# mean steps and 19.7-20.3 h per 1,601-episode arm (0.18-0.19 s/step wall,
# uniform across vanilla and every lambda — the eager PLADIS path is not
# measurably slower). Noise's SR is high (~85%), so episodes end early and the
# expensive motion-blur family is paid on fewer steps than the model assumed.
#   BUDGET 20 h per arm; 60 h for all three.
#
# CONCURRENCY: the bottleneck is single-threaded CPU (the corruption call), not
# the GPU, so a second driver over DISJOINT arms nearly halves wall clock.
# Measured 2026-08-13 with driver A live: GPU 6.9/24.5 GB at ~0% steady util,
# RAM 11/31 GB used with 18 GB available. Each eval process holds ~10.3 GB RSS,
# so TWO fit and a third would not (this box has an OOM-kill history) — driver B
# has finished, so exactly one slot is free.
# Arms are selectable as arguments; with none, this runs all three in order:
#   nohup bash experiments/sweep_n17_noise_temp.sh allxt-temp20l20 \
#        > results/sweep/driver_noise_temp.out 2>&1 &
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

# The row's arm vocabulary, in run order. A command-line selection only FILTERS
# this list — it can never introduce an arm — so an unknown name is an abort
# rather than a driver that quietly runs nothing for days.
ARMS="allxt-temp20l20 allxt-temp20l15 allxt-temp20 allxt-temp20-nagn-l20"
if [ "$#" -gt 0 ]; then SELECT="$*"; else SELECT="$ARMS"; fi
for a in $SELECT; do
  case " $ARMS " in
    *" $a "*) ;;
    *) echo "[noise-temp] ABORT: unknown arm '$a'; this row carries: $ARMS" >&2; exit 2 ;;
  esac
done
echo "[noise-temp] arms: $SELECT"

run() { # $1=tag, rest = pladis args; a tag outside $SELECT is skipped
  local tag="$1"; shift
  # A run line whose tag is not in ARMS can NEVER be selected: with no args
  # SELECT=ARMS, and explicit args are validated against ARMS above. It would
  # no-op silently on every invocation while looking wired — which is exactly
  # what happened on 2026-08-27 to the first launch of allxt-temp20-nagn-l20
  # (added as a run line, not added to ARMS; the sweep printed ALL DONE in 14 s).
  case " $ARMS " in *" $tag "*) ;; *) echo "[noise-temp] ABORT: run line for '$tag' but it is missing from ARMS — this arm would silently never run" >&2; exit 2 ;; esac
  case " $SELECT " in *" $tag "*) ;; *) return 0 ;; esac
  for S in $SUITES; do
    local out="results/sweep/n17_noise_${tag}_${S}_eplog.tsv"
    echo "[noise-temp] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis noise --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_noise_${tag}_${S}" "$@" \
      > "results/sweep/n17_noise_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_noise_${tag}_${S}.out"
  done
}

# Run order is HIGHEST DOSE FIRST, the reverse of the entmax row's. That row was
# a ladder read bottom-up for onset; this one is a matched-pair control, and the
# pair that carries the cross-axis story is lambda=2.0 (allxtext20, the cell
# that reads +2.54 z=3.2 on language and camera and -0.25 z=-0.30 here). If only
# one arm is ever run, it should be that one.
run allxt-temp20l20 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20l15 --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20    --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text

# 2026-08-28 NAG cap at the shared setting (docs/nag.md §7 Tier 1, the axis's
# decisive reading). Noise is the big-n FALLING axis: the plain temp ladder peaks
# at lambda=1.5 (86.76) and gives back -1.81pp by lambda=2 (z=-2.50, the only
# nominally significant turnover in the campaign). On `original` (the small-n
# falling axis) the cap at this same setting recovered the whole lambda=2 loss
# (+2.25 vs its uncapped twin, z=+1.96; regret vs the axis optimum 0.00pp on
# 9:9 discordants), and on the climbing axes it cost -0.78 (language, n.s.) /
# -1.23 (robot, n.s.) — so this arm is the confirmation test with real power
# (n=1601, paired SE ~0.7pp): does lambda=2 + the cap recover to the lambda=1.5
# level here too? All three comparators (temp20l15 = the axis optimum,
# temp20l20 = the uncapped twin, vanilla) are collected above.
# tau=2.5 was selected on THIS axis's own R distribution (diag_nag.py 08-26:
# clip 0.7% @lambda1 / 4.6% @1.5 / 12.5% @2 / 33.4% @3 — same rule, same pick
# as the other three axes). COST: ~20 h at this axis's measured 45 s/ep.
run allxt-temp20-nagn-l20 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-tau 2.5

echo "[noise-temp] ALL DONE $(date +%H:%M:%S)"
