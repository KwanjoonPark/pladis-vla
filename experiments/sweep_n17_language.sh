#!/bin/bash
# GR00T N1.7 x LIBERO-plus language axis, FULL curated set: all four suites,
# 1,537 episodes per arm (each curated variant exactly once, seed-0 schedule,
# paired across arms). Per-suite checkpoints from nvidia/GR00T-N1.7-LIBERO.
# Arms: vanilla, base0 (hook @ l=0), {state,action}x{text,image} + allxall @ l=1.
# Resume-safe at episode granularity (eval_arm skips episodes already logged).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
. experiments/load_machine_env.sh
MODEL_ROOT="$MODEL_ROOT_GR00T_N17"
pladis_require_clean_tree
# libero_10 first: its checkpoint is already local, buying download time
SUITES="libero_10 libero_goal libero_object libero_spatial"

# wait for the parity gate (3 DONEs), max 40 min
for _ in $(seq 1 240); do
  [ "$(grep -c "\[arm\] DONE" results/parity_gate.out 2>/dev/null)" -ge 3 ] && break
  sleep 10
done

wait_ckpt() {  # suite checkpoints are downloaded in the background; block until present
  until [ -f "$MODEL_ROOT/$1/config.json" ] && ! ls "$MODEL_ROOT/$1"/*.incomplete >/dev/null 2>&1; do
    echo "[sweep] waiting for checkpoint $1 ..."; sleep 60
  done
}

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/n17_lang_${tag}_${S}_eplog.tsv"
    wait_ckpt "$S"
    echo "[sweep] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis language --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_lang_${tag}_${S}" "$@" \
      > "results/sweep/n17_lang_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_lang_${tag}_${S}.out"
  done
}

run vanilla
run base0        --pladis-install --pladis-scale 0
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

# 2026-07-22 composition arms (resume-safe: completed arms above are skipped).
#   allxtext = {action,state}xtext — qgroup=all on the text cross-blocks
#   axt-sxi  = actionxtext + stateximage — per-kind qgroups in one pass
run allxtext --pladis-install --pladis-scale 1.0 --pladis-qgroup all --pladis-kind text
run axt-sxi  --pladis-install --pladis-scale 1.0 --pladis-cells actionxtext,stateximage

# 2026-07-23 lambda=1.5 arms (the official PLADIS recommended regime): the
# three text-locus arms plus the remaining three base cells, so lambda=1.5
# forms a complete dose row over every locus studied at lambda=1.
run actionxtext15 --pladis-install --pladis-scale 1.5 --pladis-qgroup action --pladis-kind text
run allxtext15    --pladis-install --pladis-scale 1.5 --pladis-qgroup all    --pladis-kind text
run axt-sxi15     --pladis-install --pladis-scale 1.5 --pladis-cells actionxtext,stateximage
run actionximage15 --pladis-install --pladis-scale 1.5 --pladis-qgroup action --pladis-kind image
run stateximage15  --pladis-install --pladis-scale 1.5 --pladis-qgroup state  --pladis-kind image
run statextext15   --pladis-install --pladis-scale 1.5 --pladis-qgroup state  --pladis-kind text

# 2026-07-23 temperature-softmax control (paper supplement G.1): sparse branch
# = softmax(beta*l) i.e. tau = 1/beta, at lambda=1 on the action-x-text locus.
# Strength-matched "are entmax's exact zeros special?" test vs ent15max
# (RoboCasa calibration: beta~2 matches ent15 sharpening strength).
run axt-temp15 --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 1.5 --pladis-qgroup action --pladis-kind text
run axt-temp20 --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup action --pladis-kind text
run axt-temp30 --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 3.0 --pladis-qgroup action --pladis-kind text

# 2026-07-26 lambda=2.0 arms: extend the dose ladder (1.0 -> 1.5 -> 2.0) over
# the four base cells.
run actionxtext20  --pladis-install --pladis-scale 2.0 --pladis-qgroup action --pladis-kind text
run actionximage20 --pladis-install --pladis-scale 2.0 --pladis-qgroup action --pladis-kind image
run stateximage20  --pladis-install --pladis-scale 2.0 --pladis-qgroup state  --pladis-kind image
run statextext20   --pladis-install --pladis-scale 2.0 --pladis-qgroup state  --pladis-kind text

# 2026-07-28 sharp-softmax dose row (beta=2 = ent15-matched strength, supp G.1):
# does the lambda>1 extrapolation behavior at the text loci survive replacing
# entmax with temperature sharpening? axt-temp20 (lambda=1) is the anchor;
# the allxt-temp* arms pair against allxtext{,15} (the strongest entmax cell).
run axt-temp20l15   --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup action --pladis-kind text
run axt-temp20l20   --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup action --pladis-kind text
run allxt-temp20    --pladis-install --pladis-scale 1.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all    --pladis-kind text
run allxt-temp20l15 --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all    --pladis-kind text
run allxt-temp20l20 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all    --pladis-kind text

# 2026-08-03 lambda=2.0 composite-text completion (operator request): the two
# composite text arms join the 2.0 rung so every text locus carries the full
# 1.0 -> 1.5 -> 2.0 dose ladder (actionxtext20 already collected in the 07-26
# block above; re-running it resumes to a no-op).
run allxtext20 --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text
run axt-sxi20  --pladis-install --pladis-scale 2.0 --pladis-cells actionxtext,stateximage

# 2026-08-16 denoising-step schedule row (operator's design; professor's request to
# analyse the intervention through the flow's time axis). The head integrates N=4
# Euler steps at t in {0,.25,.5,.75}; --pladis-schedule is a per-step multiplier on
# --pladis-scale (lambda_i = scale * w_i). Shape row:
#     vanilla [0,0,0,0]  all [1,1,1,1]  early [1,1,0,0]  late [0,0,1,1]
#     increasing [0,0.5,1,1.5]          decreasing [1.5,1,0.5,0]
# Two internally DOSE-MATCHED pairs (sum w: early=late=2, inc=dec=3) plus the sum=4
# parent, so shape is separable from total dose. The [0,0,0,0] and [1,1,1,1] rows
# need no run: vanilla is collected, and a flat [1,1,1,1] schedule is bit-identical
# to the unscheduled arm at the same scale (verify_step_schedule.py gate F) — here
# that is `allxtext20`.
# Base dose is lambda=2 on the all-x-text locus — the best arm this axis has
# measured (allxtext20 87.90 pooled, +2.54pp vs vanilla, z=3.2), so the shapes are
# read against an effect that exists. Effective lambda per step = 2 * w:
#     early [2,2,0,0]  late [0,0,2,2]  inc [0,1,2,3]  dec [3,2,1,0]
# NOTE the ramps peak at lambda=3, one rung above anything this axis has run (the
# ladder tops at 2.0); that peak sits on a single step, and its dose-matched mirror
# puts the same peak at the opposite end, so the inc-vs-dec contrast stays
# interpretable even if lambda=3 is harmful on its own.
run allxt-early-l2 --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 1,1,0,0
run allxt-late-l2  --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1
run allxt-inc-l2   --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0.5,1,1.5
run allxt-dec-l2   --pladis-install --pladis-scale 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 1.5,1,0.5,0

# 2026-08-17 sharp-softmax mirror of the schedule row (operator request): the same
# four shapes with the sparse branch swapped entmax-1.5 -> softmax(beta*l) at
# beta=2, the ent15-strength-matched setting of supp G.1. Everything else is held:
# same all-x-text locus, same lambda=2 base, same weights, same paired schedule.
# The matched parent is allxt-temp20l20 (all-x-text, lambda=2, beta=2), which is
# already collected and scored 87.90 pooled — the SAME value as the entmax parent
# allxtext20, so the two shape rows start from an equal footing and the entmax-vs-
# temp comparison is a clean branch swap at every point of the row. As above, the
# flat [1,1,1,1] row needs no arm: it is bit-identical to allxt-temp20l20
# (verify_step_schedule.py gate F, now asserted on the softmax branch too).
# This asks whether "the benefit lives in the late denoising steps" is a property
# of the sharpening itself or of entmax's exact zeros — the same zeros-not-special
# question the campaign has answered on the dose axis, now on the time axis.
run allxt-temp20-early-l2 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 1,1,0,0
run allxt-temp20-late-l2  --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0,1,1
run allxt-temp20-inc-l2   --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 0,0.5,1,1.5
run allxt-temp20-dec-l2   --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-schedule 1.5,1,0.5,0

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
# Here the plain ladder rises (+1.95pp from lambda=1 to 2, z=+2.72), so this arm
# asks the RISK question: does capping cost us the gain the dose bought? It reads
# against allxt-temp20l20 (same locus, same lambda, no cap — collected).
run allxt-temp20-nagn-l20 --pladis-install --pladis-scale 2.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-tau 2.5

# 2026-08-27 the A-side of the cross-machine curve (docs/nag.md §7 Tier 3).
# Machine B ran the language plateau curve self-contained (its own vanilla and
# controls, SETUP.md §0); these three arms complete the SAME curve on THIS
# machine — plain lambda=3 (does the climbing axis ever turn over here too?)
# and the NAG lambda {1.5, 3} rungs against their uncapped twins. With both
# machines carrying the full 7-arm curve, every A<->B comparison is a
# direction-replication read on independent hardware (never a paired stat).
# The lambda=2 pair (allxt-temp20l20 / allxt-temp20-nagn-l20) is collected
# above. Analysis contrasts + the climbing-side DiD are already registered
# (analyze.py, commit 1c979f5). COST: ~5.1 h/arm at the axis's ~12 s/ep.
run allxt-temp20l30       --pladis-install --pladis-scale 3.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text
run allxt-temp20-nagn-l15 --pladis-install --pladis-scale 1.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-tau 2.5
run allxt-temp20-nagn-l30 --pladis-install --pladis-scale 3.0 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-tau 2.5

# 2026-08-30 (operator request) all-x-text sharp-softmax at lambda=2.5 — the rung
# between the axis's optimum (lambda=2, 87.90) and the plain lambda=3 that did not
# turn over (88.55, +3.19 vs vanilla Bonferroni*). Fills the top of the climbing
# axis's ladder at the spacing robot already has (allxt-temp20l25 there).
# Run with the cap-OFF probe: the rollout is bit-identical to the plain arm
# (verify_nag.py gate A/I; 3/3 episodes reproduced on original), and the
# per-episode R sidecars come for free — the first R census on THIS axis, at a
# dose where the plain ladder is still rising. COST: ~5.1 h.
run allxt-temp20l25 --pladis-install --pladis-scale 2.5 --pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text --pladis-nag-probe

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
