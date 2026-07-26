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

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
