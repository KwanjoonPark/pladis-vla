#!/bin/bash
# GR00T N1.7 x LIBERO-plus language axis, FULL curated set: all four suites,
# 1,537 episodes per arm (each curated variant exactly once, seed-0 schedule,
# paired across arms). Per-suite checkpoints from nvidia/GR00T-N1.7-LIBERO.
# Arms: vanilla, base0 (hook @ l=0), {state,action}x{text,image} + allxall @ l=1.
# Resume-safe at episode granularity (eval_arm skips episodes already logged).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
MODEL_ROOT=/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO
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

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
