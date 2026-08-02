#!/bin/bash
# GR00T N1.7 x LIBERO-plus NOISE axis (Sensor Noise), FULL curated set: four
# suites, 1,601 episodes per arm (10/goal/object/spatial = 449/379/422/351),
# seed-0 paired schedule. Obs-side corruption of the AGENTVIEW camera only:
# N 1-10 motion blur / 11-20 gaussian / 21-30 zoom / 31-40 fog / 41-50 glass
# (5 families x 10 severities). Corruption draws + fixture placement are
# pinned per episode (RUNTIME_RNG_AXES; verify_noise_axis.py ALL PASSED).
# Arms: the standard lambda=1 locus grid (base0 omitted — bit-identical to
# vanilla since the 07-20 lambda=0 SDPA delegation).
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
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

echo "[noise] ALL DONE $(date +%H:%M:%S)"
