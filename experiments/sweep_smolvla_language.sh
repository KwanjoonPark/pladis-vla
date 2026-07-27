#!/bin/bash
# SmolVLA (HuggingFaceVLA/smolvla_libero) x LIBERO-plus language axis, FULL
# curated set: four suites, 1,537 episodes per arm, seed-0 paired schedule —
# the SmolVLA counterpart of sweep_n17_language.sh (same axis, same pairing).
# One checkpoint serves all suites (per_suite=False in the registry).
#
# Loci (see pladis/attn_smolvla.py): CA layers key = [image|language|state],
# SA layers key = [prefix|suffix]. Arms below put lambda=1.5 (official
# recommended regime; the GR00T dose row peaked there) on each locus:
#   vanilla / base0 (hook, lambda=0, bit-parity control)
#   axt   = kind=text    (a x t)        axi = kind=image (a x i)
#   axs   = kind=state   (a x state-key: NEW cell, GR00T s-arms' dual)
#   axpfx = kind=prefix  (a x all cross columns = GR00T allxall analogue)
#   axself= kind=self    (a x action self-attn: NEW cell)
# PRE-LAUNCH GATES (run once before this driver, results reviewed):
#   verify_smolvla_hook.py (CPU) + delivery smoke + anchor exec-horizon pick.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
. experiments/load_machine_env.sh
pladis_require_clean_tree
SUITES="libero_10 libero_goal libero_object libero_spatial"

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/smolvla_lang_${tag}_${S}_eplog.tsv"
    echo "[sweep] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh --venv lerobot experiments/eval_arm.py \
      --model smolvla --suite "$S" --axis language --episodes 0 --seed 0 \
      --out "$out" \
      --video-dir "results/sweep/videos/smolvla_lang_${tag}_${S}" "$@" \
      > "results/sweep/smolvla_lang_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/smolvla_lang_${tag}_${S}.out"
  done
}

run vanilla
run base0  --pladis-install --pladis-scale 0
run axt    --pladis-install --pladis-scale 1.5 --pladis-kind text
run axi    --pladis-install --pladis-scale 1.5 --pladis-kind image
run axs    --pladis-install --pladis-scale 1.5 --pladis-kind state
run axpfx  --pladis-install --pladis-scale 1.5 --pladis-kind prefix
run axself --pladis-install --pladis-scale 1.5 --pladis-kind self

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
