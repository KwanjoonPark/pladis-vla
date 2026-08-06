#!/bin/bash
# SmolVLA (lerobot/smolvla_libero, org-official ckpt) x LIBERO-plus language axis,
# FULL curated set: four suites, 1,537 episodes per arm, seed-0 paired schedule —
# the SmolVLA counterpart of sweep_n17_language.sh (same axis, same pairing).
# One checkpoint serves all suites (per_suite=False in the registry).
#
# Loci (see pladis/attn_smolvla.py): CA layers key = [image|language|state],
# SA layers key = [prefix|suffix]. Arms below put lambda=1.5 (official
# recommended regime; the GR00T dose row peaked there) on each locus:
#   vanilla
#   axt   = kind=text        (a x t, text block mass fixed)
#   axi   = kind=image       (a x i, image block mass fixed; mass may move
#                             BETWEEN cameras inside the image span)
#   axcam = kind=cams        (a x i split per camera: camera1/camera2 each
#                             sharpened with its OWN mass fixed)
#   axti  = kind=text-image  (text AND image blocks, each mass fixed — the
#                             maximal mass-preserving prefix intervention;
#                             replaces the old plain whole-row axpfx arm,
#                             operator decision 2026-08-02: every arm preserves
#                             modality mass)
# NO axs arm (2026-08-02): state is ONE prefix key token and mass preservation
# on a width-1 block is a bit-exact identity — install_pladis raises on it.
# NO axself arm in this round (operator arm list 2026-08-02); the driver is
# resume-safe, so it can be appended later without disturbing completed arms.
#
# NO base0 arm (dropped 2026-08-02): smolvla has no fused/eager duality — the one
# eager kernel (smolvlm_with_expert.py:504) makes the hook's lambda=0 the stock
# softmax op, bit-parity held by verify_smolvla_hook.py + the on-ckpt delivery
# smoke, so base0 == vanilla and would burn 1,537 x 4 episodes re-proving a gate
# assertion — the same call the pi05 track made (analysis/analyze.py MODELS note).
#
# Step caps are PER-SUITE, aligned to lerobot's own LIBERO eval protocol
# (lerobot envs/libero.py: spatial 280 / object 280 / goal 300 / long 520) —
# the 2026-08-02 anchor protocol uses the same caps (external validity; also
# roughly halves failure-episode wall time vs the old 720). Arms share the
# identical seed-0 schedule, so the cap cancels in every arm-vs-arm contrast.
# Instructions stay the BDDL parse (--instruction-source default): on the
# language axis the instruction IS the perturbation.
#
# PRE-LAUNCH GATES (run once before this driver, results reviewed):
#   verify_smolvla_hook.py (CPU) + official-ckpt delivery smoke + 2ep per-arm smoke.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
. experiments/load_machine_env.sh
pladis_require_clean_tree
SUITES="libero_10 libero_goal libero_object libero_spatial"

cap_of() { # lerobot envs/libero.py per-suite max_steps
  case "$1" in
    libero_spatial|libero_object) echo 280 ;;
    libero_goal)                  echo 300 ;;
    libero_10)                    echo 520 ;;
    *) echo "[sweep] unknown suite $1" >&2; exit 2 ;;
  esac
}

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/smolvla_lang_${tag}_${S}_eplog.tsv"
    echo "[sweep] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh --venv lerobot experiments/eval_arm.py \
      --model smolvla --suite "$S" --axis language --episodes 0 --seed 0 \
      --max-steps "$(cap_of "$S")" \
      --out "$out" \
      --video-dir "results/sweep/videos/smolvla_lang_${tag}_${S}" "$@" \
      > "results/sweep/smolvla_lang_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/smolvla_lang_${tag}_${S}.out"
  done
}

run vanilla
run axt   --pladis-install --pladis-scale 1.5 --pladis-kind text
run axi   --pladis-install --pladis-scale 1.5 --pladis-kind image
run axcam --pladis-install --pladis-scale 1.5 --pladis-kind cams
run axti  --pladis-install --pladis-scale 1.5 --pladis-kind text-image

# 2026-08-05 lambda=1.0 row (dose-mismatch fix, flagged 08-03): the unsuffixed
# arms above sit at lambda=1.5, unlike N1.7 whose primary row is 1.0 — append
# the 1.0 rung under "10"-suffixed tags so the tracks compare at matched dose
# and the 1.5 eplogs survive as the upper dose point.
run axt10   --pladis-install --pladis-scale 1.0 --pladis-kind text
run axi10   --pladis-install --pladis-scale 1.0 --pladis-kind image
run axcam10 --pladis-install --pladis-scale 1.0 --pladis-kind cams
run axti10  --pladis-install --pladis-scale 1.0 --pladis-kind text-image

# 2026-08-06 image-locus low-dose rung (operator request): axi's dose points
# are 1.5 and 1.0 only — probe BELOW the matched dose before opening the
# temperature row, testing whether the image-locus response is monotone in
# lambda or already saturated by 1.0.
run axi05 --pladis-install --pladis-scale 0.5 --pladis-kind image

echo "[sweep] ALL DONE $(date +%H:%M:%S)"
