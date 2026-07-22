#!/bin/bash
# OLD-BASIS base0 arm for the ROBOT axis (queued behind the main robot sweep).
#
# Old basis = the pre-07-20 eager-dense numeric path, so robot gets the same
# [fused vanilla | eager base0 | eager lambda=1] ladder as language/layout/
# original. On the current weight-space hook this path is reproduced BIT-FOR-BIT
# by --pladis-scale 1.0 --pladis-method softmax (sparse branch == dense branch
# -> blend collapses to eager dense; same fp32 softmax, same mask conversion).
#
# Three gates before the 1,550-episode arm:
#   1. main sweep finished: "ALL DONE" in results/sweep/driver_robot.out
#   2. pladis/attn_gr00t.py is still the WEIGHT-SPACE implementation (the
#      fused-anchored swap must wait until this arm completes)
#   3. 2-episode language parity vs the stored pre-07-20 base0 eplog:
#      episodes 0-1 of n17_lang_base0_libero_10 must match exactly
#      (task_name / success_once / n_steps) - proves "same basis" empirically.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL_ROOT=/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO
SUITES="libero_10 libero_goal libero_object libero_spatial"
DRIVER_OUT=results/sweep/driver_robot.out

echo "[base0] $(date '+%m-%d %H:%M:%S') waiting for main robot sweep (ALL DONE in $DRIVER_OUT)"
until grep -q "ALL DONE" "$DRIVER_OUT" 2>/dev/null; do sleep 300; done
echo "[base0] $(date '+%m-%d %H:%M:%S') main sweep ALL DONE confirmed"

if grep -q "fused-anchored" pladis/attn_gr00t.py; then
  echo "[base0] ABORT: attn_gr00t.py already swapped to fused-anchored; old-basis base0 needs the weight-space hook"
  exit 1
fi

PAR=results/sweep/robot_base0_parity_eplog.tsv
rm -f "$PAR"
echo "[base0] parity gate: 2 language eps vs stored pre-07-20 base0 ..."
bash experiments/run.sh experiments/eval_arm.py \
  --suite libero_10 --axis language --episodes 2 --seed 0 \
  --model-path "$MODEL_ROOT/libero_10" --out "$PAR" \
  --pladis-install --pladis-scale 1.0 --pladis-method softmax \
  > results/sweep/robot_base0_parity.out 2>&1
python3 - <<'EOF'
import csv, sys
def load(p):
    return {r["episode"]: r for r in csv.DictReader(open(p), delimiter="\t")}
new = load("results/sweep/robot_base0_parity_eplog.tsv")
old = load("results/sweep/n17_lang_base0_libero_10_eplog.tsv")
bad = []
for ep, r in new.items():
    o = old.get(ep)
    if o is None or any(r[k] != o[k] for k in ("task_name", "success_once", "n_steps")):
        bad.append(ep)
print(f"[base0] parity rows={len(new)} mismatches={len(bad)} {bad if bad else ''}", flush=True)
sys.exit(0 if new and not bad else 1)
EOF
if [ $? -ne 0 ]; then
  echo "[base0] ABORT: parity gate FAILED - method-softmax path is NOT the old base0 basis"
  exit 1
fi
echo "[base0] parity gate PASSED"

run() { # $1=tag, rest = pladis args  (mirrors sweep_n17_robot.sh)
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/n17_robot_${tag}_${S}_eplog.tsv"
    echo "[base0] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis robot --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_robot_${tag}_${S}" "$@" \
      > "results/sweep/n17_robot_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_robot_${tag}_${S}.out"
  done
}

run base0 --pladis-install --pladis-scale 1.0 --pladis-method softmax

echo "[base0] ALL DONE $(date '+%m-%d %H:%M:%S')"
