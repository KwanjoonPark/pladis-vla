#!/bin/bash
# GR00T N1.7 x LIBERO-plus ROBOT-init axis, full curated set: 1,550 episodes
# per arm (10/goal/object/spatial = 393/409/398/350), each variant exactly
# once (init_state_id 0 = official num_trials=1 protocol), seed-0 schedule,
# paired across arms. Runtime axis: `_initstate_<k>` swaps in a Panda{k}
# robot class with perturbed init_qpos (levels 0.1-0.5); delivery = partial
# pose offset + persistent OSC nullspace bias (gates:
# experiments/verify_robot_axis.py, all passed 2026-07-20).
# Arms: vanilla + {state,action}x{text,image} + allxall @ l=1, then an
# OLD-BASIS base0 arm. lambda=0-as-base0 stays omitted (bit-identical to
# vanilla since the 2026-07-20 lambda=0 SDPA delegation, verify_base0_parity.py);
# the old-basis arm instead reproduces the pre-07-20 eager-dense path
# bit-for-bit via --pladis-scale 1.0 --pladis-method softmax (sparse branch ==
# dense branch -> blend collapses to eager dense), giving this axis the same
# [fused vanilla | eager base0 | eager lambda=1] ladder as language/layout.
# It is gated on (1) attn_gr00t.py still being the weight-space hook and
# (2) a 2-episode language parity check against the stored pre-07-20 base0
# eplog (requires sweep_n17_language.sh results).
# Resume-safe at episode granularity.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results/sweep
MODEL_ROOT=/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO
SUITES="libero_10 libero_goal libero_object libero_spatial"

run() { # $1=tag, rest = pladis args
  local tag="$1"; shift
  for S in $SUITES; do
    local out="results/sweep/n17_robot_${tag}_${S}_eplog.tsv"
    echo "[robot] === $tag / $S ($(date +%H:%M:%S)) ==="
    bash experiments/run.sh experiments/eval_arm.py \
      --suite "$S" --axis robot --episodes 0 --seed 0 \
      --model-path "$MODEL_ROOT/$S" --out "$out" \
      --video-dir "results/sweep/videos/n17_robot_${tag}_${S}" "$@" \
      > "results/sweep/n17_robot_${tag}_${S}.out" 2>&1
    tail -1 "results/sweep/n17_robot_${tag}_${S}.out"
  done
}

run vanilla
run actionximage --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind image
run actionxtext  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
run statextext   --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind text
run stateximage  --pladis-install --pladis-scale 1.0 --pladis-qgroup state  --pladis-kind image
run allxall      --pladis-install --pladis-scale 1.0 --pladis-qgroup all    --pladis-kind all

# ---- OLD-BASIS base0 (eager-dense) arm, gated ----
if grep -q "fused-anchored" pladis/attn_gr00t.py; then
  echo "[robot] ABORT base0: attn_gr00t.py is fused-anchored; old-basis base0 needs the weight-space hook"
  exit 1
fi
REF=results/sweep/n17_lang_base0_libero_10_eplog.tsv
if [ ! -f "$REF" ]; then
  echo "[robot] ABORT base0: parity reference $REF missing (run sweep_n17_language.sh first)"
  exit 1
fi
# Parity gate: episodes 0-1 of the language axis must reproduce the stored
# pre-07-20 base0 eplog exactly (proves method-softmax == old eager-dense).
PAR=results/sweep/robot_base0_parity_eplog.tsv
rm -f "$PAR"
echo "[robot] base0 parity gate: 2 language eps vs stored pre-07-20 base0 ..."
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
bad = [e for e, r in new.items()
       if old.get(e) is None
       or any(r[k] != old[e][k] for k in ("task_name", "success_once", "n_steps"))]
print(f"[robot] base0 parity rows={len(new)} mismatches={len(bad)}", flush=True)
sys.exit(0 if new and not bad else 1)
EOF
if [ $? -ne 0 ]; then
  echo "[robot] ABORT base0: parity gate FAILED - method-softmax path is NOT the old base0 basis"
  exit 1
fi
echo "[robot] base0 parity gate PASSED"
run base0 --pladis-install --pladis-scale 1.0 --pladis-method softmax

echo "[robot] ALL DONE $(date +%H:%M:%S)"
