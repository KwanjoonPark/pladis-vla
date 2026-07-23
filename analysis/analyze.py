"""Unified post-hoc analysis of the n17 sweeps.

  python analysis/analyze.py --layout     # n17_layout_* (7 arms x 1,525 eps)
  python analysis/analyze.py --language   # n17_lang_*   (7 arms x 1,537 eps)
  python analysis/analyze.py --robot      # n17_robot_*  (7 arms x 1,550 eps)

Pairing: identical seed-0 schedule across arms -> pair by (suite, episode);
task_name equality is asserted. Test = paired McNemar, z = (n01-n10)/sqrt(disc).
Baseline severity uses n17_orig_vanilla_* (per-base-task mean over init 0-9).
Read-only; writes nothing.

Metric: `success_once`, the protocol's primary (README S2). Rollouts stop on
first contact with success, so success_at_end is evaluated at that same sim
state and the two columns agree row-for-row; any disagreement is a harness
bug and is printed as a WARN rather than silently absorbed.
"""
import argparse, csv, math, re
from collections import defaultdict
from pathlib import Path

SWEEP = Path(__file__).resolve().parent.parent / "results" / "sweep"
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
ARMS = ["vanilla", "base0", "actionxtext", "actionximage",
        "statextext", "stateximage", "allxall"]
KEY_CONTRASTS = [  # locus + each action arm vs both baselines + gates
    ("actionxtext", "actionximage"),
    ("actionxtext", "base0"), ("actionxtext", "vanilla"),
    ("actionximage", "base0"), ("actionximage", "vanilla"),
    ("statextext", "stateximage"),
    ("allxall", "base0"), ("base0", "vanilla"),
]

def layout_cat(task_name):
    if "_add_" in task_name or task_name.endswith("_add"):
        return "add"
    if "_moved_level" in task_name:
        return "moved_level"
    if re.search(r"_level\d+_sample\d+", task_name):
        return "level_sample"
    return "UNKNOWN"

def robot_level(task_name):
    """Perturbation level from the `_initstate_<k>` tail (k=1..500):
    hundreds digit -> L1..L5 = init_qpos noise strength 0.1..0.5."""
    k = int(re.search(r"_initstate_(\d+)", task_name).group(1))
    return f"L{(k - 1) // 100 + 1}"

AXES = {
    "layout": {"prefix": "n17_layout", "cat": layout_cat,
               "cats": ["add", "level_sample", "moved_level"]},
    "language": {"prefix": "n17_lang", "cat": None, "cats": []},
    "robot": {"prefix": "n17_robot", "cat": robot_level,
              "cats": ["L1", "L2", "L3", "L4", "L5"]},
}

def load(prefix, arm):
    eps = {}
    for s in SUITES:
        p = SWEEP / f"{prefix}_{arm}_{s}_eplog.tsv"
        for r in csv.DictReader(open(p), delimiter="\t"):
            r["suite"], r["succ"] = s, int(r["success_once"])
            if r["success_at_end"] != r["success_once"]:
                print(f"WARN succ_once!=at_end {arm} {s} ep{r['episode']}")
            eps[(s, int(r["episode"]))] = r
    return eps

def mcnemar(a, b, keys):
    n01 = sum(1 for k in keys if a[k]["succ"] and not b[k]["succ"])
    n10 = sum(1 for k in keys if not a[k]["succ"] and b[k]["succ"])
    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0
    z = (n01 - n10) / math.sqrt(n01 + n10)
    return n01, n10, z, math.erfc(abs(z) / math.sqrt(2))

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for name in AXES:
        g.add_argument(f"--{name}", action="store_true")
    args = ap.parse_args()
    axis = next(a for a in AXES if getattr(args, a))
    cfg = AXES[axis]

    data = {arm: load(cfg["prefix"], arm) for arm in ARMS}
    keys = sorted(data["vanilla"].keys())
    for arm in ARMS:  # schedule identity across arms
        assert set(data[arm].keys()) == set(keys), f"episode-set mismatch: {arm}"
        for k in keys:
            assert data[arm][k]["task_name"] == data["vanilla"][k]["task_name"], (arm, k)

    cats = {}
    if cfg["cat"]:
        cats = {k: cfg["cat"](data["vanilla"][k]["task_name"]) for k in keys}
        assert "UNKNOWN" not in cats.values()

    sr = lambda arm, ks: 100 * sum(data[arm][k]["succ"] for k in ks) / len(ks)
    per_suite = {s: [k for k in keys if k[0] == s] for s in SUITES}
    print(f"[{axis}] episodes/arm={len(keys)}  per-suite:",
          {s.replace('libero_', ''): len(v) for s, v in per_suite.items()},
          ({c: sum(1 for v in cats.values() if v == c) for c in cfg["cats"]}
           if cats else ""))

    print(f"\n== SR (success_once, %) — pooled + per suite ==")
    print(f"  {'arm':13s}{'pooled':>8s}"
          + "".join(f"{s.replace('libero_', ''):>9s}" for s in SUITES))
    for arm in ARMS:
        row = "".join(f"{sr(arm, per_suite[s]):9.1f}" for s in SUITES)
        print(f"  {arm:13s}{sr(arm, keys):8.2f}{row}")

    if cats:
        print(f"\n== per-category SR ==")
        print(f"  {'arm':13s}" + "".join(f"{c:>13s}" for c in cfg["cats"]))
        for arm in ARMS:
            row = "".join(f"{sr(arm, [k for k in keys if cats[k]==c]):13.1f}"
                          for c in cfg["cats"])
            print(f"  {arm:13s}{row}")

    # Bonferroni over the pooled contrast family reported below (README S6.3
    # promises the correction is noted, so compute it rather than leave it to
    # the reader): m = number of pooled contrasts tested here.
    m = len(KEY_CONTRASTS)
    print(f"\n== paired McNemar, pooled (Bonferroni m={m}, alpha=.05 -> "
          f"p<{0.05 / m:.4f}) ==")
    for a, b in KEY_CONTRASTS:
        n01, n10, z, p = mcnemar(data[a], data[b], keys)
        d = sr(a, keys) - sr(b, keys)
        mark = "*" if p < 0.05 / m else (" " if p >= 0.05 else ".")
        print(f"  {a:13s} - {b:13s} {d:+6.2f}pp  disc {n01:3d}:{n10:3d}"
              f"  z={z:+5.2f}  p={p:.4g}  p_bonf={min(1.0, p * m):.4g} {mark}")
    print("  (* survives Bonferroni; . nominal p<.05 only)")

    print("\n== key contrasts per suite ==")
    for a, b in [("actionxtext", "actionximage"), ("actionxtext", "base0"),
                 ("actionximage", "base0"), ("base0", "vanilla")]:
        print(f"  {a} - {b}:")
        for s in SUITES:
            n01, n10, z, p = mcnemar(data[a], data[b], per_suite[s])
            d = sr(a, per_suite[s]) - sr(b, per_suite[s])
            print(f"    {s:15s} {d:+6.2f}pp  disc {n01:3d}:{n10:3d}"
                  f"  z={z:+5.2f}  p={p:.4g}")

    if cats:
        print("\n== key contrasts per category ==")
        for a, b in [("actionxtext", "actionximage"), ("actionxtext", "base0"),
                     ("actionximage", "base0")]:
            print(f"  {a} - {b}:")
            for c in cfg["cats"]:
                ks = [k for k in keys if cats[k] == c]
                n01, n10, z, p = mcnemar(data[a], data[b], ks)
                d = sr(a, ks) - sr(b, ks)
                print(f"    {c:13s} {d:+6.2f}pp  disc {n01:3d}:{n10:3d}"
                      f"  z={z:+5.2f}  p={p:.4g}")

    orig = defaultdict(list)
    for s in SUITES:
        p = SWEEP / f"n17_orig_vanilla_{s}_eplog.tsv"
        for r in csv.DictReader(open(p), delimiter="\t"):
            orig[(s, r["base_task"])].append(int(r["success_once"]))
    orig_sr = {bt: 100 * sum(v) / len(v) for bt, v in orig.items()}
    print(f"\n== perturbation severity: {axis} vanilla vs original vanilla ==")
    for s in SUITES:
        ks = per_suite[s]
        o = sum(orig_sr[(s, data["vanilla"][k]["base_task"])] for k in ks) / len(ks)
        print(f"  {s:15s} orig(task-matched) {o:5.1f}  {axis} {sr('vanilla', ks):5.1f}"
              f"  drop {sr('vanilla', ks) - o:+6.1f}pp")

    print("\n== biggest per-task a_t vs a_i deltas (n>=8 variants, |delta|>=20pp) ==")
    bt_keys = defaultdict(list)
    for k in keys:
        bt_keys[(k[0], data["vanilla"][k]["base_task"])].append(k)
    rows = []
    for bt, ks in bt_keys.items():
        if len(ks) < 8:
            continue
        d = sr("actionxtext", ks) - sr("actionximage", ks)
        if abs(d) >= 20:
            rows.append((d, bt, len(ks), sr("actionxtext", ks), sr("actionximage", ks)))
    for d, bt, n, at, ai in sorted(rows, reverse=True):
        print(f"  {d:+6.1f}pp (n={n:2d}, a_t {at:4.1f} a_i {ai:4.1f})"
              f" {bt[0]}:{bt[1][:70]}")
    if not rows:
        print("  (none)")

if __name__ == "__main__":
    main()
