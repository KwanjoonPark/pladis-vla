"""Unified post-hoc analysis of the sweeps, for both model tracks.

  python analysis/analyze.py --layout                # n17_layout_*  (7 arms x 1,525 eps)
  python analysis/analyze.py --language              # n17_lang_*    (7 arms x 1,537 eps)
  python analysis/analyze.py --robot                 # n17_robot_*   (7 arms x 1,550 eps)
  python analysis/analyze.py --model pi05 --language  # pi05_lang_*  (3 arms x 1,537 eps)

Pairing: identical seed-0 schedule across arms -> pair by (suite, episode);
task_name equality is asserted. Test = paired McNemar, z = (n01-n10)/sqrt(disc).
Baseline severity uses <model>_orig_vanilla_* (per-base-task mean over init 0-9).
Read-only; writes nothing.

The two tracks study different design spaces, so the arm names and contrasts live in
MODELS below rather than being hardcoded:
  * n17  — query group x key modality, a 2x2 grid ({state,action} x {text,image}).
  * pi05 — key sub-block only. pi0.5's suffix is action-only (pi05_libero sets
    discrete_state_input=False), so the query axis collapses and every arm is
    action-row x <keys>. `text` is the direct port of the official FLUX intervention;
    `image` is the contrast with no upstream precedent.

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

MODELS = {
    "n17": {
        "tag": "n17",
        "arms": ["vanilla", "base0", "actionxtext", "actionximage",
                 "statextext", "stateximage", "allxall"],
        "key_contrasts": [  # locus + each action arm vs both baselines + gates
            ("actionxtext", "actionximage"),
            ("actionxtext", "base0"), ("actionxtext", "vanilla"),
            ("actionximage", "base0"), ("actionximage", "vanilla"),
            ("statextext", "stateximage"),
            ("allxall", "base0"), ("base0", "vanilla"),
        ],
        "locus_pair": ("actionxtext", "actionximage"),
        "suite_contrasts": [("actionxtext", "actionximage"), ("actionxtext", "base0"),
                            ("actionximage", "base0"), ("base0", "vanilla")],
        "cat_contrasts": [("actionxtext", "actionximage"), ("actionxtext", "base0"),
                          ("actionximage", "base0")],
    },
    # pi0.5 phase 1: 3 arms. base0 is NOT an arm here — it is bit-identical to vanilla
    # (verify_pi05_hook.py gate A) and is covered by verify_pi05_parity.py instead, the
    # same call sweep_n17_robot.sh:9-12 made for the n17 robot axis. Neither is there an
    # eager-dense control: pi0.5's vanilla is already on the eager path
    # (pi0_pytorch.py:447), so there is no fused-vs-eager kernel term to absorb and
    # vanilla IS the numeric control (quantified by verify_pi05_parity.py check (c)).
    "pi05": {
        "tag": "pi05",
        "arms": ["vanilla", "text", "image"],
        "key_contrasts": [
            ("text", "image"),                        # THE locus contrast
            ("text", "vanilla"), ("image", "vanilla"),
        ],
        "locus_pair": ("text", "image"),
        "suite_contrasts": [("text", "image"), ("text", "vanilla"), ("image", "vanilla")],
        "cat_contrasts": [("text", "image")],
        # phase 2 candidates; skipped until all four suite eplogs exist
        "extra_arms": ["prefix", "all", "text15", "image15", "text20", "image20"],
        "extra_contrasts": [
            ("prefix", "text"), ("prefix", "image"), ("prefix", "vanilla"),
            ("all", "prefix"), ("all", "vanilla"),
            ("text15", "vanilla"), ("text15", "text"),
            ("image15", "vanilla"), ("image15", "image"),
            ("text15", "image15"),
            ("text20", "vanilla"), ("text20", "text15"),
            ("image20", "vanilla"), ("image20", "image15"),
            ("text20", "image20"),
        ],
    },
}

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

# Axis-level metadata is model-INDEPENDENT (it describes the perturbation, not the
# intervention): `tag` builds the eplog prefix as f"{model_tag}_{axis_tag}", and cat/cats
# are the per-category breakdown. `extra_arms`/`extra_contrasts` here are keyed BY MODEL,
# because they name concrete arm tags.
AXES = {
    "layout": {"tag": "layout", "cat": layout_cat,
               "cats": ["add", "level_sample", "moved_level"]},
    "language": {"tag": "lang", "cat": None, "cats": [],
                 # extra arms are skipped until all four suite eplogs exist.
                 # 07-22 composition arms: allxtext = {action,state}xtext;
                 #   axt-sxi = actionxtext+stateximage.
                 # 07-23 lambda=1.5 row (official recommended regime) over the
                 #   text-locus arms AND the remaining base cells:
                 #   {actionxtext,allxtext,axt-sxi,actionximage,stateximage,statextext}15.
                 # 07-23 temperature control (softmax(beta*l), tau=1/beta) at
                 #   lambda=1 on a-x-t: axt-temp{15,20,30}.
                 # 07-26 lambda=2.0 row over the four base cells:
                 #   {actionxtext,actionximage,stateximage,statextext}20.
                 "extra_arms": {"n17": [
                                "allxtext", "axt-sxi",
                                "actionxtext15", "allxtext15", "axt-sxi15",
                                "actionximage15", "stateximage15", "statextext15",
                                "axt-temp15", "axt-temp20", "axt-temp30",
                                "actionxtext20", "actionximage20",
                                "stateximage20", "statextext20"]},
                 "extra_contrasts": {"n17": [
                     ("allxtext", "actionxtext"), ("allxtext", "vanilla"),
                     ("axt-sxi", "actionxtext"), ("axt-sxi", "vanilla"),
                     ("axt-sxi", "stateximage"),
                     # dose-response: each lambda=1.5 arm vs vanilla and vs
                     # its lambda=1 counterpart, plus the locus contrast at 1.5
                     ("actionxtext15", "vanilla"), ("actionxtext15", "actionxtext"),
                     ("allxtext15", "vanilla"), ("allxtext15", "allxtext"),
                     ("axt-sxi15", "vanilla"), ("axt-sxi15", "axt-sxi"),
                     ("actionximage15", "vanilla"), ("actionximage15", "actionximage"),
                     ("stateximage15", "vanilla"), ("stateximage15", "stateximage"),
                     ("statextext15", "vanilla"), ("statextext15", "statextext"),
                     ("actionxtext15", "actionximage15"),
                     # temperature vs entmax at the same locus: the paper's
                     # "exact zeros are necessary" claim, tested head-to-head
                     ("axt-temp15", "vanilla"), ("axt-temp15", "actionxtext"),
                     ("axt-temp20", "vanilla"), ("axt-temp20", "actionxtext"),
                     ("axt-temp30", "vanilla"), ("axt-temp30", "actionxtext"),
                     # lambda=2.0 dose row: each arm vs vanilla and vs its
                     # lambda=1/1.5 counterparts, plus the locus contrast at 2.0
                     ("actionxtext20", "vanilla"), ("actionxtext20", "actionxtext"),
                     ("actionxtext20", "actionxtext15"),
                     ("actionximage20", "vanilla"), ("actionximage20", "actionximage"),
                     ("actionximage20", "actionximage15"),
                     ("stateximage20", "vanilla"), ("stateximage20", "stateximage"),
                     ("stateximage20", "stateximage15"),
                     ("statextext20", "vanilla"), ("statextext20", "statextext"),
                     ("statextext20", "statextext15"),
                     ("actionxtext20", "actionximage20"),
                 ]}},
    "robot": {"tag": "robot", "cat": robot_level,
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
    ap.add_argument("--model", default="n17", choices=sorted(MODELS),
                    help="model track (selects arm names, contrasts and eplog prefix)")
    g = ap.add_mutually_exclusive_group(required=True)
    for name in AXES:
        g.add_argument(f"--{name}", action="store_true")
    args = ap.parse_args()
    axis = next(a for a in AXES if getattr(args, a))
    cfg = AXES[axis]
    mcfg = MODELS[args.model]
    prefix = f"{mcfg['tag']}_{cfg['tag']}"

    arms = list(mcfg["arms"])
    # model-level extras (phase-2 arms of that track) + axis-level extras for this model
    extra_arms = list(mcfg.get("extra_arms", [])) + list(
        cfg.get("extra_arms", {}).get(args.model, [])
    )
    for a in extra_arms:
        if all((SWEEP / f"{prefix}_{a}_{s}_eplog.tsv").exists() for s in SUITES):
            arms.append(a)
        else:
            print(f"[note] extra arm {a!r}: eplogs missing/incomplete, skipped")

    data = {arm: load(prefix, arm) for arm in arms}
    keys = sorted(data["vanilla"].keys())
    for arm in arms:  # schedule identity across arms
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
    for arm in arms:
        row = "".join(f"{sr(arm, per_suite[s]):9.1f}" for s in SUITES)
        print(f"  {arm:13s}{sr(arm, keys):8.2f}{row}")

    if cats:
        print(f"\n== per-category SR ==")
        print(f"  {'arm':13s}" + "".join(f"{c:>13s}" for c in cfg["cats"]))
        for arm in arms:
            row = "".join(f"{sr(arm, [k for k in keys if cats[k]==c]):13.1f}"
                          for c in cfg["cats"])
            print(f"  {arm:13s}{row}")

    # Bonferroni over the pooled contrast family reported below (README S6.3
    # promises the correction is noted, so compute it rather than leave it to
    # the reader): m = number of pooled contrasts tested here, INCLUDING any
    # axis-specific extra contrasts.
    contrasts = list(mcfg["key_contrasts"]) + [
        c for c in list(mcfg.get("extra_contrasts", []))
        + list(cfg.get("extra_contrasts", {}).get(args.model, []))
        if c[0] in arms and c[1] in arms
    ]
    m = len(contrasts)
    print(f"\n== paired McNemar, pooled (Bonferroni m={m}, alpha=.05 -> "
          f"p<{0.05 / m:.4f}) ==")
    for a, b in contrasts:
        n01, n10, z, p = mcnemar(data[a], data[b], keys)
        d = sr(a, keys) - sr(b, keys)
        mark = "*" if p < 0.05 / m else (" " if p >= 0.05 else ".")
        print(f"  {a:13s} - {b:13s} {d:+6.2f}pp  disc {n01:3d}:{n10:3d}"
              f"  z={z:+5.2f}  p={p:.4g}  p_bonf={min(1.0, p * m):.4g} {mark}")
    print("  (* survives Bonferroni; . nominal p<.05 only)")

    print("\n== key contrasts per suite ==")
    for a, b in [c for c in mcfg["suite_contrasts"] if c[0] in arms and c[1] in arms]:
        print(f"  {a} - {b}:")
        for s in SUITES:
            n01, n10, z, p = mcnemar(data[a], data[b], per_suite[s])
            d = sr(a, per_suite[s]) - sr(b, per_suite[s])
            print(f"    {s:15s} {d:+6.2f}pp  disc {n01:3d}:{n10:3d}"
                  f"  z={z:+5.2f}  p={p:.4g}")

    if cats:
        print("\n== key contrasts per category ==")
        for a, b in [c for c in mcfg["cat_contrasts"] if c[0] in arms and c[1] in arms]:
            print(f"  {a} - {b}:")
            for c in cfg["cats"]:
                ks = [k for k in keys if cats[k] == c]
                n01, n10, z, p = mcnemar(data[a], data[b], ks)
                d = sr(a, ks) - sr(b, ks)
                print(f"    {c:13s} {d:+6.2f}pp  disc {n01:3d}:{n10:3d}"
                      f"  z={z:+5.2f}  p={p:.4g}")

    # Severity needs the axis=none reference sweep of the SAME model. Guarded rather than
    # assumed: without the guard a missing file makes the whole analysis unusable until
    # the original sweep finishes, when everything above it is already valid.
    orig_paths = [SWEEP / f"{mcfg['tag']}_orig_vanilla_{s}_eplog.tsv" for s in SUITES]
    if not all(p.exists() for p in orig_paths):
        missing = [p.name for p in orig_paths if not p.exists()]
        print(f"\n[note] severity baseline skipped: missing {missing} "
              f"(run experiments/sweep_{mcfg['tag']}_original.sh)")
    else:
        orig = defaultdict(list)
        for s, p in zip(SUITES, orig_paths):
            for r in csv.DictReader(open(p), delimiter="\t"):
                orig[(s, r["base_task"])].append(int(r["success_once"]))
        orig_sr = {bt: 100 * sum(v) / len(v) for bt, v in orig.items()}
        print(f"\n== perturbation severity: {axis} vanilla vs original vanilla ==")
        for s in SUITES:
            ks = per_suite[s]
            o = sum(orig_sr[(s, data["vanilla"][k]["base_task"])] for k in ks) / len(ks)
            print(f"  {s:15s} orig(task-matched) {o:5.1f}  {axis} {sr('vanilla', ks):5.1f}"
                  f"  drop {sr('vanilla', ks) - o:+6.1f}pp")

    lo_a, lo_b = mcfg["locus_pair"]
    print(f"\n== biggest per-task {lo_a} vs {lo_b} deltas "
          f"(n>=8 variants, |delta|>=20pp) ==")
    bt_keys = defaultdict(list)
    for k in keys:
        bt_keys[(k[0], data["vanilla"][k]["base_task"])].append(k)
    rows = []
    for bt, ks in bt_keys.items():
        if len(ks) < 8:
            continue
        d = sr(lo_a, ks) - sr(lo_b, ks)
        if abs(d) >= 20:
            rows.append((d, bt, len(ks), sr(lo_a, ks), sr(lo_b, ks)))
    for d, bt, n, at, ai in sorted(rows, reverse=True):
        print(f"  {d:+6.1f}pp (n={n:2d}, {lo_a} {at:4.1f} {lo_b} {ai:4.1f})"
              f" {bt[0]}:{bt[1][:70]}")
    if not rows:
        print("  (none)")

if __name__ == "__main__":
    main()
