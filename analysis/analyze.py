"""Unified post-hoc analysis of the sweeps, for both model tracks.

  python analysis/analyze.py --layout                # n17_layout_*  (7 arms x 1,525 eps)
  python analysis/analyze.py --language              # n17_lang_*    (7 arms x 1,537 eps)
  python analysis/analyze.py --robot                 # n17_robot_*   (7 arms x 1,550 eps)
  python analysis/analyze.py --model pi05 --language  # pi05_lang_*  (4 arms x 1,537 eps)
  python analysis/analyze.py --model smolvla --language  # smolvla_lang_* (6 arms)

Pairing: identical seed-0 schedule across arms -> pair by (suite, episode);
task_name equality is asserted. Test = paired McNemar, z = (n01-n10)/sqrt(disc).
Baseline severity uses <model>_orig_vanilla_* (per-base-task mean over init 0-9).
Read-only; writes nothing.

The tracks study different design spaces, so the arm names and contrasts live in
MODELS below rather than being hardcoded:
  * n17  — query group x key modality, a 2x2 grid ({state,action} x {text,image}).
  * pi05 — key sub-block only. pi0.5's suffix is action-only (pi05_libero sets
    discrete_state_input=False), so the query axis collapses and every arm is
    action-row x <keys>. `text` is the direct port of the official FLUX intervention;
    `image` is the contrast with no upstream precedent.
  * smolvla — key sub-block over the CA row [image|language|state] (query axis
    collapses like pi05), plus `self` = the SA suffix rows, a locus the other
    tracks cannot express (their action self-attention is not hooked).

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
    # pi0.5: VANILLA IS THE REFERENCE (operator decision 2026-07-29). base0 is not an arm
    # — it is bit-identical to vanilla (verify_pi05_hook.py gate A, re-confirmed
    # end-to-end by verify_pi05_parity.py check (b)) and would burn 1,537 episodes
    # re-proving a 10-episode assertion; the same call sweep_n17_robot.sh:9-12 made for
    # the n17 robot axis.
    #
    # base_dense is demoted to an EXTRA arm rather than removed, because its λ=1 eplogs
    # (4 suites × 1,537 eps) are already collected and are the only direct measurement of
    # the λ>0 numeric term. What they measured: verify_pi05_parity.py check (c) found
    # 0.0000% of bf16 attention elements differing at module level, yet dense diverged
    # from vanilla in ALL 10 rollout episodes, and at sweep scale base_dense landed
    # -0.91pp below vanilla — float32 reassociation inside the λ>0 branch, below bf16
    # resolution per call and amplified by ~45k closed-loop attention calls per episode.
    #
    # Consequence for how the primary contrasts read: `text - vanilla` and
    # `image - vanilla` carry that term alongside the intervention, so a beneficial locus
    # effect is understated there (at λ=1, text - vanilla was +0.33pp against
    # text - base_dense +1.24pp). The LOCUS PAIR is unaffected — both arms run the
    # identical λ>0 path, so the term cancels within the pair — which is why
    # `text - image` is the primary contrast.
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
        # Extra arms are skipped until all four of their suite eplogs exist, so the driver
        # can be appended to while a campaign is running.
        "extra_arms": ["base_dense", "prefix", "all",
                       "text15", "image15", "text20", "image20"],
        "extra_contrasts": [
            ("base_dense", "vanilla"),          # size of the λ>0 numeric term (λ=1)
            ("text", "base_dense"), ("image", "base_dense"),
            ("prefix", "text"), ("prefix", "image"), ("prefix", "vanilla"),
            ("all", "prefix"), ("all", "vanilla"),
            # λ=1.5 — locus pair first, then vanilla and the dose step from λ=1
            ("text15", "image15"),
            ("text15", "vanilla"), ("image15", "vanilla"),
            ("text15", "text"), ("image15", "image"),
            # λ=2.0 — same, stepping from λ=1.5
            ("text20", "image20"),
            ("text20", "vanilla"), ("image20", "vanilla"),
            ("text20", "text15"), ("image20", "image15"),
        ],
    },
    # smolvla: like pi05, VANILLA IS THE REFERENCE and there is NO base0 arm — the track
    # has a single eager attention kernel (no SDPA anywhere in smolvlm_with_expert.py),
    # so the hook's λ=0 is the stock softmax op (verify_smolvla_hook.py bit-parity gate
    # + the on-ckpt delivery smoke); a base0 arm would burn 1,537×4 episodes re-proving
    # a gate assertion (dropped 2026-08-02, mirroring the pi05 decision above).
    # No qgroup axis either (suffix is action-only): every arm is action-row × <keys>,
    # tags from sweep_smolvla_language.sh. All arms are MASS-PRESERVING (operator
    # decision 2026-08-02 — the plain whole-row axpfx was dropped for moving mass
    # across modality borders): axt/axi is the cross-model locus pair (GR00T a×t/a×i,
    # pi05 text/image); axcam = image sharpened per camera (each camera's mass fixed —
    # vs axi, which lets mass move between cameras); axti = text+image each mass-fixed
    # (the maximal mass-preserving prefix arm; state is width 1 where MP = identity).
    # No axs arm: state is ONE prefix key token and the mass-preserving blend on a
    # width-1 block is a bit-exact identity (attn_smolvla install guard, 2026-08-02).
    "smolvla": {
        "tag": "smolvla",
        "arms": ["vanilla", "axt", "axi", "axcam", "axti"],
        "key_contrasts": [
            ("axt", "axi"),                       # THE locus contrast
            ("axt", "vanilla"), ("axi", "vanilla"),
            ("axcam", "vanilla"), ("axcam", "axi"),   # per-camera vs whole-image MP
            ("axti", "vanilla"), ("axti", "axt"),     # composition vs its parts
            ("axti", "axi"),
        ],
        "locus_pair": ("axt", "axi"),
        "suite_contrasts": [("axt", "axi"), ("axt", "vanilla"), ("axi", "vanilla")],
        "cat_contrasts": [("axt", "axi")],
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

def noise_cat(task_name):
    """Corruption family from the `_noise_<N>` tail (N=1..50): decade ->
    family, severity = N within decade (env_wrapper.py:283-305)."""
    n = int(re.search(r"_noise_(\d+)", task_name).group(1))
    return ["motion", "gauss", "zoom", "fog", "glass"][(n - 1) // 10]

# Axis-level metadata is model-INDEPENDENT (it describes the perturbation, not the
# intervention): `tag` builds the eplog prefix as f"{model_tag}_{axis_tag}", and cat/cats
# are the per-category breakdown. `extra_arms`/`extra_contrasts` here are keyed BY MODEL,
# because they name concrete arm tags.
AXES = {
    "layout": {"tag": "layout", "cat": layout_cat,
               "cats": ["add", "level_sample", "moved_level"],
               # 07-28 text-locus dose row (entmax), mirroring language.
               # 08-03 s-x-i / composite dose row: {stateximage}{15,20} +
               #   axt-sxi{15,20} (composite's first appearance on this axis —
               #   no lambda=1 counterpart, so its ladder anchors at 1.5).
               "extra_arms": {"n17": [
                              "actionxtext15", "actionxtext20",
                              "allxtext", "allxtext15", "allxtext20",
                              "stateximage15", "stateximage20",
                              "axt-sxi15", "axt-sxi20"]},
               "extra_contrasts": {"n17": [
                   ("actionxtext15", "vanilla"), ("actionxtext15", "actionxtext"),
                   ("actionxtext20", "vanilla"), ("actionxtext20", "actionxtext15"),
                   ("allxtext", "vanilla"), ("allxtext", "actionxtext"),
                   ("allxtext15", "vanilla"), ("allxtext15", "allxtext"),
                   ("allxtext20", "vanilla"), ("allxtext20", "allxtext15"),
                   ("stateximage15", "vanilla"), ("stateximage15", "stateximage"),
                   ("stateximage20", "vanilla"), ("stateximage20", "stateximage15"),
                   ("axt-sxi15", "vanilla"), ("axt-sxi15", "actionxtext15"),
                   ("axt-sxi15", "stateximage15"),
                   ("axt-sxi20", "vanilla"), ("axt-sxi20", "axt-sxi15"),
                   ("axt-sxi20", "actionxtext20"),
               ]}},
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
                 # 07-28 sharp-softmax dose row (beta=2, ent15-matched): the
                 #   lambda ladder at both text loci — axt-temp20l{15,20} and
                 #   allxt-temp20{,l15,l20} — head-to-head vs the entmax
                 #   counterparts (zeros-vs-sharpening in the extrapolation regime).
                 # 08-03 lambda=2.0 composite-text completion: {allxtext,axt-sxi}20
                 #   — every text locus now carries the full 1.0/1.5/2.0 ladder.
                 "extra_arms": {"n17": [
                                "allxtext", "axt-sxi",
                                "actionxtext15", "allxtext15", "axt-sxi15",
                                "actionximage15", "stateximage15", "statextext15",
                                "axt-temp15", "axt-temp20", "axt-temp30",
                                "actionxtext20", "actionximage20",
                                "stateximage20", "statextext20",
                                "allxtext20", "axt-sxi20",
                                "axt-temp20l15", "axt-temp20l20",
                                "allxt-temp20", "allxt-temp20l15", "allxt-temp20l20"]},
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
                     # 08-03 composite-text 2.0 rung: same ladder contrasts as
                     # the base cells (vs vanilla, vs lambda=1, vs lambda=1.5)
                     ("allxtext20", "vanilla"), ("allxtext20", "allxtext"),
                     ("allxtext20", "allxtext15"),
                     ("axt-sxi20", "vanilla"), ("axt-sxi20", "axt-sxi"),
                     ("axt-sxi20", "axt-sxi15"),
                     # sharp-softmax dose row: each temp arm vs vanilla, vs its
                     # entmax counterpart (zeros head-to-head at matched dose),
                     # and the within-temp dose/locus neighbors
                     ("axt-temp20l15", "vanilla"), ("axt-temp20l15", "actionxtext15"),
                     ("axt-temp20l15", "axt-temp20"),
                     ("axt-temp20l20", "vanilla"), ("axt-temp20l20", "actionxtext20"),
                     ("axt-temp20l20", "axt-temp20l15"),
                     ("allxt-temp20", "vanilla"), ("allxt-temp20", "allxtext"),
                     ("allxt-temp20l15", "vanilla"), ("allxt-temp20l15", "allxtext15"),
                     ("allxt-temp20l15", "allxt-temp20"),
                     ("allxt-temp20l20", "vanilla"), ("allxt-temp20l20", "allxt-temp20l15"),
                     ("allxt-temp20l15", "axt-temp20l15"),
                 ]}},
    "noise": {"tag": "noise", "cat": noise_cat,
              "cats": ["motion", "gauss", "zoom", "fog", "glass"]},
    "robot": {"tag": "robot", "cat": robot_level,
              "cats": ["L1", "L2", "L3", "L4", "L5"],
              # 07-28 text-locus dose row (entmax), mirroring language.
              # 08-05 far-extrapolation rungs {2.5, 3.0} at both text loci —
              #   the 07-28 row was NULL through lambda=2.0; ladder contrasts
              #   step from the lambda=2.0 arms.
              "extra_arms": {"n17": [
                             "actionxtext15", "actionxtext20",
                             "allxtext", "allxtext15", "allxtext20",
                             "actionxtext25", "actionxtext30",
                             "allxtext25", "allxtext30"]},
              "extra_contrasts": {"n17": [
                  ("actionxtext15", "vanilla"), ("actionxtext15", "actionxtext"),
                  ("actionxtext20", "vanilla"), ("actionxtext20", "actionxtext15"),
                  ("allxtext", "vanilla"), ("allxtext", "actionxtext"),
                  ("allxtext15", "vanilla"), ("allxtext15", "allxtext"),
                  ("allxtext20", "vanilla"), ("allxtext20", "allxtext15"),
                  ("actionxtext25", "vanilla"), ("actionxtext25", "actionxtext20"),
                  ("actionxtext30", "vanilla"), ("actionxtext30", "actionxtext25"),
                  ("allxtext25", "vanilla"), ("allxtext25", "allxtext20"),
                  ("allxtext30", "vanilla"), ("allxtext30", "allxtext25"),
              ]}},
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

    # `avg` is the unweighted suite mean (macro average): suites contribute
    # unequal episode counts (e.g. robot 393/409/398/350), so `pooled`
    # over-weights the larger suites; `avg` weights each suite equally.
    print(f"\n== SR (success_once, %) — pooled + suite-avg + per suite ==")
    print(f"  {'arm':13s}{'pooled':>8s}{'avg':>8s}"
          + "".join(f"{s.replace('libero_', ''):>9s}" for s in SUITES))
    for arm in arms:
        suite_srs = [sr(arm, per_suite[s]) for s in SUITES]
        row = "".join(f"{v:9.1f}" for v in suite_srs)
        print(f"  {arm:13s}{sr(arm, keys):8.2f}"
              f"{sum(suite_srs) / len(suite_srs):8.2f}{row}")

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
