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

def camera_cat(task_name):
    """Viewpoint family from the `_view_<h>_<v>_<scale>_<rot>_<vert>` tail.
    The four are disjoint and exhaust the curated set (asserted over all
    1,599 variants by verify_camera_axis.py gate A), and they differ in KIND,
    not just strength: zoom and reaim leave the camera's orientation resp.
    position fixed, so an intervention that helps one need not help another."""
    h, v, s, r, e = (int(x) for x in
                     re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_",
                               task_name).groups())
    if s != 100:
        return "zoom"
    if r or e:
        return "reaim"
    return "orbit_up" if v else "orbit"

def noise_cat(task_name):
    """Corruption family from the `_noise_<N>` tail (N=1..50): decade ->
    family, severity = N within decade (env_wrapper.py:283-305)."""
    n = int(re.search(r"_noise_(\d+)", task_name).group(1))
    return ["motion", "gauss", "zoom", "fog", "glass"][(n - 1) // 10]

# Axis-level metadata is model-INDEPENDENT (it describes the perturbation, not the
# intervention): `tag` builds the eplog prefix as f"{model_tag}_{axis_tag}", and cat/cats
# are the per-category breakdown. `extra_arms`/`extra_contrasts` here are keyed BY MODEL,
# because they name concrete arm tags.

# 2026-08-31 the SELF-ATTENTION row (docs/hopfield.md §7), identical on language /
# robot / original so it is declared once. Primary comparator = hop-dense (the
# odd-block eager-dense control; alpha=1 is bit-identical to the manual dense path),
# which cancels the fused-vs-eager kernel term the way base_dense does on pi05;
# vs vanilla is the deployable reading. Then the questions of docs/hopfield.md §5.1:
# the sign (a150 vs a050), matched first-order dose extrapolated+normalized vs direct
# (a110b5 vs a150b1), the rescale alone (-nonorm), adaptive vs static, and the
# temperature control at matched displacement (t150b1 vs a150b1 / a050b1).
HOP_ARMS = ["hop-dense", "hop-a050b1", "hop-a075b1", "hop-a125b1", "hop-a150b1",
            "hop-a200b1", "hop-t150b1", "hop-a110b5", "hop-a110b5-nonorm",
            "hop-a150b1-adap",
            # vanilla re-collected on the machine that runs the row (SETUP.md §0);
            # exists only where the row ran off machine A. `vanilla-b - vanilla`
            # is the machine term, and the deployable reading there is vs vanilla-b.
            "vanilla-b"]
_HOP_ONLY = [a for a in HOP_ARMS if a.startswith("hop-")]
HOP_CONTRASTS = (
    [("hop-dense", "vanilla"), ("vanilla-b", "vanilla"), ("hop-dense", "vanilla-b")]
    + [(a, "hop-dense") for a in _HOP_ONLY if a != "hop-dense"]
    + [(a, "vanilla") for a in _HOP_ONLY if a != "hop-dense"]
    + [(a, "vanilla-b") for a in _HOP_ONLY if a != "hop-dense"]
    + [("hop-a150b1", "hop-a050b1"), ("hop-a125b1", "hop-a075b1"),
       ("hop-a110b5", "hop-a150b1"), ("hop-a110b5-nonorm", "hop-a110b5"),
       ("hop-a150b1-adap", "hop-a150b1"),
       ("hop-a150b1", "hop-t150b1"), ("hop-a050b1", "hop-t150b1")]
)

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
                              "axt-sxi15", "axt-sxi20",
                              "allxt-late-l2", "allxt-inc-l2",
                              "allxt-temp20-late-l2"]},
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
                   # 08-18 denoising-step schedule (the language axis's 08-16/17 row
                   # carried here): late [0,0,1,1] and inc [0,0.5,1,1.5] on the same
                   # lambda=2 all-x-text base. Three readings per arm — vs vanilla; vs
                   # the flat parent allxtext20 ([2,2,2,2], same peak lambda, twice the
                   # total dose); and vs the COLLECTED FLAT ARM AT THE SAME
                   # time-integrated dose sum_i lambda_i (late [0,0,2,2] = 4 = flat
                   # lambda=1 = allxtext; inc [0,1,2,3] = 6 = flat lambda=1.5 =
                   # allxtext15) — the contrast that separates WHEN the intervention
                   # acts from HOW MUCH of it there is. late-vs-inc is the within-row
                   # dose step (both late-weighted).
                   ("allxt-late-l2", "vanilla"), ("allxt-inc-l2", "vanilla"),
                   ("allxt-late-l2", "allxtext20"), ("allxt-inc-l2", "allxtext20"),
                   ("allxt-late-l2", "allxtext"), ("allxt-inc-l2", "allxtext15"),
                   ("allxt-inc-l2", "allxt-late-l2"),
                   # 2026-08-21 sharp-softmax mirror of the LATE shape (the 08-16/17 language row's
                   # branch swap, carried here). Same all-x-text locus, same lambda=2, same
                   # [0,0,1,1] weights, entmax-1.5 -> softmax(2*l) at beta=2. On language the two
                   # branches agreed at this shape (temp-late - late -1.17pp, z=-1.59, n.s.), so the
                   # time structure there is a property of the sharpening rather than of entmax's
                   # exact zeros; this asks the same question off the language axis.
                   ("allxt-temp20-late-l2", "vanilla"),
                   ("allxt-temp20-late-l2", "allxt-late-l2"),     # branch swap at matched shape
                   # only two readings on this axis: layout carries no sharp-softmax arm
                   # (allxt-temp20{,l15,l20} were never run here), so the flat-parent and
                   # iso-dose contrasts the other axes get have no counterpart to pair with.
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
                 # 08-05 smolvla lambda=1.0 row ("10"-suffixed): the track's
                 #   unsuffixed arms are lambda=1.5, so this rung restores the
                 #   N1.7-matched dose point (ladder steps DOWN from 1.5).
                 # 08-06 smolvla image-locus low-dose rung: axi05 (lambda=0.5),
                 #   probing below the matched dose on the image locus.
                 "extra_arms": {"smolvla": [
                                "axt10", "axi10", "axcam10", "axti10", "axi05"],
                                "n17": [
                                "allxtext", "axt-sxi",
                                "actionxtext15", "allxtext15", "axt-sxi15",
                                "actionximage15", "stateximage15", "statextext15",
                                "axt-temp15", "axt-temp20", "axt-temp30",
                                "actionxtext20", "actionximage20",
                                "stateximage20", "statextext20",
                                "allxtext20", "axt-sxi20",
                                "axt-temp20l15", "axt-temp20l20",
                                "allxt-temp20", "allxt-temp20l15", "allxt-temp20l20",
                                # 08-16 denoising-step schedule row on all-x-text at
                                #   lambda=2 (this axis's best arm, allxtext20):
                                #   weights early [1,1,0,0], late [0,0,1,1],
                                #   inc [0,.5,1,1.5], dec [1.5,1,.5,0], so effective
                                #   lambda/step = 2*w. The [1,1,1,1] row IS the
                                #   parent allxtext20 (bit-identical), so it is not
                                #   a separate arm.
                                "allxt-early-l2", "allxt-late-l2",
                                "allxt-inc-l2", "allxt-dec-l2",
                                # 08-17 sharp-softmax mirror of that row: same
                                #   shapes, same lambda=2 all-x-text base, sparse
                                #   branch swapped to softmax(2*l). Parent =
                                #   allxt-temp20l20 (the flat row, bit-identical).
                                "allxt-temp20-early-l2", "allxt-temp20-late-l2",
                                "allxt-temp20-inc-l2", "allxt-temp20-dec-l2",
                                # 08-26 NAG normalization (docs/nag.md): the cap at
                                #   the shared operating point (lambda=2, tau=2.5,
                                #   rho=1) on the axis whose ladder CLIMBS — the
                                #   risk case, where the cap could cost the gain.
                                "allxt-temp20-nagn-l20",
                                # 08-28 the B-server curve (docs/nag.md §7 Tier 3,
                                #   run self-contained on its own machine per
                                #   SETUP.md §0): the lambda 1.5/3 pairs plus the
                                #   missing plain lambda=3 rung. On machine A these
                                #   eplogs never exist and the arms stay skipped.
                                "allxt-temp20l30",
                                "allxt-temp20-nagn-l15", "allxt-temp20-nagn-l30",
                                # 08-27 condition-② battery (docs/nag.md §2b —
                                #   refinement has content only with a raised
                                #   lambda and an ACTIVE cap): plain lambda=4,
                                #   cap-only at 4, and the published NAG recipe
                                #   (lambda=4, tau=2.5, rho=0.5 -> lambda_eff=2
                                #   on uncapped rows).
                                "allxt-temp20l40",
                                "allxt-temp20-nagn-l40", "allxt-temp20-nagnr-l40",
                                # 08-30 the lambda=2.5 rung (R recorded via the probe)
                                "allxt-temp20l25",
                                # 08-31 the self-attention (Hopfield) row
                                *HOP_ARMS]},
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
                     # 08-16 denoising-step schedule: WHEN in the 4-step Euler loop
                     # the intervention acts. The two primary tests are the
                     # internally dose-matched shape pairs — early vs late (sum
                     # w = 2) and increasing vs decreasing (sum w = 3) — which hold
                     # total dose fixed and vary only its position in time. Each arm
                     # is also read against vanilla and against its all-steps parent
                     # allxtext20 (sum w = 4, same lambda=2 base), which is what
                     # separates "position in time" from "less total dose".
                     ("allxt-early-l2", "allxt-late-l2"),
                     ("allxt-inc-l2", "allxt-dec-l2"),
                     ("allxt-early-l2", "vanilla"), ("allxt-late-l2", "vanilla"),
                     ("allxt-inc-l2", "vanilla"), ("allxt-dec-l2", "vanilla"),
                     ("allxt-early-l2", "allxtext20"), ("allxt-late-l2", "allxtext20"),
                     ("allxt-inc-l2", "allxtext20"), ("allxt-dec-l2", "allxtext20"),
                     # 08-17 sharp-softmax mirror: the same three readings as the
                     # entmax row (dose-matched shape pairs, vs vanilla, vs the
                     # flat parent allxt-temp20l20), plus the branch swap read
                     # shape-for-shape against its entmax twin — which is where
                     # "are exact zeros special?" gets asked on the TIME axis.
                     ("allxt-temp20-early-l2", "allxt-temp20-late-l2"),
                     ("allxt-temp20-inc-l2", "allxt-temp20-dec-l2"),
                     ("allxt-temp20-early-l2", "vanilla"), ("allxt-temp20-late-l2", "vanilla"),
                     ("allxt-temp20-inc-l2", "vanilla"), ("allxt-temp20-dec-l2", "vanilla"),
                     ("allxt-temp20-early-l2", "allxt-temp20l20"),
                     ("allxt-temp20-late-l2", "allxt-temp20l20"),
                     ("allxt-temp20-inc-l2", "allxt-temp20l20"),
                     ("allxt-temp20-dec-l2", "allxt-temp20l20"),
                     ("allxt-temp20-early-l2", "allxt-early-l2"),
                     ("allxt-temp20-late-l2", "allxt-late-l2"),
                     ("allxt-temp20-inc-l2", "allxt-inc-l2"),
                     ("allxt-temp20-dec-l2", "allxt-dec-l2"),
                     # 08-18 ISO-DOSE reading of the two rows above — no new arms,
                     # a contrast the collected data already supports. The
                     # time-integrated dose sum_i lambda_i (lambda_i = 2*w_i over
                     # the N=4 Euler steps) makes each shape equal to a FLAT arm
                     # already on this axis: early [2,2,0,0] and late [0,0,2,2] both
                     # sum to 4 = flat lambda=1 (allxtext), and inc [0,1,2,3] /
                     # dec [3,2,1,0] both sum to 6 = flat lambda=1.5 (allxtext15).
                     # early-vs-late holds dose fixed and moves it in time;
                     # shape-vs-flat holds dose fixed and CONCENTRATES it, so the
                     # two together separate "when" from "spread out or not".
                     # Same reading on the softmax branch against its own flats.
                     ("allxt-late-l2", "allxtext"), ("allxt-early-l2", "allxtext"),
                     ("allxt-inc-l2", "allxtext15"), ("allxt-dec-l2", "allxtext15"),
                     ("allxt-temp20-late-l2", "allxt-temp20"),
                     ("allxt-temp20-early-l2", "allxt-temp20"),
                     ("allxt-temp20-inc-l2", "allxt-temp20l15"),
                     ("allxt-temp20-dec-l2", "allxt-temp20l15"),
                     # 2026-08-26 NAG cap at the shared setting (lambda=2, tau=2.5,
                     # rho=1; docs/nag.md §7 Tier 1). This axis's plain ladder CLIMBS
                     # to lambda=2 (+1.95pp from lambda=1, z=+2.72), so the contrast
                     # against the uncapped twin is the risk reading: does bounding
                     # per-query magnitude cost the gain the dose bought? A dose
                     # confound is impossible — same locus, same lambda, same branch.
                     ("allxt-temp20-nagn-l20", "allxt-temp20l20"),
                     ("allxt-temp20-nagn-l20", "vanilla"),
                     # 08-28 B-server curve contrasts: each capped rung vs its
                     # uncapped twin, the plain lambda=3 rung itself (does the
                     # climbing axis EVER turn over?), and the top-rung anchors.
                     ("allxt-temp20l30", "vanilla"),
                     ("allxt-temp20l30", "allxt-temp20l20"),
                     ("allxt-temp20-nagn-l15", "allxt-temp20l15"),
                     ("allxt-temp20-nagn-l30", "allxt-temp20l30"),
                     ("allxt-temp20-nagn-l30", "vanilla"),
                     # 08-27 condition-② battery. The rho reading is the
                     # nagnr-vs-nagn pair — rho isolated at matched (lambda, tau);
                     # everything else anchors it: does plain lambda=4 finally
                     # turn the climbing axis over, does the cap hold there, and
                     # does the published recipe (overdose+cap+shrink, iso-dose
                     # with plain lambda=2 on uncapped rows) beat either plain
                     # lambda=2 or cap-at-2.
                     ("allxt-temp20l40", "vanilla"),
                     ("allxt-temp20l40", "allxt-temp20l30"),
                     ("allxt-temp20-nagn-l40", "allxt-temp20l40"),
                     ("allxt-temp20-nagn-l40", "allxt-temp20-nagn-l30"),
                     ("allxt-temp20-nagnr-l40", "allxt-temp20-nagn-l40"),
                     ("allxt-temp20-nagnr-l40", "allxt-temp20l20"),
                     ("allxt-temp20-nagnr-l40", "allxt-temp20-nagn-l20"),
                     ("allxt-temp20-nagnr-l40", "vanilla"),
                     # 2026-08-30 lambda=2.5: vs vanilla and both dose neighbours
                     ("allxt-temp20l25", "vanilla"),
                     ("allxt-temp20l25", "allxt-temp20l20"),
                     ("allxt-temp20l30", "allxt-temp20l25"),
                 ],

                 "smolvla": [
                     ("axt10", "axi10"),               # locus contrast at 1.0
                     ("axt10", "vanilla"), ("axi10", "vanilla"),
                     ("axcam10", "vanilla"), ("axti10", "vanilla"),
                     # dose step 1.0 -> 1.5 per arm (unsuffixed = 1.5)
                     ("axt", "axt10"), ("axi", "axi10"),
                     ("axcam", "axcam10"), ("axti", "axti10"),
                     # composite-vs-image at 1.0 (the 1.5 rung's only Bonf*)
                     ("axti10", "axi10"),
                     # 08-06 low-dose rung: vs vanilla, plus the 0.5 -> 1.0
                     # dose step (higher dose listed first, as in axt/axt10)
                     ("axi05", "vanilla"), ("axi10", "axi05"),
                     # 08-31 the self-attention (Hopfield) row
                     *HOP_CONTRASTS,
                 ]},
                 # The climbing-side DiD (docs/nag.md §5.1): does the cap flatten
                 # a ladder that is RISING? It should not (that is Failure A) —
                 # the lambda 1.5 -> 3 walk should be the same capped and uncapped.
                 "extra_interactions": {"n17": [
                     ("allxt-temp20-nagn-l30", "allxt-temp20-nagn-l15",
                      "allxt-temp20l30", "allxt-temp20l15",
                      "lambda 1.5->3 step, capped vs uncapped (~0 = no Failure A)"),
                 ]},},
    # noise: obs-side corruption of the agentview stream. The per-family
    # breakdown is the point of the axis — the five differ in KIND, not just
    # strength (motion/glass smear geometry, gauss/zoom soften it, fog is a
    # low-frequency additive veil), so a locus can help one and not another.
    #
    # Like camera, this axis does NOT run the track's default lambda=1 modality
    # grid (operator grid 2026-08-10, sweep_n17_noise.sh): it runs vanilla plus
    # the text-locus dose ladder at both query groups. `model_overrides` replaces
    # the model-level arms/contrasts — without it analyze.py would demand
    # base0/actionximage/statextext/stateximage/allxall eplogs this campaign
    # never produces. Only vanilla + a-x-t are mandatory so the axis is
    # analyzable while the driver is still running; the five ladder rungs join as
    # `extra_arms` the moment all four of their suite eplogs exist — which on
    # this axis also covers the two-process split (the a-x-t and all-x-t rows are
    # run by separate concurrent drivers and can finish in either order).
    # With no image arm the modality locus pair does not exist here, so the locus
    # pair is the QUERY-GROUP one: all-x-t vs a-x-t.
    "noise": {"tag": "noise", "cat": noise_cat,
              "cats": ["motion", "gauss", "zoom", "fog", "glass"],
              "model_overrides": {"n17": {
                  "arms": ["vanilla", "actionxtext"],
                  "key_contrasts": [("actionxtext", "vanilla")],
                  "locus_pair": ("allxtext", "actionxtext"),
                  "suite_contrasts": [("actionxtext", "vanilla"),
                                      ("allxtext", "vanilla"),
                                      ("allxtext20", "vanilla"),
                                      ("allxtext", "actionxtext")],
                  "cat_contrasts": [("actionxtext", "vanilla"),
                                    ("actionxtext20", "vanilla"),
                                    ("allxtext", "vanilla"),
                                    ("allxtext20", "vanilla"),
                                    ("allxtext", "actionxtext")],
              }},
              # 08-13 temperature-softmax row (beta=2, ent15-matched), all-x-t
              # only, lambda {1, 1.5, 2}: the matched-dose counterparts of the
              # entmax allxtext ladder, same shape as camera's. Delivered by the
              # CONCURRENT driver sweep_n17_noise_temp.sh, so these can complete
              # before or after their entmax pairs — extra_arms handles either
              # order. Unlike camera this row is a SYMMETRIC-NULL control: the
              # entmax ladder finished flat here (all six cells |z| <= 0.62), so
              # the pairs test that sharpening is as inert as zeros on an axis
              # where neither helps, not that they match where one does.
              "extra_arms": {"n17": [
                             "actionxtext15", "actionxtext20",
                             "allxtext", "allxtext15", "allxtext20",
                             "allxt-temp20", "allxt-temp20l15",
                             "allxt-temp20l20",
                             "allxt-late-l2", "allxt-inc-l2",
                             "allxt-temp20-late-l2", "allxt-temp20-late-l15",
                             # 08-29 entmax late at lambda=1.5 (this axis's optimum)
                             "allxt-late-l15",
                             # 08-28 NAG cap at the shared setting (docs/nag.md
                             #   §7 Tier 1): the big-n falling-axis confirmation.
                             "allxt-temp20-nagn-l20"]},
              "extra_contrasts": {"n17": [
                  # 2026-08-28 NAG cap (lambda=2, tau=2.5, rho=1). This axis's
                  # plain ladder peaks at lambda=1.5 and turns over by 2 (-1.81,
                  # z=-2.50), so the three readings are: vs the uncapped twin
                  # (the cap alone, no dose confound), vs the AXIS OPTIMUM
                  # temp20l15 (the regret of the shared setting — the number the
                  # one-lambda claim is about), and vs vanilla.
                  ("allxt-temp20-nagn-l20", "allxt-temp20l20"),
                  ("allxt-temp20-nagn-l20", "allxt-temp20l15"),
                  ("allxt-temp20-nagn-l20", "vanilla"),
                  # 2026-08-29 entmax late [0,0,1.5,1.5]: vs vanilla; the branch
                  # swap at matched shape+dose (vs allxt-temp20-late-l15 — on
                  # language the two branches agreed at lambda=2, this asks at the
                  # axis's own optimum); vs the flat parent allxtext15 (same peak
                  # lambda, twice the summed dose); vs allxt-late-l2 (dose step
                  # inside the entmax late shape).
                  ("allxt-late-l15", "vanilla"),
                  ("allxt-late-l15", "allxt-temp20-late-l15"),
                  ("allxt-late-l15", "allxtext15"),
                  ("allxt-late-l2", "allxt-late-l15"),
                  # dose ladder at each locus: vs vanilla and vs the rung below
                  ("actionxtext15", "vanilla"), ("actionxtext15", "actionxtext"),
                  ("actionxtext20", "vanilla"), ("actionxtext20", "actionxtext15"),
                  ("allxtext", "vanilla"), ("allxtext", "actionxtext"),
                  ("allxtext15", "vanilla"), ("allxtext15", "allxtext"),
                  ("allxtext20", "vanilla"), ("allxtext20", "allxtext15"),
                  # query-group locus at matched dose (the lambda=1 pair is
                  # ("allxtext", "actionxtext") above)
                  ("allxtext15", "actionxtext15"), ("allxtext20", "actionxtext20"),
                  # temperature row: each temp arm vs vanilla, vs its entmax
                  # counterpart at matched dose (zeros head-to-head), and the
                  # within-temp dose neighbor — same shape as camera/robot
                  ("allxt-temp20", "vanilla"), ("allxt-temp20", "allxtext"),
                  ("allxt-temp20l15", "vanilla"), ("allxt-temp20l15", "allxtext15"),
                  ("allxt-temp20l15", "allxt-temp20"),
                  ("allxt-temp20l20", "vanilla"), ("allxt-temp20l20", "allxtext20"),
                  ("allxt-temp20l20", "allxt-temp20l15"),
                  # 08-18 denoising-step schedule (the language axis's 08-16/17 row
                  # carried here): late [0,0,1,1] and inc [0,0.5,1,1.5] on the same
                  # lambda=2 all-x-text base. Three readings per arm — vs vanilla; vs
                  # the flat parent allxtext20 ([2,2,2,2], same peak lambda, twice the
                  # total dose); and vs the COLLECTED FLAT ARM AT THE SAME
                  # time-integrated dose sum_i lambda_i (late [0,0,2,2] = 4 = flat
                  # lambda=1 = allxtext; inc [0,1,2,3] = 6 = flat lambda=1.5 =
                  # allxtext15) — the contrast that separates WHEN the intervention
                  # acts from HOW MUCH of it there is. late-vs-inc is the within-row
                  # dose step (both late-weighted).
                  ("allxt-late-l2", "vanilla"), ("allxt-inc-l2", "vanilla"),
                  ("allxt-late-l2", "allxtext20"), ("allxt-inc-l2", "allxtext20"),
                  ("allxt-late-l2", "allxtext"), ("allxt-inc-l2", "allxtext15"),
                  ("allxt-inc-l2", "allxt-late-l2"),
                  # 2026-08-21 sharp-softmax mirror of the LATE shape (the 08-16/17 language row's
                  # branch swap, carried here). Same all-x-text locus, same lambda=2, same
                  # [0,0,1,1] weights, entmax-1.5 -> softmax(2*l) at beta=2. On language the two
                  # branches agreed at this shape (temp-late - late -1.17pp, z=-1.59, n.s.), so the
                  # time structure there is a property of the sharpening rather than of entmax's
                  # exact zeros; this asks the same question off the language axis.
                  ("allxt-temp20-late-l2", "vanilla"),
                  ("allxt-temp20-late-l2", "allxt-temp20l20"),   # flat parent [2,2,2,2]
                  ("allxt-temp20-late-l2", "allxt-temp20"),      # iso-dose flat (sum lambda = 4)
                  ("allxt-temp20-late-l2", "allxt-late-l2"),     # branch swap at matched shape
                  # 2026-08-21 dose rung under the late shape, softmax branch: same
                  # [0,0,1,1] weights and beta=2 at a lambda=1.5 base ([0,0,1.5,1.5]).
                  # Three readings only — no iso-dose flat control exists (sum lambda
                  # = 3 would need a flat lambda=0.75 arm, which no axis carries) and
                  # no entmax twin at this dose (the entmax schedule row is lambda=2
                  # only), so the dose step against the lambda=2 rung is what this arm
                  # is for: does the late shape survive one rung down?
                  ("allxt-temp20-late-l15", "vanilla"),
                  ("allxt-temp20-late-l15", "allxt-temp20l15"),      # flat parent [1.5]*4
                  ("allxt-temp20-late-l15", "allxt-temp20-late-l2"), # dose step, same shape+branch
              ]}},
    # camera: agentview re-posing (runtime `_view_` tail). The per-family
    # breakdown is the point of the axis — orbit/orbit_up move the viewpoint,
    # zoom changes scale only, reaim changes bearing only, so a locus can help
    # one and not another.
    #
    # This axis does NOT run the track's default lambda=1 modality grid
    # (operator grid 2026-08-07, sweep_n17_camera.sh): it runs vanilla plus the
    # text-locus dose ladder at both query groups. `model_overrides` therefore
    # replaces the model-level arms/contrasts — without it analyze.py would
    # demand base0/actionximage/statextext/stateximage/allxall eplogs that this
    # campaign never produces. Only vanilla + a-x-t are mandatory so the axis is
    # analyzable while the driver is still running; the five ladder rungs join
    # as `extra_arms` the moment all four of their suite eplogs exist.
    # With no image arm the modality locus pair does not exist here, so the
    # locus pair is the QUERY-GROUP one: all-x-t vs a-x-t.
    "camera": {"tag": "camera", "cat": camera_cat,
               "cats": ["orbit", "orbit_up", "zoom", "reaim"],
               "model_overrides": {"n17": {
                   "arms": ["vanilla", "actionxtext"],
                   "key_contrasts": [("actionxtext", "vanilla")],
                   "locus_pair": ("allxtext", "actionxtext"),
                   "suite_contrasts": [("actionxtext", "vanilla"),
                                       ("allxtext", "vanilla"),
                                       ("allxtext20", "vanilla"),
                                       ("allxtext", "actionxtext")],
                   "cat_contrasts": [("actionxtext", "vanilla"),
                                     ("actionxtext20", "vanilla"),
                                     ("allxtext", "vanilla"),
                                     ("allxtext20", "vanilla"),
                                     ("allxtext", "actionxtext")],
               }},
               # 08-08 temperature-softmax row (beta=2, ent15-matched), all-x-t
               # only, lambda {1, 1.5, 2}: the matched-dose counterparts of the
               # entmax allxtext ladder. Delivered by the CONCURRENT driver
               # sweep_n17_camera_temp.sh, so these can complete before or
               # after their entmax pairs — extra_arms handles either order.
               "extra_arms": {"n17": [
                              "actionxtext15", "actionxtext20",
                              "allxtext", "allxtext15", "allxtext20",
                              "allxt-temp20", "allxt-temp20l15",
                              "allxt-temp20l20",
                              "allxt-late-l2", "allxt-inc-l2",
                              "allxt-temp20-late-l2"]},
               "extra_contrasts": {"n17": [
                   # dose ladder at each locus: vs vanilla and vs the rung below
                   ("actionxtext15", "vanilla"), ("actionxtext15", "actionxtext"),
                   ("actionxtext20", "vanilla"), ("actionxtext20", "actionxtext15"),
                   ("allxtext", "vanilla"), ("allxtext", "actionxtext"),
                   ("allxtext15", "vanilla"), ("allxtext15", "allxtext"),
                   ("allxtext20", "vanilla"), ("allxtext20", "allxtext15"),
                   # query-group locus at matched dose (the lambda=1 pair is
                   # ("allxtext", "actionxtext") above)
                   ("allxtext15", "actionxtext15"), ("allxtext20", "actionxtext20"),
                   # temperature row: each temp arm vs vanilla, vs its entmax
                   # counterpart at matched dose (zeros head-to-head), and the
                   # within-temp dose neighbor — same shape as robot/language
                   ("allxt-temp20", "vanilla"), ("allxt-temp20", "allxtext"),
                   ("allxt-temp20l15", "vanilla"), ("allxt-temp20l15", "allxtext15"),
                   ("allxt-temp20l15", "allxt-temp20"),
                   ("allxt-temp20l20", "vanilla"), ("allxt-temp20l20", "allxtext20"),
                   ("allxt-temp20l20", "allxt-temp20l15"),
                   # 08-18 denoising-step schedule (the language axis's 08-16/17 row
                   # carried here): late [0,0,1,1] and inc [0,0.5,1,1.5] on the same
                   # lambda=2 all-x-text base. Three readings per arm — vs vanilla; vs
                   # the flat parent allxtext20 ([2,2,2,2], same peak lambda, twice the
                   # total dose); and vs the COLLECTED FLAT ARM AT THE SAME
                   # time-integrated dose sum_i lambda_i (late [0,0,2,2] = 4 = flat
                   # lambda=1 = allxtext; inc [0,1,2,3] = 6 = flat lambda=1.5 =
                   # allxtext15) — the contrast that separates WHEN the intervention
                   # acts from HOW MUCH of it there is. late-vs-inc is the within-row
                   # dose step (both late-weighted).
                   ("allxt-late-l2", "vanilla"), ("allxt-inc-l2", "vanilla"),
                   ("allxt-late-l2", "allxtext20"), ("allxt-inc-l2", "allxtext20"),
                   ("allxt-late-l2", "allxtext"), ("allxt-inc-l2", "allxtext15"),
                   ("allxt-inc-l2", "allxt-late-l2"),
                   # 2026-08-21 sharp-softmax mirror of the LATE shape (the 08-16/17 language row's
                   # branch swap, carried here). Same all-x-text locus, same lambda=2, same
                   # [0,0,1,1] weights, entmax-1.5 -> softmax(2*l) at beta=2. On language the two
                   # branches agreed at this shape (temp-late - late -1.17pp, z=-1.59, n.s.), so the
                   # time structure there is a property of the sharpening rather than of entmax's
                   # exact zeros; this asks the same question off the language axis.
                   ("allxt-temp20-late-l2", "vanilla"),
                   ("allxt-temp20-late-l2", "allxt-temp20l20"),   # flat parent [2,2,2,2]
                   ("allxt-temp20-late-l2", "allxt-temp20"),      # iso-dose flat (sum lambda = 4)
                   ("allxt-temp20-late-l2", "allxt-late-l2"),     # branch swap at matched shape
               ]}},
    "robot": {"tag": "robot", "cat": robot_level,
              "cats": ["L1", "L2", "L3", "L4", "L5"],
              # 07-28 text-locus dose row (entmax), mirroring language.
              # 08-05 far-extrapolation rungs {2.5, 3.0} at both text loci —
              #   the 07-28 row was NULL through lambda=2.0; ladder contrasts
              #   step from the lambda=2.0 arms.
              # 08-06 temperature-softmax row (beta=2, ent15-matched), all-x-t
              #   locus only at lambda {1, 1.5, 2, 2.5}: each temp arm vs
              #   vanilla, vs its entmax counterpart at matched dose (zeros
              #   head-to-head), and the within-temp dose neighbor.
              "extra_arms": {"n17": [
                             "actionxtext15", "actionxtext20",
                             "allxtext", "allxtext15", "allxtext20",
                             "actionxtext25", "actionxtext30",
                             "allxtext25", "allxtext30",
                             "allxt-temp20", "allxt-temp20l15",
                             "allxt-temp20l20", "allxt-temp20l25",
                             "allxt-late-l2", "allxt-inc-l2",
                             "allxt-temp20-late-l2",
                             # 08-26 NAG normalization at the shared operating
                             #   point (docs/nag.md §7 Tier 1). Like language,
                             #   this axis peaks at lambda=2, so the cap is read
                             #   where the dose already works.
                             "allxt-temp20-nagn-l20",
                             # 08-31 the self-attention (Hopfield) row
                             *HOP_ARMS]},
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
                  ("allxt-temp20", "vanilla"), ("allxt-temp20", "allxtext"),
                  ("allxt-temp20l15", "vanilla"), ("allxt-temp20l15", "allxtext15"),
                  ("allxt-temp20l15", "allxt-temp20"),
                  ("allxt-temp20l20", "vanilla"), ("allxt-temp20l20", "allxtext20"),
                  ("allxt-temp20l20", "allxt-temp20l15"),
                  ("allxt-temp20l25", "vanilla"), ("allxt-temp20l25", "allxtext25"),
                  ("allxt-temp20l25", "allxt-temp20l20"),
                  # 08-18 denoising-step schedule (the language axis's 08-16/17 row
                  # carried here): late [0,0,1,1] and inc [0,0.5,1,1.5] on the same
                  # lambda=2 all-x-text base. Three readings per arm — vs vanilla; vs
                  # the flat parent allxtext20 ([2,2,2,2], same peak lambda, twice the
                  # total dose); and vs the COLLECTED FLAT ARM AT THE SAME
                  # time-integrated dose sum_i lambda_i (late [0,0,2,2] = 4 = flat
                  # lambda=1 = allxtext; inc [0,1,2,3] = 6 = flat lambda=1.5 =
                  # allxtext15) — the contrast that separates WHEN the intervention
                  # acts from HOW MUCH of it there is. late-vs-inc is the within-row
                  # dose step (both late-weighted).
                  ("allxt-late-l2", "vanilla"), ("allxt-inc-l2", "vanilla"),
                  ("allxt-late-l2", "allxtext20"), ("allxt-inc-l2", "allxtext20"),
                  ("allxt-late-l2", "allxtext"), ("allxt-inc-l2", "allxtext15"),
                  ("allxt-inc-l2", "allxt-late-l2"),
                  # 2026-08-21 sharp-softmax mirror of the LATE shape (the 08-16/17 language row's
                  # branch swap, carried here). Same all-x-text locus, same lambda=2, same
                  # [0,0,1,1] weights, entmax-1.5 -> softmax(2*l) at beta=2. On language the two
                  # branches agreed at this shape (temp-late - late -1.17pp, z=-1.59, n.s.), so the
                  # time structure there is a property of the sharpening rather than of entmax's
                  # exact zeros; this asks the same question off the language axis.
                  ("allxt-temp20-late-l2", "vanilla"),
                  ("allxt-temp20-late-l2", "allxt-temp20l20"),   # flat parent [2,2,2,2]
                  ("allxt-temp20-late-l2", "allxt-temp20"),      # iso-dose flat (sum lambda = 4)
                  ("allxt-temp20-late-l2", "allxt-late-l2"),     # branch swap at matched shape
                  # 2026-08-26 NAG cap at the shared setting (lambda=2, tau=2.5,
                  # rho=1; docs/nag.md §7 Tier 1). vs its uncapped twin isolates the
                  # cap with no dose confound. This axis's own optimum IS lambda=2,
                  # so a negative here prices the guardrail where the dose works.
                  ("allxt-temp20-nagn-l20", "allxt-temp20l20"),
                  ("allxt-temp20-nagn-l20", "vanilla"),
                  # 08-31 the self-attention (Hopfield) row
                  *HOP_CONTRASTS,
              ]}},
    # original: axis=none — original instructions, original scenes, nothing
    # perturbed. This is the campaign's IN-DISTRIBUTION control, and the reason
    # it is worth arms rather than just a vanilla baseline: every positive we
    # have is on instruction-OOD (language), so the grounding-specificity claim
    # predicts the SAME intervention does nothing here. There are no perturbation
    # categories to stratify by, and no severity table (the baseline would be
    # this arm against itself), so both are switched off.
    #   Two eras share the prefix: the 2026-07-16 lambda=1 modality grid at 400
    # eps/arm, and the 08-16 all-x-text sharp-softmax dose row at 2,000 (init
    # 0-49). Contrasts run on the common prefix; see the length note above.
    "original": {"tag": "orig", "cat": None, "cats": [],
                 "model_overrides": {"n17": {
                     "arms": ["vanilla", "actionxtext"],
                     "key_contrasts": [("actionxtext", "vanilla")],
                     "locus_pair": ("actionxtext", "actionximage"),
                     "suite_contrasts": [("actionxtext", "vanilla"),
                                         ("actionximage", "vanilla"),
                                         ("allxt-temp20l20", "vanilla")],
                 }},
                 "extra_arms": {"n17": [
                                "base0", "actionximage", "statextext",
                                "stateximage", "allxall",
                                "allxt-temp20", "allxt-temp20l15",
                                "allxt-temp20l20",
                                "allxtext20", "allxt-late-l2", "allxt-inc-l2",
                                "allxt-temp20-late-l2",
                                # 08-26 NAG row (docs/nag.md §7 Tier 2): the missing
                                #   plain lambda=3 rung plus the four capped ones.
                                "allxt-temp20l30",
                                "allxt-temp20-nagn-l10", "allxt-temp20-nagn-l15",
                                "allxt-temp20-nagn-l20", "allxt-temp20-nagn-l30",
                                # 08-28 the missing entmax lambda=1 all-x-text rung
                                "allxtext",
                                # 08-30 ratio-diagnostic re-runs (R recorded per episode)
                                "allxt-temp20-r", "allxt-temp20l20-r",
                                "allxt-temp20-nagn-l20-r",
                                # 08-31 the lambda=2.5 rung of the falling ladder
                                "allxt-temp20l25",
                                # 08-31 the self-attention (Hopfield) row
                                *HOP_ARMS]},
                 "extra_contrasts": {"n17": [
                     # the 07-16 lambda=1 modality grid, each vs vanilla, plus
                     # the locus contrast that carries the story on language
                     ("base0", "vanilla"), ("actionximage", "vanilla"),
                     ("statextext", "vanilla"), ("stateximage", "vanilla"),
                     ("allxall", "vanilla"),
                     ("actionxtext", "actionximage"),
                     # 08-16 in-dist dose row: vs vanilla and vs the rung below
                     ("allxt-temp20", "vanilla"),
                     ("allxt-temp20l15", "vanilla"), ("allxt-temp20l15", "allxt-temp20"),
                     ("allxt-temp20l20", "vanilla"), ("allxt-temp20l20", "allxt-temp20l15"),
                     # 08-18 in-dist control for the entmax step-schedule row. The
                     # arms that carry the language axis's Bonferroni-surviving
                     # positives (late +2.86pp z=3.64, inc +2.80 z=3.37) are run
                     # where nothing is OOD; grounding-specificity predicts both
                     # go flat. allxtext20 is their flat parent AND this axis's
                     # first entmax lambda=2 rung, so it also gives the in-dist
                     # entmax-vs-softmax branch swap against allxt-temp20l20.
                     # No iso-dose flat controls here (allxtext/allxtext15 are not
                     # run in-dist): the question is "does it do anything at all",
                     # which is the vs-vanilla contrast.
                     ("allxtext20", "vanilla"), ("allxtext20", "allxt-temp20l20"),
                     ("allxt-late-l2", "vanilla"), ("allxt-inc-l2", "vanilla"),
                     ("allxt-late-l2", "allxtext20"), ("allxt-inc-l2", "allxtext20"),
                     # 2026-08-21 sharp-softmax mirror of the LATE shape (the 08-16/17 language row's
                     # branch swap, carried here). Same all-x-text locus, same lambda=2, same
                     # [0,0,1,1] weights, entmax-1.5 -> softmax(2*l) at beta=2. On language the two
                     # branches agreed at this shape (temp-late - late -1.17pp, z=-1.59, n.s.), so the
                     # time structure there is a property of the sharpening rather than of entmax's
                     # exact zeros; this asks the same question off the language axis.
                     ("allxt-temp20-late-l2", "vanilla"),
                     ("allxt-temp20-late-l2", "allxt-temp20l20"),   # flat parent, in-dist
                     ("allxt-temp20-late-l2", "allxt-late-l2"),     # branch swap at matched shape
                     # 2026-08-26 NAG row (docs/nag.md §7, Tier 2 — the plateau CURVE).
                     # Each capped rung against its uncapped twin isolates the cap at
                     # fixed dose; lambda=1 is the inertness control (the cap clips
                     # 0.8% of rows there, so this contrast should read ~0); and
                     # nagn-l20 vs allxt-temp20 is the REGRET of using the shared
                     # setting instead of this axis's own optimum, which is the
                     # quantity the "one lambda, no per-axis tuning" claim is about.
                     ("allxt-temp20l30", "vanilla"),
                     ("allxt-temp20l30", "allxt-temp20l20"),
                     ("allxt-temp20-nagn-l10", "allxt-temp20"),
                     ("allxt-temp20-nagn-l15", "allxt-temp20l15"),
                     ("allxt-temp20-nagn-l20", "allxt-temp20l20"),
                     ("allxt-temp20-nagn-l30", "allxt-temp20l30"),
                     ("allxt-temp20-nagn-l20", "allxt-temp20"),   # regret vs the axis optimum
                     ("allxt-temp20-nagn-l30", "allxt-temp20"),
                     ("allxt-temp20-nagn-l20", "vanilla"),
                     # 2026-08-28 allxtext (entmax, lambda=1, all-x-text): vs vanilla;
                     # the in-dist branch swap at matched locus+dose (vs allxt-temp20,
                     # the softmax beta=2 twin — on the perturbation axes the two
                     # branches agreed, "zeros not special"); the query-group step
                     # from action-x-text at the same branch and dose; and the entmax
                     # dose step 1 -> 2 on this locus.
                     ("allxtext", "vanilla"),
                     ("allxtext", "allxt-temp20"),
                     ("allxtext", "actionxtext"),
                     ("allxtext20", "allxtext"),
                     # 2026-08-30 each R-recorded re-run vs its parent: the probe path
                     # is bit-identical to the plain arm, so these should read 0:0 —
                     # any discordance is a same-machine reproducibility measurement.
                     ("allxt-temp20-r", "allxt-temp20"),
                     ("allxt-temp20l20-r", "allxt-temp20l20"),
                     ("allxt-temp20-nagn-l20-r", "allxt-temp20-nagn-l20"),
                     # 2026-08-31 the -r arms are the campaign's EXTENSIBLE copies (0:0 vs
                     # their parents at 400 eps, R recorded), so once ORIG_EPISODES=500
                     # extends them and vanilla to 2,000 the key original readings are
                     # re-taken among them at 5x n: the cap alone, the regret of the
                     # shared setting, and each vs vanilla. Pairs run on the episodes
                     # both arms share, so these read at 2,000 while -r-vs-parent stays 400.
                     ("allxt-temp20-nagn-l20-r", "allxt-temp20l20-r"),   # cap alone
                     ("allxt-temp20-nagn-l20-r", "allxt-temp20-r"),      # regret vs axis optimum
                     ("allxt-temp20l20-r", "allxt-temp20-r"),            # the plain lambda 1->2 loss
                     ("allxt-temp20-r", "vanilla"),
                     ("allxt-temp20l20-r", "vanilla"),
                     ("allxt-temp20-nagn-l20-r", "vanilla"),
                     # 2026-08-31 lambda=2.5: vs vanilla, the axis optimum, and both neighbours
                     ("allxt-temp20l25", "vanilla"),
                     ("allxt-temp20l25", "allxt-temp20"),
                     ("allxt-temp20l25", "allxt-temp20l20"),
                     ("allxt-temp20l30", "allxt-temp20l25"),
                     # 08-31 the self-attention (Hopfield) row
                     *HOP_CONTRASTS,
                 ]},
                 # The plateau itself: does the cap shrink the walk from this axis's
                 # optimum (lambda=1) to the shared setting (lambda=2), and to 3?
                 "extra_interactions": {"n17": [
                     ("allxt-temp20-nagn-l20", "allxt-temp20-nagn-l10",
                      "allxt-temp20l20", "allxt-temp20",
                      "lambda 1->2 step, capped vs uncapped (>0 = plateau widened)"),
                     ("allxt-temp20-nagn-l30", "allxt-temp20-nagn-l10",
                      "allxt-temp20l30", "allxt-temp20",
                      "lambda 1->3 step, capped vs uncapped"),
                     ("allxt-temp20-nagn-l15", "allxt-temp20-nagn-l10",
                      "allxt-temp20l15", "allxt-temp20",
                      "lambda 1->1.5 step (should be ~0: the cap is inert here)"),
                 ]}},
}

def arm_hosts(prefix, arm):
    """Machines that wrote this arm's suite eplogs, from the `.arm` sidecars.

    SETUP.md §0 allows cross-machine parallelism only at (model x axis) campaign
    granularity: closed-loop rollouts amplify kernel-level numeric differences, so
    two arms of one contrast produced on different machines are not validly paired.
    Nothing downstream could see that before the host was recorded, which is why it
    is surfaced here instead of left to the operator's memory of what ran where.
    An empty set = eplogs predating the record (2026-08-26), which cannot be checked.

    The provenance line is parsed inline rather than imported from
    harness/eplog.py (which owns the format, `code <git-describe> host <name>`):
    that module reaches EpisodeResult through harness.rollout -> harness.env, and
    this script must keep running with no venv, no GPU and no simulator stack.
    """
    hosts = set()
    for s in SUITES:
        sidecar = SWEEP / f"{prefix}_{arm}_{s}_eplog.tsv.arm"
        if not sidecar.exists():
            continue
        with open(sidecar) as f:
            f.readline()  # line 1 is the arm signature, never a provenance record
            for line in f:
                toks = line.split()
                hosts |= {toks[i + 1] for i, t in enumerate(toks)
                          if t == "host" and i + 1 < len(toks)}
    return hosts


def load_rstats(prefix, arm):
    """Per-episode NAG ratio rows for an arm, keyed (suite, episode); {} if none.

    Written by eval_arm next to the eplog (`<out>.rstats.tsv`) for every arm that
    ran the NAG code path — a --pladis-nag-tau arm (pre-cap R) or a
    --pladis-nag-probe re-run of a plain arm (bit-identical rollout, R recorded).
    Missing files are normal: everything collected before 2026-08-30 has none.
    """
    rows = {}
    for s_ in SUITES:
        p = SWEEP / f"{prefix}_{arm}_{s_}_eplog.tsv.rstats.tsv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p), delimiter="\t"):
            rows[(s_, int(r["episode"]))] = {k: float(v) for k, v in r.items()}
    return rows


def load_rstats_sb(prefix, arm):
    """Per (episode, step, block) rows, pooled over suites; [] if none."""
    out = []
    for s_ in SUITES:
        p = SWEEP / f"{prefix}_{arm}_{s_}_eplog.tsv.rstats_sb.tsv"
        if not p.exists():
            continue
        out += [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(p), delimiter="\t")]
    return out


def load_hopstats(prefix, arm):
    """Per-episode Hopfield census rows for an arm, keyed (suite, episode); {} if none.

    Written by eval_arm next to the eplog (`<out>.hopstats.tsv`) for every --hop-*
    arm (eta, clamp rates, realized beta_eff) and every --hop-probe run (plus E, r,
    Align and the alpha/temperature price list). Missing files are normal.
    """
    rows = {}
    for s_ in SUITES:
        p = SWEEP / f"{prefix}_{arm}_{s_}_eplog.tsv.hopstats.tsv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p), delimiter="\t"):
            rows[(s_, int(r["episode"]))] = {k: float(v) for k, v in r.items()}
    return rows


def _mean_se(xs):
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else float("nan")), float("nan")
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def rstats_section(prefix, arms, data):
    """The ratio diagnostic (2026-08-30): is harm carried by extreme R?

    For every arm with rstats sidecars: the pooled distribution of the per-episode
    summaries, then the SAME statistics split by outcome — the reading the
    run-level census cannot give. A failing arm whose failed episodes sit at
    R = 3-10 while its successes do not says the harm is unconstrained
    extrapolation, not sparsity; a flat split says the cap is not acting on the
    rows that decide the episode. Welch z on the outcome split (episodes are
    independent draws within an arm).
    """
    have = {a: load_rstats(prefix, a) for a in arms}
    have = {a: r for a, r in have.items() if r}
    if not have:
        return
    print("\n== NAG ratio diagnostic: R = ||Z_PL||_1 / ||Z_d||_1 per (head, query row), "
          "per-episode summaries pooled ==")
    print(f"  {'arm':22s} {'eps':>5s} {'mean R':>7s} {'max R':>6s} {'P>1.5':>6s} {'P>2':>6s} "
          f"{'P>3':>6s} {'P>5':>6s} {'P>10':>6s} {'clip':>6s}")
    for a, rows in have.items():
        vals = list(rows.values())
        n = len(vals)
        mean = lambda k: sum(v[k] for v in vals) / n
        clip = [v["clip_rate"] for v in vals if v["clip_rate"] == v["clip_rate"]]
        clip_s = f"{sum(clip) / len(clip):6.1%}" if clip else "   off"
        print(f"  {a:22s} {n:5d} {mean('mean_R'):7.3f} {max(v['max_R'] for v in vals):6.1f} "
              f"{mean('frac_gt_1.5'):6.1%} {mean('frac_gt_2'):6.1%} {mean('frac_gt_3'):6.1%} "
              f"{mean('frac_gt_5'):6.2%} {mean('frac_gt_10'):6.2%} {clip_s}")

    print("\n  -- by outcome (success vs fail), per arm: mean R and P(R > 3) --")
    for a, rows in have.items():
        succ = [v for v in rows.values() if v["success_once"] == 1]
        fail = [v for v in rows.values() if v["success_once"] == 0]
        if len(succ) < 2 or len(fail) < 2:
            print(f"  {a:22s} outcome split not available ({len(succ)} succ / {len(fail)} fail)")
            continue
        for key, label in (("mean_R", "mean R"), ("frac_gt_3", "P(R>3)"), ("max_R", "max R")):
            ms, ses = _mean_se([v[key] for v in succ])
            mf, sef = _mean_se([v[key] for v in fail])
            se = math.sqrt(ses ** 2 + sef ** 2)
            z = (mf - ms) / se if se else 0.0
            print(f"  {a:22s} {label:7s} succ {ms:7.4f} (n={len(succ)})  fail {mf:7.4f} (n={len(fail)})"
                  f"  fail-succ {mf - ms:+.4f}  z={z:+5.2f}")

    # same-episode comparison across arms that BOTH carry rstats (paired by episode)
    pairs = [(a, b) for i, a in enumerate(have) for b in list(have)[i + 1:]]
    for a, b in pairs:
        ks = sorted(set(have[a]) & set(have[b]))
        if len(ks) < 2:
            continue
        d = [have[a][k]["mean_R"] - have[b][k]["mean_R"] for k in ks]
        m, se = _mean_se(d)
        print(f"  paired mean R  {a} - {b}: {m:+.4f} +/- {se:.4f}  n={len(ks)}")

    print("\n  -- by denoising step and by block (pooled over episodes): mean R / P(R > 3) --")
    for a in have:
        sb = load_rstats_sb(prefix, a)
        if not sb:
            continue
        by_step, by_block = defaultdict(list), defaultdict(list)
        for r in sb:
            by_step[int(r["step"])].append(r)
            by_block[int(r["block"])].append(r)
        fmt = lambda rs: f"{sum(r['mean_R'] for r in rs) / len(rs):.3f}/{sum(r['frac_gt_3'] for r in rs) / len(rs):.1%}"
        print(f"  {a}:")
        print("     step  " + "  ".join(f"{k}: {fmt(v)}" for k, v in sorted(by_step.items())))
        print("     block " + "  ".join(f"{k}: {fmt(v)}" for k, v in sorted(by_block.items())))


def hopstats_section(prefix, arms, data):
    """The Hopfield census (docs/hopfield.md §5.1 endpoint 5): do failing episodes
    sit at lower Align / higher instability than successes, and does a hop arm's
    realized clamp rate / beta_eff differ by outcome? Welch z on the outcome split,
    as in rstats_section. Probe runs are listed too (they carry E/r/Align)."""
    have = {a: load_hopstats(prefix, a) for a in arms}
    have = {a: r for a, r in have.items() if r}
    if not have:
        return
    print("\n== Hopfield census: eta (symmetry index), E / r / Align of the baseline "
          "retrieval, norm-match clamp rate, realized beta_eff — per-episode summaries pooled ==")
    print(f"  {'arm':22s} {'eps':>5s} {'eta':>7s} {'eta_p10':>7s} {'E':>9s} {'r':>6s} "
          f"{'Align':>7s} {'clamp':>11s} {'beta_eff':>8s}")
    for a, rows in have.items():
        vals = list(rows.values())
        n = len(vals)
        mean = lambda k: sum(v[k] for v in vals if v[k] == v[k]) / max(1, sum(1 for v in vals if v[k] == v[k]))
        clamp = f"{mean('clamp_lo_rate'):5.1%}/{mean('clamp_hi_rate'):5.1%}" \
            if any(v["clamp_lo_rate"] == v["clamp_lo_rate"] for v in vals) else "        off"
        print(f"  {a:22s} {n:5d} {mean('eta_mean'):+7.3f} {mean('eta_p10'):+7.3f} {mean('E_mean'):+9.3g} "
              f"{mean('r_mean'):6.3f} {mean('align_mean'):+7.3f} {clamp:>11s} {mean('beta_eff_mean'):8.3f}")

    print("\n  -- by outcome (success vs fail), per arm --")
    for a, rows in have.items():
        succ = [v for v in rows.values() if v["success_once"] == 1]
        fail = [v for v in rows.values() if v["success_once"] == 0]
        if len(succ) < 2 or len(fail) < 2:
            print(f"  {a:22s} outcome split not available ({len(succ)} succ / {len(fail)} fail)")
            continue
        for key, label in (("eta_mean", "eta"), ("align_mean", "Align"), ("r_mean", "r"),
                           ("E_mean", "E"), ("clamp_hi_rate", "clamp_hi"), ("beta_eff_mean", "beta_eff")):
            s_ = [v[key] for v in succ if v[key] == v[key]]
            f_ = [v[key] for v in fail if v[key] == v[key]]
            if len(s_) < 2 or len(f_) < 2:
                continue
            ms, ses = _mean_se(s_)
            mf, sef = _mean_se(f_)
            se = math.sqrt(ses ** 2 + sef ** 2)
            z = (mf - ms) / se if se else 0.0
            print(f"  {a:22s} {label:9s} succ {ms:+9.4f} (n={len(s_)})  fail {mf:+9.4f} (n={len(f_)})"
                  f"  fail-succ {mf - ms:+.4f}  z={z:+5.2f}")


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

def interaction(a, b, c, d, keys):
    """Paired difference-in-differences: [a - b] - [c - d] on shared episodes.

    McNemar tests ONE 2x2 table, so it cannot answer the question the NAG row is
    for — "did the cap SHRINK the dose step" is a difference of two differences
    (docs/nag.md §5.1). Every arm of an axis runs the same seed-0 schedule, so the
    per-episode contrast x_i = (a_i - b_i) - (c_i - d_i) in {-2,...,2} is paired
    and its mean has the ordinary SE; no 2x2 approximation is involved.

    Returns (delta_pp, se_pp, z, p).
    """
    x = [(a[k]["succ"] - b[k]["succ"]) - (c[k]["succ"] - d[k]["succ"]) for k in keys]
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0, 1.0
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0.0:
        return 100 * mean, 0.0, 0.0, 1.0
    z = mean / se
    return 100 * mean, 100 * se, z, math.erfc(abs(z) / math.sqrt(2))


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
    # An axis whose driver carries a different arm list than the track default
    # overrides the model-level keys (arms / contrasts / locus_pair) here, so
    # the arm vocabulary stays defined in ONE place per (model, axis) instead
    # of being split between MODELS and a special case downstream.
    mcfg = {**MODELS[args.model], **cfg.get("model_overrides", {}).get(args.model, {})}
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
    hosts = {arm: arm_hosts(prefix, arm) for arm in arms}
    seen_hosts = set().union(*hosts.values()) if hosts else set()
    if len(seen_hosts) > 1:
        print(f"\n[HOSTS] this axis carries arms from {len(seen_hosts)} machines "
              f"{sorted(seen_hosts)}. SETUP.md §0 allows cross-machine work only at "
              f"whole-campaign granularity, so any contrast marked !host below pairs "
              f"two machines' numerics and is NOT valid:")
        for arm in arms:
            if hosts[arm]:
                print(f"         {arm:24s} {sorted(hosts[arm])}")
        unknown = [a for a in arms if not hosts[a]]
        if unknown:
            print(f"         (no host recorded, predates 2026-08-26: "
                  f"{', '.join(unknown)})")
    # Arms must be paired, but they may legitimately differ in LENGTH: env.py's
    # schedule is prefix-determined (schedule(N,seed)[:M] == schedule(M,seed)),
    # so raising --episodes extends an arm rather than reshuffling it, and the
    # original axis now mixes a 2,000-ep dose row with the 400-ep lambda=1 grid
    # of 2026-07-16. Nested is fine; CROSSING is a pairing bug. Enforced by
    # requiring each arm's episode indices to be contiguous from 0 per suite —
    # then the sets are nested by construction and the common prefix is the
    # widest honest comparison. A gap (a half-resumed arm) still aborts.
    for arm in arms:
        for s in SUITES:
            idx = sorted(k[1] for k in data[arm] if k[0] == s)
            assert idx == list(range(len(idx))), (
                f"{arm}/{s}: episode indices not contiguous from 0 "
                f"(gap at {next(i for i, v in enumerate(idx) if i != v)}) — "
                f"a partially-resumed eplog cannot be paired")
    n_common = {s: min(sum(1 for k in data[a] if k[0] == s) for a in arms)
                for s in SUITES}
    keys = sorted(k for k in data["vanilla"].keys() if k[1] < n_common[k[0]])
    short = {a: sum(1 for k in data[a]) for a in arms if len(data[a]) > len(keys)}
    if short:
        print(f"[note] arms differ in length; comparing the common prefix "
              f"({len(keys)} eps). Longer arms truncated for contrasts: "
              + ", ".join(f"{a}={n}" for a, n in sorted(short.items())))
    for arm in arms:  # schedule identity across arms, on the common prefix
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
    # Each contrast runs on the episodes ITS OWN two arms share, not on the
    # global common prefix. When arm lengths differ, the global prefix is set by
    # the shortest arm on the axis, which would silently drag an unrelated pair
    # down to it — on the original axis that would hold the 2,000-ep dose row to
    # the 400 eps of the 2026-07-16 lambda=1 grid and throw away the precision
    # the longer run was paid for. Pairing is unaffected: the schedule is
    # prefix-determined, so a pair's shared episodes are the same episodes.
    pair_keys = lambda a, b: [k for k in data[a] if k in data[b]]
    print(f"\n== paired McNemar, pooled (Bonferroni m={m}, alpha=.05 -> "
          f"p<{0.05 / m:.4f}) ==")
    for a, b in contrasts:
        ks = pair_keys(a, b)
        n01, n10, z, p = mcnemar(data[a], data[b], ks)
        d = sr(a, ks) - sr(b, ks)
        mark = "*" if p < 0.05 / m else (" " if p >= 0.05 else ".")
        n_note = "" if len(ks) == len(keys) else f"  n={len(ks)}"
        # Two arms from different machines are not paired data, whatever the
        # p-value says; flag it on the row, not only in the header block.
        if hosts[a] and hosts[b] and hosts[a] != hosts[b]:
            n_note += f"  !host {sorted(hosts[a])} vs {sorted(hosts[b])}"
        print(f"  {a:13s} - {b:13s} {d:+6.2f}pp  disc {n01:3d}:{n10:3d}"
              f"  z={z:+5.2f}  p={p:.4g}  p_bonf={min(1.0, p * m):.4g} {mark}{n_note}")
    print("  (* survives Bonferroni; . nominal p<.05 only)")

    # docs/nag.md §5.1: the plateau question is about the SHAPE of the dose
    # response, i.e. whether the cap shrinks the step between two rungs. That is a
    # 4-arm interaction, not a pair contrast, so it gets its own table.
    inter = [t for t in list(mcfg.get("extra_interactions", []))
             + list(cfg.get("extra_interactions", {}).get(args.model, []))
             if all(x in arms for x in t[:4])]
    if inter:
        print("\n== dose-step interaction, paired (docs/nag.md §5.1): "
              "[a - b] - [c - d] ==")
        for a, b, c, d, note in inter:
            ks = [k for k in data[a] if k in data[b] and k in data[c] and k in data[d]]
            delta, se, z, pv = interaction(data[a], data[b], data[c], data[d], ks)
            step_t = sr(a, ks) - sr(b, ks)
            step_c = sr(c, ks) - sr(d, ks)
            print(f"  [{a} - {b}] - [{c} - {d}]  n={len(ks)}")
            print(f"     step {step_t:+6.2f}pp vs {step_c:+6.2f}pp -> "
                  f"{delta:+6.2f}pp +/- {se:.2f}  z={z:+5.2f}  p={pv:.4g}   {note}")

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

    rstats_section(prefix, arms, data)
    hopstats_section(prefix, arms, data)

    # Severity needs the axis=none reference sweep of the SAME model. Guarded rather than
    # assumed: without the guard a missing file makes the whole analysis unusable until
    # the original sweep finishes, when everything above it is already valid.
    orig_paths = [SWEEP / f"{mcfg['tag']}_orig_vanilla_{s}_eplog.tsv" for s in SUITES]
    if axis == "original":
        pass  # the reference sweep IS this axis; a self-comparison is 0.0 by construction
    elif not all(p.exists() for p in orig_paths):
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
    # Guarded like the severity baseline above: on an axis whose locus arms are
    # extra_arms (camera), one of them can still be mid-campaign, and that must
    # not invalidate everything already printed.
    if lo_a not in arms or lo_b not in arms:
        print(f"\n[note] per-task locus deltas skipped: {lo_a}/{lo_b} "
              f"not both present in {arms}")
        return
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
