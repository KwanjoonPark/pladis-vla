# SPDX-License-Identifier: Apache-2.0
"""Generic single-arm evaluator: anchor runs, parity checks, and sweep arms
all go through this one entry point so every result shares one code path.

  bash experiments/run.sh [--venv <venv>] experiments/eval_arm.py \
      --model gr00t_n17 --axis language --episodes 100 --seed 0 \
      --out results/foo_eplog.tsv \
      [--pladis-scale 1.0 --pladis-qgroup action --pladis-kind image]

  bash experiments/run.sh --venv openpi experiments/eval_arm.py --model pi05 \
      --axis language --episodes 0 --seed 0 --max-steps 520 --exec-horizon 5 \
      --out results/foo_eplog.tsv [--pladis-install --pladis-scale 1.0 --pladis-kind text]

--model selects a ModelSpec from harness/registry.py: it resolves the
adapter loader, the PLADIS hook module, the default checkpoint path, and the
model's protocol defaults (exec horizon / step cap / n_state_tokens) for any
flag left unset. Loader and hooks are imported lazily, AFTER the resume
check, so a completed arm exits in seconds and a model never needs the other
tracks' venvs. Keeping every track on ONE evaluator is deliberate — a
sibling script would duplicate the resume ledger, the seeded schedule, the
arm signature, the git provenance and the video labelling, and a silent
divergence between the copies would leave no trace in the eplogs (the TSV
carries no arm identity, harness/eplog.py:8-15). The model-specific surface
is the registry lookup, one install branch and one tag branch.

PLADIS is installed explicitly (the registry's hook module), never via env
vars. Omitting --pladis-install gives vanilla.
  * gr00t_n17: --pladis-scale 0 with --pladis-install gives base0 — the hook is
    installed but delegates to the native fused SDPA (official PLADIS lambda=0
    semantics), so base0 is BIT-identical to vanilla.
  * pi05: lambda=0 returns the plain softmax the stock gemma eager path computes, so
    base0 is likewise bit-identical (verify_pi05_hook.py gate A). pi0.5's suffix is
    action-only, so the qgroup axis does not exist there; the locus axis is --pladis-kind
    over the key sub-blocks (text / image / prefix / all).
  * smolvla: same situation as pi05 — a single eager kernel (no SDPA anywhere in
    smolvlm_with_expert.py), lambda=0 is the stock softmax op (verify_smolvla_hook.py
    bit-parity gate), so no base0 arm exists in its driver. Locus axis is --pladis-kind
    over the CA key sub-blocks (text / image / state / prefix) or the SA rows (self).


Resume: episodes already in --out are skipped (eplog is the ledger).
"""

from __future__ import annotations

import argparse
import os
import re
import time

import torch

from harness.env import LiberoPlusTaskSet, LiberoPlusSession
from harness.eplog import EpisodeLogger
from harness.registry import MODELS, resolve_hooks, resolve_loader
from harness.rollout import run_episode

# The hooks spell the same transform differently (attn_gr00t_n17.py:74 "ent15max"
# vs attn_pi05.py:53 / attn_smolvla.py:88 "entmax15"). --pladis-method therefore
# defaults to None and is resolved per track BEFORE the arm signature is built, so
# every pre-existing gr00t_n17 signature string stays byte-identical and its eplogs
# still resume (harness/eplog.py:62-80 aborts on any signature change).
_DEFAULT_METHOD = {"gr00t_n17": "ent15max", "pi05": "entmax15", "smolvla": "entmax15"}
_METHOD_ALIAS = {
    "gr00t_n17": {"entmax15": "ent15max"},
    "pi05": {"ent15max": "entmax15"},
    "smolvla": {"ent15max": "entmax15"},
}
# The step axis needs a per-step index published by a hook on the denoising loop;
# only attn_gr00t_n17 has one. Accepting the flag silently on the other tracks would
# log a scheduled arm that ran on every step.
_SCHEDULE_TRACK_MSG = (
    "[arm] --pladis-schedule is gr00t_n17-only: the pi05/smolvla hooks carry no "
    "denoising-step probe, so a schedule cannot be enforced there."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gr00t_n17", choices=sorted(MODELS),
                   help="registry key (harness/registry.py); fills unset "
                        "protocol defaults below")
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--axis", default="language", help="language|light|... or 'none'")
    p.add_argument("--episodes", type=int, required=True,
                   help="0 = every curated task exactly once")
    p.add_argument("--model-path", default=None,
                   help="default: the registry's checkpoint root for --suite")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    # defaults of None resolve from the ModelSpec (gr00t_n17: official
    # examples/LIBERO protocol — 720 env-step cap, execute 8 of the 16-step
    # decoded chunk)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--exec-horizon", type=int, default=None,
                   help="execute first k of each chunk (gr00t_n17 official: 8)")
    p.add_argument("--video-dir", default=None,
                   help="record one mp4 (agentview+wrist) per episode into this dir")
    p.add_argument("--instruction-source", default="bddl",
                   choices=["bddl", "task-meta"],
                   help="where the episode instruction comes from. 'bddl' (default): "
                        "liberoplus's own BDDL parse via env.language_instruction — "
                        "the harness contract for every perturbation axis. "
                        "'task-meta': the training-distribution string parsed from the "
                        "task FILENAME (libero benchmark grab_language_from_filename), "
                        "--axis none only — for checkpoints fine-tuned on filename-"
                        "derived strings, where the BDDL parse is out-of-distribution "
                        "phrasing (smolvla anchor protocol, 2026-08-02)")
    p.add_argument("--pladis-install", action="store_true")
    p.add_argument("--pladis-scale", type=float, default=0.0)
    p.add_argument("--pladis-qgroup", default="all", choices=["all", "state", "action"],
                   help="gr00t_n17 only: query row group (pi0.5's suffix is action-only)")
    p.add_argument("--pladis-kind", default="all",
                   choices=["all", "text", "image", "prefix", "state", "self",
                            "cams", "text-image"],
                   help="key group. gr00t_n17: which cross blocks (all|text|image). "
                        "pi05: which key sub-block of the joint-attention row "
                        "('prefix' = the whole conditioning span as one "
                        "mass-preserving block). smolvla: CA key sub-block "
                        "(text|image), multi-block mass-preserving 'cams' "
                        "(per-camera) / 'text-image', whole-row 'prefix', or "
                        "'self' = SA layers. Each hook rejects kinds outside its "
                        "own set at install")
    p.add_argument("--pladis-cells", default=None,
                   help="gr00t_n17 only: comma-separated {qgroup}x{kind} cells with "
                        "per-kind qgroups (e.g. actionxtext,stateximage); overrides "
                        "qgroup/kind")
    p.add_argument("--pladis-method", default=None,
                   help="sparse transform; defaults per track (gr00t_n17 ent15max / "
                        "pi05 entmax15). sparsemax and softmax are valid on both")
    p.add_argument("--pladis-beta", type=float, default=1.0,
                   help="sparse-branch inverse temperature: sparse = method(beta*logits). "
                        "With --pladis-method softmax and beta>1 this is the paper's "
                        "S G.1 temperature-sharpened softmax control (tau = 1/beta)")
    p.add_argument("--pladis-sparse-backend", default="entmax",
                   choices=["entmax", "adasplash"],
                   help="pi05 only: implementation of the sparse transform. `entmax` is "
                        "exact/sorting-based (default, reproduces existing eplogs); "
                        "`adasplash` is the Triton kernel — identical support, ~5x faster "
                        "at this hook's shapes. Recorded in the arm signature")
    p.add_argument("--pladis-n-state-tokens", type=int, default=None,
                   help="leading state query rows; splits the [state; action] "
                        "sequence for --pladis-qgroup (default: per model)")
    p.add_argument("--pladis-schedule", default="all",
                   help="gr00t_n17 only: per-denoising-step MULTIPLIER on "
                        "--pladis-scale, one weight per step, e.g. '1,1,0,0' (early) "
                        "or '0,0.5,1,1.5' (increasing ramp); default 'all' = one "
                        "strength everywhere. The head runs N=4 Euler steps at "
                        "t in {0,.25,.5,.75}; a zero-weight step takes the vanilla "
                        "fused-SDPA path, so this is the TIME coordinate of the locus")
    # pi0.5 key-axis geometry: [image(0:ni) | language(ni:ni+nl) | suffix]. Defaults are
    # the real pi05_libero layout — 3 image slots x 256 + max_token_len 200 = 968 prefix,
    # suffix = action_horizon 10. Re-validated against the live key_len at run time
    # (attn_pi05.py raises on mismatch) and asserted by verify_pi05_delivery.py.
    p.add_argument("--pladis-n-img-prefix", type=int, default=768,
                   help="pi05 only: width of the image key block")
    p.add_argument("--pladis-n-lang", type=int, default=200,
                   help="pi05 only: width of the language key block")
    p.add_argument("--pladis-max-suffix-query", type=int, default=100,
                   help="pi05 only: only blend forwards whose query length is <= this, "
                        "so the large-query prefix VLM pass stays dense")
    # smolvla key-axis geometry: CA key = [image(0:ni) | language(ni:ni+nl_live) | state].
    # Defaults are the official lerobot/smolvla_libero layout — 2 cams x 64 tokens = 128,
    # tokenizer_max_length 48 (fixed "max_length" padding => static prefix 177 = 128+48+1).
    # The live language width is derived per inference from the recorded prefix pass;
    # these two locate/bound it and are re-validated at run time (attn_smolvla.py
    # _geometry raises on mismatch) and by the delivery assert below.
    p.add_argument("--pladis-n-img", type=int, default=128,
                   help="smolvla only: width of the CA image key block")
    p.add_argument("--pladis-n-lang-max", type=int, default=48,
                   help="smolvla only: upper bound of the derived language block width")
    args = p.parse_args()

    if args.pladis_method is None:
        args.pladis_method = _DEFAULT_METHOD[args.model]
    else:
        args.pladis_method = _METHOD_ALIAS[args.model].get(
            args.pladis_method, args.pladis_method
        )

    # Fill protocol defaults from the ModelSpec. For gr00t_n17 the resolved
    # values reproduce the historical literals byte-for-byte — the arm
    # signature (and with it every existing eplog's resume) depends on that.
    spec = MODELS[args.model]
    if args.model_path is None:
        args.model_path = spec.default_model_path(args.suite)
    if args.max_steps is None:
        args.max_steps = spec.default_max_steps
    if args.exec_horizon is None:
        args.exec_horizon = spec.default_exec_horizon
    if args.pladis_n_state_tokens is None:
        args.pladis_n_state_tokens = spec.default_n_state_tokens

    # The BDDL-parsed instruction IS the perturbation on the language axis (README:
    # the RLinf bug that silently evaluated original instructions is why the contract
    # exists), so the task-meta override is hard-restricted to unperturbed anchors.
    if args.instruction_source != "bddl" and args.axis != "none":
        raise SystemExit(
            "[arm] --instruction-source task-meta is restricted to --axis none: on a "
            "perturbation axis the BDDL parse is the treatment, and overriding it would "
            "run original instructions while the eplog claims a perturbed arm."
        )

    # Reject cross-track flag combinations BEFORE the model load (30s+) — and before an
    # arm can start. Same spirit as the hooks' own empty-install guards: an arm whose
    # locus flags do not mean what the operator thinks must never consume a sweep.
    # (Kinds outside a hook's own set are rejected by that hook at install.)
    _SMOLVLA_GEOM_DEFAULTS = (128, 48)
    if args.model == "pi05":
        if args.pladis_kind in ("cams", "text-image", "state", "self"):
            raise SystemExit(
                "[arm] --pladis-kind cams/text-image/state/self is smolvla-only; "
                "pi05 kinds are text|image|prefix|all (attn_pi05 rejects the rest "
                "only AFTER the model load)."
            )
        if args.pladis_qgroup != "all":
            raise SystemExit(
                "[arm] --pladis-qgroup is gr00t_n17-only: pi0.5's suffix is action-only "
                "(state is embedded as discrete language keys, not a query row), so the "
                "query-group axis collapses. Drop the flag; every pi05 arm is action-row."
            )
        if args.pladis_cells:
            raise SystemExit("[arm] --pladis-cells is gr00t_n17-only (per-kind qgroups).")
        if args.pladis_n_state_tokens != 1:
            raise SystemExit("[arm] --pladis-n-state-tokens is gr00t_n17-only.")
        if args.pladis_schedule != "all":
            raise SystemExit(_SCHEDULE_TRACK_MSG)
        if (args.pladis_n_img, args.pladis_n_lang_max) != _SMOLVLA_GEOM_DEFAULTS:
            raise SystemExit(
                "[arm] --pladis-n-img/--pladis-n-lang-max are smolvla-only "
                "(pi05 geometry flags: --pladis-n-img-prefix/--pladis-n-lang)."
            )
    elif args.model == "gr00t_n17":
        if args.pladis_schedule != "all":
            # Parse here, before the model load, so a typo ('1-1-0-0', 'early') costs
            # a second rather than a suite's worth of startup. The length-vs-N and
            # all-zero checks need the live head and stay in install_pladis.
            from pladis.attn_gr00t_n17 import parse_schedule

            try:
                parse_schedule(args.pladis_schedule)
            except ValueError as exc:
                raise SystemExit(f"[arm] --pladis-schedule {args.pladis_schedule!r}: {exc}")
            if not args.pladis_install:
                raise SystemExit(
                    "[arm] --pladis-schedule without --pladis-install: a schedule with "
                    "no hook is a vanilla arm wearing an intervention's name."
                )
        if args.pladis_kind not in ("all", "text", "image"):
            raise SystemExit(
                "[arm] gr00t_n17 selects whole cross blocks: --pladis-kind all|text|"
                "image (its state axis is --pladis-qgroup). Other kinds name key-"
                "column spans of a joint/prefix row and are pi05/smolvla-only."
            )
        if (args.pladis_n_img, args.pladis_n_lang_max) != _SMOLVLA_GEOM_DEFAULTS:
            raise SystemExit("[arm] --pladis-n-img/--pladis-n-lang-max are smolvla-only.")
    elif args.model == "smolvla":
        if args.pladis_cells:
            raise SystemExit("[arm] --pladis-cells is gr00t_n17-only (per-kind qgroups).")
        if args.pladis_schedule != "all":
            raise SystemExit(_SCHEDULE_TRACK_MSG)
        if args.pladis_qgroup == "state":
            raise SystemExit(
                "[arm] --pladis-qgroup state is invalid for smolvla: the expert suffix "
                "is action-only; state is a prefix KEY token — use --pladis-kind state."
            )
        if args.pladis_install and args.pladis_kind == "all":
            # attn_smolvla would reject this too, but only AFTER the model load —
            # and the sweep would already have paid one load per suite (bug B1).
            raise SystemExit(
                "[arm] smolvla needs an explicit --pladis-kind in "
                "{text,image,cams,text-image,prefix,self} — 'all' is gr00t_n17/pi05 "
                "vocabulary. The mass-preserving whole-prefix arm is 'text-image'; "
                "'prefix' is the plain whole-row blend."
            )
        if (args.pladis_n_img_prefix, args.pladis_n_lang,
                args.pladis_max_suffix_query) != (768, 200, 100):
            raise SystemExit(
                "[arm] --pladis-n-img-prefix/--pladis-n-lang/--pladis-max-suffix-query "
                "are pi05-only (smolvla geometry: --pladis-n-img/--pladis-n-lang-max)."
            )
    return args, spec


def _git_describe() -> str:
    """Commit that produced this run (+ -dirty marker), for the .arm sidecar —
    multi-server campaigns must be attributable to exact code versions."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _model_tag(model_path: str) -> str:
    """Human-readable VLA name from the checkpoint path.
    .../GR00T-N1.7-LIBERO/libero_10 (HF repo nvidia/GR00T-N1.7-LIBERO, one
    subdir per suite) -> "GR00T N1.7 (libero_10)". Unknown layouts fall back
    to the last two path components."""
    p = os.path.normpath(model_path)
    sub, repo = os.path.basename(p), os.path.basename(os.path.dirname(p))
    m = re.search(r"(?i)gr00t[-_ ]?n(\d+(?:\.\d+)?)", f"{repo} {sub}")
    if m:
        model = f"GR00T N{m.group(1)}"
        return f"{model} ({sub})" if re.search(r"(?i)gr00t", repo) else model
    return f"{repo}/{sub}"


def _task_meta_instruction(base_task: str) -> str:
    """Training-distribution instruction: the task-FILENAME parse both the LIBERO
    benchmark loader and lerobot's eval stack feed the policy
    (libero/benchmark/__init__.py:43-52 ``grab_language_from_filename``, replicated
    verbatim on ``base_task + ".bddl"`` so no libero import is needed; the upper-case
    branch strips the LIBERO-100 ``*_SCENE<d>_`` prefix).

    Why it exists (2026-08-02): smolvla checkpoints are fine-tuned on these strings
    (lerobot/libero ``meta/tasks.parquet``), while liberoplus's BDDL parse phrases the
    same tasks differently ("pick the akita black bowl ..." vs trained "pick up the
    black bowl ..."; goal: "open the middle layer of the drawer" vs trained "open the
    middle drawer of the cabinet") — out-of-distribution phrasing that collapsed the
    smolvla anchors (spatial 41 vs paper 90). Verified per-suite against the anchor
    eplogs' instruction column."""
    x = base_task + ".bddl"
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8:].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7:].split("_"))
    else:
        language = " ".join(x.split("_"))
    return language[: language.find(".bddl")]


def _assert_pi05_delivery(sess, ts, first_spec, model) -> None:
    """Prove the PLADIS blend actually fires, BEFORE logging a single episode.

    attn_pi05._uses_eager_gemma reads ``model.config._attn_implementation``; the openpi
    policy object has no such attribute, so it returns None and install_pladis proceeds
    with its guard collapsed onto assert_delivered() (attn_pi05.py:165-176, 234-243).
    Handing it the gemma submodule instead would RAISE, because that config only becomes
    "eager" during the first suffix step (pi0_pytorch.py:447). So the only correct order
    is: install -> run one throwaway chunk -> assert.

    Without this an arm can burn all 1,537 episodes running as vanilla while the eplog
    and the .arm sidecar both claim it was an intervention.

    The warm-up reuses the FIRST scheduled episode's reset, and its RNG effect is
    contained: run_episode reseeds from (episode_seed, step) before every chunk
    (harness/rollout.py) and sess.reset() re-runs for real when the loop starts, so
    the logged rollout is bit-identical to one without the warm-up.
    """
    from pladis.attn_pi05 import CFG, assert_delivered

    raw_obs, instruction = sess.reset(first_spec, ts.init_states_of(first_spec.task_name))
    with torch.no_grad():
        model.predict_action_batch(model.wrap_obs(raw_obs, instruction), mode="eval")
    assert_delivered()
    print(f"[arm] PLADIS delivered: {CFG.n_calls} blended forward(s), "
          f"(query_len, key_len) seen = {sorted(CFG.seen_shapes)}", flush=True)


def _assert_smolvla_delivery(sess, ts, first_spec, model) -> None:
    """smolvla counterpart of :func:`_assert_pi05_delivery` — prove the blend fires
    BEFORE logging a single episode.

    The failure mode it guards: install_pladis binds the replacement to one
    SmolVLMWithExpertModel INSTANCE (attn_smolvla.py, review F2); a hook bound to an
    object the serving path never calls raises nothing on its own, and the arm would
    burn 1,537 episodes as silent vanilla while the .arm sidecar claims an
    intervention. assert_delivered() (attn_smolvla.py) converts that silence into a
    hard error; the CA/SA call census printed here is the same evidence the on-ckpt
    delivery smoke checks (official ckpt: 80 CA or 80 SA blended calls per chunk).

    The warm-up surface differs from pi05's (predict_chunk through the checkpoint's
    own pre/post processors vs predict_action_batch(mode="eval")) — two explicit
    branches rather than a ModelSpec hook, same call as _assert_pi05_delivery made.
    RNG containment is identical: run_episode reseeds the global stream before every
    chunk (harness/rollout.py) and sess.reset() re-runs for real when the loop
    starts, so the logged rollout is bit-identical to one without the warm-up."""
    from pladis.attn_smolvla import CFG, assert_delivered

    raw_obs, instruction = sess.reset(first_spec, ts.init_states_of(first_spec.task_name))
    with torch.no_grad():
        model.predict_chunk(model.wrap_obs(raw_obs, instruction))
    assert_delivered()
    print(f"[arm] PLADIS delivered: CA={CFG.n_calls_ca} SA={CFG.n_calls_sa} blended "
          f"forward(s), prefix_len={CFG.prefix_len}", flush=True)


def _assert_n17_step_delivery(sess, ts, first_spec, model) -> None:
    """gr00t_n17 counterpart for --pladis-schedule: prove the blend fired at exactly
    the weighted denoising steps, at the right strengths, BEFORE logging an episode.

    Block selection is already proven by install_pladis's non-empty return, but the
    step gate is enforced at RUN time by a forward pre-hook, so nothing before the
    first inference can tell whether it fires. A probe bound to a DiT the serving
    path never calls, or a schedule naming steps the loop never reaches, would run
    the arm as vanilla (or at full schedule) for all 1,537 episodes while the eplog
    and .arm sidecar claim a partial schedule.

    RNG containment is the same as the pi05/smolvla warm-ups: run_episode reseeds the
    global stream before every chunk (harness/rollout.py:131) and sess.reset() re-runs
    for real when the loop starts, so the logged rollout is bit-identical to one
    without the warm-up.
    """
    from pladis.attn_gr00t_n17 import assert_delivered

    raw_obs, instruction = sess.reset(first_spec, ts.init_states_of(first_spec.task_name))
    with torch.no_grad():
        model.predict_chunk(model.wrap_obs(raw_obs, instruction))
    print(f"[arm] PLADIS step schedule delivered: {assert_delivered()}", flush=True)


def main():
    args, spec = parse_args()
    axis = None if args.axis == "none" else args.axis

    # Everything that determines what an episode row means. The eplog is the
    # resume ledger and carries no arm identity of its own, so this is what
    # stops a re-run with different flags from appending into another arm's
    # file (harness/eplog.py).
    if not args.pladis_install:
        pladis_clause = "pladis=off"
    elif args.model == "pi05":
        # The geometry belongs in the signature: WHICH key sub-block gets sharpened is
        # the experiment, so an eplog produced with a wrong n_lang must not resume into
        # a correct one. No qgroup/cells/ns here — they do not exist for pi0.5.
        # The sparse backend is APPENDED ONLY WHEN NON-DEFAULT. adasplash and entmax 1.3
        # keep identical support (verified at all three block widths; see attn_pi05.py),
        # but they are still different kernels, so an arm must not silently mix them —
        # hence it belongs in the signature. Emitting nothing for the default keeps every
        # signature written before 2026-07-31 byte-identical, so the 12,296 episodes of
        # the language campaign still resume (harness/eplog.py:62-80 aborts on any change).
        backend_clause = (
            "" if args.pladis_sparse_backend == "entmax"
            else f",be{args.pladis_sparse_backend}"
        )
        pladis_clause = (
            f"pladis=scale{args.pladis_scale:g},{args.pladis_method},"
            f"b{args.pladis_beta:g},k{args.pladis_kind},"
            f"ni{args.pladis_n_img_prefix},nl{args.pladis_n_lang},"
            f"msq{args.pladis_max_suffix_query}{backend_clause}"
        )
    elif args.model == "smolvla":
        # Geometry in the signature for the same reason as pi05: WHICH key sub-block
        # gets sharpened is the experiment. No qgroup/cells — the suffix is action-only.
        # Reshaping this clause is safe as of 2026-08-02: every existing smolvla eplog
        # (anchors + gates) carries "pladis=off", no installed smolvla eplog exists.
        pladis_clause = (
            f"pladis=scale{args.pladis_scale:g},{args.pladis_method},"
            f"b{args.pladis_beta:g},k{args.pladis_kind},"
            f"ni{args.pladis_n_img},nlmax{args.pladis_n_lang_max},"
            f"ns{args.pladis_n_state_tokens}"
        )
    else:
        # The denoising-step schedule is part of the locus, so it belongs in the
        # signature — but APPENDED ONLY WHEN NON-DEFAULT, the same append-only
        # discipline as pi05's backend_clause: the 44k+ episodes already logged on
        # this track were written before the step axis existed and must keep
        # resuming byte-identically (harness/eplog.py:62-80 aborts on any change).
        if args.pladis_schedule == "all":
            steps_clause = ""
        else:
            from pladis.attn_gr00t_n17 import fmt_schedule, parse_schedule

            steps_clause = f",sched{fmt_schedule(parse_schedule(args.pladis_schedule))}"
        pladis_clause = (
            f"pladis=scale{args.pladis_scale:g},{args.pladis_method},"
            f"b{args.pladis_beta:g},"
            + (f"cells[{args.pladis_cells}],"
               if args.pladis_cells
               else f"q{args.pladis_qgroup},k{args.pladis_kind},")
            + f"ns{args.pladis_n_state_tokens}"
            + steps_clause
        )

    arm_signature = "|".join(
        [
            f"suite={args.suite}",
            f"axis={args.axis}",
            f"seed={args.seed}",
            f"model={os.path.normpath(args.model_path)}",
            # DELIBERATE ASYMMETRY — do not "clean up". model_kind is emitted only for
            # tracks that postdate the field, so that every eplog written before it
            # existed still resumes byte-identically (eplog.py:76 aborts otherwise, and
            # there are in-flight gr00t_n17 sweeps). smolvla is also exempt: its anchor
            # eplogs (results/smolvla_*_eplog.tsv, 2026-07-3x) predate the field, and
            # the checkpoint path in model= already separates the tracks.
            *([f"model_kind={args.model}"]
              if args.model not in ("gr00t_n17", "smolvla") else []),
            f"max_steps={args.max_steps}",
            f"exec_horizon={args.exec_horizon}",
            pladis_clause,
            # emitted only when non-default, so every eplog written before the flag
            # existed keeps its byte-identical signature and still resumes — the same
            # append-only discipline as pi05's backend_clause above.
            *([f"instr={args.instruction_source}"]
              if args.instruction_source != "bddl" else []),
        ]
    )
    print(f"[arm] signature {arm_signature}", flush=True)
    code_version = _git_describe()
    print(f"[arm] code {code_version}", flush=True)

    ts = LiberoPlusTaskSet(args.suite, axis)
    n_eps = len(ts.task_names) if args.episodes == 0 else args.episodes
    sched = ts.schedule(n_eps, seed=args.seed)
    log = EpisodeLogger(args.out, resume=True, arm_signature=arm_signature,
                        provenance=code_version)
    todo = [s for s in sched if s.episode not in log.done_episodes]
    print(f"[arm] {len(todo)}/{len(sched)} episodes to run -> {args.out}", flush=True)
    if not todo:
        # resume no-op: exit before the model load — sweep drivers re-invoke
        # every arm on every run, and completed arms should cost seconds.
        log.close()
        print(f"[arm] DONE 0 eps (resume: all {len(sched)} already logged)", flush=True)
        return

    model = resolve_loader(spec)(args.model_path)
    if args.pladis_install and args.model == "pi05":
        # pi0.5's hook has its own kwarg surface (key-block geometry, sparse
        # backend, no qgroup axis). No block list to print — delivery is proven
        # (and printed) by _assert_pi05_delivery below, before any episode is
        # logged.
        resolve_hooks(spec).install_pladis(
            model,
            pladis_scale=args.pladis_scale,
            method=args.pladis_method,
            beta=args.pladis_beta,
            kind=args.pladis_kind,
            n_img_prefix=args.pladis_n_img_prefix,
            n_lang=args.pladis_n_lang,
            max_suffix_query=args.pladis_max_suffix_query,
            sparse_backend=args.pladis_sparse_backend,
        )
    elif args.pladis_install and args.model == "smolvla":
        # smolvla's kwarg surface: kind + CA key geometry, no qgroup/cells (the suffix
        # is action-only). install_pladis prints its own status line, and delivery is
        # proven by _assert_smolvla_delivery below — no block list to print here.
        resolve_hooks(spec).install_pladis(
            model,
            pladis_scale=args.pladis_scale,
            method=args.pladis_method,
            beta=args.pladis_beta,
            kind=args.pladis_kind,
            n_img=args.pladis_n_img,
            n_lang_max=args.pladis_n_lang_max,
            n_state_tokens=args.pladis_n_state_tokens,
        )
    elif args.pladis_install:
        hooks = resolve_hooks(spec)
        if args.pladis_cells:
            install_cells = getattr(hooks, "install_pladis_cells", None)
            if install_cells is None:
                raise SystemExit(
                    f"[arm] hook module {spec.hook_module} has no install_pladis_cells "
                    f"— --pladis-cells is not supported for model {spec.name!r}"
                )
            installed = install_cells(
                model,
                args.pladis_cells,
                pladis_scale=args.pladis_scale,
                method=args.pladis_method,
                beta=args.pladis_beta,
                n_state_tokens=args.pladis_n_state_tokens,
                schedule=args.pladis_schedule,
            )
        else:
            installed = hooks.install_pladis(
                model,
                pladis_scale=args.pladis_scale,
                method=args.pladis_method,
                beta=args.pladis_beta,
                kind=args.pladis_kind,
                qgroup=args.pladis_qgroup,
                n_state_tokens=args.pladis_n_state_tokens,
                schedule=args.pladis_schedule,
            )
        print(f"[arm] PLADIS installed on blocks {installed}", flush=True)
    else:
        print("[arm] vanilla (no hook)", flush=True)

    # model/arm tag for the video header, e.g.
    # "GR00T N1.7 (libero_10) | action x text (scale=1)"
    model_tag = _model_tag(args.model_path)
    if not args.pladis_install:
        arm_tag = "vanilla"
    elif args.pladis_scale == 0:
        arm_tag = "base0 (hook s=0)"
    elif args.pladis_cells:
        arm_tag = f"{args.pladis_cells} (s={args.pladis_scale:g})"
    elif args.model in ("pi05", "smolvla"):
        # no query-group axis on either — the locus IS the key sub-block ("all x text"
        # would misname a smolvla arm: there is no qgroup to cross with)
        arm_tag = f"{args.pladis_kind} keys (s={args.pladis_scale:g})"
    else:
        arm_tag = f"{args.pladis_qgroup} x {args.pladis_kind} (s={args.pladis_scale:g})"
    if args.pladis_install and args.pladis_schedule != "all":
        # the schedule is part of what a reviewer watching the video is judging
        arm_tag += f" sched {args.pladis_schedule}"
    video_label = f"{model_tag} | {arm_tag}"

    from harness.env import RUNTIME_RNG_AXES

    sess = LiberoPlusSession(seed=args.seed,
                             per_episode_np_seed=axis in RUNTIME_RNG_AXES)
    if args.model == "pi05" and args.pladis_install:
        _assert_pi05_delivery(sess, ts, todo[0], model)
    elif args.model == "smolvla" and args.pladis_install:
        _assert_smolvla_delivery(sess, ts, todo[0], model)
    elif (args.model == "gr00t_n17" and args.pladis_install
          and args.pladis_schedule != "all"):
        _assert_n17_step_delivery(sess, ts, todo[0], model)

    instruction_map = (
        (lambda spec: _task_meta_instruction(spec.base_task))
        if args.instruction_source == "task-meta" else None
    )

    t0, n_succ, n_run = time.time(), 0, 0
    for spec in todo:
        r = run_episode(
            sess,
            spec,
            ts.init_states_of(spec.task_name),
            model,
            episode_seed=args.seed * 1_000_003 + spec.episode,
            max_steps=args.max_steps,
            exec_horizon=args.exec_horizon,
            video_dir=args.video_dir,
            video_label=video_label,
            video_suite={
                "libero_10": "long",
                "libero_spatial": "spatial",
                "libero_object": "object",
                "libero_goal": "goal",
            }.get(args.suite, args.suite),
            instruction_map=instruction_map,
        )
        log.log(r)
        n_run += 1
        n_succ += r.success_once
        if n_run % 10 == 0:
            print(
                f"[arm] {n_run}/{len(todo)} running-SR={n_succ / n_run:.3f} "
                f"({(time.time() - t0) / n_run:.1f}s/ep)",
                flush=True,
            )
    sess.close()
    log.close()
    print(f"[arm] DONE {n_run} eps, SR={n_succ / max(n_run, 1):.4f}", flush=True)


if __name__ == "__main__":
    main()
