# SPDX-License-Identifier: Apache-2.0
"""Generic single-arm evaluator: anchor runs, parity checks, and sweep arms
all go through this one entry point so every result shares one code path.

  bash experiments/run.sh experiments/eval_arm.py \
      --axis language --episodes 100 --seed 0 --out results/foo_eplog.tsv \
      [--pladis-scale 1.0 --pladis-qgroup action --pladis-kind image]

  bash experiments/run.sh --venv openpi experiments/eval_arm.py --model pi05 \
      --axis language --episodes 0 --seed 0 --max-steps 520 --exec-horizon 5 \
      --out results/foo_eplog.tsv [--pladis-install --pladis-scale 1.0 --pladis-kind text]

Two model tracks share this evaluator (`--model`): gr00t_n17 and pi05. Keeping them on
ONE evaluator is deliberate — a sibling script would duplicate the resume ledger, the
seeded schedule, the arm signature, the git provenance and the video labelling, and a
silent divergence between the two copies would leave no trace in the eplogs (the TSV
carries no arm identity, harness/eplog.py:8-15). The model-specific surface is one
import branch, one install branch and one tag branch.

PLADIS is installed explicitly (pladis/attn_gr00t_n17.py, pladis/attn_pi05.py), never
via env vars. Omitting --pladis-install gives vanilla.
  * gr00t_n17: --pladis-scale 0 with --pladis-install gives base0 — the hook is
    installed but delegates to the native fused SDPA (official PLADIS lambda=0
    semantics), so base0 is BIT-identical to vanilla.
  * pi05: lambda=0 returns the plain softmax the stock gemma eager path computes, so
    base0 is likewise bit-identical (verify_pi05_hook.py gate A). pi0.5's suffix is
    action-only, so the qgroup axis does not exist there; the locus axis is --pladis-kind
    over the key sub-blocks (text / image / prefix / all).

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
from harness.rollout import run_episode

MODEL = os.environ.get(
    "GR00T_MODEL_PATH",
    "/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO/libero_10",
)

# The two hooks spell the same transform differently (attn_gr00t_n17.py:74 "ent15max"
# vs attn_pi05.py:53 "entmax15"). --pladis-method therefore defaults to None and is
# resolved per track BEFORE the arm signature is built, so every pre-existing gr00t_n17
# signature string stays byte-identical and its eplogs still resume
# (harness/eplog.py:62-80 aborts on any signature change).
_DEFAULT_METHOD = {"gr00t_n17": "ent15max", "pi05": "entmax15"}
_METHOD_ALIAS = {
    "gr00t_n17": {"entmax15": "ent15max"},
    "pi05": {"ent15max": "entmax15"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gr00t_n17", choices=["gr00t_n17", "pi05"],
                   help="model track; selects the loader and the PLADIS hook")
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--axis", default="language", help="language|light|... or 'none'")
    p.add_argument("--episodes", type=int, required=True,
                   help="0 = every curated task exactly once")
    p.add_argument("--model-path", default=MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    # official examples/LIBERO protocol: 720 env-step cap, execute 8 of the
    # 16-step decoded chunk (receding horizon)
    p.add_argument("--max-steps", type=int, default=720)
    p.add_argument("--exec-horizon", type=int, default=8,
                   help="execute first k of each chunk (official protocol: 8)")
    p.add_argument("--video-dir", default=None,
                   help="record one mp4 (agentview+wrist) per episode into this dir")
    p.add_argument("--pladis-install", action="store_true")
    p.add_argument("--pladis-scale", type=float, default=0.0)
    p.add_argument("--pladis-qgroup", default="all", choices=["all", "state", "action"],
                   help="gr00t_n17 only: query row group (pi0.5's suffix is action-only)")
    p.add_argument("--pladis-kind", default="all",
                   choices=["all", "text", "image", "prefix"],
                   help="key group. gr00t_n17: which cross blocks. pi05: which key "
                        "sub-block of the joint-attention row ('prefix' = the whole "
                        "conditioning span as one mass-preserving block; gr00t_n17 has "
                        "no 'prefix')")
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
    p.add_argument("--pladis-n-state-tokens", type=int, default=1,
                   help="gr00t_n17 only: leading state query rows; splits the "
                        "[state; action] sequence for --pladis-qgroup (N1.7: 1)")
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
    args = p.parse_args()

    if args.pladis_method is None:
        args.pladis_method = _DEFAULT_METHOD[args.model]
    else:
        args.pladis_method = _METHOD_ALIAS[args.model].get(
            args.pladis_method, args.pladis_method
        )

    # Reject cross-track flag combinations BEFORE the model load (30s+) — and before an
    # arm can start. Same spirit as the hooks' own empty-install guards: an arm whose
    # locus flags do not mean what the operator thinks must never consume a sweep.
    if args.model == "pi05":
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
    else:
        if args.pladis_kind == "prefix":
            raise SystemExit(
                "[arm] --pladis-kind prefix is pi05-only (it names a key-column span of "
                "the joint-attention row; gr00t_n17 selects whole cross blocks)."
            )
    return args


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
    (rollout.py:148) and sess.reset() re-runs for real when the loop starts, so the
    logged rollout is bit-identical to one without the warm-up.
    """
    from pladis.attn_pi05 import CFG, assert_delivered

    raw_obs, instruction = sess.reset(first_spec, ts.init_states_of(first_spec.task_name))
    with torch.no_grad():
        model.predict_action_batch(model.wrap_obs(raw_obs, instruction), mode="eval")
    assert_delivered()
    print(f"[arm] PLADIS delivered: {CFG.n_calls} blended forward(s), "
          f"(query_len, key_len) seen = {sorted(CFG.seen_shapes)}", flush=True)


def main():
    args = parse_args()
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
    else:
        pladis_clause = (
            f"pladis=scale{args.pladis_scale:g},{args.pladis_method},"
            f"b{args.pladis_beta:g},"
            + (f"cells[{args.pladis_cells}],"
               if args.pladis_cells
               else f"q{args.pladis_qgroup},k{args.pladis_kind},")
            + f"ns{args.pladis_n_state_tokens}"
        )

    arm_signature = "|".join(
        [
            f"suite={args.suite}",
            f"axis={args.axis}",
            f"seed={args.seed}",
            f"model={os.path.normpath(args.model_path)}",
            # DELIBERATE ASYMMETRY — do not "clean up". model_kind is emitted only for
            # non-default tracks so that every eplog written before the pi0.5 track
            # existed still resumes byte-identically (eplog.py:76 aborts otherwise, and
            # there are in-flight gr00t_n17 sweeps). Adding it unconditionally would
            # invalidate all of them.
            *([f"model_kind={args.model}"] if args.model != "gr00t_n17" else []),
            f"max_steps={args.max_steps}",
            f"exec_horizon={args.exec_horizon}",
            pladis_clause,
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

    if args.model == "pi05":
        from harness.model_pi05 import load_pi05

        model = load_pi05(args.model_path)
    else:
        from harness.model_gr00t import load_gr00t_n1d7

        model = load_gr00t_n1d7(args.model_path)

    if args.model == "pi05":
        if args.pladis_install:
            from pladis.attn_pi05 import install_pladis as install_pladis_pi05

            install_pladis_pi05(
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
        else:
            print("[arm] vanilla (no hook)", flush=True)
    elif args.pladis_install:
        if args.pladis_cells:
            from pladis.attn_gr00t_n17 import install_pladis_cells

            installed = install_pladis_cells(
                model,
                args.pladis_cells,
                pladis_scale=args.pladis_scale,
                method=args.pladis_method,
                beta=args.pladis_beta,
                n_state_tokens=args.pladis_n_state_tokens,
            )
        else:
            from pladis.attn_gr00t_n17 import install_pladis

            installed = install_pladis(
                model,
                pladis_scale=args.pladis_scale,
                method=args.pladis_method,
                beta=args.pladis_beta,
                kind=args.pladis_kind,
                qgroup=args.pladis_qgroup,
                n_state_tokens=args.pladis_n_state_tokens,
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
    elif args.model == "pi05":
        # no query-group axis on pi0.5 — the locus IS the key sub-block
        arm_tag = f"{args.pladis_kind} keys (s={args.pladis_scale:g})"
    else:
        arm_tag = f"{args.pladis_qgroup} x {args.pladis_kind} (s={args.pladis_scale:g})"
    video_label = f"{model_tag} | {arm_tag}"

    sess = LiberoPlusSession(seed=args.seed)
    if args.model == "pi05" and args.pladis_install:
        _assert_pi05_delivery(sess, ts, todo[0], model)

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
