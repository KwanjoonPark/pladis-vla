# SPDX-License-Identifier: Apache-2.0
"""Generic single-arm evaluator: anchor runs, parity checks, and sweep arms
all go through this one entry point so every result shares one code path.

  bash experiments/run.sh experiments/eval_arm.py \
      --axis language --episodes 100 --seed 0 --out results/foo_eplog.tsv \
      [--pladis-scale 1.0 --pladis-qgroup action --pladis-kind image]

PLADIS is installed explicitly (pladis/attn_gr00t.py), never via env vars.
--pladis-scale 0 with --pladis-install gives base0: the hook is installed but
delegates to the native fused SDPA (official PLADIS lambda=0 semantics), so
base0 is BIT-identical to vanilla. Omitting --pladis-install gives vanilla.
Resume: episodes already in --out are skipped (eplog is the ledger).
"""

from __future__ import annotations

import argparse
import os
import re
import time

from harness.env import LiberoPlusTaskSet, LiberoPlusSession
from harness.eplog import EpisodeLogger
from harness.model_gr00t import load_gr00t_n1d7
from harness.rollout import run_episode

MODEL = os.environ.get(
    "GR00T_MODEL_PATH",
    "/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO/libero_10",
)


def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--pladis-qgroup", default="all", choices=["all", "state", "action"])
    p.add_argument("--pladis-kind", default="all", choices=["all", "text", "image"])
    p.add_argument("--pladis-cells", default=None,
                   help="comma-separated {qgroup}x{kind} cells with per-kind qgroups "
                        "(e.g. actionxtext,stateximage); overrides qgroup/kind")
    p.add_argument("--pladis-method", default="ent15max")
    p.add_argument("--pladis-beta", type=float, default=1.0,
                   help="sparse-branch inverse temperature: sparse = method(beta*logits). "
                        "With --pladis-method softmax and beta>1 this is the paper's "
                        "S G.1 temperature-sharpened softmax control (tau = 1/beta)")
    p.add_argument("--pladis-n-state-tokens", type=int, default=1,
                   help="leading state query rows; splits the [state; action] "
                        "sequence for --pladis-qgroup (N1.7: 1)")
    return p.parse_args()


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


def main():
    args = parse_args()
    axis = None if args.axis == "none" else args.axis

    # Everything that determines what an episode row means. The eplog is the
    # resume ledger and carries no arm identity of its own, so this is what
    # stops a re-run with different flags from appending into another arm's
    # file (harness/eplog.py).
    arm_signature = "|".join(
        [
            f"suite={args.suite}",
            f"axis={args.axis}",
            f"seed={args.seed}",
            f"model={os.path.normpath(args.model_path)}",
            f"max_steps={args.max_steps}",
            f"exec_horizon={args.exec_horizon}",
            "pladis=off" if not args.pladis_install else (
                f"pladis=scale{args.pladis_scale:g},{args.pladis_method},"
                f"b{args.pladis_beta:g},"
                + (f"cells[{args.pladis_cells}],"
                   if args.pladis_cells
                   else f"q{args.pladis_qgroup},k{args.pladis_kind},")
                + f"ns{args.pladis_n_state_tokens}"
            ),
        ]
    )
    print(f"[arm] signature {arm_signature}", flush=True)

    ts = LiberoPlusTaskSet(args.suite, axis)
    n_eps = len(ts.task_names) if args.episodes == 0 else args.episodes
    sched = ts.schedule(n_eps, seed=args.seed)
    log = EpisodeLogger(args.out, resume=True, arm_signature=arm_signature)
    todo = [s for s in sched if s.episode not in log.done_episodes]
    print(f"[arm] {len(todo)}/{len(sched)} episodes to run -> {args.out}", flush=True)
    if not todo:
        # resume no-op: exit before the model load — sweep drivers re-invoke
        # every arm on every run, and completed arms should cost seconds.
        log.close()
        print(f"[arm] DONE 0 eps (resume: all {len(sched)} already logged)", flush=True)
        return

    model = load_gr00t_n1d7(args.model_path)
    if args.pladis_install:
        if args.pladis_cells:
            from pladis.attn_gr00t import install_pladis_cells

            installed = install_pladis_cells(
                model,
                args.pladis_cells,
                pladis_scale=args.pladis_scale,
                method=args.pladis_method,
                beta=args.pladis_beta,
                n_state_tokens=args.pladis_n_state_tokens,
            )
        else:
            from pladis.attn_gr00t import install_pladis

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
    else:
        arm_tag = f"{args.pladis_qgroup} x {args.pladis_kind} (s={args.pladis_scale:g})"
    video_label = f"{model_tag} | {arm_tag}"

    sess = LiberoPlusSession(seed=args.seed)
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
