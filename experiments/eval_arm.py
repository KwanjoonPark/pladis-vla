# SPDX-License-Identifier: Apache-2.0
"""Generic single-arm evaluator: anchor runs, parity checks, and sweep arms
all go through this one entry point so every result shares one code path.

  bash experiments/run.sh experiments/eval_arm.py \
      --axis language --episodes 100 --seed 0 --out results/foo_eplog.tsv \
      [--pladis-scale 1.0 --pladis-qgroup action --pladis-kind image]

PLADIS is installed explicitly (pladis/attn_gr00t.py), never via env vars.
--pladis-scale 0 with --pladis-install gives the hook-installed exact-dense
arm (base0). Omitting --pladis-install entirely gives vanilla.
Resume: episodes already in --out are skipped (eplog is the ledger).
"""

from __future__ import annotations

import argparse
import time

from harness.env import LiberoPlusTaskSet, LiberoPlusSession
from harness.eplog import EpisodeLogger
from harness.model_gr00t import load_gr00t_n1d7
from harness.rollout import run_episode

MODEL = "/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO/libero_10"
BACKBONE = (
    "/home/reallab/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/"
    "snapshots/9ce19a195e423419c349abfc86fd07178b230561"
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
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--exec-horizon", type=int, default=None,
                   help="execute first k of each chunk (Isaac-GR00T protocol: 8)")
    p.add_argument("--pladis-install", action="store_true")
    p.add_argument("--pladis-scale", type=float, default=0.0)
    p.add_argument("--pladis-qgroup", default="all", choices=["all", "state", "action"])
    p.add_argument("--pladis-kind", default="all", choices=["all", "text", "image"])
    p.add_argument("--pladis-method", default="ent15max")
    return p.parse_args()


def main():
    args = parse_args()
    axis = None if args.axis == "none" else args.axis

    model = load_gr00t_n1d7(args.model_path, BACKBONE)
    if args.pladis_install:
        from pladis.attn_gr00t import install_pladis

        installed = install_pladis(
            model,
            pladis_scale=args.pladis_scale,
            method=args.pladis_method,
            kind=args.pladis_kind,
            qgroup=args.pladis_qgroup,
        )
        print(f"[arm] PLADIS installed on blocks {installed}", flush=True)
    else:
        print("[arm] vanilla (no hook)", flush=True)

    ts = LiberoPlusTaskSet(args.suite, axis)
    n_eps = len(ts.task_names) if args.episodes == 0 else args.episodes
    sched = ts.schedule(n_eps, seed=args.seed)
    log = EpisodeLogger(args.out, resume=True)
    todo = [s for s in sched if s.episode not in log.done_episodes]
    print(f"[arm] {len(todo)}/{len(sched)} episodes to run -> {args.out}", flush=True)

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
