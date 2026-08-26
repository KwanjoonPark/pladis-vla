# SPDX-License-Identifier: Apache-2.0
"""Phase 0 of docs/nag.md: measure the L1 ratio R, then pick tau from it.

    R = ||Z_d + lambda*(Z_s - Z_d)||_1 / ||Z_d||_1        per (head, query row)

tau=2.5 is a Flux/SD3.5 number. Our two branches differ by a SHARPENING OPERATOR,
not by a prompt, so nothing licenses transferring it: a tau nothing exceeds makes
a NAG arm bit-identical to its uncapped control (docs/nag.md §2a), and a tau that
bites at every dose only rescales the ladder (Failure A of §5.2). Both are
sweep-sized mistakes, and both are visible here for ~1 GPU-hour.

One rollout prices the WHOLE dose ladder. The blend is affine in lambda through Z,
so from the arm's own two features
    R(l) = ||Z_d + (l/lambda)*(Z_PL - Z_d)||_1 / ||Z_d||_1
is exact for every other rung (gate H of verify_nag.py) — the rungs are therefore
compared on ONE trajectory, not on four different ones.

The report answers the two questions §6 pre-registers:
  * tau* = the smallest candidate with clip(1) <= 5%, clip(1.5) <= 10%, clip(3) >= 20%
    — inert where each axis's ladder is healthy, active where it turns over;
  * the cross-axis ranking of clip rate at matched (lambda, tau), which decides
    whether `noise` (20 h/arm) is funded at all.

Run (per axis; the trajectory is the --scale arm's, i.e. the shared operating point):

    bash experiments/run.sh experiments/diag_nag.py \\
      --suite libero_10 --axis language --episodes 6

(--axis none = the unperturbed set; --model-path defaults to the registry's.)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from harness.env import RUNTIME_RNG_AXES, LiberoPlusSession, LiberoPlusTaskSet
from harness.registry import MODELS, resolve_loader
from harness.rollout import run_episode
from pladis.attn_gr00t_n17 import NAG, NAG_CANDIDATE_TAUS, install_pladis

# docs/nag.md §6: the rule is stated on these three rungs, so they are always priced.
LADDER = (1.0, 1.5, 2.0, 3.0)
RULE = ((1.0, 0.05, "<="), (1.5, 0.10, "<="), (3.0, 0.20, ">="))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True)
    p.add_argument("--axis", required=True,
                   help="perturbation axis, or 'none' for the unperturbed set")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    # Optional: the registry already knows this track's checkpoint root (and
    # run.sh exports MODEL_ROOT_GR00T_N17 from machine.env), so the diag is
    # runnable as `--suite S --axis A`. Passing $WS from an interactive shell is
    # exactly how this gets run against /models/... by accident.
    p.add_argument("--model-path", default=None,
                   help="default: the registry's per-suite checkpoint path")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--exec-horizon", type=int, default=None)
    # The trajectory-generating arm. Default 2.0 = the shared operating point the
    # campaign proposes, so R is measured on the states that arm actually visits.
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--kind", default="text", choices=["all", "text", "image"])
    p.add_argument("--qgroup", default="all", choices=["all", "state", "action"])
    p.add_argument("--method", default="softmax", choices=["ent15max", "sparsemax", "softmax"])
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--n-state-tokens", type=int, default=None)
    a = p.parse_args()
    spec = MODELS["gr00t_n17"]  # NAG lives on this track only (docs/nag.md §4)
    a.max_steps = a.max_steps or spec.default_max_steps
    a.exec_horizon = a.exec_horizon or spec.default_exec_horizon
    a.n_state_tokens = a.n_state_tokens or spec.default_n_state_tokens
    a.model_path = a.model_path or spec.default_model_path(a.suite)
    if not os.path.isdir(a.model_path):
        raise SystemExit(f"[diag] no checkpoint at {a.model_path!r} — pass --model-path "
                         f"or set MODEL_ROOT_GR00T_N17 in experiments/machine.env")
    return a, spec


def _quantiles(hist, qs=(0.5, 0.9, 0.99)):
    """Read quantiles off the census histogram; values above its top clamp to it."""
    hi = NAG._HIST_HI
    total = float(hist.sum())
    if total == 0:
        return {q: float("nan") for q in qs}
    cum, out, edge = 0.0, {}, hi / len(hist)
    it = iter(sorted(qs))
    want = next(it, None)
    for i, c in enumerate(hist.tolist()):
        cum += c
        while want is not None and cum / total >= want:
            out[want] = (i + 1) * edge
            want = next(it, None)
    for q in qs:
        out.setdefault(q, hi)  # saturated: the true quantile is >= the top bin
    return out


def _agg(keys, field):
    """Sum a census field over a set of keys."""
    return sum(field.get(k, 0) for k in keys)


def report(args) -> None:
    lambdas = sorted({k[3] for k in NAG.p_n})
    print(f"\n=== NAG ratio census: {args.axis}/{args.suite}, {args.episodes} eps, "
          f"trajectory = lambda {args.scale:g} {args.qgroup}x{args.kind} "
          f"{args.method} beta={args.beta:g} ===", flush=True)

    print("\n-- clip rate by candidate tau (all steps, all blocks) --")
    head = "  lambda |  slots  | mean R | p50   p90   p99   max  | " + \
           "  ".join(f"t={t:g}" for t in NAG_CANDIDATE_TAUS)
    print(head)
    rates = {}
    for lam in lambdas:
        keys = [k for k in NAG.p_n if k[3] == lam]
        n = _agg(keys, NAG.p_n)
        mean = _agg(keys, NAG.p_sum) / n
        mx = max(NAG.p_max[k] for k in keys)
        hist = sum((NAG.p_hist[k] for k in keys[1:]), NAG.p_hist[keys[0]].clone())
        q = _quantiles(hist)
        ex = [sum(NAG.p_exceed[k][i] for k in keys) / n for i in range(len(NAG_CANDIDATE_TAUS))]
        rates[lam] = ex
        print(f"  {lam:6.2f} | {n:7d} | {mean:6.3f} | "
              f"{q[0.5]:5.2f} {q[0.9]:5.2f} {q[0.99]:5.2f} {mx:5.2f} | "
              + "  ".join(f"{r:5.1%}" for r in ex))

    print("\n-- clip rate by denoising step (the campaign's time axis) --")
    for lam in lambdas:
        for step in sorted({k[0] for k in NAG.p_n if k[3] == lam}):
            keys = [k for k in NAG.p_n if k[3] == lam and k[0] == step]
            n = _agg(keys, NAG.p_n)
            ex = [sum(NAG.p_exceed[k][i] for k in keys) / n for i in range(len(NAG_CANDIDATE_TAUS))]
            print(f"  lambda {lam:4.2f} step {step} | mean R "
                  f"{_agg(keys, NAG.p_sum) / n:6.3f} | "
                  + "  ".join(f"t={t:g}:{r:5.1%}" for t, r in zip(NAG_CANDIDATE_TAUS, ex)))

    print("\n-- clip rate by block --")
    for lam in (args.scale,):
        for blk in sorted({k[1] for k in NAG.p_n if k[3] == lam}):
            keys = [k for k in NAG.p_n if k[3] == lam and k[1] == blk]
            n = _agg(keys, NAG.p_n)
            ex = [sum(NAG.p_exceed[k][i] for k in keys) / n for i in range(len(NAG_CANDIDATE_TAUS))]
            print(f"  lambda {lam:4.2f} block {blk:2d} | "
                  + "  ".join(f"t={t:g}:{r:5.1%}" for t, r in zip(NAG_CANDIDATE_TAUS, ex)))

    print("\n-- docs/nag.md §6 pre-registered tau rule --")
    missing = [lam for lam, _, _ in RULE if lam not in rates]
    if missing:
        print(f"  INCONCLUSIVE: the rule is stated on lambdas {[r[0] for r in RULE]}, "
              f"but {missing} were not priced (use --probe covering them).")
        return
    picked = None
    for i, tau in enumerate(NAG_CANDIDATE_TAUS):
        checks = [(lam, rates[lam][i], lim, op) for lam, lim, op in RULE]
        ok = all(r <= lim if op == "<=" else r >= lim for _, r, lim, op in checks)
        detail = "  ".join(f"clip({lam:g})={r:.1%}{op}{lim:.0%}" for lam, r, lim, op in checks)
        print(f"  tau={tau:<4g} {'PASS' if ok else 'fail'}  {detail}")
        if ok and picked is None:
            picked = tau
    if picked is None:
        print("\n  NO CANDIDATE SATISFIES THE RULE -> do not launch the row (docs/nag.md §6). "
              "Either the cap cannot bite at this locus (Failure B) or it bites where the "
              "ladder is healthy (Failure A); report the census instead.")
    else:
        print(f"\n  tau* = {picked:g}  (smallest candidate that is inert at the axes' own "
              f"optima and active at lambda=3)")


def main():
    args, spec = parse_args()
    axis = None if args.axis == "none" else args.axis
    ts = LiberoPlusTaskSet(args.suite, axis)
    sched = ts.schedule(args.episodes, seed=args.seed)
    print(f"[diag] {len(sched)} episodes, {args.axis}/{args.suite}, "
          f"model={args.model_path}", flush=True)

    model = resolve_loader(spec)(args.model_path)
    # probe BEFORE install: install_pladis reads NAG.probe to decide whether the
    # denoising-step index has to be published (the census is keyed by step).
    NAG.reset()
    NAG.probe, NAG.probe_scales = True, tuple(sorted(set(LADDER) | {args.scale}))
    installed = install_pladis(
        model, pladis_scale=args.scale, method=args.method, beta=args.beta,
        kind=args.kind, qgroup=args.qgroup, n_state_tokens=args.n_state_tokens,
    )
    print(f"[diag] measuring on blocks {installed}, ladder {NAG.probe_scales}", flush=True)

    sess = LiberoPlusSession(seed=args.seed,
                             per_episode_np_seed=axis in RUNTIME_RNG_AXES)
    t0 = time.time()
    for i, ep in enumerate(sched, 1):
        with torch.no_grad():
            r = run_episode(sess, ep, ts.init_states_of(ep.task_name), model,
                            episode_seed=args.seed * 1_000_003 + ep.episode,
                            max_steps=args.max_steps, exec_horizon=args.exec_horizon)
        print(f"[diag] {i}/{len(sched)} {'OK ' if r.success_once else 'FAIL'} "
              f"({(time.time() - t0) / i:.1f}s/ep)", flush=True)
    sess.close()
    report(args)


if __name__ == "__main__":
    main()
