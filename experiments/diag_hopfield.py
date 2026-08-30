# SPDX-License-Identifier: Apache-2.0
"""Phase 0 of docs/hopfield.md: measure the self-attention structure, then price the grid.

Runs a VANILLA rollout (the Hopfield processor in probe mode returns the fused
output bit-identically) and records, per odd block and denoising step:

  * eta_bar — the realized symmetry index (paper Eq. 36) per head: symmetric-dominant
    (eta ~ 1: adaptive control self-extinguishes, alpha must move far from 1) or
    circulation-heavy;
  * E, r, Align — the paper's stability measures (Eq. 27-29) of the baseline retrieval
    Xi = P X, per episode and split by outcome;
  * the PRICE LIST — from the same logits and values, for every alpha in HOP_ALPHA_GRID
    and every temperature in HOP_TEMP_GRID: the relative displacement
    d = ||Z_alpha - Z||_2 / ||Z||_2 per (head, query row), the norm-match clamp rate at
    every beta in HOP_BETA_GRID, the linearity d(alpha)/(alpha-1), and the
    fused-vs-eager floor ||fused - Z||_2/||Z||_2 on the same rows (gate H of
    verify_hopfield.py: the priced numbers equal what a processor run there computes).

The pre-registered rules (docs/hopfield.md §6) are evaluated at the end: both signs of
alpha are swept; the magnitude is the grid point whose median displacement is closest
to the even-block reference (the NAG probe's displacement of the --pladis-* arm, when
one is passed — default: the all-x-text sharp-softmax lambda=2 arm); the temperature
control is the grid point matched to that alpha; nothing launches below 3x the floor.

Run (per axis):

    bash experiments/run.sh experiments/diag_hopfield.py \\
      --suite libero_goal --axis language --episodes 40

    # without the even-block reference arm (pure vanilla trajectory):
    bash experiments/run.sh experiments/diag_hopfield.py --suite libero_goal --axis robot \\
      --episodes 40 --no-cross-ref

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
from pladis.attn_gr00t_n17 import (
    HOP,
    HOP_ALPHA_GRID,
    HOP_BETA_GRID,
    HOP_TEMP_GRID,
    NAG,
    _hist_quantiles,
    assert_hopfield_delivered,
    install_hopfield,
    install_pladis,
)

FLOOR_MULT = 3.0  # docs/hopfield.md §6: launch only if displacement clears 3x the floor
ETA_ADAPTIVE_CUTOFF = 0.9  # median eta above this at every block -> drop the adaptive arm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True)
    p.add_argument("--axis", required=True,
                   help="perturbation axis, or 'none' for the unperturbed set")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-path", default=None,
                   help="default: the registry's per-suite checkpoint path")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--exec-horizon", type=int, default=None)
    p.add_argument("--n-state-tokens", type=int, default=None)
    # The even-block STRENGTH reference: the cross-block arm whose displacement the
    # alpha grid is matched against. Recorded by the NAG probe on the same rollout.
    # NOTE: with it installed the trajectory is that arm's, not vanilla's — the
    # diagnostics are then "on the states the campaign's best arm visits". Pass
    # --no-cross-ref for the pure vanilla trajectory.
    p.add_argument("--no-cross-ref", action="store_true")
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--kind", default="text", choices=["all", "text", "image"])
    p.add_argument("--qgroup", default="all", choices=["all", "state", "action"])
    p.add_argument("--method", default="softmax", choices=["ent15max", "sparsemax", "softmax"])
    p.add_argument("--beta", type=float, default=2.0)
    a = p.parse_args()
    spec = MODELS["gr00t_n17"]  # the Hopfield processor lives on this track only
    a.max_steps = a.max_steps or spec.default_max_steps
    a.exec_horizon = a.exec_horizon or spec.default_exec_horizon
    a.n_state_tokens = a.n_state_tokens or spec.default_n_state_tokens
    a.model_path = a.model_path or spec.default_model_path(a.suite)
    if not os.path.isdir(a.model_path):
        raise SystemExit(f"[diag] no checkpoint at {a.model_path!r} — pass --model-path "
                         f"or set MODEL_ROOT_GR00T_N17 in experiments/machine.env")
    return a, spec


class _Acc:
    """Run-level accumulation of the per-episode census (the census is cleared per
    episode so the outcome split is available)."""

    def __init__(self):
        self.eps = []  # (success, summary, rows)
        self.eta_hist = {}
        self.al_hist = {}
        self.p_hist = {}
        self.p_n, self.p_sum, self.p_lo, self.p_hi = {}, {}, {}, {}
        self.f_n, self.f_sum = {}, {}
        self.cross_n, self.cross_sum = 0, 0.0

    @staticmethod
    def _add(dst, src):
        for k, v in src.items():
            if isinstance(v, dict):
                d = dst.setdefault(k, {})
                for b, c in v.items():
                    d[b] = d.get(b, 0) + c
            elif isinstance(v, torch.Tensor):
                dst[k] = v.clone() if k not in dst else dst[k] + v
            else:
                dst[k] = dst.get(k, 0) + v

    def take(self, success: int, episode: int = -1, n_steps: int = -1, eps_tsv=None):
        summary, rows = HOP.episode_stats()
        self.eps.append((success, summary, rows))
        if eps_tsv is not None:
            # One row per episode WITH its length: the outcome split of the per-episode
            # means is confounded by episode length (a failed episode runs to the step
            # cap and its mean is dominated by the stuck states), and only the
            # per-episode rows can separate "unstable retrieval causes failure" from
            # "failure produces unstable-looking states" (docs/hopfield.md §6).
            if eps_tsv.tell() == 0:
                eps_tsv.write("\t".join(["episode", "success_once", "n_steps"] + list(summary)) + "\n")
            eps_tsv.write("\t".join([str(episode), str(success), str(n_steps)]
                                    + [f"{summary[k]:.6g}" for k in summary]) + "\n")
            eps_tsv.flush()
        self._add(self.eta_hist, HOP.eta_hist)
        self._add(self.al_hist, HOP.al_hist)
        self._add(self.p_hist, HOP.p_hist)
        self._add(self.p_n, HOP.p_n)
        self._add(self.p_sum, HOP.p_sum)
        self._add(self.p_lo, HOP.p_lo)
        self._add(self.p_hi, HOP.p_hi)
        self._add(self.f_n, HOP.f_n)
        self._add(self.f_sum, HOP.f_sum)
        self.cross_n += sum(NAG.p_disp_n.values())
        self.cross_sum += sum(NAG.p_disp_sum.values())
        HOP.clear_episode()
        NAG.clear_episode()

    def hist(self, hists, keys):
        hs = [hists[k] for k in keys if k in hists]
        return sum(hs[1:], hs[0].clone()) if hs else None


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def report(args, acc: _Acc) -> None:
    cells = sorted({(r["step"], r["block"]) for _, _, rows in acc.eps for r in rows})
    steps = sorted({s for s, _ in cells})
    blocks = sorted({b for _, b in cells})
    n_eps = len(acc.eps)
    n_succ = sum(s for s, _, _ in acc.eps)
    print(f"\n=== Hopfield phase-0 census: {args.axis}/{args.suite}, {n_eps} eps "
          f"({n_succ} succ / {n_eps - n_succ} fail), "
          f"{'vanilla trajectory' if args.no_cross_ref else 'trajectory of the cross-ref arm'} ===",
          flush=True)

    # -- 1. eta by block (all steps) and by step (all blocks) -------------------
    print("\n-- realized symmetry index eta (paper Eq. 36), per head; 1 = symmetric-dominant --")
    eta_med = {}
    print("  block |  p10   p50   p90 |  mean")
    for b in blocks:
        h = acc.hist(acc.eta_hist, [c for c in cells if c[1] == b])
        q = _hist_quantiles(h, -1.0, 1.0, (0.1, 0.5, 0.9))
        m = _mean([r["eta_mean"] for _, _, rows in acc.eps for r in rows if r["block"] == b])
        eta_med[b] = q[0.5]
        print(f"  {b:5d} | {q[0.1]:+.3f} {q[0.5]:+.3f} {q[0.9]:+.3f} | {m:+.3f}")
    print("  step  |  p10   p50   p90")
    for s in steps:
        h = acc.hist(acc.eta_hist, [c for c in cells if c[0] == s])
        q = _hist_quantiles(h, -1.0, 1.0, (0.1, 0.5, 0.9))
        print(f"  {s:5d} | {q[0.1]:+.3f} {q[0.5]:+.3f} {q[0.9]:+.3f}")

    # -- 2. stability measures, pooled and by outcome ---------------------------
    print("\n-- Hopfield stability of the baseline retrieval (Eq. 27-29), per episode --")
    for key, label in (("E_mean", "E (energy)"), ("r_mean", "r (instability)"),
                       ("align_mean", "Align"), ("align_p10", "Align p10"), ("eta_mean", "eta")):
        succ = [sm[key] for s, sm, _ in acc.eps if s == 1]
        fail = [sm[key] for s, sm, _ in acc.eps if s == 0]
        line = f"  {label:16s} all {_mean(succ + fail):+9.4f}"
        if len(succ) >= 2 and len(fail) >= 2:
            ms, mf = _mean(succ), _mean(fail)
            vs = sum((x - ms) ** 2 for x in succ) / (len(succ) - 1) / len(succ)
            vf = sum((x - mf) ** 2 for x in fail) / (len(fail) - 1) / len(fail)
            z = (mf - ms) / ((vs + vf) ** 0.5) if vs + vf > 0 else 0.0
            line += f"   succ {ms:+9.4f} (n={len(succ)})  fail {mf:+9.4f} (n={len(fail)})  fail-succ z={z:+5.2f}"
        print(line)
    print("  Align by block (all episodes):  " + "  ".join(
        f"{b}:{_mean([r['align_mean'] for _, _, rows in acc.eps for r in rows if r['block'] == b]):+.3f}"
        for b in blocks))

    # -- 3. the price list ------------------------------------------------------
    floor_n = sum(acc.f_n.values())
    floor = sum(acc.f_sum.values()) / floor_n if floor_n else float("nan")
    cross = acc.cross_sum / acc.cross_n if acc.cross_n else float("nan")
    print(f"\n-- price list: relative displacement d = ||Z_a - Z||_2/||Z||_2 per (head, row), "
          f"all odd blocks, all steps --")
    print(f"  fused-vs-eager floor (same rows): mean {floor:.2e}  -> launch threshold "
          f"{FLOOR_MULT:g}x = {FLOOR_MULT * floor:.2e}")
    if acc.cross_n:
        print(f"  even-block reference: NAG-probe displacement of the cross-ref arm "
              f"(lambda {args.scale:g} {args.qgroup}x{args.kind} {args.method} beta={args.beta:g}): "
              f"mean {cross:.4f}")
    else:
        print("  even-block reference: (not recorded — --no-cross-ref)")
    head = "  kind   value | mean d   p50     p90   | d/(a-1) | " + \
           "  ".join(f"clip@b{b:g} (lo/hi)" for b in HOP_BETA_GRID)
    print(head)
    price = {}
    for kind, grid in (("alpha", HOP_ALPHA_GRID), ("temp", HOP_TEMP_GRID)):
        for v in grid:
            ks = [c + (kind, v) for c in cells]
            n = sum(acc.p_n.get(k, 0) for k in ks)
            if not n:
                continue
            mean = sum(acc.p_sum.get(k, 0.0) for k in ks) / n
            q = _hist_quantiles(acc.hist(acc.p_hist, ks), 0.0, 2.0, (0.5, 0.9))
            price[(kind, v)] = (mean, q[0.5], q[0.9])
            lin = f"{mean / (v - 1.0):7.3f}" if kind == "alpha" else "   -   "
            clips = []
            for b in HOP_BETA_GRID:
                lo = sum(acc.p_lo.get(k, {}).get(b, 0) for k in ks) / n
                hi = sum(acc.p_hi.get(k, {}).get(b, 0) for k in ks) / n
                clips.append(f"{lo:5.1%}/{hi:5.1%}")
            print(f"  {kind:5s} {v:6.2f} | {mean:6.4f} {q[0.5]:6.4f} {q[0.9]:6.4f} | {lin} | "
                  + "  ".join(f"{c:>15s}" for c in clips))

    print("\n-- displacement by block at alpha=1.5 and by step (mean d) --")
    for s_or_b, label, idx in ((steps, "step ", 0), (blocks, "block", 1)):
        parts = []
        for x in s_or_b:
            ks = [c + ("alpha", 1.5) for c in cells if c[idx] == x]
            n = sum(acc.p_n.get(k, 0) for k in ks)
            parts.append(f"{x}:{(sum(acc.p_sum.get(k, 0.0) for k in ks) / n if n else float('nan')):.4f}")
        print(f"  {label} " + "  ".join(parts))

    # -- 4. the pre-registered decisions ---------------------------------------
    print("\n-- docs/hopfield.md §6 pre-registered rules --")
    thr = FLOOR_MULT * floor
    above = [(v, price[("alpha", v)][1]) for v in HOP_ALPHA_GRID
             if ("alpha", v) in price and price[("alpha", v)][1] > thr]
    if not above:
        print(f"  NO alpha on the grid clears {FLOOR_MULT:g}x the floor -> DO NOT LAUNCH "
              f"(inert by construction; report the census instead).")
    else:
        print(f"  alphas clearing the floor: {[v for v, _ in above]}")
        if acc.cross_n:
            for sign, cand in (("alpha < 1", [(v, d) for v, d in above if v < 1]),
                               ("alpha > 1", [(v, d) for v, d in above if v > 1])):
                if not cand:
                    print(f"  {sign}: no grid point clears the floor")
                    continue
                v, d = min(cand, key=lambda vd: abs(vd[1] - cross))
                print(f"  {sign}: alpha* = {v:g}  (median d {d:.4f} vs even-block reference {cross:.4f})")
                tc = [(t, price[("temp", t)][1]) for t in HOP_TEMP_GRID if ("temp", t) in price]
                if tc:
                    t, dt = min(tc, key=lambda td: abs(td[1] - d))
                    print(f"           matched temperature control: temp* = {t:g} (median d {dt:.4f})")
        else:
            print("  (no even-block reference recorded: pick alpha by hand from the price list)")
    if eta_med and min(eta_med.values()) > ETA_ADAPTIVE_CUTOFF:
        print(f"  median eta > {ETA_ADAPTIVE_CUTOFF:g} at EVERY block -> adaptive arm would be a "
              f"near-copy of hop-dense; drop it.")
    else:
        lo_b = min(eta_med, key=eta_med.get) if eta_med else None
        print(f"  adaptive control has room: lowest median eta {eta_med.get(lo_b, float('nan')):+.3f} "
              f"at block {lo_b}")


def main():
    args, spec = parse_args()
    axis = None if args.axis == "none" else args.axis
    ts = LiberoPlusTaskSet(args.suite, axis)
    sched = ts.schedule(args.episodes, seed=args.seed)
    print(f"[diag] {len(sched)} episodes, {args.axis}/{args.suite}, model={args.model_path}",
          flush=True)

    model = resolve_loader(spec)(args.model_path)
    HOP.reset()
    NAG.reset()
    if not args.no_cross_ref:
        # the even-block reference arm, with the NAG probe recording its displacement
        NAG.probe, NAG.record_episode = True, True
        blocks = install_pladis(
            model, pladis_scale=args.scale, method=args.method, beta=args.beta,
            kind=args.kind, qgroup=args.qgroup, n_state_tokens=args.n_state_tokens,
        )
        print(f"[diag] even-block reference arm on blocks {blocks}", flush=True)
    HOP.probe, HOP.record_episode = True, True
    installed = install_hopfield(model, alpha=1.0, beta=0.0, n_state_tokens=args.n_state_tokens)
    print(f"[diag] Hopfield probe on blocks {installed}", flush=True)

    sess = LiberoPlusSession(seed=args.seed,
                             per_episode_np_seed=axis in RUNTIME_RNG_AXES)
    acc = _Acc()
    os.makedirs("results/diag", exist_ok=True)
    eps_path = f"results/diag/hop_{args.axis}_{args.suite}_eps.tsv"
    eps_tsv = open(eps_path, "w")
    print(f"[diag] per-episode rows -> {eps_path}", flush=True)
    t0 = time.time()
    for i, ep in enumerate(sched, 1):
        with torch.no_grad():
            r = run_episode(sess, ep, ts.init_states_of(ep.task_name), model,
                            episode_seed=args.seed * 1_000_003 + ep.episode,
                            max_steps=args.max_steps, exec_horizon=args.exec_horizon)
        if i == 1:
            print(f"[diag] delivery: {assert_hopfield_delivered()}", flush=True)
        acc.take(int(r.success_once), episode=ep.episode, n_steps=int(r.n_steps), eps_tsv=eps_tsv)
        print(f"[diag] {i}/{len(sched)} {'OK ' if r.success_once else 'FAIL'} "
              f"({(time.time() - t0) / i:.1f}s/ep)", flush=True)
    sess.close()
    report(args, acc)


if __name__ == "__main__":
    main()
