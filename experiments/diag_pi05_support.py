# SPDX-License-Identifier: Apache-2.0
"""How MUCH does the intervention actually sparsify? (§5 gate 5c — a MEASUREMENT.)

This is not pass/fail. It exists because an entmax15 blend can be installed, deliver, and
still change almost nothing: if entmax15 keeps 190 of the 200 language columns, the `text`
arm is a rounding perturbation wearing an intervention's flags. A null result would then
be uninterpretable — "locus does not matter" and "the dose was too small" look identical
in the eplog. Nothing else in the gate ladder measures this.

Three questions, three consumers:

  (i)  text / image — is the dose meaningful? Support size relative to block width, per
       layer and per denoise step. Feeds the λ choice for phase 1/2.
  (ii) prefix — does the language block SURVIVE? entmax15 over the whole 968-column
       conditioning span normalizes image (768) and language (200) together, so image can
       crowd language out entirely. If it does, `prefix` is just "image-only + language
       ablation" and does not deserve a 1,537-episode arm. This is the check that decides
       whether prefix enters phase 2 at all.
  (iii) all — for the record, how much mass migrates out of the conditioning blocks into
       the action<->action self-attention columns (the off-method behaviour that made
       `all` a reference arm rather than the "both modalities" arm).

Method: wrap pladis.attn_pi05._sparse, which receives the ALREADY-SLICED block logits, so
the support count it sees is exactly the quantity of interest. Aggregated incrementally
(180 calls per chunk x ~50 chunks x N episodes is too many to keep as tensors).

Run: bash experiments/run.sh --venv openpi experiments/diag_pi05_support.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from harness.env import LiberoPlusSession, LiberoPlusTaskSet
from harness.model_pi05 import load_pi05
from harness.rollout import run_episode
from pladis import attn_pi05

N_IMG, N_LANG, SUFFIX = 768, 200, 10
PREFIX = N_IMG + N_LANG


class SupportStats:
    """Incremental support-size accumulator over every blended attention row."""

    def __init__(self):
        self.rows = 0
        self.width = None
        self.support = []           # per-row support counts (subsampled)
        self.lang_support = []      # prefix/all only: surviving columns inside language
        self.lang_mass = []         # prefix/all only: mass share landing on language
        self._calls = 0

    def add(self, p: torch.Tensor, lo: int, hi: int) -> None:
        """p: sparse distribution over the block, shape [B, H, Q, hi-lo], sums to 1."""
        self._calls += 1
        # subsample: every 7th call keeps the cost negligible while still covering all
        # 18 layers and all 10 denoise steps many times over
        if self._calls % 7:
            return
        p = p.detach().float()
        self.width = p.shape[-1]
        nz = (p > 0).sum(-1).flatten()
        self.rows += nz.numel()
        self.support.append(nz.cpu().numpy())
        # language columns sit at [768:968] of the FULL key axis; inside this block they
        # are at [768-lo : 968-lo], present only when the block spans them
        a, b = max(N_IMG - lo, 0), min(PREFIX - lo, self.width)
        if hi > N_IMG and b > a:
            sub = p[..., a:b]
            self.lang_support.append((sub > 0).sum(-1).flatten().cpu().numpy())
            self.lang_mass.append(sub.sum(-1).flatten().cpu().numpy())

    def report(self, kind: str) -> None:
        if not self.support:
            print(f"  {kind:7s} (no blended rows recorded)")
            return
        s = np.concatenate(self.support)
        q = np.percentile(s, [5, 25, 50, 75, 95])
        print(f"  {kind:7s} block width {self.width:4d} | support "
              f"p5={q[0]:.0f} p25={q[1]:.0f} med={q[2]:.0f} p75={q[3]:.0f} p95={q[4]:.0f} "
              f"| median kept {100 * q[2] / self.width:5.1f}%  ({self.rows} rows)")
        if self.lang_support:
            ls = np.concatenate(self.lang_support)
            lm = np.concatenate(self.lang_mass)
            dead = 100.0 * float((ls == 0).mean())
            print(f"          language sub-block ({N_LANG} cols): survivors "
                  f"med={np.median(ls):.0f} p95={np.percentile(ls, 95):.0f} | "
                  f"rows with ZERO language weight {dead:5.1f}% | "
                  f"median language mass share {np.median(lm):.4f}")
            if dead > 50:
                print("          ^^ language is crowded out on most rows: this arm is "
                      "effectively 'image-only + language ablation', NOT 'both modalities'")


class BlendStats:
    """λ>1 extrapolation accounting, measured on the BLEND OUTPUT (not the sparse branch).

    PLADIS is parameterized `λ·sparse + (1−λ)·dense` (pipeline_flux.py:109). At λ>1 the
    dense coefficient goes negative, so every column entmax dropped (p=0) comes out at
    `(1−λ)·dense < 0` — a NEGATIVE attention weight. That is intended, CFG-style
    extrapolation, and the row still sums to the same mass because the surviving columns
    are pushed above dense by exactly the deficit. But π0.5 has no precedent for it (the
    n17 λ=1.5 arms ran on another server), and negative weights on the action expert's
    joint attention are worth quantifying BEFORE spending 4,600 episodes on them.

    Reports, per λ and kind: what fraction of the block goes negative, how far, how much
    mass is routed through the negative lobe, and whether the full row still normalizes.
    """

    def __init__(self):
        self.rows = 0
        self.neg_frac = []      # fraction of block columns with w < 0
        self.min_w = []         # most negative weight in the block, per row
        self.neg_mass = []      # total |negative| mass in the block, per row
        self.row_sum = []       # full-row sum, must stay ~1 (normalization check)
        self._calls = 0

    def add(self, w: torch.Tensor, lo: int, hi: int) -> None:
        """w: blended weights over the FULL key axis, [B, H, Q, K], float32."""
        self._calls += 1
        if self._calls % 7:     # same subsampling as SupportStats
            return
        w = w.detach().float()
        sub = w[..., lo:hi]
        width = sub.shape[-1]
        neg = sub < 0
        self.rows += sub.shape[:-1].numel()
        self.neg_frac.append((neg.sum(-1).flatten() / width).cpu().numpy())
        self.min_w.append(sub.amin(-1).flatten().cpu().numpy())
        self.neg_mass.append(sub.clamp(max=0).sum(-1).neg().flatten().cpu().numpy())
        self.row_sum.append(w.sum(-1).flatten().cpu().numpy())

    def report(self) -> None:
        if not self.neg_frac:
            print("          (no blended rows recorded)")
            return
        nf = np.concatenate(self.neg_frac) * 100.0
        mw = np.concatenate(self.min_w)
        nm = np.concatenate(self.neg_mass)
        rs = np.concatenate(self.row_sum)
        print(f"          negative w: rows affected {100.0 * float((nf > 0).mean()):5.1f}% | "
              f"block cols negative med={np.median(nf):5.1f}% p95={np.percentile(nf, 95):5.1f}% | "
              f"min w med={np.median(mw):+.4f} p05={np.percentile(mw, 5):+.4f}")
        print(f"          negative mass med={np.median(nm):.4f} p95={np.percentile(nm, 95):.4f} | "
              f"row sum med={np.median(rs):.6f} (max dev {np.abs(rs - 1.0).max():.2e})")


def _install_probe(stats: SupportStats, blend: "BlendStats"):
    """Record the sparse block distribution AND the blended output, then delegate."""
    orig = attn_pi05._sparse
    orig_blend = attn_pi05._pladis_blend

    def probe(z, method):
        p = orig(z, method)
        lo, hi = _block_of(attn_pi05.CFG)
        stats.add(p, lo, hi)
        return p

    def blend_probe(scores, query_len, key_len):
        w = orig_blend(scores, query_len, key_len)
        cfg = attn_pi05.CFG
        # Only rows the blend actually touched: the dense-return paths (λ=0, the
        # large-query PaliGemma prefix pass) carry no extrapolation to measure.
        if cfg.scale != 0.0 and query_len is not None and query_len <= cfg.max_suffix_query:
            lo, hi = _block_of(cfg)
            blend.add(w, lo, hi)
        return w

    attn_pi05._sparse = probe
    attn_pi05._pladis_blend = blend_probe
    return orig, orig_blend


def _block_of(cfg):
    if cfg.kind == "text":
        return cfg.n_img_prefix, cfg.n_img_prefix + cfg.n_lang
    if cfg.kind == "image":
        return 0, cfg.n_img_prefix
    if cfg.kind == "prefix":
        return 0, cfg.n_img_prefix + cfg.n_lang
    return 0, cfg.n_img_prefix + cfg.n_lang + SUFFIX  # all


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get(
        "MODEL_ROOT_PI05", "/home/reallab/parkkwanjoon/workspace/models/pi05_libero"))
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--scales", default="1.0,1.5,2.0")
    p.add_argument("--kinds", default="text,image,prefix,all")
    args = p.parse_args()

    ts = LiberoPlusTaskSet(args.suite, "language")
    specs = ts.schedule(args.episodes, seed=0)
    sess = LiberoPlusSession(seed=0)
    model = load_pi05(args.model_path)

    print(f"\n=== entmax15 support size on {args.suite} language variants, "
          f"{args.episodes} episode(s)/cell ===")
    print("key axis at a suffix step: [image 0:768 | language 768:968 | suffix 968:978]\n")

    for scale in [float(x) for x in args.scales.split(",")]:
        print(f"λ = {scale:g}")
        for kind in args.kinds.split(","):
            stats = SupportStats()
            blend = BlendStats()
            orig, orig_blend = _install_probe(stats, blend)
            try:
                attn_pi05.install_pladis(
                    model, pladis_scale=scale, method="entmax15", kind=kind,
                    n_img_prefix=N_IMG, n_lang=N_LANG,
                )
                for spec in specs:
                    run_episode(sess, spec, ts.init_states_of(spec.task_name), model,
                                episode_seed=spec.episode, max_steps=520, exec_horizon=5)
                stats.report(kind)
                blend.report()
            finally:
                attn_pi05._sparse = orig
                attn_pi05._pladis_blend = orig_blend
        print()

    sess.close()
    print("NOTE: support size is λ-independent (entmax15 is applied to β*logits, and λ only "
          "mixes the result with dense), so the per-λ SUPPORT rows above should agree — they "
          "are repeated as a consistency check, not as a dose sweep. The `negative w` rows "
          "ARE λ-dependent: at λ>1 every dropped column lands at (1-λ)*dense < 0.")
    print("DIAGNOSTIC COMPLETE (measurement, not a gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
