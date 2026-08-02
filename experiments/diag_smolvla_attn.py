# SPDX-License-Identifier: Apache-2.0
"""SmolVLA attention-map diagnostic: WHERE and HOW MUCH can PLADIS bite? (MEASUREMENT.)

Reference frame is the ORIGINAL SDXL PLADIS, not pi0.5: SDXL sharpens the whole
cross-attention row (PLADIS/pipeline/pipeline_sdxl.py:69-105), which is well-defined
because SDXL cross keys are text-ONLY — the row IS the text conditional. GR00T N1.7
inherits that directly (single-modality keys per cross block, attn_gr00t_n17.py).
SmolVLA's CA row mixes three modalities in one key axis [image 128 | text 48 | state 1],
so "sharpen the text conditional" must be spelled as the mass-preserving sub-block form:

    w[sub] = dense[sub] + λ·(m·p − dense[sub]),   m = Σ_sub dense,  p = entmax15(β·z_sub)

which by dense[sub] = m·softmax_block(sub) IS the SDXL whole-row blend applied to the
text-only sub-problem with the row's text-mass share m held fixed (verify_smolvla_hook.py
gate: FLUX identity). kind=prefix is instead the LITERAL whole-row entmax over mixed
modalities — no upstream precedent; whether text even survives it is question (ii) below.

Questions, and the sweep decision each feeds:
  (i)   headroom — is the dense text conditional already peaked? entmax15 support of the
        text block (of ~n_real tokens, β=1 and 2) + top-1 share. If support ≈ n_real,
        β=1 entmax is a rounding perturbation and the λ ladder must start higher (or β>1);
        if top-1 ≈ 1 already, a×t has nothing to redistribute. Feeds λ/β for EVERY arm.
  (ii)  axpfx viability — whole-row entmax15: do text columns survive next to 128 image
        columns, or is kind=prefix effectively "image sharpening + text ablation"?
  (iii) axs / axself surface — dense mass on the 1-column state block and the SA suffix
        block: an arm whose locus carries ~0 dense mass cannot move behavior.
  (iv)  mechanism — paired ID-vs-OOD phrasing on IDENTICAL (task, init, pinned noise):
        does the OOD BDDL phrasing ("akita ...", the string that collapsed the anchors,
        2026-08-02: spatial 41→87 on the swap alone) measurably diffuse the text
        conditional (entropy ↑, text mass ↓) — the regime text sharpening claims to fix?

Method: install the hook at λ=0 (bit-parity op — measured rollouts ARE vanilla; the
episodes replay the anchor/A2 schedule bit-identically, which doubles as a free parity
check) and wrap attn_smolvla._maybe_blend, which receives the full masked fp32 logits of
every attention call. Layer identity is recovered from call order (prefix pass = 16
consecutive q==k calls, then 16 layers x 10 flow steps in order) and ASSERTED against the
shape classification (even idx = SA, odd = CA) every call.

Run: bash experiments/run.sh --venv lerobot experiments/diag_smolvla_attn.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from entmax import entmax15

from harness.env import LiberoPlusSession, LiberoPlusTaskSet
from harness.model_smolvla import load_smolvla
from harness.rollout import run_episode
from pladis import attn_smolvla

from experiments.eval_arm import _task_meta_instruction

N_IMG, N_LANG_MAX, N_STATE = 128, 48, 1
P = N_IMG + N_LANG_MAX + N_STATE          # official ckpt: constant 177
NEG = torch.finfo(torch.float32).min / 4  # same finite floor as attn_smolvla._blend_rows
REAL = -1e30                              # logits above this = unmasked (real token)


class RolloutStats:
    """Incremental per-call accumulator for ONE rollout. Records are per sampled call:
    (layer, flow_step, chunk_idx, *scalars). Aggregated with numpy at report time."""

    def __init__(self, subsample: int = 5, betas=(1.0, 2.0)):
        self.sub = subsample
        self.betas = betas
        self.ca = []       # rows: layer, step, chunk, m_img, m_txt, m_state, Hn, top1,
                           #       sup_txt@b1, sup_txt@b2, sup_img@b1,
                           #       row_sup, row_txt_surv, row_txt_mass
        self.sa = []       # rows: layer, step, chunk, m_suffix, sup_suffix@b1
        self.n_real_txt = None
        self._calls = 0

    @staticmethod
    def _entmax_support(z: torch.Tensor, beta: float) -> torch.Tensor:
        p = entmax15((beta * z).clamp_min(NEG), dim=-1)
        return (p > 0).sum(-1).float()

    def add(self, layer: int, step: int, chunk: int, kind: str, logits: torch.Tensor):
        self._calls += 1
        if self._calls % self.sub:
            return
        z = logits.detach().float()
        dense = torch.softmax(z, dim=-1)
        if kind == "ca":
            n_real = int((z[..., N_IMG:N_IMG + N_LANG_MAX] > REAL).any((0, 1, 2)).sum())
            self.n_real_txt = n_real
            m_img = dense[..., :N_IMG].sum(-1).mean()
            m_txt = dense[..., N_IMG:N_IMG + N_LANG_MAX].sum(-1).mean()
            m_st = dense[..., -N_STATE:].sum(-1).mean()
            # text CONDITIONAL (renormalized sub-block): entropy over real tokens + top-1
            pt = dense[..., N_IMG:N_IMG + N_LANG_MAX]
            pt = pt / pt.sum(-1, keepdim=True).clamp_min(1e-12)
            H = -(pt * pt.clamp_min(1e-12).log()).sum(-1).mean()
            Hn = float(H) / max(np.log(max(n_real, 2)), 1e-9)
            top1 = pt.max(-1).values.mean()
            zt = z[..., N_IMG:N_IMG + N_LANG_MAX]
            sup_t1 = self._entmax_support(zt, self.betas[0]).mean()
            sup_t2 = self._entmax_support(zt, self.betas[1]).mean()
            sup_i1 = self._entmax_support(z[..., :N_IMG], self.betas[0]).mean()
            # question (ii): whole-row entmax15 (what kind=prefix computes)
            prow = entmax15(z.clamp_min(NEG), dim=-1)
            row_sup = (prow > 0).sum(-1).float().mean()
            tsub = prow[..., N_IMG:N_IMG + N_LANG_MAX]
            row_txt_surv = (tsub > 0).sum(-1).float().mean()
            row_txt_mass = tsub.sum(-1).mean()
            self.ca.append([layer, step, chunk, float(m_img), float(m_txt), float(m_st),
                            Hn, float(top1), float(sup_t1), float(sup_t2), float(sup_i1),
                            float(row_sup), float(row_txt_surv), float(row_txt_mass)])
        else:
            m_suf = dense[..., P:].sum(-1).mean()
            sup_s = self._entmax_support(z[..., P:], self.betas[0]).mean()
            self.sa.append([layer, step, chunk, float(m_suf), float(sup_s)])

    def arrays(self):
        return np.array(self.ca), np.array(self.sa)


def install_probe(collect):
    """Wrap _maybe_blend; recover (layer, flow step, chunk) from call order and ASSERT
    it against the shape classification. collect(layer, step, chunk, kind, logits)."""
    orig = attn_smolvla._maybe_blend
    state = {"prefix_run": 0, "denoise_i": 0, "chunk": -1}

    def probe(masked):
        q, k = masked.shape[-2], masked.shape[-1]
        if q == k:                                     # prefix pass call
            state["prefix_run"] += 1
        else:
            if state["prefix_run"]:                    # first denoise call of a chunk
                assert state["prefix_run"] == 16, state["prefix_run"]
                state["prefix_run"] = 0
                state["denoise_i"] = 0
                state["chunk"] += 1
            i = state["denoise_i"]
            state["denoise_i"] += 1
            layer, step = i % 16, i // 16
            is_sa = k == P + q
            assert k == P or is_sa, (q, k)
            assert (layer % 2 == 0) == is_sa, (layer, q, k)   # even=SA, odd=CA
            collect(layer, step, state["chunk"], "sa" if is_sa else "ca", masked)
        return orig(masked)

    attn_smolvla._maybe_blend = probe
    return orig


def report_rollout(tag: str, res, st: RolloutStats):
    ca, sa = st.arrays()
    print(f"\n--- {tag}: success={res.success_once} n_steps={res.n_steps} "
          f"n_real_txt={st.n_real_txt} ({len(ca)} CA / {len(sa)} SA sampled calls) ---")
    print("  CA layer |  m_img  m_txt  m_state |  H_norm top1 | sup_txt b1/b2 (of real)"
          " | sup_img b1 (/128) | row: sup txt_surv txt_mass")
    for L in sorted(set(ca[:, 0].astype(int))):
        r = ca[ca[:, 0] == L]
        print(f"       {L:2d}  |  {r[:,3].mean():.3f}  {r[:,4].mean():.3f}   {r[:,5].mean():.3f}"
              f"  |   {r[:,6].mean():.2f}  {r[:,7].mean():.2f} |"
              f"    {r[:,8].mean():5.1f}/{r[:,9].mean():4.1f}"
              f"        |   {r[:,10].mean():5.1f}      |"
              f" {r[:,11].mean():5.1f}  {r[:,12].mean():4.1f}   {r[:,13].mean():.3f}")
    prof = [f"s{s}:{ca[ca[:,1]==s][:,4].mean():.3f}" for s in sorted(set(ca[:, 1].astype(int)))]
    print(f"  m_txt by flow step: {' '.join(prof)}")
    print(f"  SA: m_suffix={sa[:,3].mean():.3f} sup_suffix b1={sa[:,4].mean():.1f} (/{50})")
    return ca, sa


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get(
        "MODEL_ROOT_SMOLVLA",
        "/home/reallab/parkkwanjoon/workspace/models/smolvla_libero_official"))
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", default="on_the_cookie_box,on_the_stove,"
                                      "in_the_top_drawer,on_the_ramekin",
                   help="base-task substrings; defaults = 3 instruction-flip tasks "
                        "(old-anchor 0-2/10 -> A2 8-10/10) + one ID-weak control "
                        "(on_the_ramekin, A2 3/10 = capability failure)")
    p.add_argument("--inits", default="0,1")
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--exec-horizon", type=int, default=10)
    args = p.parse_args()

    ts = LiberoPlusTaskSet(args.suite, None)
    inits = {int(x) for x in args.inits.split(",")}
    subs = args.tasks.split(",")
    specs = [s for s in ts.schedule(100, seed=0)
             if s.init_state_id in inits and any(t in s.base_task for t in subs)]
    print(f"{len(specs)} (task, init) pairs x 2 phrasings on {args.suite}")

    sess = LiberoPlusSession(seed=0)
    model = load_smolvla(args.model_path)
    # λ=0: bit-parity op — every measured rollout is exactly the vanilla anchor episode
    attn_smolvla.install_pladis(model, pladis_scale=0.0, kind="text")

    rows = []   # (task, init, phrasing, success, m_txt, Hn, top1, sup_b1, m_txt_chunk0, Hn_chunk0)
    orig = attn_smolvla._maybe_blend
    try:
        for spec in specs:
            for phrasing, imap in [("OOD/bddl", None),
                                   ("ID/task-meta",
                                    lambda s: _task_meta_instruction(s.base_task))]:
                st = RolloutStats()
                orig = install_probe(st.add)
                try:
                    res = run_episode(sess, spec, ts.init_states_of(spec.task_name), model,
                                      episode_seed=spec.episode,   # seed 0 anchor schedule
                                      max_steps=args.max_steps,
                                      exec_horizon=args.exec_horizon,
                                      instruction_map=imap)
                finally:
                    attn_smolvla._maybe_blend = orig
                short = spec.base_task.split("black_bowl_")[-1][:24]
                ca, _ = report_rollout(f"{short} init{spec.init_state_id} {phrasing}",
                                       res, st)
                c0 = ca[ca[:, 2] == 0]
                rows.append((short, spec.init_state_id, phrasing, res.success_once,
                             ca[:, 4].mean(), ca[:, 6].mean(), ca[:, 7].mean(),
                             ca[:, 8].mean(), c0[:, 4].mean(), c0[:, 6].mean()))
    finally:
        attn_smolvla._maybe_blend = orig
        sess.close()

    print("\n=== paired ID-vs-OOD summary (same task+init+pinned noise; "
          "whole-episode / chunk-0) ===")
    print(f"{'task':26s} init  {'phrasing':13s} succ  m_txt   Hn   top1  sup_b1 "
          f"| c0: m_txt   Hn")
    for r in rows:
        print(f"{r[0]:26s}  {r[1]}   {r[2]:13s}  {r[3]}   {r[4]:.3f}  {r[5]:.2f}  "
              f"{r[6]:.2f}  {r[7]:5.1f} |    {r[8]:.3f}  {r[9]:.2f}")
    ood = np.array([r[4:] for r in rows if r[2] == "OOD/bddl"], dtype=float)
    idm = np.array([r[4:] for r in rows if r[2] == "ID/task-meta"], dtype=float)
    if len(ood) == len(idm) and len(ood):
        d = idm.mean(0) - ood.mean(0)
        print(f"\nmean(ID) - mean(OOD): m_txt {d[0]:+.4f}  Hn {d[1]:+.3f}  "
              f"top1 {d[2]:+.3f}  sup_b1 {d[3]:+.2f}  | c0: m_txt {d[4]:+.4f}  Hn {d[5]:+.3f}")
    print("\nDIAGNOSTIC COMPLETE (measurement, not a gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
