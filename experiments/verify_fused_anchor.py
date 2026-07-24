# SPDX-License-Identifier: Apache-2.0
"""Verification gates for the fused-anchored PLADIS processor (attn_gr00t_fused.py).

Three claims, on one Attention module in the N1.7 DiT configuration (heads=32,
head_dim=48; bool (B,S) key masks as AlternateVLDiT passes them):

  A. base0 == vanilla WITH the correction machinery running: the ungated
     processor at pladis_scale=0 must be torch.equal to AttnProcessor2_0 on
     every case (self / cross / cross+bool-mask). This is substantive — the
     entmax/delta path executes and must contribute exactly zero.
  B. qgroup rows outside the group are bit-identical to vanilla at scale=1
     (their correction row is zeroed; only selected rows may differ).
  C. Same method as the shipped weight-space processor: at scale=1 ent15max the
     two implementations agree to the dtype rounding floor (rel diff bound).

Run (CPU pre-check, safe while a sweep holds the GPU):
    CUDA_VISIBLE_DEVICES= bash experiments/run.sh experiments/verify_fused_anchor.py cpu
Definitive run after the sweep (bf16 cuda, mirrors verify_base0_parity.py):
    bash experiments/run.sh experiments/verify_fused_anchor.py cuda
"""

import sys

import torch
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0

from pladis.attn_gr00t import PLADISAttnProcessor as WeightSpace
from pladis.attn_gr00t_fused import PLADISAttnProcessor as FusedAnchor

REL_BOUND = {torch.float32: 1e-5, torch.bfloat16: 2e-2}


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    torch.manual_seed(0)
    dim, heads, head_dim = 1536, 32, 48  # gr00t_n1d7 diffusion_model_cfg
    dtypes = [torch.bfloat16] if dev == "cuda" else [torch.float32, torch.bfloat16]

    ok = True
    for dt in dtypes:
        attn = (
            Attention(
                query_dim=dim,
                heads=heads,
                dim_head=head_dim,
                dropout=0.2,
                bias=True,
                cross_attention_dim=dim,
                out_bias=True,
            )
            .to(dev, dt)
            .eval()
        )
        B, Lq, S = 2, 17, 261  # [state; action] query rows; VL key tokens
        hs = torch.randn(B, Lq, dim, device=dev, dtype=dt)
        ehs = torch.randn(B, S, dim, device=dev, dtype=dt)
        key_mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
        key_mask[:, :40] = True
        key_mask[1, 40:] = True
        cases = {
            "self (no mask)": dict(hidden_states=hs),
            "cross (no mask)": dict(hidden_states=hs, encoder_hidden_states=ehs),
            "cross (bool key mask)": dict(
                hidden_states=hs, encoder_hidden_states=ehs, attention_mask=key_mask
            ),
        }

        with torch.no_grad():
            for name, kw in cases.items():
                attn.set_processor(AttnProcessor2_0())
                ref = attn(**kw)

                # A: ungated scale=0 (entmax + delta computed) must be bit-vanilla
                attn.set_processor(FusedAnchor(pladis_scale=0.0))
                out0 = attn(**kw)
                bit = torch.equal(ref, out0)
                maxdiff = (ref.float() - out0.float()).abs().max().item()
                print(f"[A {dt} {dev}] {name:22s} base0 bit-exact={bit} max|diff|={maxdiff:.3e}",
                      flush=True)
                ok &= bit

                # B: qgroup=action scale=1 -> state row (row 0) stays bit-vanilla
                attn.set_processor(
                    FusedAnchor(pladis_scale=1.0, qgroup="action", n_state_tokens=1)
                )
                treated = attn(**kw)
                assert torch.isfinite(treated.float()).all(), f"{name}: scale>0 not finite"
                rows_ok = torch.equal(ref[:, :1], treated[:, :1])
                rows_ne = not torch.equal(ref[:, 1:], treated[:, 1:])
                print(f"[B {dt} {dev}] {name:22s} state-row bit-vanilla={rows_ok} "
                      f"action-rows-changed={rows_ne}", flush=True)
                ok &= rows_ok and rows_ne

                # C: fused-anchored vs weight-space at scale=1 = same method
                attn.set_processor(WeightSpace(pladis_scale=1.0))
                w = attn(**kw)
                attn.set_processor(FusedAnchor(pladis_scale=1.0))
                f = attn(**kw)
                rel = ((w.float() - f.float()).norm() / w.float().norm()).item()
                bound = REL_BOUND[dt]
                print(f"[C {dt} {dev}] {name:22s} weight-vs-fused rel={rel:.3e} "
                      f"(bound {bound:g})", flush=True)
                ok &= rel < bound

    print("[fused-anchor] PASS" if ok else "[fused-anchor] FAIL", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
