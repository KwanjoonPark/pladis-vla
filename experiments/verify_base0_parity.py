# SPDX-License-Identifier: Apache-2.0
"""Bit-exact parity check: PLADISAttnProcessor(pladis_scale=0) vs diffusers
AttnProcessor2_0 on one Attention module in the N1.7 DiT configuration
(heads=32, head_dim=48, bf16, cuda; bool (B,S) key masks as AlternateVLDiT
passes them).

Official PLADIS keeps lambda=0 on the untouched fused-SDPA path (the swap is
gated on do_sparse_guidance = pladis_scale > 0, PLADIS/pipeline/
pipeline_sdxl.py:1215,1707), so base0 must equal vanilla byte-for-byte:
torch.equal on every case, plus a finiteness guard on the lambda>0 branch.

Run: bash experiments/run.sh experiments/verify_base0_parity.py
"""

import torch
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0

from pladis.attn_gr00t import PLADISAttnProcessor


def main():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    dim, heads, head_dim = 1536, 32, 48  # gr00t_n1d7 diffusion_model_cfg
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
    key_mask[:, :40] = True  # text-ish key group
    key_mask[1, 40:] = True  # different pattern per batch row
    cases = {
        "self (no mask)": dict(hidden_states=hs),
        "cross (no mask)": dict(hidden_states=hs, encoder_hidden_states=ehs),
        "cross (bool key mask)": dict(
            hidden_states=hs, encoder_hidden_states=ehs, attention_mask=key_mask
        ),
    }

    ok = True
    with torch.no_grad():
        for name, kw in cases.items():
            attn.set_processor(AttnProcessor2_0())
            ref = attn(**kw)
            attn.set_processor(PLADISAttnProcessor(pladis_scale=0.0))
            out = attn(**kw)
            bit = torch.equal(ref, out)
            maxdiff = (ref.float() - out.float()).abs().max().item()
            print(f"[parity] {name:22s} bit-exact={bit} max|diff|={maxdiff:.3e}", flush=True)
            ok &= bit
            # regression guard: the restructured lambda>0 manual branch still runs
            attn.set_processor(
                PLADISAttnProcessor(pladis_scale=1.0, qgroup="action", n_state_tokens=1)
            )
            treated = attn(**kw)
            assert torch.isfinite(treated.float()).all(), f"{name}: lambda>0 output not finite"

    print("[parity] PASS" if ok else "[parity] FAIL", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
