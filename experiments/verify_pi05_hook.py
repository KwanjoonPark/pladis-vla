# SPDX-License-Identifier: Apache-2.0
"""CPU smoke gates for pladis/attn_pi05.py (no openpi model needed).

What CAN be proven on any machine with torch+transformers+entmax:
  A. lambda=0 and prefix-pass (q_len > max_suffix_query) outputs are BIT-identical
     to the stock gemma eager_attention_forward (torch.equal, fp32 and bf16).
  B. The kind={text,image} blend equals the official FLUX formulation
     m * (lambda*entmax15(sub) + (1-lambda)*softmax_block(sub))  (pipeline_flux.py:105-113)
     via the identity dense[sub] == m * softmax_block(sub).
  C. Row normalization and block mass are preserved for ANY lambda.
  D. method=softmax, beta=1 collapses to dense (the eager-dense control).
  E. Defenses fire: geometry that does not fit raises; qgroup=state raises;
     assert_delivered raises on zero deliveries and is exempt at lambda=0.
  F. The monkeypatch intercepts a REAL transformers GemmaAttention forward
     (dispatch integration), and install is idempotent (no double wrap).

What CANNOT be proven here and stays gated on the openpi machine (4.53.2 pin):
the real pi0.5 expert being a transformers gemma module on the eager path, and
the true [image|language|suffix] geometry — run the delivery gate there.

Run: bash experiments/run.sh experiments/verify_pi05_hook.py
"""

import torch
from entmax import entmax15
from types import SimpleNamespace

import transformers.models.gemma.modeling_gemma as mg
from transformers.models.gemma.modeling_gemma import repeat_kv

# capture the STOCK function before any install wraps it
STOCK = mg.eager_attention_forward
assert not getattr(STOCK, "_pladis_wrapped", False), "gemma eager already wrapped; run in a fresh process"

from pladis.attn_pi05 import CFG, install_pladis, assert_delivered

# tiny joint-attention geometry: [image 12 | language 6 | suffix 4]
NI, NL, NS = 12, 6, 4
SEQ = NI + NL + NS
B, H, KVH, D = 2, 4, 2, 16
SCALING = D ** -0.5

eager_model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))


def mk_module():
    return SimpleNamespace(num_key_value_groups=H // KVH, training=False)


def mk_qkv(q_len, k_len, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, q_len, D, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(B, KVH, k_len, D, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(B, KVH, k_len, D, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v


def mk_mask(q_len, k_len, dtype, n_masked=2):
    # additive mask, last n_masked language columns masked (padding-like)
    m = torch.zeros(B, 1, q_len, k_len, dtype=dtype)
    m[..., NI + NL - n_masked : NI + NL] = torch.finfo(dtype).min
    return m


def scores_of(q, k, mask):
    ks = repeat_kv(k, H // KVH)
    s = torch.matmul(q, ks.transpose(2, 3)) * SCALING
    return s if mask is None else s + mask[:, :, :, : ks.shape[-2]]


# ---------------------------------------------------------------- gate A: bit parity
def gate_A():
    install_pladis(eager_model, pladis_scale=0.0, kind="text", n_img_prefix=NI, n_lang=NL)
    for dtype in (torch.float32, torch.bfloat16):
        for q_len, tag in ((NS, "suffix-shaped, lambda=0"), (SEQ, "any-shape, lambda=0")):
            q, k, v = mk_qkv(q_len, SEQ, dtype)
            mask = mk_mask(q_len, SEQ, dtype)
            o_ref, w_ref = STOCK(mk_module(), q, k, v, mask, SCALING, dropout=0.0)
            o_new, w_new = mg.eager_attention_forward(mk_module(), q, k, v, mask, SCALING, dropout=0.0)
            assert torch.equal(o_ref, o_new) and torch.equal(w_ref, w_new), f"A: {tag} {dtype} not bit-equal"
    # prefix pass: lambda>0 but q_len > max_suffix_query must stay bit-vanilla
    install_pladis(eager_model, pladis_scale=1.5, kind="text", n_img_prefix=NI, n_lang=NL, max_suffix_query=NS)
    q, k, v = mk_qkv(SEQ, SEQ, torch.float32)  # SEQ > NS -> prefix gate
    mask = mk_mask(SEQ, SEQ, torch.float32)
    o_ref, w_ref = STOCK(mk_module(), q, k, v, mask, SCALING, dropout=0.0)
    o_new, w_new = mg.eager_attention_forward(mk_module(), q, k, v, mask, SCALING, dropout=0.0)
    assert torch.equal(o_ref, o_new) and torch.equal(w_ref, w_new), "A: prefix pass not bit-equal"
    print("PASS gate A: lambda=0 + prefix pass bit-identical to stock (fp32, bf16)")


# ------------------------------------------------- gate B: FLUX-formulation identity
def gate_B():
    for kind, lo, hi in (("text", NI, NI + NL), ("image", 0, NI)):
        for lam in (1.0, 1.5, 2.5):
            install_pladis(eager_model, pladis_scale=lam, kind=kind, n_img_prefix=NI, n_lang=NL)
            q, k, v = mk_qkv(NS, SEQ, torch.float32, seed=1)
            _, w = mg.eager_attention_forward(mk_module(), q, k, v, None, SCALING, dropout=0.0)
            s = scores_of(q, k, None).to(torch.float32)
            dense = torch.softmax(s, dim=-1)
            sub = s[..., lo:hi]
            m = dense[..., lo:hi].sum(-1, keepdim=True)
            flux = dense.clone()  # official FLUX pattern, pipeline_flux.py:105-113
            flux[..., lo:hi] = m * (lam * entmax15(sub, dim=-1) + (1 - lam) * torch.softmax(sub, dim=-1))
            assert torch.allclose(w, flux, atol=1e-6), f"B: {kind} lam={lam} != FLUX formulation"
    print("PASS gate B: kind blend == official FLUX m*(lam*entmax + (1-lam)*softmax_block)")


# ------------------------------------------- gate C: row/block mass preservation
def gate_C():
    for kind, lo, hi in (("text", NI, NI + NL), ("image", 0, NI)):
        for lam in (1.0, 1.5, 2.5):
            install_pladis(eager_model, pladis_scale=lam, kind=kind, n_img_prefix=NI, n_lang=NL)
            q, k, v = mk_qkv(NS, SEQ, torch.float32, seed=2)
            mask = mk_mask(NS, SEQ, torch.float32)
            _, w = mg.eager_attention_forward(mk_module(), q, k, v, mask, SCALING, dropout=0.0)
            s = scores_of(q, k, mask).to(torch.float32)
            dense = torch.softmax(s, dim=-1)
            m = dense[..., lo:hi].sum(-1)
            assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5), f"C: rows not normalized {kind} {lam}"
            assert torch.allclose(w[..., lo:hi].sum(-1), m, atol=1e-5), f"C: block mass changed {kind} {lam}"
            masked = w[..., NI + NL - 2 : NI + NL]
            assert masked.abs().max() < 1e-6, f"C: masked columns got weight {kind} {lam}"
    print("PASS gate C: rows sum to 1, block mass preserved, masked columns stay 0 (any lambda)")


# ------------------------------------------------- gate D: beta=1 softmax collapse
def gate_D():
    for kind in ("all", "text"):
        install_pladis(eager_model, pladis_scale=1.7, method="softmax", beta=1.0, kind=kind,
                       n_img_prefix=NI, n_lang=NL)
        q, k, v = mk_qkv(NS, SEQ, torch.float32, seed=3)
        mask = mk_mask(NS, SEQ, torch.float32)
        _, w = mg.eager_attention_forward(mk_module(), q, k, v, mask, SCALING, dropout=0.0)
        dense = torch.softmax(scores_of(q, k, mask).to(torch.float32), dim=-1)
        assert torch.allclose(w, dense, atol=1e-6), f"D: beta=1 softmax did not collapse to dense ({kind})"
    print("PASS gate D: method=softmax beta=1 collapses to dense for any lambda (eager-dense control)")


# --------------------------------------------------------------- gate E: defenses
def gate_E():
    install_pladis(eager_model, pladis_scale=1.5, kind="text", n_img_prefix=768, n_lang=200)
    q, k, v = mk_qkv(NS, SEQ, torch.float32)  # key_len=22 << needed 968 -> must raise
    try:
        mg.eager_attention_forward(mk_module(), q, k, v, None, SCALING, dropout=0.0)
        raise AssertionError("E: bad geometry did NOT raise")
    except RuntimeError:
        pass
    try:
        install_pladis(eager_model, pladis_scale=1.5, qgroup="state")
        raise AssertionError("E: qgroup=state did NOT raise")
    except ValueError:
        pass
    install_pladis(eager_model, pladis_scale=1.5, kind="text", n_img_prefix=NI, n_lang=NL)
    CFG.n_calls = 0
    try:
        assert_delivered()
        raise AssertionError("E: assert_delivered passed with zero deliveries")
    except RuntimeError:
        pass
    install_pladis(eager_model, pladis_scale=0.0, kind="text", n_img_prefix=NI, n_lang=NL)
    assert_delivered()  # lambda=0 exempt by design
    print("PASS gate E: geometry raise, qgroup=state raise, assert_delivered guard + lambda=0 exemption")


# ------------------------------------- gate F: real GemmaAttention dispatch + idempotence
def gate_F():
    from transformers.models.gemma.configuration_gemma import GemmaConfig

    cfg = GemmaConfig(hidden_size=H * D, num_attention_heads=H, num_key_value_heads=KVH,
                      head_dim=D, intermediate_size=64, num_hidden_layers=1,
                      attn_implementation="eager")
    attn = mg.GemmaAttention(cfg, layer_idx=0).eval()
    rotary = mg.GemmaRotaryEmbedding(config=cfg)
    h = torch.randn(1, SEQ, H * D)
    pos = torch.arange(SEQ)[None]
    cos, sin = rotary(h, pos)

    install_pladis(eager_model, pladis_scale=1.5, kind="text", n_img_prefix=NI, n_lang=NL,
                   max_suffix_query=SEQ)  # let the tiny full-seq pass count as a delivery
    before = CFG.n_calls
    with torch.no_grad():
        attn(hidden_states=h, position_embeddings=(cos, sin), attention_mask=None)
    assert CFG.n_calls == before + 1, "F: real GemmaAttention forward did not route through the hook"
    assert_delivered()
    # idempotence: re-install must not double-wrap
    install_pladis(eager_model, pladis_scale=1.5, kind="text", n_img_prefix=NI, n_lang=NL)
    assert mg.eager_attention_forward._orig is STOCK, "F: double wrap detected"
    print("PASS gate F: patch intercepts a real GemmaAttention forward; install idempotent")


def main():
    torch.manual_seed(0)
    gate_A(); gate_B(); gate_C(); gate_D(); gate_E(); gate_F()
    print("ALL GATES PASSED (CPU smoke; openpi-machine delivery gate still required)")


if __name__ == "__main__":
    main()
