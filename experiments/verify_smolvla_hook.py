# SPDX-License-Identifier: Apache-2.0
"""CPU smoke gates for pladis/attn_smolvla.py (no checkpoint needed).

  A. lambda=0 calls and non-intervened passes (VLM prefix pass; lambda>0 too)
     are BIT-identical to lerobot's stock eager_attention_forward
     (torch.equal, fp32 and bf16).
  B. Every kind's blend equals the official FLUX mass-preserving formulation
     at the OUTPUT level (sub-block kinds) / the plain PLADIS blend
     (kind="prefix", whole row), lambda in {1.0, 1.5, 2.5}.
  C. Rows stay normalized, block mass is preserved, masked columns stay zero.
  D. method=softmax beta=1 collapses to dense for any lambda.
  E. Defenses: denoise call before any prefix pass raises; geometry that fits
     neither CA nor SA raises; derived language width out of bounds raises;
     qgroup=state raises; assert_delivered guards (kind='self' counts SA
     only) with the lambda=0 exemption.
  F. Self-calibration: the language width follows the per-episode prefix pass
     (padding="longest"), and a stale prefix length raises.
  G. The patch binds to ONE real SmolVLMWithExpertModel instance (class left
     untouched), and lerobot's get_attention_interface still resolves
     ``self.eager_attention_forward`` (source assertion).

Still gated on the real checkpoint (GPU): the on-model delivery gate and the
n_img=128 ground-truth probe.

Run: bash experiments/run.sh experiments/verify_smolvla_hook.py
"""

import inspect

import torch
from entmax import entmax15

from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

STOCK = SmolVLMWithExpertModel.eager_attention_forward  # unbound stock function

from pladis.attn_smolvla import (
    CFG,
    _blend_rows,
    assert_delivered,
    install_pladis,
)

B, H, KVH, D = 2, 4, 2, 16
NI, NS = 12, 1          # tiny image/state blocks
NL_A, NL_B = 6, 8       # two "episodes" with different language widths
P_A, P_B = NI + NL_A + NS, NI + NL_B + NS
S = 4                   # suffix (action chunk) length


class _StubExpert:
    num_attention_heads = H
    num_key_value_heads = KVH

    def get_attention_interface(self):  # same resolution rule as lerobot
        return self.eager_attention_forward

    def eager_attention_forward(self, *a):
        return STOCK(self, *a)


def mk_qkv(q_len, k_len, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, q_len, H, D, generator=g).to(dtype)
    k = torch.randn(B, k_len, KVH, D, generator=g).to(dtype)
    v = torch.randn(B, k_len, KVH, D, generator=g).to(dtype)
    return q, k, v


def mk_mask(q_len, k_len, n_masked=2):
    m = torch.ones(B, q_len, k_len, dtype=torch.bool)
    if n_masked:
        m[..., NI + 4 : NI + 4 + n_masked] = False  # padding-like language cols
    return m


def scores_of(stub, q, k, mask):
    ks = k[:, :, :, None, :].expand(B, k.shape[1], KVH, H // KVH, D).reshape(B, k.shape[1], H, D)
    w = torch.matmul(
        q.to(torch.float32).transpose(1, 2), ks.to(torch.float32).transpose(1, 2).transpose(2, 3)
    ) * D**-0.5
    return torch.where(mask[:, None, :, :], w, torch.finfo(torch.float32).min)


def calibrate(stub, p_len):
    """Run a prefix-shaped pass so the hook records prefix_len = p_len."""
    q, k, v = mk_qkv(p_len, p_len)
    stub.eager_attention_forward(mk_mask(p_len, p_len, 0), B, D, q, k, v)


def gate_A():
    stub = _StubExpert()
    install_pladis(stub, pladis_scale=0.0, kind="text", n_img=NI, n_state_tokens=NS)
    for dtype in (torch.float32, torch.bfloat16):
        for q_len, k_len, tag in ((P_A, P_A, "prefix"), (S, P_A, "CA"), (S, P_A + S, "SA")):
            q, k, v = mk_qkv(q_len, k_len, dtype)
            mask = mk_mask(q_len, k_len)
            ref = STOCK(_StubExpert(), mask, B, D, q, k, v)
            out = stub.eager_attention_forward(mask, B, D, q, k, v)
            assert torch.equal(ref, out), f"A: lambda=0 {tag} {dtype} not bit-equal"
    # lambda>0: the prefix pass itself must stay bit-vanilla
    install_pladis(stub, pladis_scale=1.5, kind="text", n_img=NI, n_state_tokens=NS)
    q, k, v = mk_qkv(P_A, P_A)
    mask = mk_mask(P_A, P_A, 0)
    assert torch.equal(
        STOCK(_StubExpert(), mask, B, D, q, k, v), stub.eager_attention_forward(mask, B, D, q, k, v)
    ), "A: prefix pass at lambda>0 not bit-equal"
    print("PASS gate A: lambda=0 + prefix pass bit-identical to stock (fp32, bf16)")


def gate_B():
    cases = [
        ("image", 0, NI, S, P_A, False),
        ("text", NI, NI + NL_A, S, P_A, False),
        ("state", NI + NL_A, P_A, S, P_A, False),
        ("self", P_A, P_A + S, S, P_A + S, False),
        ("prefix", 0, P_A, S, P_A, True),
    ]
    for kind, lo, hi, q_len, k_len, whole in cases:
        for lam in (1.0, 1.5, 2.5):
            stub = _StubExpert()
            install_pladis(stub, pladis_scale=lam, kind=kind, n_img=NI, n_state_tokens=NS)
            calibrate(stub, P_A)
            q, k, v = mk_qkv(q_len, k_len, seed=1)
            mask = mk_mask(q_len, k_len, 0)  # finite scores -> the FLUX identity is exact
            out = stub.eager_attention_forward(mask, B, D, q, k, v)
            w = scores_of(stub, q, k, mask)
            dense = torch.softmax(w, dim=-1)
            ref_probs = dense.clone()
            if whole:
                ref_probs = dense + lam * (entmax15(w, dim=-1) - dense)
            else:
                sub = w[..., lo:hi]
                m = dense[..., lo:hi].sum(-1, keepdim=True)
                ref_probs[..., lo:hi] = m * (
                    lam * entmax15(sub, dim=-1) + (1 - lam) * torch.softmax(sub, dim=-1)
                )
            ks = k[:, :, :, None, :].expand(B, k_len, KVH, H // KVH, D).reshape(B, k_len, H, D)
            vs = v[:, :, :, None, :].expand(B, k_len, KVH, H // KVH, D).reshape(B, k_len, H, D)
            ref = torch.matmul(ref_probs.to(vs.dtype), vs.permute(0, 2, 1, 3))
            ref = ref.permute(0, 2, 1, 3).reshape(B, -1, H * D)
            assert torch.allclose(out, ref, atol=1e-6), f"B: {kind} lam={lam} != FLUX/PLADIS form"
    print("PASS gate B: all five kinds match the FLUX mass-preserving / plain PLADIS form")


def gate_C():
    for kind, lo, hi, k_len in (
        ("text", NI, NI + NL_A, P_A),
        ("self", P_A, P_A + S, P_A + S),
    ):
        install_pladis(_StubExpert(), pladis_scale=1.5, kind=kind, n_img=NI, n_state_tokens=NS)
        q, k, v = mk_qkv(S, k_len, seed=2)
        w = scores_of(_StubExpert(), q, k, mk_mask(S, k_len))
        dense = torch.softmax(w, dim=-1)
        out = _blend_rows(w, lo, hi, False)
        assert torch.allclose(out.sum(-1), torch.ones_like(out.sum(-1)), atol=1e-5), f"C: rows {kind}"
        assert torch.allclose(
            out[..., lo:hi].sum(-1), dense[..., lo:hi].sum(-1), atol=1e-5
        ), f"C: block mass {kind}"
        assert out[..., NI + 4 : NI + 6].abs().max() < 1e-6, f"C: masked cols {kind}"
    print("PASS gate C: rows sum to 1, block mass preserved, masked columns stay 0")


def gate_D():
    for kind, q_len, k_len in (("text", S, P_A), ("self", S, P_A + S), ("prefix", S, P_A)):
        stub = _StubExpert()
        install_pladis(stub, pladis_scale=1.7, method="softmax", beta=1.0, kind=kind,
                       n_img=NI, n_state_tokens=NS)
        calibrate(stub, P_A)
        q, k, v = mk_qkv(q_len, k_len, seed=3)
        mask = mk_mask(q_len, k_len)
        out = stub.eager_attention_forward(mask, B, D, q, k, v)
        ref = STOCK(_StubExpert(), mask, B, D, q, k, v)
        assert torch.allclose(out, ref, atol=1e-6), f"D: beta=1 softmax != dense ({kind})"
    print("PASS gate D: method=softmax beta=1 collapses to dense for any lambda")


def _expect_raise(fn, what):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError(f"E/F: {what} did NOT raise")


def gate_E():
    stub = _StubExpert()
    install_pladis(stub, pladis_scale=1.5, kind="text", n_img=NI, n_state_tokens=NS)
    q, k, v = mk_qkv(S, P_A)
    _expect_raise(lambda: stub.eager_attention_forward(mk_mask(S, P_A, 0), B, D, q, k, v),
                  "denoise before prefix pass")
    calibrate(stub, P_A)
    q2, k2, v2 = mk_qkv(S, P_A + 2)
    _expect_raise(lambda: stub.eager_attention_forward(mk_mask(S, P_A + 2, 0), B, D, q2, k2, v2),
                  "geometry fitting neither CA nor SA")
    # derived language width above n_lang_max
    install_pladis(stub, pladis_scale=1.5, kind="text", n_img=NI, n_lang_max=4, n_state_tokens=NS)
    calibrate(stub, P_A)  # implies n_lang=6 > 4
    q3, k3, v3 = mk_qkv(S, P_A)
    _expect_raise(lambda: stub.eager_attention_forward(mk_mask(S, P_A, 0), B, D, q3, k3, v3),
                  "language width out of bounds")
    try:
        install_pladis(stub, qgroup="state")
        raise AssertionError("E: qgroup=state did NOT raise")
    except ValueError:
        pass
    # assert_delivered: kind='self' must not count CA hits
    # (n_lang_max=48 restores the bound narrowed above — CFG args persist by design)
    install_pladis(stub, pladis_scale=1.5, kind="self", n_img=NI, n_lang_max=48, n_state_tokens=NS)
    calibrate(stub, P_A)
    q4, k4, v4 = mk_qkv(S, P_A)
    stub.eager_attention_forward(mk_mask(S, P_A, 0), B, D, q4, k4, v4)  # CA call
    _expect_raise(assert_delivered, "assert_delivered with zero SA deliveries")
    install_pladis(stub, pladis_scale=0.0, kind="text", n_img=NI, n_state_tokens=NS)
    assert_delivered()  # lambda=0 exempt
    print("PASS gate E: geometry/qgroup/assert_delivered defenses all fire")


def gate_F():
    stub = _StubExpert()
    install_pladis(stub, pladis_scale=1.5, kind="text", n_img=NI, n_state_tokens=NS)
    for p_len in (P_A, P_B):  # episode A then episode B (longer instruction)
        calibrate(stub, p_len)
        q, k, v = mk_qkv(S, p_len)
        stub.eager_attention_forward(mk_mask(S, p_len, 0), B, D, q, k, v)
    assert CFG.n_calls_ca == 2, "F: CA blend did not fire for both widths"
    q, k, v = mk_qkv(S, P_A)  # stale geometry from episode A must now raise
    _expect_raise(lambda: stub.eager_attention_forward(mk_mask(S, P_A, 0), B, D, q, k, v),
                  "stale prefix length")
    print("PASS gate F: language width follows each episode's prefix pass")


def gate_G():
    src = inspect.getsource(SmolVLMWithExpertModel.get_attention_interface)
    assert "self.eager_attention_forward" in src, (
        "G: lerobot's get_attention_interface no longer resolves "
        "self.eager_attention_forward — instance patching is broken, re-port the hook"
    )
    obj = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    torch.nn.Module.__init__(obj)
    obj.num_attention_heads = H
    obj.num_key_value_heads = KVH
    install_pladis(obj, pladis_scale=1.5, kind="text", n_img=NI, n_state_tokens=NS)
    fn = obj.get_attention_interface()
    assert getattr(fn, "_pladis_wrapped", False), "G: instance dispatch not wrapped"
    calibrate(obj, P_A)
    q, k, v = mk_qkv(S, P_A)
    obj.eager_attention_forward(mk_mask(S, P_A, 0), B, D, q, k, v)
    assert_delivered()
    obj2 = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    torch.nn.Module.__init__(obj2)
    fn2 = obj2.get_attention_interface()
    assert not getattr(fn2, "_pladis_wrapped", False), "G: class method was mutated"
    print("PASS gate G: real-instance dispatch intercepted; class and other instances untouched")


def main():
    torch.manual_seed(0)
    gate_A(); gate_B(); gate_C(); gate_D(); gate_E(); gate_F(); gate_G()
    print("ALL GATES PASSED (CPU smoke; real-checkpoint delivery gate still required)")


if __name__ == "__main__":
    main()
