# SPDX-License-Identifier: Apache-2.0
"""CPU smoke gates for the Hopfield circulation control of pladis/attn_gr00t_n17.py
(no checkpoint). docs/hopfield.md §8.

Per odd (self-attention) block, with square logits L over [state; action]:

    L_skew = (L - L^T)/2,  L_a = L + (alpha-1)*L_skew,  Z = softmax(L)V,  Z_a = softmax(L_a)V
    Z_b    = Z + beta*(Z_a - Z),   Z_out = Z_b * clamp(||Z_a|| / ||Z_b||, 0.25, 4)

  A. beta=0 (every qgroup; zero-weight schedule steps; probe on/off) is `torch.equal`
     to diffusers' AttnProcessor2_0 — the fused path, bit-identical to vanilla.
  B. alpha=1 with beta in {0.5, 1, 2} is `torch.equal` to the MANUAL dense path (the
     softmax/beta=1 PLADIS processor), with zero clamp events: the odd-block
     eager-dense control (docs/hopfield.md §2b). Also documents why the code uses
     L + (alpha-1)*L_skew: (L+L^T)/2 + (L-L^T)/2 is only 1e-6-close to L.
  C. Decomposition identities on the processor's own logits: L_sym + L_skew == L,
     L_skew^T == -L_skew exactly, xi^T L_skew xi at rounding level, eta in [-1, 1],
     and the recorded eta equals a direct computation.
  D. qgroup: rows outside the group are `torch.equal` to the dense output for
     alpha != 1, beta in {1, 2}, norm on/off, adaptive on/off.
  E. Schedule: zero-weight steps `torch.equal` to vanilla; weighted steps
     `torch.equal` to an unscheduled processor at beta*w.
  F. Every rejected setting raises (beta<0, dead flags at beta=0, temp with alpha,
     probe with an intervention, cross call, even block, empty install, wrong-length
     / all-zero schedule, block-set collision, double install); PLADIS on the even
     blocks and Hopfield on the odd blocks install together with ONE step pre-hook.
  G. assert_hopfield_delivered(): raises with nothing installed, with a never-fired
     probe, with an unreached step, with a missing (step, block) cell and with a blend
     at a zero-weight step; passes with the census on a full loop.
  H. Probe: eta / E / r / Align against a float64 reference on the same tensors; the
     priced displacement for a grid alpha (and temperature) equals the displacement
     of a processor actually run there; the priced clamp counts at (alpha, beta)
     equal what a processor run at (alpha, beta) records. That equality is what lets
     diag_hopfield.py price the whole grid from ONE vanilla rollout.
  I. Adaptive: q == k (L_skew = 0) gives eta_bar = 1 and an output equal to the
     dense control; random logits give eta_bar in (-1, 1) and the recorded
     beta_eff == beta*(1 - eta_bar).

Still gated on the real checkpoint (GPU): verify_base0_parity.py (beta=0 self case at
N1.7 shapes, and the raise on both cross cases) and eval_arm's own warm-up census.

Run: bash experiments/run.sh experiments/verify_hopfield.py
"""

import math

import numpy as np
import torch

from diffusers.models.attention_processor import AttnProcessor2_0

from pladis.attn_gr00t_n17 import (
    HOP,
    HOP_ALPHA_GRID,
    HOP_BETA_GRID,
    HOP_ROW_COLS,
    HOP_SUMMARY_COLS,
    HOP_TEMP_GRID,
    SCHED,
    HopfieldAttnProcessor,
    PLADISAttnProcessor,
    assert_delivered,
    assert_hopfield_delivered,
    fmt_hop,
    install_hopfield,
    install_pladis,
    validate_hopfield,
)
from experiments.verify_nag import _bare, _dense, _heads
from experiments.verify_step_schedule import (
    HEAD_DIM,
    HEADS,
    _MiniModel,
    _attn,
    _fresh_model,
    _inputs,
    _raises,
    _run_loop,
)

N_QUERY, N_STATE = 6, 2
_R_MIN, _R_MAX = 0.25, 4.0


def _self_attn():
    return _attn(cross=False)


def _self_inputs(n_query=N_QUERY):
    h, _ = _inputs(n_query=n_query)
    return h


def _reset():
    HOP.reset()
    SCHED.reset()


def _logits(attn, h):
    """The processor's own pre-softmax logits, recomputed from the module weights."""
    q = attn.to_q(h).view(1, -1, HEADS, HEAD_DIM).transpose(1, 2)
    k = attn.to_k(h).view(1, -1, HEADS, HEAD_DIM).transpose(1, 2)
    return (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)).float()


def _row_rel(a, b):
    """||a - b||_2 / ||b||_2 per (batch, head, row)."""
    return (torch.linalg.vector_norm(a - b, dim=-1)
            / torch.linalg.vector_norm(b, dim=-1).clamp_min(1e-6))


def gate_A():
    torch.manual_seed(0)
    attn, h = _self_attn(), _self_inputs()
    ref = AttnProcessor2_0()(attn, h)
    for qgroup in ("all", "state", "action"):
        _reset()
        got = HopfieldAttnProcessor(alpha=1.0, beta=0.0, qgroup=qgroup, n_state_tokens=N_STATE)(attn, h)
        assert torch.equal(got, ref), f"A[{qgroup}]: beta=0 left the fused path"
    # zero-weight steps of a scheduled arm
    _reset()
    proc = HopfieldAttnProcessor(alpha=1.5, beta=2.0, schedule=(0, 0, 1, 1))
    for step in (0, 1):
        SCHED.current = step
        assert torch.equal(proc(attn, h), ref), f"A: zero-weight step {step} left the fused path"
    # probe: same output, ledgers filled
    _reset()
    HOP.probe = True
    SCHED.current = 2
    got = HopfieldAttnProcessor(alpha=1.0, beta=0.0, block_idx=3)(attn, h)
    assert torch.equal(got, ref), "A[probe]: probe changed the output"
    assert HOP.n_applied == {(2, 3): 1}, HOP.n_applied
    assert HOP.eta_n and HOP.E_n and HOP.p_n and HOP.f_n, "A[probe]: a ledger stayed empty"
    _reset()
    print("PASS gate A: beta=0 / zero-weight steps / probe are bit-identical to AttnProcessor2_0")


def gate_B():
    torch.manual_seed(0)
    attn, h = _bare(_self_attn()), _self_inputs()
    _reset()
    z_d = _dense(attn, h, None)
    for beta in (0.5, 1.0, 2.0):
        for norm in ("l2", "off"):
            _reset()
            HOP.record_episode = True
            SCHED.current = 0
            got = _heads(HopfieldAttnProcessor(alpha=1.0, beta=beta, norm=norm, block_idx=1)(attn, h))
            assert torch.equal(got, z_d), f"B[beta{beta:g}/{norm}]: alpha=1 is not the dense path bitwise"
            if norm == "l2":
                assert sum(HOP.a_lo.values()) == 0 and sum(HOP.a_hi.values()) == 0, \
                    f"B[beta{beta:g}]: clamp fired on the identity"
    # why the L + (alpha-1)L_skew form: the textbook decomposition does not round-trip
    L = _logits(attn, h)
    sym, skew = 0.5 * (L + L.transpose(-2, -1)), 0.5 * (L - L.transpose(-2, -1))
    assert torch.allclose(sym + skew, L, atol=1e-6)
    assert torch.equal(L + 0.0 * skew, L), "B: L + 0*skew must be L bitwise"
    _reset()
    print("PASS gate B: alpha=1 is torch.equal to the manual dense path for beta in "
          "{0.5,1,2}, norm on/off, no clamp events")


def gate_C():
    torch.manual_seed(0)
    attn, h = _self_attn(), _self_inputs()
    L = _logits(attn, h)
    sym, skew = 0.5 * (L + L.transpose(-2, -1)), 0.5 * (L - L.transpose(-2, -1))
    assert torch.allclose(sym + skew, L, atol=1e-6)
    assert torch.equal(skew.transpose(-2, -1), -skew), "C: L_skew is not exactly antisymmetric"
    xi = torch.randn(1, HEADS, N_QUERY, 1)
    quad = (xi.transpose(-2, -1) @ skew @ xi).abs().max()
    assert quad < 1e-5 * (xi.norm() ** 2), f"C: xi^T L_skew xi = {quad:.2e}"
    s2, n2 = sym.pow(2).sum((-2, -1)), skew.pow(2).sum((-2, -1))
    eta = (s2 - n2) / (s2 + n2)
    assert bool((eta >= -1).all() and (eta <= 1).all())
    _reset()
    HOP.probe = True
    SCHED.current = 0
    HopfieldAttnProcessor(alpha=1.0, beta=0.0, block_idx=1)(attn, h)
    rec = HOP.eta_sum[(0, 1)] / HOP.eta_n[(0, 1)]
    assert abs(rec - float(eta.mean())) < 1e-5, (rec, float(eta.mean()))
    assert HOP.eta_n[(0, 1)] == HEADS
    _reset()
    print(f"PASS gate C: decomposition identities hold; recorded eta_bar {rec:+.4f} == direct")


def gate_D():
    torch.manual_seed(0)
    attn, h = _bare(_self_attn()), _self_inputs()
    _reset()
    z_d = _dense(attn, h, None)
    for qgroup, untouched in (("state", slice(N_STATE, None)), ("action", slice(0, N_STATE))):
        for beta in (1.0, 2.0):
            for norm in ("l2", "off"):
                for adaptive in (False, True):
                    _reset()
                    SCHED.current = 0
                    got = _heads(HopfieldAttnProcessor(
                        alpha=1.5, beta=beta, qgroup=qgroup, n_state_tokens=N_STATE,
                        norm=norm, adaptive=adaptive)(attn, h))
                    tag = f"{qgroup}/beta{beta:g}/{norm}/adap{int(adaptive)}"
                    assert torch.equal(got[:, :, untouched], z_d[:, :, untouched]), \
                        f"D[{tag}]: rows outside the group are not bit-identical to dense"
                    assert not torch.equal(got, z_d), f"D[{tag}]: nothing changed at all"
    _reset()
    print("PASS gate D: qgroup rows outside the selection stay bit-exact "
          "(alpha=1.5, beta {1,2}, norm on/off, adaptive on/off)")


def gate_E():
    torch.manual_seed(0)
    attn, h = _self_attn(), _self_inputs()
    ref = AttnProcessor2_0()(attn, h)
    for weights in ((0, 0, 1, 1), (0, 0.5, 1, 1.5), (1.5, 1, 0.5, 0)):
        _reset()
        proc = HopfieldAttnProcessor(alpha=1.5, beta=2.0, schedule=weights)
        for step, w in enumerate(weights):
            SCHED.current = step
            got = proc(attn, h)
            if w == 0:
                assert torch.equal(got, ref), f"E[{weights}]: zero-weight step {step} intervened"
            else:
                plain = HopfieldAttnProcessor(alpha=1.5, beta=2.0 * w)
                assert torch.equal(got, plain(attn, h)), \
                    f"E[{weights}]: step {step} != unscheduled processor at beta={2.0 * w:g}"
                assert not torch.equal(got, ref), f"E[{weights}]: step {step} did not intervene"
    _reset()
    SCHED.current = None
    _raises(lambda: HopfieldAttnProcessor(alpha=1.5, beta=2.0, schedule=(0, 0, 1, 1))(attn, h),
            what="scheduled processor with no step index")
    SCHED.current = 4
    _raises(lambda: HopfieldAttnProcessor(alpha=1.5, beta=2.0, schedule=(0, 0, 1, 1))(attn, h),
            what="step beyond the schedule")
    _reset()
    print("PASS gate E: zero-weight steps == vanilla (bit-exact); weighted steps == "
          "unscheduled processor at beta*w")


def gate_F():
    _reset()
    v = validate_hopfield
    _raises(lambda: v(1.0, -1.0), what="beta < 0")
    _raises(lambda: v(1.0, 1.0, temp=0.0), what="temp <= 0")
    _raises(lambda: v(1.0, 1.0, norm="l1"), what="unknown norm")
    for kw in (dict(alpha=1.5), dict(temp=2.0), dict(adaptive=True), dict(norm="off"),
               dict(schedule="1,1,0,0")):
        msg = _raises(lambda kw=kw: v(beta=0.0, **{"alpha": 1.0, **kw}), what=f"dead flag {kw} at beta=0")
        assert "dead" in msg, msg
    _raises(lambda: v(1.5, 1.0, temp=2.0), what="temp with alpha")
    _raises(lambda: v(1.0, 1.0, probe=True), what="probe with beta > 0")
    _raises(lambda: v(1.5, 0.0, probe=True), what="probe with alpha != 1")
    assert v(1.0, 0.0, probe=True) is None
    assert v(1.0, 0.0) is None
    note = v(1.0, 2.0)
    assert note and "EAGER-DENSE" in note, note
    assert v(1.5, 2.0) is None and v(1.0, 2.0, temp=2.0) is None
    assert fmt_hop(1.5, 5.0, "all", 1) == "a1.5,b5,qall,ns1"
    assert fmt_hop(1.0, 1.0, "action", 1, 2.0, (0, 0, 1, 1), True, "off") == \
        "a1,b1,qaction,ns1,t2,s0-0-1-1,adap,norm-off"

    # processor-level defenses
    attn, (h, enc) = _attn(cross=True), _inputs(n_query=N_QUERY)
    _raises(lambda: HopfieldAttnProcessor(alpha=1.5, beta=1.0)(attn, h, encoder_hidden_states=enc),
            what="Hopfield processor on a cross call")
    _raises(lambda: HopfieldAttnProcessor(alpha=1.5, beta=1.0, qgroup="rows"), what="bad qgroup")

    # install-level defenses
    m = _fresh_model(); HOP.reset()
    _raises(lambda: install_hopfield(m, alpha=1.5, beta=1.0, blocks=[0]), what="even block")
    m = _fresh_model(); HOP.reset()
    _raises(lambda: install_hopfield(m, alpha=1.5, beta=1.0, blocks=[9]), what="block out of range")
    m = _fresh_model(); HOP.reset()
    _raises(lambda: install_hopfield(_MiniModel(n_blocks=1).eval(), alpha=1.5, beta=1.0),
            what="DiT with no odd block")
    m = _fresh_model(); HOP.reset()
    _raises(lambda: install_hopfield(m, alpha=1.5, beta=1.0, schedule="1,1,0"), what="3 weights on N=4")
    m = _fresh_model(); HOP.reset()
    _raises(lambda: install_hopfield(m, alpha=1.5, beta=1.0, schedule="0,0,0,0"), what="all-zero schedule")
    m = _fresh_model(); HOP.reset()
    _raises(lambda: install_hopfield(m, alpha=1.5, beta=0.0), what="dead alpha at beta=0 (install)")
    m = _fresh_model(); HOP.reset()
    install_pladis(m, pladis_scale=1.0, kind="all", blocks=[1])  # a PLADIS processor on an odd block
    _raises(lambda: install_hopfield(m, alpha=1.5, beta=1.0, blocks=[1]), what="block-set collision")
    m = _fresh_model(); HOP.reset()
    install_hopfield(m, alpha=1.5, beta=1.0)
    _raises(lambda: install_hopfield(m, alpha=1.5, beta=1.0), what="double install")

    # coexistence: PLADIS on the even/text blocks + Hopfield on the odd blocks, one pre-hook
    m = _fresh_model(); HOP.reset()
    h, enc = _inputs(n_query=N_QUERY)
    ref = _run_loop(_fresh_model(seed=0), h, enc)
    m = _fresh_model(seed=0); HOP.reset()
    even = install_pladis(m, pladis_scale=1.0, kind="text", schedule="0,0,1,1")
    n_hooks = len(m.action_head.model._forward_pre_hooks)
    odd = install_hopfield(m, alpha=1.5, beta=1.0)
    assert even == [0, 4] and odd == [1, 3, 5, 7], (even, odd)
    assert len(m.action_head.model._forward_pre_hooks) == n_hooks == 1, "F: step probe stacked"
    outs = _run_loop(m, h, enc)
    assert all(not torch.equal(o, r) for o, r in zip(outs, ref)), "F: combined arm changed nothing"
    assert_delivered()
    census = assert_hopfield_delivered()
    assert "blocks [1, 3, 5, 7]" in census, census
    _reset()
    print("PASS gate F: every invalid setting raises; PLADIS(even) + Hopfield(odd) coexist "
          "on one DiT with one pre-hook")


def gate_G():
    _reset()
    _raises(assert_hopfield_delivered, what="nothing installed")
    h, enc = _inputs(n_query=N_QUERY)
    m = _fresh_model(); HOP.reset()
    install_hopfield(m, alpha=1.5, beta=1.0)
    msg = _raises(assert_hopfield_delivered, what="probe never fired")
    assert "never fired" in msg, msg
    m.action_head.model(h, enc, timestep=torch.full((1,), 500))  # only step 2 visited
    msg = _raises(assert_hopfield_delivered, what="unreached steps")
    assert "never reached" in msg, msg
    HOP.clear_episode()  # drop the step-2-only call; SCHED.seen keeps {2} (harmless)
    _run_loop(m, h, enc)
    census = assert_hopfield_delivered()
    assert "calls/step {0: 4, 1: 4, 2: 4, 3: 4}" in census, census
    # a missing (step, block) cell
    del HOP.n_applied[(2, 3)]
    msg = _raises(assert_hopfield_delivered, what="missing cell")
    assert "(2, 3)" in msg, msg
    # a blend at a zero-weight step of a scheduled arm
    m = _fresh_model(); HOP.reset()
    install_hopfield(m, alpha=1.5, beta=1.0, schedule="0,0,1,1")
    _run_loop(m, h, enc)
    assert set(s for s, _ in HOP.n_applied) == {2, 3} and set(s for s, _ in HOP.n_skipped) == {0, 1}
    assert_hopfield_delivered()
    HOP.n_applied[(0, 1)] = 1
    msg = _raises(assert_hopfield_delivered, what="blend at a zero-weight step")
    assert "zero-weight" in msg, msg
    _reset()
    print(f"PASS gate G: delivery census — {census}")


def gate_H():
    torch.manual_seed(0)
    attn, h = _bare(_self_attn()), _self_inputs()
    cell = (1, 3)

    # --- the probe run, and its ledgers frozen before any other processor runs ---
    _reset()
    HOP.probe = True
    SCHED.current = cell[0]
    HopfieldAttnProcessor(alpha=1.0, beta=0.0, block_idx=cell[1])(attn, h)
    eta_rec = HOP.eta_sum[cell] / HOP.eta_n[cell]
    E_rec, r_rec = HOP.E_sum[cell] / HOP.E_n[cell], HOP.r_sum[cell] / HOP.E_n[cell]
    al_rec = HOP.al_sum[cell] / HOP.E_n[cell]
    priced = {k: HOP.p_sum[k] / HOP.p_n[k] for k in HOP.p_n}
    clip = {k: (dict(HOP.p_lo[k]), dict(HOP.p_hi[k])) for k in HOP.p_n}
    floor = HOP.f_sum[cell] / HOP.f_n[cell]
    summary, rows = HOP.episode_stats()
    assert set(summary) == set(HOP_SUMMARY_COLS) and len(rows) == 1
    assert set(rows[0]) == set(HOP_ROW_COLS) and rows[0]["step"] == 1 and rows[0]["block"] == 3
    assert summary["n_calls"] == 1 and abs(summary["eta_mean"] - eta_rec) < 1e-9
    assert summary["floor_mean"] == floor and abs(floor) < 1e-4, floor  # CPU fused vs manual
    HOP.clear_episode()
    assert HOP.probe and HOP.episode_stats() == ({}, [])

    # --- float64 reference of the paper's diagnostics on the same tensors ---
    with torch.no_grad():
        W = {n: attn.__getattr__(n).weight.double().numpy() for n in ("to_q", "to_k", "to_v")}
    x = h[0].double().numpy()  # (L, D)
    q = (x @ W["to_q"].T).reshape(N_QUERY, HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = (x @ W["to_k"].T).reshape(N_QUERY, HEADS, HEAD_DIM).transpose(1, 0, 2)
    L = q @ k.transpose(0, 2, 1) / math.sqrt(HEAD_DIM)  # (H, L, L)
    sym, skew = 0.5 * (L + L.transpose(0, 2, 1)), 0.5 * (L - L.transpose(0, 2, 1))
    s2, n2 = (sym ** 2).sum((1, 2)), (skew ** 2).sum((1, 2))
    eta = ((s2 - n2) / (s2 + n2)).mean()
    P = np.exp(L - L.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    xi = P @ x[None]  # (H, L, D)
    hf = sym @ xi
    lam = xi * hf
    E = (-0.5 * lam.sum(1)).mean()
    r = (lam < 0).mean()
    al = (lam.sum(1) / (np.linalg.norm(xi, axis=1) * np.linalg.norm(hf, axis=1))).mean()
    for name, got, want in (("eta", eta_rec, eta), ("E", E_rec, E), ("r", r_rec, r), ("Align", al_rec, al)):
        assert abs(got - want) <= 1e-4 * max(1.0, abs(want)), f"H[{name}]: {got} vs {want}"

    # --- priced displacement == displacement of a processor actually run there ---
    _reset()
    z_d = _dense(attn, h, None)  # already head-split (verify_nag._dense)
    for kind, grid in (("alpha", HOP_ALPHA_GRID), ("temp", HOP_TEMP_GRID)):
        for v in grid:
            _reset()
            kw = dict(alpha=v) if kind == "alpha" else dict(alpha=1.0, temp=v)
            got = _heads(HopfieldAttnProcessor(beta=1.0, norm="off", **kw)(attn, h))
            d = float(_row_rel(got, z_d).mean())
            want = priced[cell + (kind, v)]
            assert abs(d - want) < 1e-5, f"H[{kind}={v:g}]: run {d:.6f} vs priced {want:.6f}"
    # --- priced clamp counts == what the arm's norm-match records ---
    for kind, v in (("alpha", 2.0), ("alpha", 0.5), ("temp", 3.0)):
        for beta in HOP_BETA_GRID:
            _reset()
            HOP.record_episode = True
            SCHED.current = cell[0]
            kw = dict(alpha=v) if kind == "alpha" else dict(alpha=1.0, temp=v)
            HopfieldAttnProcessor(beta=beta, norm="l2", block_idx=cell[1], **kw)(attn, h)
            lo, hi = HOP.a_lo.get(cell, 0), HOP.a_hi.get(cell, 0)
            plo, phi = clip[cell + (kind, v)]
            assert (lo, hi) == (plo[beta], phi[beta]), \
                f"H[{kind}={v:g}/beta{beta:g}]: arm clamp ({lo},{hi}) vs priced ({plo[beta]},{phi[beta]})"
    _reset()
    print(f"PASS gate H: probe stats == float64 reference (eta {eta:+.3f}, E {E:+.3g}, r {r:.3f}, "
          f"Align {al:+.3f}); priced grid == processors run at each point; clamp counts agree")


def gate_I():
    torch.manual_seed(0)
    attn, h = _bare(_self_attn()), _self_inputs()
    # (1) symmetric logits: q == k -> L_skew == 0 -> eta_bar = 1 -> identity
    with torch.no_grad():
        attn.to_k.weight.copy_(attn.to_q.weight)
    _reset()
    z_d = _dense(attn, h, None)
    HOP.record_episode = True
    SCHED.current = 0
    got = _heads(HopfieldAttnProcessor(alpha=1.5, beta=2.0, adaptive=True, block_idx=1)(attn, h))
    eta_bar = HOP.eta_sum[(0, 1)] / HOP.eta_n[(0, 1)]
    assert eta_bar > 1 - 1e-6, eta_bar
    assert torch.allclose(got, z_d, atol=1e-6), "I: adaptive at eta_bar=1 did not vanish"
    beff = HOP.beff_sum[(0, 1)] / HOP.beff_n[(0, 1)]
    assert abs(beff) < 1e-6, beff
    # (2) random logits: eta_bar in (-1, 1), beta_eff == beta*(1 - eta_bar), and the
    #     output equals the STATIC processor at (1 + (alpha-1)*eta_bar, beta*(1-eta_bar))
    torch.manual_seed(1)
    attn, h = _bare(_self_attn()), _self_inputs()
    _reset()
    HOP.record_episode = True
    SCHED.current = 0
    got = _heads(HopfieldAttnProcessor(alpha=1.5, beta=2.0, adaptive=True, block_idx=1)(attn, h))
    eta_bar = HOP.eta_sum[(0, 1)] / HOP.eta_n[(0, 1)]
    beff = HOP.beff_sum[(0, 1)] / HOP.beff_n[(0, 1)]
    assert -1 < eta_bar < 1 and abs(beff - 2.0 * (1 - eta_bar)) < 1e-6, (eta_bar, beff)
    _reset()
    static = _heads(HopfieldAttnProcessor(alpha=1 + 0.5 * eta_bar, beta=beff)(attn, h))
    assert torch.allclose(got, static, atol=1e-5), "I: adaptive != static at (alpha_eff, beta_eff)"
    _reset()
    print(f"PASS gate I: adaptive vanishes at eta_bar=1; at eta_bar={eta_bar:+.3f} it equals "
          f"the static arm at (alpha_eff, beta_eff={beff:.3f})")


def main():
    torch.manual_seed(0)
    gate_A(); gate_B(); gate_C(); gate_D(); gate_E(); gate_F(); gate_G(); gate_H(); gate_I()
    _reset()
    print("ALL GATES PASSED (CPU smoke; alpha/temperature selection is "
          "experiments/diag_hopfield.py, on-checkpoint delivery is eval_arm's warm-up)")


if __name__ == "__main__":
    main()
