# SPDX-License-Identifier: Apache-2.0
"""CPU smoke gates for the NAG stages of pladis/attn_gr00t_n17.py (no checkpoint).

NAG (arXiv:2505.21179 Eq. 8-10) in the mapping of docs/nag.md §1 — the DENSE
branch is the positive baseline, so with lambda the blend strength:

    Z_PL     = Z_d + lambda*(Z_s - Z_d)
    R[i]     = ||Z_PL[i]||_1 / (||Z_d[i]||_1 + eps)     per (head, query row)
    Z_NPL[i] = min(R[i], tau)/R[i] * Z_PL[i]            NORMALIZATION
    Z_final  = rho*Z_NPL + (1 - rho)*Z_d                REFINEMENT

  A. tau=off with the NAG code path armed (probe) is `torch.equal` to the pre-NAG
     path at the same lambda — for both sparse branches and all three qgroups.
     This is what lets a NAG arm be compared against collected arms at all.
  B. The formula, checked against Z_PL and Z_d recovered from the SAME processor
     at other settings (lambda with tau off; lambda=0): R matches a per-(head,row)
     reference, the cap fires exactly on R > tau, the census counts what fired,
     and uncapped rows equal Z_d + rho*lambda*(Z_s - Z_d) — docs/nag.md §2b, the
     identity that makes "refinement only" a dose rescale rather than a new arm.
  C. tau=1 pins every capped row's L1 magnitude to the dense branch's, and leaves
     rows with R <= 1 bit-identical to the uncapped output.
  D. qgroup: rows outside the selected group are bit-identical to dense even at
     rho=0.5, where blending them would return rho*x + (1-rho)*x != x.
  E. Step schedule composition: a zero-weight step stays bit-identical to
     diffusers' AttnProcessor2_0 with NAG armed; the weighted steps carry the cap.
  F. The three settings that would silently be a different arm all raise: tau < 1,
     rho outside (0,1], rho<1 with no tau, NAG at scale=0 — plus conflicting caps
     across the cells of one arm.
  G. assert_nag_delivered() reports the clip census and raises when the cap never
     fired (that arm is bit-identical to its own uncapped control).
  I. Per-episode R statistics (eval_arm --pladis-nag-probe / rstats sidecars):
     recording with the cap off leaves the output bit-identical; the summary's
     mean/max/P(R>t) match a direct computation on the same tensors; the per
     (step, block) rows sum back to the summary; clear_episode() empties the
     ledgers and keeps the arm settings; the cap ledger's clip rate is reported.
  H. The probe's off-lambda reconstruction is EXACT: R measured at lambda' from an
     arm running at lambda equals R measured by an arm actually running at lambda'.
     experiments/diag_nag.py prices the whole dose ladder on one trajectory with
     it, so a wrong reconstruction would pick tau off a fictional distribution.

Still gated on the real checkpoint (GPU): verify_base0_parity.py for lambda=0
bit-parity, eval_arm's own NAG warm-up on the live serving path, and
experiments/diag_nag.py for the tau that these gates take as given.

Run: bash experiments/run.sh experiments/verify_nag.py
"""

import torch

from diffusers.models.attention_processor import AttnProcessor2_0

from pladis.attn_gr00t_n17 import (
    NAG,
    NAG_CANDIDATE_TAUS,
    R_THRESHOLDS,
    SCHED,
    PLADISAttnProcessor,
    assert_nag_delivered,
    fmt_nag,
    install_pladis,
    validate_nag,
)
from experiments.verify_step_schedule import (
    HEADS,
    HEAD_DIM,
    _attn,
    _fresh_model,
    _inputs,
    _raises,
)

BRANCHES = (("ent15max", 1.0), ("softmax", 2.0))  # entmax-1.5 and the beta=2 mirror
N_QUERY, N_STATE = 6, 2


def _bare(attn):
    """Make the processor's return value BE Z (heads merged): identity to_out.

    Attention.to_out[0] is a Linear(inner_dim, query_dim) and here inner_dim ==
    query_dim, so an identity weight with no bias turns the processor's output into
    the attention output itself. That is what lets these gates check the NAG algebra
    against real Z tensors without re-implementing the blend (a second
    implementation would only test itself).
    """
    with torch.no_grad():
        attn.to_out[0].weight.copy_(torch.eye(attn.to_out[0].weight.shape[0]))
        if attn.to_out[0].bias is not None:
            attn.to_out[0].bias.zero_()
    assert not attn.residual_connection and attn.rescale_output_factor == 1.0
    return attn


def _heads(z):
    """(1, L, H*D) -> (1, H, L, D): undo the merge the processor does at the end."""
    b, l, _ = z.shape
    return z.view(b, l, HEADS, HEAD_DIM).transpose(1, 2)


def _l1(z):
    return z.abs().sum(dim=-1, keepdim=True)


def _run(proc, attn, h, enc):
    return proc(attn, h, encoder_hidden_states=enc)


def _dense(attn, h, enc):
    """Z_d as the NAG branch itself computes it: the MANUAL path, not fused SDPA.

    ``method="softmax", beta=1`` makes the sparse branch equal the dense one
    (attn_gr00t_n17.py's own integration sanity check), so the blend collapses to
    `dense` exactly at any lambda while still running the manual softmax/matmul.
    Taking the reference from `pladis_scale=0` instead would compare against
    F.scaled_dot_product_attention, whose kernel is bit-DIFFERENT from the manual
    path — that difference is a property of the fused kernel, not of NAG, and it
    would make every bit-parity assertion below fail for the wrong reason.
    """
    return _heads(_run(
        PLADISAttnProcessor(pladis_scale=1.0, method="softmax", beta=1.0, qgroup="all"),
        attn, h, enc))


def _probe(on: bool):
    NAG.reset()
    NAG.probe = on


def gate_A():
    torch.manual_seed(0)
    attn, (h, enc) = _attn(), _inputs(n_query=N_QUERY)
    for method, beta in BRANCHES:
        for qgroup in ("all", "state", "action"):
            for scale in (1.0, 2.0):
                kw = dict(pladis_scale=scale, method=method, beta=beta,
                          qgroup=qgroup, n_state_tokens=N_STATE)
                _probe(False)
                ref = _run(PLADISAttnProcessor(**kw), attn, h, enc)
                _probe(True)  # same processor settings, NAG code path, cap off
                got = _run(PLADISAttnProcessor(**kw), attn, h, enc)
                tag = f"{method}b{beta:g}/{qgroup}/lambda{scale:g}"
                assert torch.equal(got, ref), f"A[{tag}]: NAG path is not bit-identical with the cap off"
                assert NAG.p_n, f"A[{tag}]: probe recorded no ratios"
                assert not NAG.n_clipped, f"A[{tag}]: cap fired with tau off"
    _probe(False)
    print("PASS gate A: tau=off is bit-identical to the pre-NAG path "
          "(2 branches x 3 qgroups x 2 doses)")


def gate_B():
    torch.manual_seed(0)
    attn, (h, enc) = _bare(_attn()), _inputs(n_query=N_QUERY)
    for method, beta in BRANCHES:
        for lam in (1.5, 2.0, 4.0):
            kw = dict(method=method, beta=beta, qgroup="all")
            _probe(True)
            z_pl = _heads(_run(PLADISAttnProcessor(pladis_scale=lam, **kw), attn, h, enc))
            _probe(False)
            z_d = _dense(attn, h, enc)
            ratio = _l1(z_pl) / (_l1(z_d) + 1e-6)
            assert ratio.shape == (1, HEADS, N_QUERY, 1), ratio.shape

            for tau in NAG_CANDIDATE_TAUS:
                for rho in (1.0, 0.5, 0.25):
                    NAG.reset()
                    got = _heads(_run(
                        PLADISAttnProcessor(pladis_scale=lam, nag_tau=tau, nag_rho=rho, **kw),
                        attn, h, enc))
                    capped = ratio > tau
                    want = torch.where(capped, tau / ratio, torch.ones_like(ratio)) * z_pl
                    want = rho * want + (1.0 - rho) * z_d
                    tag = f"{method}b{beta:g}/lambda{lam:g}/tau{tau:g}/rho{rho:g}"
                    assert torch.allclose(got, want, atol=1e-6), \
                        f"B[{tag}]: output != NAG reference (max {(got - want).abs().max():.2e})"
                    assert sum(NAG.n_clipped.values()) == int(capped.sum()), \
                        f"B[{tag}]: census counted {sum(NAG.n_clipped.values())}, cap fired {int(capped.sum())}"
                    # docs/nag.md §2b: wherever the cap is inactive, NAG IS the plain
                    # arm at dose rho*lambda -- the reason no refinement-only arm runs.
                    if not bool(capped.any()):
                        plain = _heads(_run(
                            PLADISAttnProcessor(pladis_scale=rho * lam, **kw), attn, h, enc))
                        assert torch.allclose(got, plain, atol=1e-6), \
                            f"B[{tag}]: uncapped NAG != plain arm at scale={rho * lam:g}"
    _probe(False)
    print("PASS gate B: R, cap, refinement and census match the reference; "
          "uncapped NAG == the plain arm at scale=rho*lambda")


def gate_C():
    torch.manual_seed(0)
    attn, (h, enc) = _bare(_attn()), _inputs(n_query=N_QUERY)
    kw = dict(method="softmax", beta=2.0, qgroup="all", pladis_scale=4.0)
    _probe(True)
    z_pl = _heads(_run(PLADISAttnProcessor(**kw), attn, h, enc))
    _probe(False)
    z_d = _dense(attn, h, enc)
    ratio = _l1(z_pl) / (_l1(z_d) + 1e-6)
    assert bool((ratio > 1.0).any()), "C: no row exceeds the dense magnitude at lambda=4"

    NAG.reset()
    got = _heads(_run(PLADISAttnProcessor(nag_tau=1.0, **kw), attn, h, enc))
    capped = (ratio > 1.0).squeeze(-1)
    l1_got, l1_d = _l1(got).squeeze(-1), _l1(z_d).squeeze(-1)
    assert torch.allclose(l1_got[capped], l1_d[capped], rtol=1e-5), \
        "C: capped rows are not pinned to the dense branch's L1 magnitude"
    assert torch.equal(got[~capped], z_pl[~capped]), \
        "C: rows with R <= 1 were not left bit-identical"
    print(f"PASS gate C: tau=1 pins {int(capped.sum())}/{capped.numel()} rows to "
          f"||Z_d||_1 and leaves the rest untouched")


def gate_D():
    torch.manual_seed(0)
    attn, (h, enc) = _bare(_attn()), _inputs(n_query=N_QUERY)
    _probe(False)
    z_d = _dense(attn, h, enc)
    for qgroup, untouched in (("state", slice(N_STATE, None)), ("action", slice(0, N_STATE))):
        for rho in (1.0, 0.5):
            NAG.reset()
            got = _heads(_run(
                PLADISAttnProcessor(pladis_scale=4.0, method="softmax", beta=2.0,
                                    qgroup=qgroup, n_state_tokens=N_STATE,
                                    nag_tau=1.0, nag_rho=rho),
                attn, h, enc))
            assert torch.equal(got[:, :, untouched], z_d[:, :, untouched]), \
                f"D[{qgroup}/rho{rho:g}]: rows outside the group are not bit-identical to dense"
            # and identical to what the SAME arm computes without NAG, which is the
            # invariant an arm's untouched rows actually have to satisfy
            _probe(False)
            off = _heads(_run(
                PLADISAttnProcessor(pladis_scale=4.0, method="softmax", beta=2.0,
                                    qgroup=qgroup, n_state_tokens=N_STATE),
                attn, h, enc))
            assert torch.equal(got[:, :, untouched], off[:, :, untouched]), \
                f"D[{qgroup}/rho{rho:g}]: NAG perturbed rows its own arm leaves dense"
            assert not torch.equal(got, z_d), f"D[{qgroup}/rho{rho:g}]: nothing changed at all"
    print("PASS gate D: qgroup rows outside the selection stay bit-exact under the "
          "cap and under rho<1")


def gate_E():
    torch.manual_seed(0)
    attn, (h, enc) = _attn(), _inputs(n_query=N_QUERY)
    ref = AttnProcessor2_0()(attn, h, encoder_hidden_states=enc)
    NAG.reset()
    SCHED.reset()
    proc = PLADISAttnProcessor(pladis_scale=4.0, method="softmax", beta=2.0,
                               qgroup="all", schedule=(0, 0, 1, 1), nag_tau=1.0)
    for step in range(4):
        SCHED.current = step
        got = _run(proc, attn, h, enc)
        if step < 2:
            assert torch.equal(got, ref), f"E: zero-weight step {step} left the fused SDPA path"
        else:
            assert not torch.equal(got, ref), f"E: weighted step {step} did not intervene"
    steps = {k[0] for k in NAG.n}
    assert steps == {2, 3}, f"E: cap recorded at steps {sorted(steps)}, expected the late pair"
    assert sum(NAG.n_clipped.values()) > 0, "E: cap never fired on the weighted steps"
    SCHED.reset()
    NAG.reset()
    print("PASS gate E: zero-weight steps stay vanilla with NAG armed; the cap acts "
          "only on the weighted ones")


def gate_F():
    NAG.reset()
    msgs = [
        _raises(lambda: validate_nag(2.0, 0.9, 1.0), what="tau < 1"),
        _raises(lambda: validate_nag(2.0, 1.5, 0.0), what="rho = 0"),
        _raises(lambda: validate_nag(2.0, 1.5, 1.5), what="rho > 1"),
        _raises(lambda: validate_nag(2.0, None, 0.5), what="rho without tau"),
        _raises(lambda: validate_nag(0.0, 1.5, 1.0), what="NAG at scale 0"),
        _raises(lambda: PLADISAttnProcessor(pladis_scale=2.0, nag_tau=0.5),
                what="processor with tau < 1"),
    ]
    assert "scale=1" in msgs[3], msgs[3]  # names the equivalent plain arm, rho*lambda
    m = _fresh_model(seed=1)
    NAG.reset()
    _raises(lambda: install_pladis(m, pladis_scale=0.0, kind="text", nag_tau=1.5),
            what="install of NAG at scale 0")
    # one arm, two cells, two different caps -> the census refuses to arm twice
    NAG.reset()
    NAG.arm(1.25, 1.0)
    _raises(lambda: NAG.arm(2.0, 1.0), what="conflicting taus in one arm")
    NAG.reset()
    print("PASS gate F: tau<1 / rho outside (0,1] / rho-without-tau / scale=0 / "
          "conflicting caps all raise")


def gate_G():
    NAG.reset()
    _raises(assert_nag_delivered, what="delivery asserted with no cap armed")
    m = _fresh_model(seed=1)
    h, enc = _inputs(n_query=N_QUERY)
    # a tau nothing can exceed: the arm would be bit-identical to its own control
    install_pladis(m, pladis_scale=2.0, kind="text", method="softmax", beta=2.0,
                   nag_tau=1e9)
    m.action_head.model(h, enc, timestep=torch.tensor([0]))
    msg = _raises(assert_nag_delivered, what="cap that never fired")
    assert "never fired" in msg, msg

    NAG.reset()
    SCHED.reset()
    m = _fresh_model(seed=1)
    install_pladis(m, pladis_scale=4.0, kind="text", method="softmax", beta=2.0,
                   nag_tau=1.0)
    m.action_head.model(h, enc, timestep=torch.tensor([0]))
    census = assert_nag_delivered()
    assert "clip rate" in census and fmt_nag(1.0, 1.0) in census, census
    NAG.reset()
    SCHED.reset()
    print(f"PASS gate G: {census}")


def gate_H():
    torch.manual_seed(0)
    attn, (h, enc) = _attn(), _inputs(n_query=N_QUERY)
    ladder = (1.0, 1.5, 2.0, 3.0)
    kw = dict(method="softmax", beta=2.0, qgroup="all")

    # one arm at lambda=2 that prices the whole ladder from its own two features
    NAG.reset()
    NAG.probe, NAG.probe_scales = True, ladder
    _run(PLADISAttnProcessor(pladis_scale=2.0, **kw), attn, h, enc)
    priced = {k[3]: (NAG.p_sum[k], NAG.p_max[k], tuple(NAG.p_exceed[k])) for k in NAG.p_n}
    assert set(priced) == set(ladder), sorted(priced)

    for lam in ladder:  # ...against arms that actually run at each rung
        NAG.reset()
        NAG.probe, NAG.probe_scales = True, ()
        _run(PLADISAttnProcessor(pladis_scale=lam, **kw), attn, h, enc)
        (key,) = list(NAG.p_n)
        assert key[3] == lam, key
        got, want = priced[lam], (NAG.p_sum[key], NAG.p_max[key], tuple(NAG.p_exceed[key]))
        assert abs(got[0] - want[0]) < 1e-3 and abs(got[1] - want[1]) < 1e-5, \
            f"H[lambda={lam:g}]: reconstructed R != measured R ({got} vs {want})"
        assert got[2] == want[2], f"H[lambda={lam:g}]: exceedance counts differ"
    NAG.reset()
    print(f"PASS gate H: one arm prices the ladder {ladder} exactly (R is affine in "
          f"lambda through Z, not through the norm)")


def gate_I():
    torch.manual_seed(0)
    attn, (h, enc) = _bare(_attn()), _inputs(n_query=N_QUERY)
    kw = dict(pladis_scale=3.0, method="softmax", beta=2.0, qgroup="all")

    # references first, with recording OFF: _dense() runs a processor too, and
    # with the census armed it would record a second cell and double n_slots
    NAG.reset()
    ref = _run(PLADISAttnProcessor(**kw), attn, h, enc)
    z_d = _dense(attn, h, enc)
    # (1) recording with the cap off changes nothing
    NAG.reset(); NAG.record_episode, NAG.probe = True, True
    SCHED.current = 2
    z_pl = _heads(_run(PLADISAttnProcessor(block_idx=4, **kw), attn, h, enc))
    assert torch.equal(_heads(ref), z_pl), "I: episode recording perturbed the output"
    ratio = (_l1(z_pl) / (_l1(z_d) + 1e-6)).reshape(-1)

    # (2) summary matches a direct computation; (3) rows sum to the summary
    summary, rows = NAG.episode_stats()
    assert summary["n_slots"] == ratio.numel() == HEADS * N_QUERY
    assert abs(summary["mean_R"] - float(ratio.mean())) < 1e-5, summary
    assert abs(summary["max_R"] - float(ratio.max())) < 1e-5, summary
    for t in R_THRESHOLDS:
        want = float((ratio > t).float().mean())
        assert abs(summary[f"frac_gt_{t:g}"] - want) < 1e-6, (t, summary[f"frac_gt_{t:g}"], want)  # float32 mean
    assert summary["clip_rate"] != summary["clip_rate"], "I: clip rate should be NaN with no cap"
    assert len(rows) == 1 and rows[0]["step"] == 2 and rows[0]["block"] == 4, rows
    assert rows[0]["n_slots"] == summary["n_slots"]
    assert abs(rows[0]["frac_gt_3"] - summary["frac_gt_3"]) < 1e-6

    # (4) clear keeps the settings, empties the ledgers
    NAG.clear_episode()
    assert NAG.record_episode and NAG.probe and not NAG.p_n and not NAG.n
    assert NAG.episode_stats() == ({}, [])

    # (5) with a cap: pre-cap R is what gets recorded, and the clip rate is reported
    NAG.reset(); NAG.record_episode = True; NAG.arm(1.5, 1.0)
    SCHED.current = 0
    _run(PLADISAttnProcessor(nag_tau=1.5, block_idx=0, **kw), attn, h, enc)
    summary, rows = NAG.episode_stats()
    assert abs(summary["mean_R"] - float(ratio.mean())) < 1e-5, "I: capped arm did not record pre-cap R"
    assert abs(summary["clip_rate"] - float((ratio > 1.5).float().mean())) < 1e-6, summary["clip_rate"]
    NAG.reset(); SCHED.reset()
    print(f"PASS gate I: per-episode R stats exact (mean {summary['mean_R']:.3f}, "
          f"clip@1.5 {summary['clip_rate']:.0%}); recording is bit-inert; clear keeps settings")


def main():
    torch.manual_seed(0)
    gate_A(); gate_B(); gate_C(); gate_D(); gate_E(); gate_F(); gate_G(); gate_H(); gate_I()
    NAG.reset()
    SCHED.reset()
    print("ALL GATES PASSED (CPU smoke; tau selection is experiments/diag_nag.py, "
          "on-checkpoint delivery is eval_arm's NAG warm-up)")


if __name__ == "__main__":
    main()
