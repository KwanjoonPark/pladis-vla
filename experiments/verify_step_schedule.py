# SPDX-License-Identifier: Apache-2.0
"""CPU smoke gates for the denoising-step schedule of pladis/attn_gr00t_n17.py
(no checkpoint needed).

``--pladis-schedule`` is a per-step MULTIPLIER on ``--pladis-scale``: the effective
strength at step i is ``lambda_i = scale * schedule[i]``, so at scale=1 the vector
is the lambda schedule itself (all [1,1,1,1] / early [1,1,0,0] / late [0,0,1,1] /
increasing [0,.5,1,1.5] / decreasing [1.5,1,.5,0]).

  A. The bucket -> step-index map is EXACT for every step of N in
     {1,2,3,4,8,16}: the head emits int(i/N * buckets) and the probe inverts it
     as round(bucket*N/buckets). A drifting map would silently re-target the
     intervention onto neighbouring steps.
  B. Schedule parsing/formatting round-trips (fractional weights included), and
     an empty schedule raises instead of becoming "no gating".
  C. Per-step strength at the processor level: a zero-weight step is bit-identical
     (torch.equal) to diffusers' AttnProcessor2_0; a weighted step equals a plain
     processor at the SAME effective lambda (w=0.5 at scale 1 == lambda 0.5, w=1.5
     == lambda 1.5); schedule=None (the default) is unchanged.
  D. The pre-hook publishes the right index on a real nn.Module DiT forward, and
     a batch carrying mixed timesteps raises rather than gating on row 0.
  E. install_pladis defenses: a length != N schedule raises, an all-zero effective
     schedule raises (vanilla wearing an intervention's name), conflicting per-cell
     schedules raise, and a schedule with no resolvable head raises.
  F. End-to-end on a mini DiT, for all five shapes of the 08-16 design and BOTH
     sparse branches the row is run on (entmax-1.5, and the 08-17 sharp-softmax
     counterpart at beta=2): the loop blends exactly at the non-zero steps with
     the right lambda, the zero steps are bit-identical to vanilla, [1,1,1,1] is
     bit-identical to the unscheduled arm at the same scale and branch, and
     assert_delivered() reports the census.
  G. assert_delivered() converts the silent failures into errors: never-fired
     probe, weighted step never reached, no blended call.

Still gated on the real checkpoint (GPU): verify_base0_parity.py for lambda=0
bit-parity, and eval_arm's own _assert_n17_step_delivery warm-up on the live
serving path.

Run: bash experiments/run.sh experiments/verify_step_schedule.py
"""

import torch
from torch import nn

from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from pladis import attn_gr00t_n17 as HOOK
from pladis.attn_gr00t_n17 import (
    SCHED,
    PLADISAttnProcessor,
    assert_delivered,
    fmt_schedule,
    install_pladis,
    parse_schedule,
)

DIM, CROSS, HEADS, HEAD_DIM = 16, 24, 2, 8
BUCKETS = 1000
# the arm shapes the 08-16 language row runs (operator's design)
SHAPES = {
    "all": (1, 1, 1, 1),
    "early": (1, 1, 0, 0),
    "late": (0, 0, 1, 1),
    "inc": (0, 0.5, 1, 1.5),
    "dec": (1.5, 1, 0.5, 0),
}


def _attn(cross: bool = True) -> Attention:
    a = Attention(
        query_dim=DIM,
        cross_attention_dim=CROSS if cross else None,
        heads=HEADS,
        dim_head=HEAD_DIM,
        bias=False,
    )
    return a.eval()


def _inputs(n_query: int = 5, n_key: int = 7):
    return torch.randn(1, n_query, DIM), torch.randn(1, n_key, CROSS)


class _MiniDiT(nn.Module):
    """Minimal stand-in for AlternateVLDiT: even blocks cross-attend, odd blocks
    self-attend, and forward takes ``timestep`` as a kwarg exactly as the action
    head passes it (gr00t_n1d7.py:400-408 in the pinned checkout)."""

    def __init__(self, n_blocks: int = 8):
        super().__init__()
        blocks = []
        for idx in range(n_blocks):
            blk = nn.Module()
            blk.attn1 = _attn(cross=(idx % 2 == 0))
            blocks.append(blk)
        self.transformer_blocks = nn.ModuleList(blocks)
        self.attend_text_every_n_blocks = 2

    def forward(self, hidden_states, encoder_hidden_states, timestep=None, **_):
        h = hidden_states
        for idx, blk in enumerate(self.transformer_blocks):
            h = blk.attn1(
                h,
                encoder_hidden_states=encoder_hidden_states if idx % 2 == 0 else None,
            )
        return h


class _MiniModel(nn.Module):
    """`model.action_head.model` shape, the layout both resolvers walk."""

    def __init__(self, n_steps: int = 4, n_blocks: int = 8):
        super().__init__()
        head = nn.Module()
        head.model = _MiniDiT(n_blocks)
        head.num_inference_timesteps = n_steps
        head.num_timestep_buckets = BUCKETS
        self.action_head = head


def _fresh_model(n_steps: int = 4, n_blocks: int = 8, seed: int = 0) -> _MiniModel:
    torch.manual_seed(seed)
    m = _MiniModel(n_steps, n_blocks).eval()
    SCHED.reset()
    return m


def _raises(fn, *, what: str) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the gate asserts the raise, any type
        return str(exc)
    raise AssertionError(f"expected a raise: {what}")


def gate_A():
    for n in (1, 2, 3, 4, 8, 16):
        for i in range(n):
            bucket = int(i / n * BUCKETS)  # gr00t_n1d7.py:384-385
            assert round(bucket * n / BUCKETS) == i, (n, i, bucket)
    assert [int(i / 4 * BUCKETS) for i in range(4)] == [0, 250, 500, 750]
    print("PASS gate A: bucket <-> step map exact for N in {1,2,3,4,8,16}")


def gate_B():
    assert parse_schedule(None) is None and parse_schedule("all") is None
    assert parse_schedule("1,1,0,0") == (1.0, 1.0, 0.0, 0.0)
    assert parse_schedule(" 0 , 0.5 , 1 , 1.5 ") == (0.0, 0.5, 1.0, 1.5)
    assert parse_schedule([1.5, 1, 0.5, 0]) == (1.5, 1.0, 0.5, 0.0)
    assert fmt_schedule(None) == "all"
    assert fmt_schedule((0.0, 0.5, 1.0, 1.5)) == "0-0.5-1-1.5"
    assert fmt_schedule((1.0, 1.0, 0.0, 0.0)) == "1-1-0-0"
    _raises(lambda: parse_schedule(","), what="empty schedule")
    print("PASS gate B: schedule parse/format round-trip (fractional weights kept)")


def gate_C():
    torch.manual_seed(0)
    attn = _attn()
    h, enc = _inputs()
    ref = AttnProcessor2_0()(attn, h, encoder_hidden_states=enc)

    SCHED.reset()
    proc = PLADISAttnProcessor(pladis_scale=1.0, qgroup="all", schedule=SHAPES["inc"])
    SCHED.current = 0  # weight 0 -> vanilla
    assert torch.equal(proc(attn, h, encoder_hidden_states=enc), ref), \
        "C: zero-weight step left the fused SDPA path"
    for step, lam in ((1, 0.5), (2, 1.0), (3, 1.5)):
        SCHED.current = step
        plain = PLADISAttnProcessor(pladis_scale=lam, qgroup="all")
        got = proc(attn, h, encoder_hidden_states=enc)
        assert torch.equal(got, plain(attn, h, encoder_hidden_states=enc)), \
            f"C: step {step} != unscheduled processor at lambda={lam}"
        assert not torch.equal(got, ref), f"C: step {step} did not blend"

    # schedule=None is the pre-schedule behavior: blends regardless of SCHED.current
    plain = PLADISAttnProcessor(pladis_scale=1.5, qgroup="all")
    SCHED.current = None
    assert not torch.equal(plain(attn, h, encoder_hidden_states=enc), ref), \
        "C: default (unscheduled) processor changed behavior"
    # armed schedule + no observed step = hard error, never a silent default
    _raises(lambda: proc(attn, h, encoder_hidden_states=enc), what="probe never fired")
    # N raised after install -> no weight for the new step
    SCHED.current = 4
    _raises(lambda: proc(attn, h, encoder_hidden_states=enc), what="step beyond schedule")
    print("PASS gate C: zero weight == vanilla (bit-exact); weighted step == plain "
          "processor at lambda = scale*w; default path unchanged")


def gate_D():
    m = _fresh_model()
    HOOK._install_step_probe(m.action_head.model, m.action_head)
    h, enc = _inputs()
    for i, bucket in enumerate((0, 250, 500, 750)):
        m.action_head.model(h, enc, timestep=torch.full((1,), bucket))
        assert SCHED.current == i, (bucket, SCHED.current)
    assert SCHED.seen == {0, 1, 2, 3}
    n_hooks = len(m.action_head.model._forward_pre_hooks)
    HOOK._install_step_probe(m.action_head.model, m.action_head)
    assert len(m.action_head.model._forward_pre_hooks) == n_hooks, "D: probe stacked"
    _raises(
        lambda: m.action_head.model(h, enc, timestep=torch.tensor([0, 250])),
        what="mixed timesteps in one batch",
    )
    print("PASS gate D: pre-hook publishes the step index; mixed-timestep batch raises")


def gate_E():
    m = _fresh_model()
    _raises(
        lambda: install_pladis(m, pladis_scale=1.0, kind="text", schedule="1,1,0"),
        what="3 weights on an N=4 head",
    )
    m = _fresh_model()
    _raises(
        lambda: install_pladis(m, pladis_scale=1.0, kind="text", schedule="0,0,0,0"),
        what="all-zero schedule",
    )
    m = _fresh_model()
    _raises(
        lambda: install_pladis(m, pladis_scale=0.0, kind="text", schedule="1,1,0,0"),
        what="scale=0 zeroing every weight",
    )
    m = _fresh_model()
    install_pladis(m, pladis_scale=1.0, kind="text", schedule="1,1,0,0")
    _raises(
        lambda: install_pladis(m, pladis_scale=1.0, kind="image", schedule="0,0,1,1"),
        what="two different schedules on one model",
    )
    bare = _MiniDiT(4).eval()
    SCHED.reset()
    _raises(
        lambda: install_pladis(bare, pladis_scale=1.0, kind="text", schedule="1,1,0,0"),
        what="schedule with no resolvable action head",
    )
    print("PASS gate E: bad length / all-zero / conflicting / unresolvable schedules raise")


def _run_loop(model, h, enc, n_steps: int = 4):
    return [
        model.action_head.model(h, enc, timestep=torch.full((1,), int(i / n_steps * BUCKETS)))
        for i in range(n_steps)
    ]


def gate_F():
    torch.manual_seed(0)
    h, enc = _inputs()

    vanilla = _fresh_model(seed=1)
    ref = _run_loop(vanilla, h, enc)

    # Both sparse branches the shape row is run on. The schedule gates lambda
    # (attn_gr00t_n17.py:231) and the branch is only reached once lambda != 0
    # (:305-331), so the two are orthogonal by construction — but the 08-17 temp
    # row burns ~19h on that composition, so it is asserted rather than argued.
    # beta=2 is the ent15-strength-matched setting the campaign uses; beta=1 would
    # make sparse == dense and turn every "did blend" assertion below into a
    # false negative, which is exactly the silent failure this gate exists for.
    for method, beta in (("ent15max", 1.0), ("softmax", 2.0)):
        # scale 1 = the shape row as written; scale 2 = the lambda base the 08-16
        # language campaign runs (allxt-*-l2), where inc/dec peak at lambda=3
        for scale in (1.0, 2.0):
            # the unscheduled arm at this scale: what "all" must reproduce bit-for-bit
            flat = _fresh_model(seed=1)
            install_pladis(flat, pladis_scale=scale, kind="text", method=method, beta=beta)
            ref_flat = _run_loop(flat, h, enc)

            for name, weights in SHAPES.items():
                tag = f"{name}@{method}b{beta:g}@lambda{scale:g}"
                m = _fresh_model(seed=1)
                installed = install_pladis(
                    m, pladis_scale=scale, kind="text", method=method, beta=beta,
                    schedule=",".join(f"{w:g}" for w in weights),
                )
                assert installed == [0, 4], installed  # text cross blocks of an 8-block DiT
                outs = _run_loop(m, h, enc)
                active = {i for i, w in enumerate(weights) if w != 0}
                for i in range(4):
                    if i in active:
                        assert not torch.equal(outs[i], ref[i]), \
                            f"F[{tag}]: step {i} did not blend"
                    else:
                        assert torch.equal(outs[i], ref[i]), \
                            f"F[{tag}]: step {i} not bit-identical to vanilla"
                if name == "all":
                    for i in range(4):
                        assert torch.equal(outs[i], ref_flat[i]), \
                            f"F[{tag}]: flat schedule != unscheduled arm at same scale"
                assert set(SCHED.n_applied) == active
                assert set(SCHED.n_skipped) == set(range(4)) - active
                assert sum(SCHED.n_applied.values()) == 2 * len(active)  # 2 text blocks
                assert all(abs(SCHED.lam[i] - scale * weights[i]) < 1e-9 for i in active)
                census = assert_delivered()
                assert f"schedule={fmt_schedule(weights)}" in census
                print(f"PASS gate F[{tag}]: {census}")


def gate_G():
    SCHED.reset()
    _raises(assert_delivered, what="assert_delivered with no schedule armed")
    m = _fresh_model()
    install_pladis(m, pladis_scale=1.0, kind="text", schedule="1,1,0,0")
    _raises(assert_delivered, what="probe never fired")
    h, enc = _inputs()
    m.action_head.model(h, enc, timestep=torch.full((1,), 500))  # only step 2 visited
    _raises(assert_delivered, what="weighted steps never reached")
    print("PASS gate G: never-fired probe and unreached weighted step both raise")


def main():
    torch.manual_seed(0)
    gate_A(); gate_B(); gate_C(); gate_D(); gate_E(); gate_F(); gate_G()
    SCHED.reset()
    print("ALL GATES PASSED (CPU smoke; on-checkpoint delivery still gated by "
          "eval_arm's warm-up and verify_base0_parity.py)")


if __name__ == "__main__":
    main()
