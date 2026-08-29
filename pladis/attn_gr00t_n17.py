# SPDX-License-Identifier: Apache-2.0
"""PLADIS test-time sparse-attention guidance for the GR00T N1.7 action-head DiT,
with query-group (state vs action) gating.

Ported from Isaac-GR00T-latest/gr00t/model/modules/pladis_attn.py so the RLinf
rollout worker needs no modified gr00t package. PLADIS (arXiv:2503.07677) blends
the dense softmax map with a sparse (entmax15/sparsemax) map:

    attn = dense + lambda * (sparse - dense)

``pladis_scale == 0`` -> delegate to the same fused ``F.scaled_dot_product_attention``
call diffusers' AttnProcessor2_0 makes, so base0 is BIT-identical to vanilla. This is
the official PLADIS semantics: lambda=0 never leaves the native SDPA path (the repo
gates the processor swap on ``do_sparse_guidance = pladis_scale > 0``,
PLADIS/pipeline/pipeline_sdxl.py:1215,1707); only the lambda>0 branch switches to the
manual torch softmax/entmax implementation (theirs: pipeline_sdxl.py:69-105).

New here vs the Isaac-GR00T original: ``qgroup`` restricts the blend to a QUERY
row group. The N1.7 DiT query sequence is ``[state (n_state_tokens); action (H)]``
(gr00t_n1d7.py builds ``sa_embs = cat(state_features, action_features)``), so
    qgroup="state"  -> blend only query row(s) 0:n_state_tokens
    qgroup="action" -> blend only query rows n_state_tokens:
    qgroup="all"    -> blend every query row (original behavior)
Key-group selection stays per-block: AlternateVLDiT's even (cross) blocks attend
to text keys when ``idx % (2*attend_text_every_n_blocks) == 0``, else image keys,
so ``kind`` picks blocks and the {state,action}x{image,text} cells compose as
(qgroup, kind) pairs.

Also new here (2026-08-16): ``schedule`` gives the blend a per-DENOISING-STEP
strength. The action head integrates the flow with N=4 Euler steps at
t in {0, .25, .5, .75} (``num_inference_timesteps``, gr00t_n1d7.py:341-421 in the
pinned checkout), and every block runs once per step, so an arm's locus has a
TIME coordinate on top of (query group x key modality). ``schedule`` is a
per-step MULTIPLIER on ``pladis_scale`` — the effective strength at step i is

    lambda_i = pladis_scale * schedule[i]

so with ``pladis_scale=1`` the vector IS the lambda schedule:
    None                -> lambda at every step (the pre-2026-08-16 behavior)
    (1, 1, 1, 1)        -> "all", numerically identical to None
    (1, 1, 0, 0)        -> "early": the two noisiest steps, vanilla afterwards
    (0, 0, 1, 1)        -> "late": the dose-matched complement (sum lambda = 2)
    (0, .5, 1, 1.5)     -> "increasing" ramp (sum lambda = 3)
    (1.5, 1, .5, 0)     -> "decreasing", its dose-matched mirror
A step whose lambda_i is 0 takes the same fused-SDPA path lambda=0 takes, so it
is bit-identical to vanilla there. The step index of the forward in flight is
published by a DiT pre-hook (:func:`_install_step_probe`) into :data:`SCHED`.

Also new here (2026-08-26): ``nag_tau``/``nag_rho`` add the two stabilization
stages of NAG (arXiv:2505.21179, Eq. 8-10) on top of the blend, in the mapping
of ``docs/nag.md`` §1 — the DENSE branch is NAG's positive baseline ``Z+``,
because our un-guided setting is lambda=0 (vanilla), exactly as NAG's is phi=0:

    Z_PL      = Z_d + lambda*(Z_s - Z_d)          # the blend, in output space
    R[i]      = ||Z_PL[i]||_1 / (||Z_d[i]||_1 + eps)     # per (head, query row)
    Z_NPL[i]  = min(R[i], tau)/R[i] * Z_PL[i]     # NORMALIZATION (magnitude cap)
    Z_final   = rho*Z_NPL + (1 - rho)*Z_d         # REFINEMENT (pull to baseline)

``nag_tau=None`` (the default) leaves the pre-NAG code path untouched, and the
NAG path with ``nag_tau=None, nag_rho=1`` is bit-identical to it (gate A of
verify_nag.py) — the cap contributes a factor of exactly 1.0 on uncapped rows.
Two consequences from docs/nag.md §2 that the interface enforces rather than
documents: refinement alone (rho<1, tau off) is the SAME arm as scale=rho*lambda
(so it raises), and lambda=0 with NAG armed is rejected (base0 must stay on the
fused-SDPA parity path, and R == 1 there makes the cap a no-op anyway).
"""

from __future__ import annotations

import math
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

try:
    from entmax import entmax15, sparsemax
except Exception as exc:  # pragma: no cover - surfaced only if entmax missing
    raise ImportError(
        "PLADIS needs the `entmax` package (pip install entmax) for the sparse branch."
    ) from exc

_VALID_QGROUPS = ("all", "state", "action")


class _StepSchedule:
    """Denoising-step census + the index of the DiT forward in flight.

    One module-level instance (:data:`SCHED`) shared by every installed processor —
    the processors are per-block, the schedule is per-inference. Same role as
    attn_pi05.CFG / attn_smolvla.CFG: it is what turns "the arm silently ran as
    something else" into a hard error (:func:`assert_delivered`).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.weights: Optional[tuple] = None  # None = one strength at every step
        self.n_steps: Optional[int] = None  # num_inference_timesteps at install
        self.current: Optional[int] = None  # step index of the forward in flight
        self.seen: set = set()  # every step index the probe has observed
        self.n_applied: dict = {}  # step -> blended attention calls
        self.n_skipped: dict = {}  # step -> calls that fell back to dense SDPA
        self.lam: dict = {}  # step -> effective lambda actually used

    def arm(self, weights: tuple, n_steps: int) -> None:
        # install_pladis_cells installs one processor set per kind, so arm() is
        # called once per cell; the same schedule twice is fine, two different
        # ones would mean half the blocks run on a schedule nobody asked for.
        if self.weights is not None and self.weights != weights:
            raise ValueError(
                f"conflicting PLADIS step schedules on one model: {fmt_schedule(self.weights)} "
                f"then {fmt_schedule(weights)} — every cell of an arm must share one schedule."
            )
        self.weights = weights
        self.n_steps = n_steps

    @property
    def active_steps(self) -> frozenset:
        """Steps whose weight is non-zero (the ones that must blend)."""
        if self.weights is None:
            return frozenset()
        return frozenset(i for i, w in enumerate(self.weights) if w != 0.0)


SCHED = _StepSchedule()


def fmt_schedule(weights: Optional[tuple]) -> str:
    """Canonical schedule string for signatures/logs: ``(0,.5,1,1.5)`` -> ``0-0.5-1-1.5``."""
    return "all" if weights is None else "-".join(f"{w:g}" for w in weights)


def parse_schedule(schedule) -> Optional[tuple]:
    """``None``/``"all"`` -> None (single strength everywhere); ``"1,1,0,0"`` -> tuple.

    Values are per-step MULTIPLIERS on ``pladis_scale``, so with scale=1 the string
    is the lambda schedule itself ("0,0.5,1,1.5"). Length is validated against the
    head's N at install time — it cannot be checked here.
    """
    if schedule is None or schedule == "" or schedule == "all":
        return None
    if isinstance(schedule, (list, tuple)):
        vals = [float(w) for w in schedule]
    else:
        vals = [float(tok) for tok in str(schedule).replace(" ", "").split(",") if tok]
    if not vals:
        raise ValueError("empty PLADIS step schedule — use 'all' for one strength everywhere.")
    return tuple(vals)


def assert_delivered() -> str:
    """Prove a step-scheduled arm blended at EXACTLY its non-zero-weight steps.

    The failure modes this converts into a hard error, each of which would burn a
    full sweep while the eplog and the .arm sidecar claim a schedule:
      * the pre-hook never fired -> no step index -> the processors would have to
        guess (they raise instead, but only if they run at all);
      * a non-zero step the loop never reaches -> the arm ran weaker than claimed;
      * no blended call at all -> the arm ran as vanilla.
    Returns the census line the caller prints.
    """
    if SCHED.weights is None:
        raise RuntimeError("assert_delivered() called on an arm with no step schedule.")
    active = SCHED.active_steps
    if not SCHED.seen:
        raise RuntimeError(
            "PLADIS step probe never fired: no DiT forward was observed, so the "
            f"schedule={fmt_schedule(SCHED.weights)} was never enforced."
        )
    missing = sorted(active - SCHED.seen)
    if missing:
        raise RuntimeError(
            f"steps {missing} carry non-zero weight but were never reached by the "
            f"denoising loop (observed {sorted(SCHED.seen)}, N={SCHED.n_steps}) — "
            f"this arm ran weaker than it claims."
        )
    if not SCHED.n_applied:
        raise RuntimeError(
            f"no blended attention call at any weighted step of "
            f"{fmt_schedule(SCHED.weights)} — the arm ran as vanilla."
        )
    stray = sorted(set(SCHED.n_applied) - active)
    if stray:
        raise RuntimeError(
            f"blend fired at zero-weight steps {stray}; schedule is "
            f"{fmt_schedule(SCHED.weights)}."
        )
    applied = {k: SCHED.n_applied[k] for k in sorted(SCHED.n_applied)}
    skipped = {k: SCHED.n_skipped[k] for k in sorted(SCHED.n_skipped)}
    lam = {k: round(SCHED.lam[k], 6) for k in sorted(SCHED.lam)}
    return (f"schedule={fmt_schedule(SCHED.weights)} of N={SCHED.n_steps}; "
            f"effective lambda/step {lam}; blended calls/step {applied}; "
            f"dense-SDPA calls/step {skipped}")


# Guards a zero-magnitude baseline row in the ratio ONLY. It is deliberately not
# added to the second denominator (the operator's deck writes min(R,tau)/(R+eps)):
# that form multiplies every UNCAPPED row by R/(R+eps) < 1, a silent shrink of the
# rows the method says pass through unchanged, which would break both the tau-off
# nesting and the untouched-row bit-parity of the qgroup split (docs/nag.md §3).
_NAG_EPS = 1e-6

# The tau grid docs/nag.md §6 pre-registers. The census counts exceedances against
# every candidate in one pass, so ONE diag run at a given lambda yields the clip
# rate of all of them — the selection rule is then read off, not re-measured.
NAG_CANDIDATE_TAUS = (1.0, 1.1, 1.25, 1.5, 2.0, 2.5)

# The DIAGNOSTIC grid (2026-08-30, professor's request): how often does the
# blend push a query row's output to 1.5x, 2x, ... 10x the dense branch's L1
# magnitude? An arm that hurts while its rows sit at R=3-10 says the harm is
# unconstrained extrapolation, not sparsity — which is the claim the cap tests.
# The census counts exceedances on the union of both grids in one pass.
R_THRESHOLDS = (1.5, 2.0, 2.5, 3.0, 5.0, 10.0)
_EXCEED_GRID = tuple(sorted(set(NAG_CANDIDATE_TAUS) | set(R_THRESHOLDS)))


class _NagCensus:
    """Per-inference census of the L1 ratio R and of the cap that acts on it.

    Same role as :data:`SCHED` for the step schedule: it is what turns "the arm
    silently ran as something else" into a hard error (:func:`assert_nag_delivered`),
    and in ``probe`` mode it is the measurement instrument of experiments/diag_nag.py.
    Keyed by (denoising step, block index, query-row group) so a reading can be
    marginalized any of the three ways docs/nag.md §6 asks for.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tau: Optional[float] = None  # None = cap off (probe may still record)
        self.rho: float = 1.0
        self.probe: bool = False  # record R even with the cap off (diag_nag.py)
        # Per-EPISODE R statistics (eval_arm --pladis-nag-probe / any tau arm):
        # record the arm's own-lambda R into the probe ledger regardless of tau, so
        # eval_arm can snapshot it after each episode next to that episode's
        # outcome. That join is what "do failing episodes carry extreme R" needs
        # and what the run-level census (diag_nag.py) cannot give.
        self.record_episode: bool = False
        # Extra lambdas to evaluate R at, from the SAME dense/sparse pair the arm
        # computed: R(l) = ||Z_d + (l/lam)*(Z_PL - Z_d)||_1 / ||Z_d||_1. That is what
        # lets one rollout price every rung of the dose ladder on ONE trajectory,
        # instead of one rollout per lambda on four different ones.
        self.probe_scales: tuple = ()
        # cap ledger, key = (step, block, qgroup) -- only the arm's own lambda
        self.n: dict = {}  # key -> query-row slots the cap saw
        self.n_clipped: dict = {}  # key -> slots the cap actually shrank
        self.r_max: dict = {}  # key -> max R
        # probe ledger, key = (step, block, qgroup, lambda) -- measurement only
        self.p_n: dict = {}
        self.p_sum: dict = {}
        self.p_max: dict = {}
        self.p_exceed: dict = {}  # key -> exceedance counts over _EXCEED_GRID
        self.p_hist: dict = {}  # key -> histogram of R

    def clear_episode(self) -> None:
        """Drop both ledgers, keep the arm settings (tau/rho/probe flags)."""
        self.n, self.n_clipped, self.r_max = {}, {}, {}
        self.p_n, self.p_sum, self.p_max, self.p_exceed, self.p_hist = {}, {}, {}, {}, {}

    # 0..8 in 0.05 steps; every candidate tau lands ON an edge. The top matters:
    # the first measurement (language/libero_10) put p90 at 2.65 and the max at 12.4
    # for lambda=2, so a 4.0 ceiling silently reported a saturated p99 as "4.00".
    # Exceedance counts are exact regardless -- only the quantiles read the histogram.
    _HIST_HI, _HIST_BINS = 8.0, 160

    def arm(self, tau: Optional[float], rho: float) -> None:
        if self.tau is not None and tau is not None and self.tau != tau:
            raise ValueError(
                f"conflicting NAG thresholds on one model: tau={self.tau} then {tau} "
                f"— every cell of an arm must share one cap."
            )
        if self.rho != 1.0 and rho != 1.0 and self.rho != rho:
            raise ValueError(
                f"conflicting NAG blend weights on one model: rho={self.rho} then {rho}."
            )
        if tau is not None:
            self.tau = float(tau)
        if rho != 1.0:
            self.rho = float(rho)

    def record_cap(self, key, ratio: torch.Tensor, clipped: torch.Tensor) -> None:
        """One (step, block, qgroup) cell of the arm's own cap activity."""
        self.n[key] = self.n.get(key, 0) + ratio.numel()
        self.n_clipped[key] = self.n_clipped.get(key, 0) + int(clipped.sum())
        self.r_max[key] = max(self.r_max.get(key, 0.0), float(ratio.max()))

    def record_probe(self, key, ratio: torch.Tensor) -> None:
        """One (step, block, qgroup, lambda) cell of the measured R distribution."""
        r = ratio.detach().reshape(-1).float()
        self.p_n[key] = self.p_n.get(key, 0) + r.numel()
        self.p_sum[key] = self.p_sum.get(key, 0.0) + float(r.sum())
        self.p_max[key] = max(self.p_max.get(key, 0.0), float(r.max()))
        ex = self.p_exceed.setdefault(key, [0] * len(_EXCEED_GRID))
        # one sync for all thresholds instead of one per threshold
        counts = (r.unsqueeze(0) > r.new_tensor(_EXCEED_GRID).unsqueeze(1)).sum(dim=1).tolist()
        for i, c in enumerate(counts):
            ex[i] += int(c)
        h = torch.histc(r.clamp(max=self._HIST_HI), bins=self._HIST_BINS,
                        min=0.0, max=self._HIST_HI).cpu()
        prev = self.p_hist.get(key)
        self.p_hist[key] = h if prev is None else prev + h

    def exceed(self, key, threshold: float) -> int:
        """Slots with R > threshold in one probe cell (threshold must be on the grid)."""
        return self.p_exceed[key][_EXCEED_GRID.index(threshold)]

    @classmethod
    def quantiles(cls, hist: torch.Tensor, qs=(0.5, 0.9, 0.99)) -> dict:
        """Quantiles read off a census histogram; values past its top clamp to it."""
        total = float(hist.sum())
        if total == 0:
            return {q: float("nan") for q in qs}
        edge, cum, out = cls._HIST_HI / len(hist), 0.0, {}
        it, want = iter(sorted(qs)), None
        want = next(it, None)
        for i, c in enumerate(hist.tolist()):
            cum += c
            while want is not None and cum / total >= want:
                out[want] = (i + 1) * edge
                want = next(it, None)
        for q in qs:
            out.setdefault(q, cls._HIST_HI)
        return out

    def episode_stats(self) -> tuple:
        """Summarize the probe ledger for ONE episode: (summary dict, per-(step, block) rows).

        The summary carries the diagnostic the professor asked for — mean R, max R,
        P(R > t) for t in R_THRESHOLDS, quantiles, and the cap's clip rate when a tau
        is armed. The rows carry the same per (denoising step, block), which is where
        the run-level census found the structure (flat in step, steep in block).
        Aggregation is over query rows AND heads; qgroup/lambda are fixed per arm.
        """
        keys = sorted(self.p_n)
        if not keys:
            return {}, []
        n = sum(self.p_n[k] for k in keys)
        hist = sum((self.p_hist[k] for k in keys[1:]), self.p_hist[keys[0]].clone())
        q = self.quantiles(hist)
        summary = {
            "n_slots": n,
            "mean_R": sum(self.p_sum[k] for k in keys) / n,
            "max_R": max(self.p_max[k] for k in keys),
            "p50_R": q[0.5], "p90_R": q[0.9], "p99_R": q[0.99],
        }
        for t in R_THRESHOLDS:
            summary[f"frac_gt_{t:g}"] = sum(self.exceed(k, t) for k in keys) / n
        n_cap = sum(self.n.values())
        summary["clip_rate"] = (sum(self.n_clipped.values()) / n_cap) if n_cap else float("nan")
        rows = []
        for k in keys:
            step, block = k[0], k[1]
            nk = self.p_n[k]
            row = {"step": step, "block": block, "n_slots": nk,
                   "mean_R": self.p_sum[k] / nk, "max_R": self.p_max[k]}
            for t in (2.0, 3.0, 5.0):
                row[f"frac_gt_{t:g}"] = self.exceed(k, t) / nk
            ck = k[:3]
            row["clip_rate"] = (self.n_clipped.get(ck, 0) / self.n[ck]) if self.n.get(ck) else float("nan")
            rows.append(row)
        return summary, rows

    @property
    def clip_rate(self) -> float:
        """Fraction of (head, query row) slots the cap shrank, over the whole run."""
        total = sum(self.n.values())
        return 0.0 if not total else sum(self.n_clipped.values()) / total


NAG = _NagCensus()


def fmt_nag(tau: Optional[float], rho: float) -> str:
    """Canonical NAG string for signatures/logs: ``off`` or ``tau=1.25,rho=1``."""
    return "off" if tau is None else f"tau={tau:g},rho={rho:g}"


def validate_nag(pladis_scale: float, tau: Optional[float], rho: float) -> None:
    """Reject the three NAG settings that would silently be a different arm.

    Called at install AND from eval_arm's argument layer, so a bad combination
    dies before a checkpoint is loaded rather than after 1,537 episodes.
    """
    if tau is not None and tau < 1.0:
        # tau < 1 caps the guided output BELOW the dense branch's own magnitude:
        # no longer a guardrail on the extrapolation but an attenuation of vanilla.
        raise ValueError(f"NAG tau must be >= 1, got {tau}.")
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"NAG rho must lie in (0, 1], got {rho}.")
    if rho != 1.0 and tau is None:
        # docs/nag.md §2(b): with no cap, rho only rescales the dose --
        # Z = Z_d + (rho*lambda)*(Z_s - Z_d) -- so this is the plain arm at
        # scale=rho*lambda, and the dose ladder has already run those rungs.
        raise ValueError(
            f"NAG rho={rho:g} without a tau is the plain arm at "
            f"scale={rho * pladis_scale:g} under another name (docs/nag.md §2b); "
            f"pass --pladis-nag-tau or use --pladis-scale directly."
        )
    if tau is not None and pladis_scale == 0.0:
        raise ValueError(
            "NAG with pladis_scale=0 is rejected: base0 must stay on the fused-SDPA "
            "bit-parity path, and R == 1 makes the cap a no-op there anyway."
        )


def assert_nag_delivered() -> str:
    """Prove the cap actually fired; return the census line the caller prints.

    The failure this converts into a hard error: a tau nothing exceeds makes the
    arm bit-identical to its own control (docs/nag.md §2a), so it would burn a
    full sweep to re-measure an arm we already have.
    """
    if NAG.tau is None:
        raise RuntimeError("assert_nag_delivered() called on an arm with no NAG cap.")
    if not NAG.n:
        raise RuntimeError(
            f"NAG is armed ({fmt_nag(NAG.tau, NAG.rho)}) but no attention call "
            f"recorded a ratio — the processors never ran, so the cap was never applied."
        )
    total, clipped = sum(NAG.n.values()), sum(NAG.n_clipped.values())
    if clipped == 0:
        r_max = max(NAG.r_max.values())
        raise RuntimeError(
            f"NAG cap never fired: max R observed {r_max:.4f} <= tau={NAG.tau:g} over "
            f"{total} query-row slots. This arm is bit-identical to its uncapped "
            f"control (docs/nag.md §2a) — pick tau from the diag, do not run it."
        )
    per_step: dict = {}
    for key, n in NAG.n.items():
        acc = per_step.setdefault(key[0], [0, 0])
        acc[0] += n
        acc[1] += NAG.n_clipped.get(key, 0)
    rates = {k: round(v[1] / v[0], 4) for k, v in sorted(per_step.items())}
    return (f"nag={fmt_nag(NAG.tau, NAG.rho)}; clip rate {clipped}/{total} = "
            f"{clipped / total:.2%}; per step {rates}; "
            f"max R {max(NAG.r_max.values()):.3f}")


class PLADISAttnProcessor:
    """Dense/sparse-extrapolation attention processor (single forward pass)."""

    def __init__(
        self,
        pladis_scale: float = 1.5,
        method: str = "ent15max",
        beta: float = 1.0,
        qgroup: str = "all",
        n_state_tokens: int = 1,
        schedule: Optional[tuple] = None,
        nag_tau: Optional[float] = None,
        nag_rho: float = 1.0,
        block_idx: int = -1,
    ) -> None:
        self.pladis_scale = float(pladis_scale)
        self.method = method
        # beta scales the logits of the sparse branch ONLY (a temperature on the
        # entmax/sparsemax reference); the dense softmax branch is left untouched.
        self.beta = float(beta)
        if qgroup not in _VALID_QGROUPS:
            raise ValueError(f"qgroup must be one of {_VALID_QGROUPS}, got {qgroup!r}")
        self.qgroup = qgroup
        self.n_state_tokens = int(n_state_tokens)
        # None = one strength at every denoising step (untouched code path); a tuple
        # scales it per step, keyed on SCHED.current (published per DiT forward by
        # _install_step_probe).
        self.schedule = schedule
        # NAG (docs/nag.md): None = cap off, which keeps the pre-NAG code path.
        validate_nag(self.pladis_scale, nag_tau, nag_rho)
        self.nag_tau = None if nag_tau is None else float(nag_tau)
        self.nag_rho = float(nag_rho)
        # census key component: WHICH block a ratio came from. Only the diag reads
        # it, but it costs nothing to carry and cannot be recovered afterwards.
        self.block_idx = int(block_idx)

    def _nag(self, z_blend: torch.Tensor, z_dense: torch.Tensor, lam: float, step) -> torch.Tensor:
        """NAG normalization + refinement on the attention OUTPUT (docs/nag.md §1).

        ``z_blend`` is Z_PL (the blend already applied), ``z_dense`` is Z_d, NAG's
        positive baseline. Both are (B, H, Lq, D_head) — the same shape the paper's
        Algorithm 1 reduces over with ``p=1, dim=-1``, so the norm is per
        (batch, head, query row) over the head channels, not over the merged heads.
        """
        # float32 for the ratio and the cap: an L1 norm over a head's bf16 channels
        # is not worth the precision, and f32->bf16 round-trips exactly, so the
        # uncapped rows stay bit-identical to the pre-NAG path.
        num = z_blend.float().abs().sum(dim=-1, keepdim=True)
        den = z_dense.float().abs().sum(dim=-1, keepdim=True)
        ratio = num / (den + _NAG_EPS)
        key = (step, self.block_idx, self.qgroup)
        if NAG.probe or NAG.record_episode:
            NAG.record_probe(key + (lam,), ratio)
            # Every other rung of the ladder, from the same two features: the blend
            # is affine in lambda, so Z(l) = Z_d + (l/lam)*(Z_PL - Z_d) exactly.
            if NAG.probe and NAG.probe_scales:
                delta = z_blend.float() - z_dense.float()
                for other in NAG.probe_scales:
                    if other == lam:
                        continue
                    r_other = (z_dense.float() + (other / lam) * delta).abs().sum(
                        dim=-1, keepdim=True) / (den + _NAG_EPS)
                    NAG.record_probe(key + (other,), r_other)
        if self.nag_tau is None:
            return z_blend  # probe mode: measure, change nothing
        # The published form (Algorithm 1): where(R > tau, tau/R, 1). Uncapped rows
        # get a factor of EXACTLY 1.0, which is what makes tau-off nesting and the
        # qgroup bit-parity hold; min(R,tau)/(R+eps) would shrink them all slightly.
        clipped = ratio > self.nag_tau
        scale = torch.where(clipped, self.nag_tau / ratio, torch.ones_like(ratio))
        NAG.record_cap(key, ratio, clipped)
        z = z_blend.float() * scale
        if self.nag_rho != 1.0:
            # Refinement pulls EVERY row toward the baseline, not only capped ones
            # (Eq. 10) — which is exactly why it is a dose rescale wherever the cap
            # is inactive (docs/nag.md §2b).
            z = self.nag_rho * z + (1.0 - self.nag_rho) * z_dense.float()
        return z.to(z_blend.dtype)

    def _lambda_now(self) -> float:
        """Effective blend strength for the denoising step in flight.

        A missing step index means the pre-hook did not fire. Defaulting either way
        would be silent: the full strength runs a scheduled arm at every step,
        zero runs vanilla under an intervention's name. Raise instead.
        """
        if self.schedule is None:
            return self.pladis_scale
        step = SCHED.current
        if step is None:
            raise RuntimeError(
                "PLADIS step schedule is armed but no denoising-step index was "
                "observed — the DiT forward pre-hook did not fire, so "
                f"schedule={fmt_schedule(self.schedule)} cannot be enforced."
            )
        if step >= len(self.schedule):
            # N was raised after install (open_loop_eval.py:310 does exactly this);
            # silently reusing the last weight would run an unrequested schedule.
            raise RuntimeError(
                f"denoising step {step} has no weight in schedule "
                f"{fmt_schedule(self.schedule)} (len={len(self.schedule)}) — "
                f"num_inference_timesteps changed after install."
            )
        lam = self.pladis_scale * self.schedule[step]
        if lam == 0.0:
            SCHED.n_skipped[step] = SCHED.n_skipped.get(step, 0) + 1
        else:
            SCHED.n_applied[step] = SCHED.n_applied.get(step, 0) + 1
            SCHED.lam[step] = lam
        return lam

    def _split_point(self, n_query: int) -> Optional[int]:
        """``n_state_tokens`` if this processor gates a query group, else None.

        A wrong n_state_tokens mis-slices the two groups SILENTLY (no shape error —
        cat() reassembles any split), so the whole state/action contrast would be
        meaningless. Check the split is non-degenerate against the live query length.
        """
        if self.qgroup == "all":
            return None
        ns = self.n_state_tokens
        if not 0 < ns < n_query:
            raise ValueError(
                f"n_state_tokens={ns} does not split a {n_query}-row query "
                f"sequence into non-empty [state; action] groups — the "
                f"qgroup={self.qgroup!r} arm would be degenerate."
            )
        return ns

    def _sparse(self, logits: torch.Tensor) -> torch.Tensor:
        z = self.beta * logits
        if self.method == "sparsemax":
            return sparsemax(z, dim=-1)
        elif self.method == "ent15max":
            return entmax15(z, dim=-1)
        elif self.method == "softmax":
            # alpha=1 entmax == softmax; with beta=1 the sparse branch equals the dense
            # branch so PLADIS collapses to vanilla for ANY scale -> integration sanity check.
            return torch.softmax(z, dim=-1)
        else:
            raise ValueError(f"Unknown PLADIS method: {self.method}")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # --- PLADIS: dense/sparse extrapolation in place of scaled_dot_product_attention ---
        lam = self._lambda_now()
        if lam == 0.0:
            # Official lambda=0 semantics: stay on the fused SDPA path, byte-for-byte
            # the call AttnProcessor2_0 makes (bool mask passed through untouched).
            # base0 == vanilla bit-exact; the hook only exercises install plumbing.
            # An unscheduled denoising step takes this same path, which is what makes
            # "intervene on steps {0,1}" mean "vanilla on {2,3}" exactly.
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            scale_factor = 1.0 / math.sqrt(query.size(-1))
            logits = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (B, H, Lq, Lk)
            # softmax/entmax in float32 for numerical stability under bf16 autocast.
            logits = logits.float()
            if attention_mask is not None:
                # SDPA treats a BOOL mask as True=attend / False=-inf. Adding a bool
                # tensor would add 0/1 instead and silently disable masking (this is
                # what the AlternateVLDiT text/image key masks are), so convert to an
                # additive float mask. Large-finite instead of -inf keeps entmax on
                # the sparse branch NaN-free (matches the pi05 hook's clamp).
                neg = torch.finfo(torch.float32).min / 4
                if attention_mask.dtype == torch.bool:
                    attention_mask = (~attention_mask).to(torch.float32) * neg
                logits = (logits + attention_mask.float()).clamp_min(neg)
            dense = torch.softmax(logits, dim=-1)
            sparse = self._sparse(logits)
            attn_weight = dense + lam * (sparse - dense)
            # Query rows are [state(0:n_state_tokens); action(n_state_tokens:)].
            # Keep the intervention only on the selected group; the rest stays dense.
            ns = self._split_point(attn_weight.shape[-2])
            if self.nag_tau is None and not NAG.probe:
                # Pre-NAG path, kept verbatim: lambda>0 arms carry no bit-parity gate,
                # so re-associating these ops would make 44k+ collected episodes
                # non-reproducible. NAG never touches it.
                if ns is not None:
                    if self.qgroup == "state":
                        attn_weight = torch.cat(
                            [attn_weight[..., :ns, :], dense[..., ns:, :]], dim=-2
                        )
                    else:  # action
                        attn_weight = torch.cat(
                            [dense[..., :ns, :], attn_weight[..., ns:, :]], dim=-2
                        )
                attn_weight = attn_weight.to(value.dtype)
                hidden_states = torch.matmul(attn_weight, value)
            else:
                # NAG needs the dense branch as a FEATURE, not just as a map, so the
                # blend is taken into output space first (docs/nag.md §1). Row i of a
                # matmul depends only on row i of the left operand, so the selected
                # rows here are bit-identical to the slice-then-matmul above, and the
                # qgroup select moves AFTER the cap: blending rho on the untouched
                # rows would return rho*x + (1-rho)*x, which is not bit-exactly x.
                z_blend = torch.matmul(attn_weight.to(value.dtype), value)
                z_dense = torch.matmul(dense.to(value.dtype), value)
                hidden_states = self._nag(z_blend, z_dense, lam, SCHED.current)
                if ns is not None:
                    if self.qgroup == "state":
                        hidden_states = torch.cat(
                            [hidden_states[..., :ns, :], z_dense[..., ns:, :]], dim=-2
                        )
                    else:  # action
                        hidden_states = torch.cat(
                            [z_dense[..., :ns, :], hidden_states[..., ns:, :]], dim=-2
                        )
        # -------------------------------------------------------------------------------

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def _find_alternate_dit(model):
    """Return the AlternateVLDiT/DiT module holding ``transformer_blocks``.

    Accepts either the top GR00T model, the action head, or the DiT itself.
    """
    # top model -> action_head -> model (the DiT)
    for path in ("action_head.model", "model", ""):
        obj = model
        ok = True
        for attr in filter(None, path.split(".")):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and hasattr(obj, "transformer_blocks"):
            return obj
    raise AttributeError("Could not locate a DiT with `transformer_blocks` on the given model.")


def _find_action_head(model):
    """Return the action head (the module owning the flow-matching loop).

    Accepts the harness adapter (``OfficialGr00tPolicy.action_head``), the top GR00T
    model, or the head itself. It is the only place that knows N and the bucket
    count, which is what makes a step index recoverable.
    """
    for path in ("action_head", "model.action_head", ""):
        obj = model
        ok = True
        for attr in filter(None, path.split(".")):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and hasattr(obj, "num_inference_timesteps") and hasattr(
            obj, "num_timestep_buckets"
        ):
            return obj
    raise AttributeError(
        "Could not locate an action head with `num_inference_timesteps` — a step "
        "schedule cannot be mapped onto denoising steps without it."
    )


def _install_step_probe(dit, head) -> None:
    """Publish the denoising-step index of each DiT forward into ``SCHED.current``.

    The head calls the DiT once per Euler step with ``timestep=`` holding the
    discretized bucket ``int(i/N * num_timestep_buckets)`` (gr00t_n1d7.py:383-408 in
    the pinned checkout), so the index inverts exactly as ``round(bucket*N/buckets)``.
    That map is STATELESS on purpose: a call counter would drift the moment an
    inference raises, a warm-up chunk runs, or N changes at runtime, and the drift
    would silently re-target the intervention onto the wrong steps.

    N and the bucket count are read live from the head (not frozen at install) so a
    runtime `num_inference_timesteps` override stays consistent with the gate.
    """
    if getattr(dit, "_pladis_step_probe", None) is not None:
        return  # install_pladis_cells installs per kind; one probe per DiT is enough

    def _probe(module, args, kwargs):
        ts = kwargs.get("timestep")
        if ts is None and len(args) >= 3:
            ts = args[2]  # DiT.forward(hidden_states, encoder_hidden_states, timestep)
        if ts is None:
            raise RuntimeError(
                "DiT forward carried no `timestep`; the PLADIS step schedule has "
                "nothing to key on."
            )
        flat = ts.reshape(-1)
        if flat.numel() > 1 and not bool((flat == flat[0]).all()):
            raise RuntimeError(
                "batched inference with mixed timesteps — one schedule cannot gate "
                "rows that are at different denoising steps."
            )
        n_steps = int(head.num_inference_timesteps)
        buckets = int(head.num_timestep_buckets)
        step = int(round(int(flat[0].item()) * n_steps / buckets))
        SCHED.current = step
        SCHED.seen.add(step)
        return None

    dit._pladis_step_probe = dit.register_forward_pre_hook(_probe, with_kwargs=True)


def cross_block_indices(dit, kind: str = "text") -> List[int]:
    """Even (cross-attention) block indices of the DiT, optionally split by target.

    kind: "all" (every even/cross block), "text" (even blocks that cross-attend to
    language tokens), or "image" (even blocks that cross-attend to image tokens).
    Text/image split follows AlternateVLDiT.forward: a cross block attends to text
    when ``idx % (2 * attend_text_every_n_blocks) == 0``, else to image.
    """
    n = len(dit.transformer_blocks)
    even = [i for i in range(n) if i % 2 == 0]
    if kind == "all":
        return even
    if kind not in ("text", "image"):
        raise ValueError(f"kind must be all|text|image, got {kind}")
    # NOT a soft default: with every==1 the rule collapses (every even block is
    # a text block) and kind="image" would silently select ZERO blocks — the
    # arm would then run as plain vanilla while being logged as an intervention.
    every = getattr(dit, "attend_text_every_n_blocks", None)
    if not every or every < 2:
        raise ValueError(
            f"attend_text_every_n_blocks={every!r} gives no text/image alternation "
            f"on this DiT, so kind={kind!r} is not a well-defined key group. "
            f"Use kind='all' or pass explicit `blocks=`."
        )
    text = [i for i in even if i % (2 * every) == 0]
    if kind == "text":
        return text
    return [i for i in even if i not in set(text)]


def install_pladis(
    model,
    pladis_scale: float = 1.5,
    method: str = "ent15max",
    beta: float = 1.0,
    blocks: Optional[List[int]] = None,
    kind: str = "text",
    qgroup: str = "all",
    n_state_tokens: int = 1,
    schedule=None,
    nag_tau: Optional[float] = None,
    nag_rho: float = 1.0,
) -> List[int]:
    """Install PLADISAttnProcessor on selected cross blocks; returns the block idxs used.

    If ``blocks`` is given it is used verbatim (must be even/cross indices). Otherwise
    all cross blocks of ``kind`` (text|image|all) are targeted. ``qgroup`` restricts
    the blend to state/action query rows, ``schedule`` scales it per denoising step
    (see module docstring); both default to "everything", leaving the pre-existing
    code path. ``nag_tau``/``nag_rho`` add the NAG cap and refinement on top
    (docs/nag.md); ``nag_tau=None`` (the default) leaves that path untouched too.
    """
    dit = _find_alternate_dit(model)
    validate_nag(pladis_scale, nag_tau, nag_rho)
    sched = parse_schedule(schedule)
    if sched is not None:
        head = _find_action_head(model)
        n_steps = int(head.num_inference_timesteps)
        # The length must MATCH N, not be truncated or padded: a 4-weight schedule on
        # a model reconfigured to N=2 would drop the tail (a different arm than the
        # one requested), and a short one would leave later steps undefined.
        if len(sched) != n_steps:
            raise ValueError(
                f"step schedule {fmt_schedule(sched)} has {len(sched)} weights but the "
                f"head runs N={n_steps} denoising steps — one weight per step is "
                f"required (use 'all' for a flat schedule)."
            )
        # An all-zero effective schedule is vanilla wearing an intervention's name;
        # same policy as the empty-install guard below.
        if all(pladis_scale * w == 0.0 for w in sched):
            raise ValueError(
                f"schedule {fmt_schedule(sched)} at scale={pladis_scale} gives lambda=0 "
                f"at every step — this arm would be bit-identical to vanilla."
            )
        _install_step_probe(dit, head)
        SCHED.arm(sched, n_steps)
    if nag_tau is not None or NAG.probe:
        # The census is keyed by denoising step, so a NAG arm needs the step index
        # even without a schedule. The probe is numerically inert (it only publishes
        # SCHED.current), which is why arming it here cannot change what an arm
        # computes — it is simply not installed when NAG is off.
        _install_step_probe(dit, _find_action_head(model))
        NAG.arm(nag_tau, nag_rho)
    if blocks is None:
        blocks = cross_block_indices(dit, kind=kind)
    targets = set(blocks)
    installed = []
    for idx, block in enumerate(dit.transformer_blocks):
        if idx in targets:
            block.attn1.set_processor(
                PLADISAttnProcessor(
                    pladis_scale=pladis_scale,
                    method=method,
                    beta=beta,
                    qgroup=qgroup,
                    n_state_tokens=n_state_tokens,
                    schedule=sched,
                    nag_tau=nag_tau,
                    nag_rho=nag_rho,
                    block_idx=idx,
                )
            )
            installed.append(idx)

    # An empty install is indistinguishable from vanilla at rollout time: the
    # arm would consume a full sweep and be reported as an intervention while
    # having changed nothing. Never let it start.
    if not installed:
        raise RuntimeError(
            f"PLADIS install selected no blocks (kind={kind!r}, blocks={blocks!r}, "
            f"n_layers={len(dit.transformer_blocks)}) — this arm would be "
            f"bit-identical to vanilla."
        )

    msg = (
        f"[PLADIS] installed on blocks {installed} "
        f"(scale={pladis_scale}, method={method}, beta={beta}, kind={kind}, "
        f"qgroup={qgroup}, n_state_tokens={n_state_tokens}, "
        f"schedule={fmt_schedule(sched)}, nag={fmt_nag(nag_tau, nag_rho)}, "
        f"n_layers={len(dit.transformer_blocks)})"
    )
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)  # survives SIGTERM before stdout buffer flush
    return installed


def install_pladis_cells(
    model,
    cells: str,
    pladis_scale: float = 1.5,
    method: str = "ent15max",
    beta: float = 1.0,
    n_state_tokens: int = 1,
    schedule=None,
    nag_tau: Optional[float] = None,
    nag_rho: float = 1.0,
) -> List[int]:
    """Install a (possibly different) qgroup per key-kind block set.

    ``cells`` is a comma-separated list of ``{qgroup}x{kind}`` cells, e.g.
    ``"actionxtext,stateximage"`` = blend action rows on text blocks AND state
    rows on image blocks in the same pass. Kinds must be disjoint (one
    processor per block): express {action,state} on one kind as qgroup=all.
    """
    parsed = []
    for cell in cells.split(","):
        qgroup, sep, kind = cell.strip().partition("x")
        if not sep or qgroup not in _VALID_QGROUPS or kind not in ("all", "text", "image"):
            raise ValueError(f"bad cell {cell!r}: expected {{qgroup}}x{{kind}}")
        parsed.append((qgroup, kind))
    kinds = [k for _, k in parsed]
    if len(set(kinds)) != len(kinds) or ("all" in kinds and len(kinds) > 1):
        raise ValueError(f"cells must target disjoint kinds, got {kinds} "
                         "(use qgroup=all for both row groups on one kind)")
    installed: List[int] = []
    for qgroup, kind in parsed:
        installed += install_pladis(
            model,
            pladis_scale=pladis_scale,
            method=method,
            beta=beta,
            kind=kind,
            qgroup=qgroup,
            n_state_tokens=n_state_tokens,
            schedule=schedule,
            nag_tau=nag_tau,
            nag_rho=nag_rho,
        )
    return sorted(installed)
