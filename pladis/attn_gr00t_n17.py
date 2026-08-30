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

Also new here (2026-08-31): a SECOND processor, :class:`HopfieldAttnProcessor`,
for the ODD blocks — the self-attention blocks PLADIS never touches. It is the
symmetric/skew circulation control of Cho, Han & Jin (ICML 2026) in the mapping
of docs/hopfield.md §1 / docs/loci.md §1.1. Per odd block and head, with the
square logits ``L = QK^T/sqrt(d)`` over the ``[state; action]`` tokens:

    L_skew = (L - L^T)/2                          # circulation; xi^T L_skew xi == 0
    L_a    = L + (alpha - 1) * L_skew             # == L_sym + alpha*L_skew; alpha=1 -> L bitwise
    Z      = softmax(L) V,   Z_a = softmax(L_a) V
    Z_b    = Z + beta * (Z_a - Z)
    Z_out  = Z_b * clamp(||Z_a||_2 / ||Z_b||_2, 0.25, 4)   # per (b, head, row); ref code Alg. 2

``beta == 0`` (and every zero-weight schedule step) takes the fused-SDPA path,
bit-identical to vanilla; ``alpha == 1`` is bit-identical to the manual dense
path (the odd-block eager-dense control). ``--hop-probe`` returns the fused
output and records the paper's stability diagnostics (eta, E, r, Align on
``Xi = P X``) plus a price list of the (alpha, temperature) grid computed from
the same logits, so phase 0 (docs/hopfield.md §6) costs a vanilla rollout. The
two processors live on disjoint block sets and share only the step probe.
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
        # relative L2 displacement of the blend, ||Z_PL - Z_d||_2 / ||Z_d||_2 per
        # (head, query row), same key as the probe ledger. Recorded only; the
        # NAG report never reads it. It is the even-block STRENGTH reference the
        # Hopfield phase 0 matches its alpha grid against (docs/hopfield.md §6):
        # R is a magnitude ratio, and a magnitude can stay put while the feature
        # moves, so the match needs a displacement measured the same way on both.
        self.p_disp_sum: dict = {}
        self.p_disp_n: dict = {}

    def clear_episode(self) -> None:
        """Drop both ledgers, keep the arm settings (tau/rho/probe flags)."""
        self.n, self.n_clipped, self.r_max = {}, {}, {}
        self.p_n, self.p_sum, self.p_max, self.p_exceed, self.p_hist = {}, {}, {}, {}, {}
        self.p_disp_sum, self.p_disp_n = {}, {}

    def record_disp(self, key, z_blend: torch.Tensor, z_dense: torch.Tensor) -> None:
        """Relative L2 displacement of the blended output from the dense one."""
        zd = z_dense.detach().float()
        d = torch.linalg.vector_norm(z_blend.detach().float() - zd, dim=-1)
        d = d / torch.linalg.vector_norm(zd, dim=-1).clamp_min(_NAG_EPS)
        self.p_disp_sum[key] = self.p_disp_sum.get(key, 0.0) + float(d.sum())
        self.p_disp_n[key] = self.p_disp_n.get(key, 0) + d.numel()

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


def _split_point(qgroup: str, n_state_tokens: int, n_query: int) -> Optional[int]:
    """``n_state_tokens`` if ``qgroup`` gates a query-row group, else None.

    A wrong n_state_tokens mis-slices the two groups SILENTLY (no shape error —
    cat() reassembles any split), so the whole state/action contrast would be
    meaningless. Check the split is non-degenerate against the live query length.
    """
    if qgroup == "all":
        return None
    ns = int(n_state_tokens)
    if not 0 < ns < n_query:
        raise ValueError(
            f"n_state_tokens={ns} does not split a {n_query}-row query "
            f"sequence into non-empty [state; action] groups — the "
            f"qgroup={qgroup!r} arm would be degenerate."
        )
    return ns


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
            NAG.record_disp(key + (lam,), z_blend, z_dense)
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
        """``n_state_tokens`` if this processor gates a query group, else None
        (module-level :func:`_split_point`, shared with the Hopfield processor)."""
        return _split_point(self.qgroup, self.n_state_tokens, n_query)

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


# =============================================================================
# Hopfield circulation control on the ODD (self-attention) blocks
# (docs/hopfield.md; docs/loci.md §1.1) — 2026-08-31
# =============================================================================

# The grids phase 0 prices from ONE vanilla rollout (docs/hopfield.md §6): for
# every alpha and every temperature, the relative displacement of the retrieved
# feature and the norm-match clamp rate at every beta, from the same logits and
# values the rollout computed. The arm values are read off this list, not
# transferred from the paper (whose operating points are SDXL/SD3 numbers).
HOP_ALPHA_GRID = (0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0)
HOP_TEMP_GRID = (1.25, 1.5, 2.0, 3.0)
HOP_BETA_GRID = (0.5, 1.0, 2.0, 5.0)
# Reference code (paper Appendix A.1, Algorithm 2): eps, r_min, r_max of the
# norm-match. eps guards a zero row in clamp_min ONLY (the docs/nag.md §3 lesson).
_HOP_EPS = 1e-6
_HOP_R_MIN, _HOP_R_MAX = 0.25, 4.0
_VALID_HOP_NORMS = ("l2", "off")

# Sidecar columns (eval_arm's _HopStatsWriter reads these, so the writer and the
# census cannot drift apart).
_HOP_DISP_COLS = ([f"disp_a{a:g}" for a in HOP_ALPHA_GRID]
                  + [f"disp_t{t:g}" for t in HOP_TEMP_GRID])
HOP_SUMMARY_COLS = (["n_calls", "eta_mean", "eta_p10", "eta_p50", "eta_p90", "eta_min",
                     "E_mean", "r_mean", "align_mean", "align_p10",
                     "clamp_lo_rate", "clamp_hi_rate", "beta_eff_mean", "floor_mean"]
                    + _HOP_DISP_COLS)
HOP_ROW_COLS = (["step", "block", "n_calls", "eta_mean", "E_mean", "r_mean", "align_mean",
                 "clamp_lo_rate", "clamp_hi_rate", "beta_eff_mean", "floor_mean"]
                + _HOP_DISP_COLS)


def _hist(x: torch.Tensor, lo: float, hi: float, bins: int) -> torch.Tensor:
    return torch.histc(x.detach().float().reshape(-1).clamp(lo, hi),
                       bins=bins, min=lo, max=hi).cpu()


def _hist_quantiles(hist: torch.Tensor, lo: float, hi: float, qs) -> dict:
    """Quantiles read off a histogram over [lo, hi]; values past the top clamp to it."""
    total = float(hist.sum())
    if total == 0:
        return {q: float("nan") for q in qs}
    edge, cum, out = (hi - lo) / len(hist), 0.0, {}
    it = iter(sorted(qs))
    want = next(it, None)
    for i, c in enumerate(hist.tolist()):
        cum += c
        while want is not None and cum / total >= want:
            out[want] = lo + (i + 1) * edge
            want = next(it, None)
    for q in qs:
        out.setdefault(q, hi)
    return out


def _eta(logits: torch.Tensor, skew: torch.Tensor) -> torch.Tensor:
    """Realized symmetry index per (batch, head), paper Eq. 36: in [-1, 1]."""
    sym = 0.5 * (logits + logits.transpose(-2, -1))
    s2 = sym.pow(2).sum(dim=(-2, -1))
    n2 = skew.pow(2).sum(dim=(-2, -1))
    return (s2 - n2) / (s2 + n2).clamp_min(_HOP_EPS)


def _row_norm(z: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(z, dim=-1, keepdim=True)


class _HopCensus:
    """Per-inference census of the Hopfield processors (delivery + diagnostics).

    Same role as :data:`SCHED`/:data:`NAG`: one module-level instance, keyed by
    (denoising step, block), that turns "the arm silently ran as something else"
    into a hard error (:func:`assert_hopfield_delivered`) and, in probe mode, is
    the instrument of experiments/diag_hopfield.py. It keeps its OWN schedule
    weights and reads only ``SCHED.current`` — ``SCHED.arm()`` refuses two
    different weight vectors on one model, and a combined arm legitimately runs
    the cross blocks on ``0,0,1,1`` and the self blocks flat.
    """

    _BINS = 200

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.alpha: float = 1.0
        self.beta: float = 0.0
        self.temp: float = 1.0
        self.adaptive: bool = False
        self.norm: str = "l2"
        self.qgroup: str = "all"
        self.weights: Optional[tuple] = None  # own per-step multiplier on beta
        self.n_steps: Optional[int] = None
        self.blocks: tuple = ()
        self.probe: bool = False  # bit-identical rollout, diagnostics recorded
        self.record_episode: bool = False  # per-episode stats (hopstats sidecars)
        self.clear_episode()

    def clear_episode(self) -> None:
        """Drop every ledger, keep the arm settings."""
        # delivery, key = (step, block)
        self.n_applied: dict = {}
        self.n_skipped: dict = {}
        # arm ledger, key = (step, block): norm-match clamp events, realized beta
        self.a_n: dict = {}
        self.a_lo: dict = {}
        self.a_hi: dict = {}
        self.beff_sum: dict = {}
        self.beff_n: dict = {}
        # diagnostics, key = (step, block)
        self.d_calls: dict = {}
        self.eta_n: dict = {}
        self.eta_sum: dict = {}
        self.eta_min: dict = {}
        self.eta_hist: dict = {}
        self.E_sum: dict = {}
        self.E_n: dict = {}
        self.r_sum: dict = {}
        self.al_sum: dict = {}
        self.al_hist: dict = {}
        # price list, key = (step, block, kind, value), kind in {"alpha", "temp"}
        self.p_n: dict = {}
        self.p_sum: dict = {}
        self.p_hist: dict = {}
        self.p_lo: dict = {}  # key -> {beta: count of ratio < r_min}
        self.p_hi: dict = {}  # key -> {beta: count of ratio > r_max}
        # fused-vs-eager floor, key = (step, block)
        self.f_n: dict = {}
        self.f_sum: dict = {}

    def arm(self, alpha, beta, temp, adaptive, norm, qgroup, weights, n_steps, blocks) -> None:
        if self.blocks:
            raise ValueError(
                f"Hopfield processors are already installed on blocks {list(self.blocks)} "
                f"— one install per model (reset the census first)."
            )
        self.alpha, self.beta, self.temp = float(alpha), float(beta), float(temp)
        self.adaptive, self.norm, self.qgroup = bool(adaptive), norm, qgroup
        self.weights, self.n_steps, self.blocks = weights, int(n_steps), tuple(blocks)

    @property
    def active_steps(self) -> frozenset:
        """Steps the arm must blend at: the non-zero weights, or every step."""
        if self.n_steps is None:
            return frozenset()
        if self.weights is None:
            return frozenset(range(self.n_steps))
        return frozenset(i for i, w in enumerate(self.weights) if w != 0.0)

    # -- recording -------------------------------------------------------------
    def record_delivery(self, key, applied: bool) -> None:
        ledger = self.n_applied if applied else self.n_skipped
        ledger[key] = ledger.get(key, 0) + 1

    def record_eta(self, key, eta: torch.Tensor) -> None:
        e = eta.detach().float().reshape(-1)
        self.eta_n[key] = self.eta_n.get(key, 0) + e.numel()
        self.eta_sum[key] = self.eta_sum.get(key, 0.0) + float(e.sum())
        self.eta_min[key] = min(self.eta_min.get(key, 1.0), float(e.min()))
        h = _hist(e, -1.0, 1.0, self._BINS)
        prev = self.eta_hist.get(key)
        self.eta_hist[key] = h if prev is None else prev + h

    def record_arm(self, key, raw_ratio: Optional[torch.Tensor], beta_eff) -> None:
        """One arm call: clamp events of the norm-match (pre-clamp ratio) + beta."""
        if raw_ratio is not None:
            r = raw_ratio.detach().reshape(-1)
            # one sync for both bounds
            lo_hi = torch.stack([(r < _HOP_R_MIN).sum(), (r > _HOP_R_MAX).sum()]).tolist()
            self.a_n[key] = self.a_n.get(key, 0) + r.numel()
            self.a_lo[key] = self.a_lo.get(key, 0) + int(lo_hi[0])
            self.a_hi[key] = self.a_hi.get(key, 0) + int(lo_hi[1])
        b = float(beta_eff) if not isinstance(beta_eff, torch.Tensor) else float(beta_eff.item())
        self.beff_sum[key] = self.beff_sum.get(key, 0.0) + b
        self.beff_n[key] = self.beff_n.get(key, 0) + 1

    def record_diag(self, key, E: torch.Tensor, r: torch.Tensor, align: torch.Tensor) -> None:
        """Paper Eq. 27-29 per retrieved feature column, aggregated over (b, h, column)."""
        self.d_calls[key] = self.d_calls.get(key, 0) + 1
        n = E.numel()
        self.E_n[key] = self.E_n.get(key, 0) + n
        self.E_sum[key] = self.E_sum.get(key, 0.0) + float(E.detach().float().sum())
        self.r_sum[key] = self.r_sum.get(key, 0.0) + float(r.detach().float().sum())
        a = align.detach().float().reshape(-1)
        self.al_sum[key] = self.al_sum.get(key, 0.0) + float(a.sum())
        h = _hist(a, -1.0, 1.0, self._BINS)
        prev = self.al_hist.get(key)
        self.al_hist[key] = h if prev is None else prev + h

    def record_price(self, key, disp: torch.Tensor, lo: dict, hi: dict) -> None:
        d = disp.detach().float().reshape(-1)
        self.p_n[key] = self.p_n.get(key, 0) + d.numel()
        self.p_sum[key] = self.p_sum.get(key, 0.0) + float(d.sum())
        h = _hist(d, 0.0, 2.0, self._BINS)
        prev = self.p_hist.get(key)
        self.p_hist[key] = h if prev is None else prev + h
        plo, phi = self.p_lo.setdefault(key, {}), self.p_hi.setdefault(key, {})
        for b in lo:
            plo[b] = plo.get(b, 0) + int(lo[b])
            phi[b] = phi.get(b, 0) + int(hi[b])

    def record_floor(self, key, d: torch.Tensor) -> None:
        f = d.detach().float().reshape(-1)
        self.f_n[key] = self.f_n.get(key, 0) + f.numel()
        self.f_sum[key] = self.f_sum.get(key, 0.0) + float(f.sum())

    # -- reading ---------------------------------------------------------------
    @staticmethod
    def _sum_hist(hists, keys):
        hs = [hists[k] for k in keys if k in hists]
        if not hs:
            return None
        return sum(hs[1:], hs[0].clone())

    def _stats(self, keys) -> dict:
        """Aggregate over a set of (step, block) keys into the summary fields."""
        nan = float("nan")
        out = {c: nan for c in HOP_SUMMARY_COLS}
        out["n_calls"] = sum(self.n_applied.get(k, 0) for k in keys)
        n_eta = sum(self.eta_n.get(k, 0) for k in keys)
        if n_eta:
            out["eta_mean"] = sum(self.eta_sum[k] for k in keys if k in self.eta_sum) / n_eta
            out["eta_min"] = min(self.eta_min[k] for k in keys if k in self.eta_min)
            q = _hist_quantiles(self._sum_hist(self.eta_hist, keys), -1.0, 1.0, (0.1, 0.5, 0.9))
            out["eta_p10"], out["eta_p50"], out["eta_p90"] = q[0.1], q[0.5], q[0.9]
        n_E = sum(self.E_n.get(k, 0) for k in keys)
        if n_E:
            out["E_mean"] = sum(self.E_sum[k] for k in keys if k in self.E_sum) / n_E
            out["r_mean"] = sum(self.r_sum[k] for k in keys if k in self.r_sum) / n_E
            out["align_mean"] = sum(self.al_sum[k] for k in keys if k in self.al_sum) / n_E
            out["align_p10"] = _hist_quantiles(
                self._sum_hist(self.al_hist, keys), -1.0, 1.0, (0.1,))[0.1]
        n_a = sum(self.a_n.get(k, 0) for k in keys)
        if n_a:
            out["clamp_lo_rate"] = sum(self.a_lo.get(k, 0) for k in keys) / n_a
            out["clamp_hi_rate"] = sum(self.a_hi.get(k, 0) for k in keys) / n_a
        n_b = sum(self.beff_n.get(k, 0) for k in keys)
        if n_b:
            out["beta_eff_mean"] = sum(self.beff_sum.get(k, 0.0) for k in keys) / n_b
        n_f = sum(self.f_n.get(k, 0) for k in keys)
        if n_f:
            out["floor_mean"] = sum(self.f_sum.get(k, 0.0) for k in keys) / n_f
        for kind, grid, col in (("alpha", HOP_ALPHA_GRID, "disp_a"), ("temp", HOP_TEMP_GRID, "disp_t")):
            for v in grid:
                pk = [k + (kind, v) for k in keys]
                n_p = sum(self.p_n.get(k, 0) for k in pk)
                if n_p:
                    out[f"{col}{v:g}"] = sum(self.p_sum.get(k, 0.0) for k in pk) / n_p
        return out

    def price_quantiles(self, keys, kind: str, value: float, qs=(0.5, 0.9)) -> dict:
        pk = [k + (kind, value) for k in keys]
        return _hist_quantiles(self._sum_hist(self.p_hist, pk) if any(k in self.p_hist for k in pk)
                               else torch.zeros(self._BINS), 0.0, 2.0, qs)

    def price_clip_rate(self, keys, kind: str, value: float, beta: float) -> tuple:
        pk = [k + (kind, value) for k in keys]
        n = sum(self.p_n.get(k, 0) for k in pk)
        if not n:
            return float("nan"), float("nan")
        lo = sum(self.p_lo.get(k, {}).get(beta, 0) for k in pk)
        hi = sum(self.p_hi.get(k, {}).get(beta, 0) for k in pk)
        return lo / n, hi / n

    def episode_stats(self) -> tuple:
        """(summary dict over the whole episode, one row per (step, block))."""
        keys = sorted(set(self.n_applied) | set(self.eta_n) | set(self.E_n))
        if not keys:
            return {}, []
        summary = self._stats(keys)
        rows = []
        for k in keys:
            s = self._stats([k])
            row = {"step": k[0], "block": k[1]}
            row.update({c: s[c] for c in HOP_ROW_COLS if c not in ("step", "block")})
            rows.append(row)
        return summary, rows


HOP = _HopCensus()


def fmt_hop(alpha: float, beta: float, qgroup: str, n_state_tokens: int, temp: float = 1.0,
            schedule: Optional[tuple] = None, adaptive: bool = False, norm: str = "l2") -> str:
    """Canonical arm string for signatures/logs — append-only: every optional
    part is emitted only when it is not its default."""
    s = f"a{alpha:g},b{beta:g},q{qgroup},ns{int(n_state_tokens)}"
    if temp != 1.0:
        s += f",t{temp:g}"
    if schedule is not None:
        s += f",s{fmt_schedule(schedule)}"
    if adaptive:
        s += ",adap"
    if norm == "off":
        s += ",norm-off"
    return s


def validate_hopfield(alpha: float, beta: float, temp: float = 1.0, adaptive: bool = False,
                      norm: str = "l2", schedule=None, probe: bool = False) -> Optional[str]:
    """Reject every Hopfield setting that would silently be a different arm.

    Called at install AND from eval_arm's argument layer (like validate_nag), so a
    bad combination dies before a checkpoint is loaded. Returns a notice string
    for the one legal-but-special setting (alpha=1, beta>0: the eager-dense
    control), None otherwise.
    """
    alpha, beta, temp = float(alpha), float(beta), float(temp)
    sched = parse_schedule(schedule)
    if beta < 0.0:
        raise ValueError(f"Hopfield beta must be >= 0, got {beta:g}.")
    if temp <= 0.0:
        raise ValueError(f"Hopfield temperature must be > 0, got {temp:g}.")
    if norm not in _VALID_HOP_NORMS:
        raise ValueError(f"Hopfield norm must be one of {_VALID_HOP_NORMS}, got {norm!r}.")
    if probe:
        if beta != 0.0 or alpha != 1.0 or temp != 1.0 or adaptive or norm != "l2" or sched is not None:
            raise ValueError(
                "Hopfield probe is the intervention-OFF measurement (alpha=1, beta=0, no "
                "temp/adaptive/norm/schedule): it returns the fused output bit-identically "
                "and only records. Run the arm without --hop-probe to intervene."
            )
        return None
    if beta == 0.0:
        dead = []
        if alpha != 1.0:
            dead.append(f"alpha={alpha:g}")
        if temp != 1.0:
            dead.append(f"temp={temp:g}")
        if adaptive:
            dead.append("adaptive")
        if norm != "l2":
            dead.append(f"norm={norm}")
        if sched is not None:
            dead.append(f"schedule={fmt_schedule(sched)}")
        if dead:
            raise ValueError(
                f"Hopfield beta=0 takes the fused-SDPA path (bit-identical to vanilla), so "
                f"{', '.join(dead)} would be dead flags: the signature would claim an "
                f"intervention that changes nothing. Set beta > 0 or drop them."
            )
        return None
    if temp != 1.0 and alpha != 1.0:
        raise ValueError(
            f"Hopfield temp={temp:g} with alpha={alpha:g}: the temperature control replaces "
            f"the skew-scaled retrieval, so the two are one arm each, never both."
        )
    if alpha == 1.0 and temp == 1.0:
        return (f"[HOP] alpha=1, beta={beta:g}: Z_a == Z bitwise, so this arm is the "
                f"odd-block EAGER-DENSE control (docs/hopfield.md §2b), not an intervention.")
    return None


def assert_hopfield_delivered() -> str:
    """Prove the Hopfield processors ran at every (active step, odd block) cell.

    Failure modes converted into a hard error, each of which would burn a full
    sweep under an arm name that claims an intervention: the pre-hook never
    fired; an active step the loop never reached; a (step, block) cell that
    never ran; a blend at a zero-weight step. Returns the census line.
    """
    if not HOP.blocks:
        raise RuntimeError("assert_hopfield_delivered() called with no Hopfield install.")
    if not SCHED.seen:
        raise RuntimeError(
            "Hopfield step probe never fired: no DiT forward was observed, so nothing "
            "proves the odd blocks ran through the Hopfield processor."
        )
    active = HOP.active_steps
    missing = sorted(active - SCHED.seen)
    if missing:
        raise RuntimeError(
            f"steps {missing} are active for the Hopfield arm but were never reached "
            f"(observed {sorted(SCHED.seen)}, N={HOP.n_steps}) — the arm ran weaker than it claims."
        )
    stray = sorted({s for (s, _) in HOP.n_applied if s not in active})
    if stray:
        raise RuntimeError(
            f"Hopfield blend fired at zero-weight steps {stray}; schedule is "
            f"{fmt_schedule(HOP.weights)}."
        )
    cells = [(s, b) for s in sorted(active) for b in HOP.blocks if HOP.n_applied.get((s, b), 0) == 0]
    if cells:
        raise RuntimeError(
            f"Hopfield processor never ran at (step, block) cells {cells[:8]}"
            f"{' ...' if len(cells) > 8 else ''} — the arm would be partly vanilla."
        )
    per_step = {}
    for (s, _), n in HOP.n_applied.items():
        per_step[s] = per_step.get(s, 0) + n
    skipped = {}
    for (s, _), n in HOP.n_skipped.items():
        skipped[s] = skipped.get(s, 0) + n
    what = "probe" if HOP.probe else fmt_hop(HOP.alpha, HOP.beta, HOP.qgroup, 1, HOP.temp,
                                              HOP.weights, HOP.adaptive, HOP.norm)
    line = (f"hop={what}; blocks {list(HOP.blocks)}; calls/step "
            f"{dict(sorted(per_step.items()))}; fused calls/step {dict(sorted(skipped.items()))}")
    n_b = sum(HOP.beff_n.values())
    if HOP.adaptive and n_b:
        line += f"; mean realized beta_eff {sum(HOP.beff_sum.values()) / n_b:.4f}"
    return line


class HopfieldAttnProcessor:
    """Symmetric/skew circulation control for one ODD (self-attention) block.

    docs/hopfield.md §1/§3. Prologue identical to :class:`PLADISAttnProcessor`
    (diffusers' AttnProcessor2_0); the attention core is replaced by

        beta_now == 0  ->  fused F.scaled_dot_product_attention   (bit-identical to vanilla)
        else           ->  Z_out = norm_match( Z + beta*(Z_a - Z) ),  Z_a = softmax(L_a) V

    with ``L_a = L + (alpha-1)*L_skew`` (``temp`` != 1: ``L_a = temp*L`` instead —
    the paper's temperature control through the same blend and norm-match).
    ``qgroup`` rows outside the group take Z; ``schedule`` multiplies beta per
    denoising step; ``adaptive`` scales (alpha-1) by the realized symmetry
    eta_bar and beta by (1 - eta_bar). In ``HOP.probe`` mode the fused output is
    returned and the diagnostics/price list are recorded on the side.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        temp: float = 1.0,
        qgroup: str = "all",
        n_state_tokens: int = 1,
        schedule: Optional[tuple] = None,
        adaptive: bool = False,
        norm: str = "l2",
        block_idx: int = -1,
    ) -> None:
        if qgroup not in _VALID_QGROUPS:
            raise ValueError(f"qgroup must be one of {_VALID_QGROUPS}, got {qgroup!r}")
        validate_hopfield(alpha, beta, temp, adaptive, norm, schedule, probe=HOP.probe)
        self.alpha, self.beta, self.temp = float(alpha), float(beta), float(temp)
        self.qgroup, self.n_state_tokens = qgroup, int(n_state_tokens)
        self.schedule = parse_schedule(schedule)
        self.adaptive, self.norm = bool(adaptive), norm
        self.block_idx = int(block_idx)

    def _beta_now(self, step) -> float:
        """Effective injection strength at the denoising step in flight (cf. _lambda_now)."""
        if self.schedule is None:
            return self.beta
        if step is None:
            raise RuntimeError(
                "Hopfield step schedule is armed but no denoising-step index was observed "
                f"— the DiT forward pre-hook did not fire, so schedule="
                f"{fmt_schedule(self.schedule)} cannot be enforced."
            )
        if step >= len(self.schedule):
            raise RuntimeError(
                f"denoising step {step} has no weight in Hopfield schedule "
                f"{fmt_schedule(self.schedule)} (len={len(self.schedule)}) — "
                f"num_inference_timesteps changed after install."
            )
        return self.beta * self.schedule[step]

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
        if encoder_hidden_states is not None or attention_mask is not None:
            # A square cross block would "work" silently with a meaningless transpose;
            # the decomposition is defined only for keys == queries (loci.md §1.1).
            raise RuntimeError(
                f"HopfieldAttnProcessor (block {self.block_idx}) received a cross-attention "
                f"call (encoder_hidden_states/attention_mask set): the sym/skew "
                f"decomposition needs QK^T square with keys == queries — install it on the "
                f"odd self-attention blocks only."
            )
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # --- Hopfield circulation control in place of scaled_dot_product_attention ---
        step = SCHED.current
        cell = (step, self.block_idx)
        beta_now = self._beta_now(step)
        if beta_now == 0.0 and not HOP.probe:
            # Fused path, byte-for-byte the call AttnProcessor2_0 makes: beta=0 and
            # every zero-weight schedule step are bit-identical to vanilla.
            HOP.record_delivery(cell, applied=False)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = self._core(query, key, value, hidden_states, cell, beta_now)
        # ------------------------------------------------------------------------------

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def _core(self, query, key, value, x, cell, beta_now: float) -> torch.Tensor:
        # Logits exactly as the PLADIS branch forms them (bf16 matmul, then f32);
        # transpose and 0.5*(L - L^T) are exact in f32.
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        logits = logits.float()
        if logits.shape[-1] != logits.shape[-2]:
            raise RuntimeError(
                f"Hopfield block {self.block_idx}: logits are {tuple(logits.shape[-2:])}, "
                f"not square — keys and queries differ, the decomposition is undefined."
            )
        skew = 0.5 * (logits - logits.transpose(-2, -1))
        dense = torch.softmax(logits, dim=-1)
        # Z_d in the SAME expression as the NAG branch (:692): one numeric
        # convention for both output-space stages.
        z_d = torch.matmul(dense.to(value.dtype), value)

        if HOP.probe:
            fused = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            HOP.record_delivery(cell, applied=True)
            self._measure(logits, skew, dense, value, x, z_d, fused, cell)
            return fused

        eta_bar = None
        if self.adaptive or HOP.record_episode:
            eta = _eta(logits, skew)  # (B, H)
            eta_bar = eta.mean()  # 0-dim tensor: no .item() on the arm path (docs/hopfield.md §3)
            if HOP.record_episode:
                HOP.record_eta(cell, eta)

        if self.temp != 1.0:
            la = self.temp * logits
        elif self.alpha == 1.0:
            la = logits  # exact identity -> Z_a == Z_d bitwise: the eager-dense control
        elif self.adaptive:
            la = logits + ((self.alpha - 1.0) * eta_bar) * skew  # Eq. 38-39
        else:
            la = logits + (self.alpha - 1.0) * skew
        z_a = torch.matmul(torch.softmax(la, dim=-1).to(value.dtype), value)

        beta_eff = beta_now * (1.0 - eta_bar) if self.adaptive else beta_now  # Eq. 41
        zb = z_d.float() + beta_eff * (z_a.float() - z_d.float())
        raw_ratio = None
        if self.norm == "l2":
            # Reference code: match to the PERTURBED retrieval's norm (Alg. 2, ref = ||Hx||).
            ref = _row_norm(z_a.float()).clamp_min(_HOP_EPS)
            cur = _row_norm(zb).clamp_min(_HOP_EPS)
            raw_ratio = ref / cur
            zb = zb * raw_ratio.clamp(_HOP_R_MIN, _HOP_R_MAX)
        HOP.record_delivery(cell, applied=True)
        if HOP.record_episode:
            HOP.record_arm(cell, raw_ratio, beta_eff)
        out = zb.to(value.dtype)
        # Query rows outside the group take Z_d, selected AFTER the norm-match, so
        # they are bit-identical to the dense output (the :694-702 pattern).
        ns = _split_point(self.qgroup, self.n_state_tokens, out.shape[-2])
        if ns is not None:
            if self.qgroup == "state":
                out = torch.cat([out[..., :ns, :], z_d[..., ns:, :]], dim=-2)
            else:  # action
                out = torch.cat([z_d[..., :ns, :], out[..., ns:, :]], dim=-2)
        return out

    @staticmethod
    def _measure(logits, skew, dense, value, x, z_d, fused, cell) -> None:
        """Probe mode: the paper's diagnostics on Xi = P X, plus the price list."""
        HOP.record_eta(cell, _eta(logits, skew))
        # Xi = P X per head over the block's INPUT features (paper Eq. 14, 1536-wide);
        # columns xi in R^L; local field h = L_sym xi (Eq. 25); lambda = xi (.) h (Eq. 26).
        xf = x.detach().float().unsqueeze(1)  # (B, 1, L, D_in)
        xi = torch.matmul(dense, xf)  # (B, H, L, D_in)
        lsym = 0.5 * (logits + logits.transpose(-2, -1))
        h = torch.matmul(lsym, xi)
        lam = xi * h
        E = -0.5 * lam.sum(dim=-2)  # (B, H, D_in), Eq. 27
        r = (lam < 0).float().mean(dim=-2)  # Eq. 28
        align = lam.sum(dim=-2) / (
            torch.linalg.vector_norm(xi, dim=-2) * torch.linalg.vector_norm(h, dim=-2)
        ).clamp_min(_HOP_EPS)  # Eq. 29
        HOP.record_diag(cell, E, r, align)

        zd_f = z_d.float()
        zd_norm = _row_norm(zd_f).clamp_min(_HOP_EPS)
        HOP.record_floor(cell, _row_norm(fused.float() - zd_f) / zd_norm)
        for kind, grid in (("alpha", HOP_ALPHA_GRID), ("temp", HOP_TEMP_GRID)):
            for v in grid:
                la = logits + (v - 1.0) * skew if kind == "alpha" else v * logits
                # the exact expression the arm path uses for Z_a, so the priced
                # displacement IS the arm's (gate H of verify_hopfield.py)
                z_c = torch.matmul(torch.softmax(la, dim=-1).to(value.dtype), value).float()
                delta = z_c - zd_f
                disp = _row_norm(delta) / zd_norm
                ref = _row_norm(z_c).clamp_min(_HOP_EPS)
                lo, hi = {}, {}
                for b in HOP_BETA_GRID:
                    ratio = ref / _row_norm(zd_f + b * delta).clamp_min(_HOP_EPS)
                    lo_hi = torch.stack([(ratio < _HOP_R_MIN).sum(), (ratio > _HOP_R_MAX).sum()]).tolist()
                    lo[b], hi[b] = int(lo_hi[0]), int(lo_hi[1])
                HOP.record_price(cell + (kind, v), disp, lo, hi)


def self_block_indices(dit) -> List[int]:
    """Odd (self-attention) block indices of the AlternateVLDiT (dit.py:380-388)."""
    n = len(dit.transformer_blocks)
    odd = [i for i in range(n) if i % 2 == 1]
    if not odd:
        raise ValueError(f"DiT has {n} block(s) and no odd (self-attention) block.")
    return odd


def install_hopfield(
    model,
    alpha: float = 1.0,
    beta: float = 0.0,
    temp: float = 1.0,
    qgroup: str = "all",
    n_state_tokens: int = 1,
    schedule=None,
    adaptive: bool = False,
    norm: str = "l2",
    blocks: Optional[List[int]] = None,
) -> List[int]:
    """Install HopfieldAttnProcessor on the odd (self) blocks; returns the block idxs.

    Reads ``HOP.probe`` (set BEFORE install, like NAG.probe). Coexists with
    install_pladis on the even blocks: the block sets are disjoint and the step
    probe is installed once per DiT. Raises on every configuration that would be
    vanilla under an intervention's name (docs/hopfield.md §4).
    """
    dit = _find_alternate_dit(model)
    notice = validate_hopfield(alpha, beta, temp, adaptive, norm, schedule, probe=HOP.probe)
    sched = parse_schedule(schedule)
    # N is needed for the per-step delivery census even without a schedule.
    head = _find_action_head(model)
    n_steps = int(head.num_inference_timesteps)
    if sched is not None:
        if len(sched) != n_steps:
            raise ValueError(
                f"Hopfield step schedule {fmt_schedule(sched)} has {len(sched)} weights but "
                f"the head runs N={n_steps} denoising steps — one weight per step is required."
            )
        if all(float(beta) * w == 0.0 for w in sched):
            raise ValueError(
                f"Hopfield schedule {fmt_schedule(sched)} at beta={beta:g} gives beta=0 at "
                f"every step — this arm would be bit-identical to vanilla."
            )
    _install_step_probe(dit, head)  # idempotent (:781); shared with install_pladis

    if blocks is None:
        blocks = self_block_indices(dit)
    even = [i for i in blocks if i % 2 == 0]
    if even:
        raise ValueError(
            f"Hopfield blocks {even} are even (cross-attention) indices: the decomposition "
            f"is undefined there (keys are VLM tokens). Odd blocks only."
        )
    n = len(dit.transformer_blocks)
    bad = [i for i in blocks if not 0 <= i < n]
    if bad:
        raise ValueError(f"Hopfield blocks {bad} outside the DiT's {n} blocks.")
    for i in blocks:
        if isinstance(getattr(dit.transformer_blocks[i].attn1, "processor", None), PLADISAttnProcessor):
            raise ValueError(
                f"block {i} already carries a PLADISAttnProcessor — the two processors must "
                f"live on disjoint block sets (PLADIS even/cross, Hopfield odd/self)."
            )
    HOP.arm(alpha, beta, temp, adaptive, norm, qgroup, sched, n_steps, blocks)
    installed = []
    for i in blocks:
        dit.transformer_blocks[i].attn1.set_processor(
            HopfieldAttnProcessor(
                alpha=alpha, beta=beta, temp=temp, qgroup=qgroup,
                n_state_tokens=n_state_tokens, schedule=sched, adaptive=adaptive,
                norm=norm, block_idx=i,
            )
        )
        installed.append(i)
    if not installed:
        raise RuntimeError("Hopfield install selected no blocks — this arm would be vanilla.")
    what = "probe" if HOP.probe else fmt_hop(alpha, beta, qgroup, n_state_tokens, temp,
                                              sched, adaptive, norm)
    msg = (f"[HOP] installed on blocks {installed} (hop={what}, "
           f"n_layers={n}, N={n_steps})")
    if notice:
        msg += "\n" + notice
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)
    return installed
