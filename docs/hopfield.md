# Hopfield circulation control on the self-attention blocks

How the symmetric/skew decomposition of Cho, Han & Jin (ICML 2026, *Balancing
Fidelity and Diversity in Diffusion Models via Symmetric Attention
Decomposition: Hopfield Perspective*) maps onto GR00T N1.7's action-head DiT,
what is algebraically new versus a re-parametrization of something already
collected, and the arm design that separates the two. Companion of
`docs/nag.md` (same section numbering) and of `docs/loci.md` §1.1.

**Why we want it.** Every positive this campaign has is on the *cross*
blocks (condition retrieval), and only on the language axis; the self blocks —
where the retrieved condition is turned into a trajectory — have never been
intervened on. The operator's 260821 deck §3 states the hypothesis this row
tests: *sparse cross-attention improves condition retrieval, while the
symmetric–skew structure of self-attention determines whether the retrieved
condition is converted into a coherent and adaptive action trajectory.*

**Provenance.** The method is the paper's (§0); the mapping onto the
`[state; action]` self-attention (§1), the dose algebra (§2) and the
placement inside the hook (§3) are this document's. Notation: `α` = skew
(circulation) scale, `β` = injection strength, as in the paper; the entmax
family parameter of the PLADIS branch is not involved here.

Status: **implemented, gated, phase 0 not yet measured** (2026-08-31). What
landed: `HopfieldAttnProcessor` + census in `pladis/attn_gr00t_n17.py`,
`--hop-*` on `eval_arm.py` (signature clause appended only when installed),
`experiments/verify_hopfield.py` (CPU gates §8), `experiments/diag_hopfield.py`
(phase 0, §6). Arms (§7) are registered in the drivers only after phase 0
fixes their values. Launch is the operator's.

## 0. The method as published

For a self-attention layer over features `X ∈ R^{L×d_in}` with
`Q = XW_Q`, `K = XW_K`, the pre-softmax matrix `QKᵀ = XWXᵀ` (`W = W_Q W_Kᵀ`)
is read as an associative memory over the L positions (Eq. 6–8). Its
symmetric part defines a Hopfield energy, its skew part a circulation that
contributes no energy (Eq. 18–23):

```
M_sym  = (QKᵀ + (QKᵀ)ᵀ)/2 = XSXᵀ ,   M_skew = (QKᵀ − (QKᵀ)ᵀ)/2 = XNXᵀ
E_X(ξ) = −½ ξᵀ M_sym ξ ,             ξᵀ M_skew ξ = 0   for every real ξ
```

Retrieval is `Ξ = softmax(M) X` (Eq. 13–16) and the output `Ξ W_V` (Eq. 17).
Stability of a retrieved feature column `ξ ∈ R^L` is measured against its
local field `h = M_sym ξ` (Eq. 25–29):

```
λ = ξ ⊙ h            E = −½ Σ_a λ_a         r = (1/L) Σ_a 1[λ_a < 0]         Align = cos(ξ, h)
```

Control (§5, Eq. 30–34, Algorithm 1) scales the circulation and blends the
result back into the baseline retrieval:

```
M_α     = M_sym + α·M_skew
Ξ_α     = softmax(M_α) X
Ξ_blend = Ξ + β·(Ξ_α − Ξ)
```

followed by "a normalization step that matches the baseline feature scale".
The reference code (Appendix A.1, Algorithm 2) fixes what that means:

```
Hx     = softmax(x_q (S + α·N) x_kᵀ) · x_v         # Ξ_α   (perturbed retrieval)
Hx_org = softmax(x_q (S + N)   x_kᵀ) · x_v         # Ξ     (baseline)
Hx_new = Hx_org + β·(Hx − Hx_org)
ref    = ‖Hx‖₂      (per row, clamp_min 1e-6)       # <- the norm of Ξ_α, NOT of Ξ
cur    = ‖Hx_new‖₂
Hx     = Hx_new · clamp(ref/cur, 0.25, 4.0)
```

Adaptive control (§6.1, Eq. 36–42): the realized symmetry index
`η_M = (‖M_sym‖²_F − ‖M_skew‖²_F)/(‖M_sym‖²_F + ‖M_skew‖²_F) ∈ [−1, 1]` per
(sample, head), aggregated to one scalar `η̄` per attention call, then
`α_eff = (α−1)·η̄` (i.e. `M_adap = M + α_eff·M_skew`) and `β_eff = β(1−η̄)`.

Published operating points: SDXL `(α, β) ∈ {(1.05, 5–7.5), (1.10, 5–6),
(1.15, 3–4)}`; SD3 (a transformer) `(0.95, 3), (0.97, 4), (0.97, 2)` — note
`α < 1` there. Gains are on the *worst-20 %* subsets (Table 3b), with
degradation on the *top-20 %* (Table 4): the intervention is state-dependent.
The paper's own control (Fig. 9) is global temperature `QKᵀ/τ`, which it
reports as less selective than circulation control at comparable strength.

## 1. The mapping onto GR00T N1.7

The action-head DiT alternates cross (even) and self (odd) blocks
(`docs/loci.md` §1.1). Odd blocks self-attend over `[state(1); action(40)]`
with no mask, so per head the logits `L = QKᵀ/√48` are 41×41 and queries and
keys are the same tokens — exactly the setting of §0. Even blocks have VLM
keys (non-square) and stay the PLADIS locus. Per odd block, per head:

```
L_skew = (L − Lᵀ)/2
L_α    = L + (α−1)·L_skew                 ≡ L_sym + α·L_skew
P      = softmax(L),       P_α = softmax(L_α)
Z      = P·V,              Z_α = P_α·V     # V = X W_V^h
Z_b    = Z + β·(Z_α − Z)
Z_out  = Z_b · clamp(‖Z_α‖₂ / ‖Z_b‖₂, 0.25, 4)    per (batch, head, query row)
```

`Z = Ξ W_V^h` by associativity, so the blend is the paper's up to floating
point; the norm-match ratio is measured on the 48 head channels instead of on
the 1536 input channels — the one registered deviation (`docs/loci.md` §5).
The diagnostics of §0 are computed on `Ξ = P·X` (the block's input features,
1536-wide), exactly as published.

Three axes of this repo apply unchanged:

| axis | mechanism |
|---|---|
| query group (`--hop-qgroup`) | rows outside `{state, action}` take `Z` (the dense output), selected after the norm-match, as the NAG stage does (`docs/nag.md` §3) |
| denoising step (`--hop-schedule`) | per-step multiplier on β; a zero-weight step takes the fused-SDPA path |
| temperature control (`--hop-temp τ`) | `Z_α := softmax(τ·L)·V` in place of the skew-scaled retrieval, same blend, same norm-match — the paper's Fig. 9 control, run through the same code path so it differs from a Hopfield arm in nothing but the perturbed logits |

## 2. What is algebraically new, and what is not

**(a) β = 0 is the fused path.** No blend, no eager softmax: the processor
delegates to `F.scaled_dot_product_attention` exactly as `pladis_scale = 0`
does, so a hop arm at β=0 (and every zero-weight step) is bit-identical to
vanilla. This is why `--hop-probe` — which returns that fused output and only
*measures* — is a vanilla rollout with a diagnostic attached.

**(b) α = 1 is bit-identical to the eager dense path.** With the
`L + (α−1)·L_skew` form the added term is exactly `0.0`, so `P_α == P`,
`Z_α == Z`, `Z_b == Z + β·0 == Z`, the ratio is `1.0` exactly and the output
is the manual-softmax dense output. That arm (`hop-dense`) is the self-block
counterpart of the even-block eager-dense control: no collected arm has ever
run the odd blocks on the eager path, and the fused-vs-eager kernel term is
real on this track (README §1.4, §7). Every `hop − hop-dense` contrast
cancels it; `hop − vanilla` carries it.

**(c) The first-order dose is δ = β(α−1).** Expanding the softmax in α−1,
`Z_b − Z = β(Z_α − Z) ≈ β(α−1)·J_P[L_skew]·V`. So `(α, β)` pairs with equal
`β(α−1)` are first-order duplicates, the way ρ·λ was for NAG
(`docs/nag.md` §2b), and a ladder is a ladder in δ, not in α and β
separately. The sign of δ is the direction: `α < 1` damps the circulation
(toward the energy landscape — the "fidelity" side, the SD3 setting),
`α > 1` injects it ("diversity", the SDXL setting). Second order — and the
norm-match — is where equal-δ pairs differ, which §7 tests with ONE matched
pair rather than a grid.

**(d) β = 1 arms are un-normalized α-retrievals; β ≠ 1 arms are normalized.**
Because the reference norm-matches to `‖Z_α‖` (not to `‖Z‖`), `β = 1` gives
`Z_b = Z_α` and the ratio is identically 1. So the (α, β=1) family and the
(α, β>1) family are two families, not one ladder: the first is "replace the
retrieval by the circulation-scaled one", the second is "extrapolate through
it and rescale to its magnitude". `--hop-norm off` on a β>1 arm is what
separates extrapolation from the rescale.

**(e) Adaptive control is self-extinguishing.** `β_eff = β(1−η̄)` and
`α_eff = (α−1)η̄`: at `η̄ → 1` (symmetric-dominant realized attention) the
intervention vanishes, at `η̄ < 0` the injected skew flips sign. Whether it
does anything on this DiT is therefore a *measurement* (η̄ per block/step,
§6), not a design choice — an adaptive arm launched without it could be a
re-run of `hop-dense`.

**(f) Cost.** Two extra softmaxes and one extra `(41×41)·(41×48)` matmul per
head per odd block per step; far below the 40 ms/step scale of the sweep cost
model. The probe's `Ξ = P·X` is a `(41×41)·(41×1536)` per head and runs only
in probe mode.

## 3. Placement inside the hook

- **Odd blocks only, self-attention only.** The processor raises on
  `encoder_hidden_states`, on an `attention_mask`, and on `L_q ≠ L_k`: a
  square cross block would otherwise "work" silently with a meaningless
  transpose. Installing on an even index raises; installing on a block that
  already carries a `PLADISAttnProcessor` raises (block sets must be disjoint).
- **Dtype.** Logits as the PLADIS branch computes them (bf16 `q@kᵀ·scale`,
  then f32); transpose and `0.5·(L − Lᵀ)` in f32 (exact operations); both
  softmaxes in f32; `P.to(value.dtype) @ value` — the same expression as the
  NAG branch's `Z_d`, so the two output-space stages share one numeric
  convention. Blend and norm-match in f32, one cast to `value.dtype` at the
  end. Norms are never taken on bf16.
- **Norm-match.** `ref = ‖Z_α‖₂.clamp_min(1e-6)`, `cur = ‖Z_b‖₂.clamp_min(1e-6)`,
  `ratio = (ref/cur).clamp(0.25, 4)` — the reference code's constants; ε
  guards a zero row and is not added anywhere else (the `docs/nag.md` §3
  lesson). The census counts how often either clamp bound binds.
- **Query groups.** Norm-match on the full `Z_b`, THEN rows outside the
  group are taken from `Z` by `torch.cat` — bit-identical to the untouched
  rows of the dense path (gate D).
- **Schedule.** The Hopfield census keeps its OWN weight vector and reads only
  the step index the (single, idempotent) DiT pre-hook publishes. It does not
  go through `SCHED.arm()`, whose conflict guard would refuse a combined arm
  whose cross schedule (`0,0,1,1`) differs from its self schedule (`all`).
- **Adaptive η̄** is kept as a 0-dim tensor (no `.item()` in the arm path —
  16 blocks × 4 steps × ~90 chunks per episode would otherwise be ~6k GPU
  syncs); only the census syncs, once per call, when it is recording.
- **RNG.** Nothing on this path consumes random numbers; the three seeding
  layers of the harness are untouched.

## 4. Interface

`eval_arm.py`, gr00t_n17-only (raised on pi05/smolvla):

```
--hop-install                     install HopfieldAttnProcessor on the odd blocks
--hop-alpha  <float>  default 1   skew (circulation) scale α
--hop-beta   <float>  default 0   injection strength β; 0 = fused path (bit-parity)
--hop-temp   <float>  default 1   temperature control: Z_α = softmax(τ·L)V; requires α = 1
--hop-qgroup {all,state,action}   query rows the blend is written to
--hop-schedule <w,w,w,w>|all      per-step multiplier on β
--hop-adaptive                    η̄-adaptive (α_eff, β_eff) per attention call
--hop-norm   {l2,off}  default l2 the reference norm-match, or none
--hop-probe                       α=β=… ignored: bit-identical rollout, diagnostics recorded
```

Rejected at the argument layer AND at install (`validate_hopfield`): β<0;
τ≤0; β=0 with any of {α≠1, τ≠1, adaptive, norm off, schedule} — dead flags
on a fused-path arm whose signature would claim an intervention; τ≠1 with
α≠1 (two perturbations in one arm); probe with anything non-default; any
`--hop-*` without `--hop-install`. `α=1, β>0` is *allowed* and announced as
the eager-dense control.

Signature clause, appended as a new `|` element only under `--hop-install`:
`hop=a{α},b{β},q{qgroup},ns{n_state}[,t{τ}][,s{sched}][,adap][,norm-off]`;
a probe writes `hop=probe` (a separate ledger on purpose — under the vanilla
signature a collected arm would resume as a no-op and record nothing). Every
eplog written before this row keeps a byte-identical signature.

Sidecars, per episode, for every hop arm with β>0 and for probes:
`<out>.hopstats.tsv` (η̄ mean/quantiles, E, r, Align, clamp rates, realized
β_eff, and in probe mode the displacement priced for the α and τ grids) and
`<out>.hopstats_sb.tsv` per (episode, step, block).

## 5. The objective

> Does the symmetric/skew structure of the self blocks carry test-time
> leverage that the cross blocks do not — on the axes where cross-attention
> sparsification was null or harmful (robot, layout, camera, noise) — and does
> it add to the cross-block gain where that exists (language)?

### 5.1 Endpoints

All arms of an axis share one episode set; every statistic is paired.

1. **Primary — `hop − hop-dense`** per axis, paired McNemar: the intervention
   with the odd-block eager term cancelled. `hop − vanilla` is reported next
   to it as the deployable number.
2. **Direction** — the sign of δ that helps, if any, per axis: the α<1 and
   α>1 rungs against each other and against `hop-dense`.
3. **Specificity** — `hop − hop-temp` at matched median displacement (§6):
   whether the circulation structure matters or any comparable perturbation
   of the self blocks does the same (the paper's Fig. 9, our
   zeros-not-special question restated).
4. **Additivity** — the 4-arm interaction
   `[combined − allxt-late-l2] − [hop − vanilla]` on language (`analyze.py`'s
   `extra_interactions`), the deck's "retrieval + dynamics" claim.
5. **Diagnostic correlation** (free, from the probe): per-episode `Align`,
   `E`, `r` of the vanilla rollout split by outcome (Welch z), the paper's
   Table 1 restated on success instead of aesthetics.

### 5.2 The outcomes, named in advance

| outcome | reading |
|---|---|
| α>1 helps on robot/OOD axes, null on language | the deck's "adaptive dynamics" reading: skew lets the chunk re-aim under a perturbed state; cross = retrieval, self = dynamics dissociate by axis |
| α<1 helps | a coherence lever (the symmetric landscape), the SD3 direction; opposite to the deck's guess but a positive all the same |
| both directions null at matched displacement | the self-block structure is not a test-time lever; joins the noise/camera ceiling story |
| hop ≈ hop-temp at matched displacement | nothing Hopfield-specific — a generic perturbation of the self blocks, to be reported as such |
| hop ≠ hop-temp | structure-specific; the paper's selectivity claim transfers |
| interaction > 0 on language | the two loci are additive: the deck's composite hypothesis |

**Inert by construction** is the outcome §6 exists to catch before a launch:
if the α grid's displacement does not clear the fused-vs-eager floor measured
on the same tensors, the row is not launched.

## 6. Phase 0 — measure first (`experiments/diag_hopfield.py`)

`--hop-probe` runs the vanilla rollout (bit-identical) and records, per
(block, step) and per episode:

1. **η̄ per head** — is the realized self-attention symmetric-dominant
   (η̄ ≈ 1, adaptive control self-extinguishes, α must move far from 1) or
   circulation-heavy (η̄ ≪ 1)?
2. **E, r, Align** of the baseline retrieval, split by outcome.
3. **The price list.** From the SAME logits and V, without changing the
   rollout: for every α in `HOP_ALPHA_GRID = (0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2)`
   and every τ in `HOP_TEMP_GRID = (1.25, 1.5, 2, 3)`, the per-row relative
   displacement `d = ‖Z_α − Z‖₂/‖Z‖₂` (mean, quantiles) and the norm-match
   clamp rate at β ∈ {0.5, 1, 2, 5}; the linearity check `d(α)/(α−1)`; and
   the fused-vs-eager floor `‖fused − Z‖₂/‖Z‖₂` on the same rows. The
   even-block reference: the NAG probe's own displacement of the
   `allxt-temp20l20` blend on the text blocks, recorded on the same rollout
   when the diag is run with that arm's flags.

**Pre-registered rules.** Both signs of δ are swept. The magnitude of α is
the grid point whose median displacement is closest to the even-block
`allxt-temp20l20` displacement (the strength the cross-block campaign found
effective); the temperature control τ is the grid point whose median
displacement matches that α's. If no grid point clears **3× the
fused-vs-eager floor**, the row is not launched. η̄ decides whether an
adaptive arm is meaningful: if the median η̄ exceeds 0.9 at every block, the
adaptive arm is dropped as a near-copy of `hop-dense`.

Measurement tables: to be filled by the first run (language and robot,
`libero_goal`, 40 episodes each).

## 7. Arms

Provisional, at the paper's operating points; α and τ are replaced by the
§6 values before any `run` line is committed. Language and robot axes first
(one axis per machine, SETUP.md §0); ~5 h/arm.

| arm | flags | reads against |
|---|---|---|
| `hop-dense` | `--hop-install --hop-alpha 1 --hop-beta 1` | vanilla (the odd-block eager term) |
| `hop-a090b5`, `hop-a095b5`, `hop-a105b5`, `hop-a110b5` | `--hop-beta 5`, α ∈ {.90, .95, 1.05, 1.10} | δ ∈ {−.5, −.25, +.25, +.5}; each vs `hop-dense` and vs vanilla; the two signs against each other |
| `hop-a075b1`, `hop-a125b1` | `--hop-beta 1` | matched-δ (±.25) un-normalized family, vs its β=5 twin |
| `hop-a105b5-nonorm` | `--hop-norm off` | vs `hop-a105b5`: extrapolation vs rescale |
| `hop-a110b5-adap` | `--hop-adaptive` | vs `hop-a110b5` |
| `hop-t{τ}b1` | `--hop-temp τ --hop-beta 1` | vs `hop-dense` and vs the hop arm it is matched to |
| later: `hop-{best}-late` (`--hop-schedule 0,0,1,1`), `allxt-late-l2-hop-{best}` | | the late-step and the combined arm, once a direction exists |

## 8. Gates (CPU, no checkpoint) — `experiments/verify_hopfield.py`

- **A.** β=0 (all qgroups, zero-weight schedule steps, probe on/off) is
  `torch.equal` to diffusers' `AttnProcessor2_0`.
- **B.** α=1, β ∈ {0.5, 1, 2} is `torch.equal` to the manual dense path
  (the `method=softmax, β=1` processor), with zero clamp events.
- **C.** Decomposition identities on the processor's own logits:
  `L_sym + L_skew == L`, `L_skewᵀ == −L_skew` exactly, `|ξᵀL_skewξ|` at
  rounding level, `η ∈ [−1, 1]`, recorded η equals a direct computation.
- **D.** qgroup: rows outside the group are `torch.equal` to the dense
  output for α≠1, β ∈ {1, 2}, norm on/off, adaptive.
- **E.** Schedule: zero-weight steps `torch.equal` to vanilla; weighted steps
  `torch.equal` to an unscheduled processor at β·w.
- **F.** Every rejected setting raises; PLADIS on the even blocks and
  Hopfield on the odd blocks install together on one DiT with one pre-hook.
- **G.** `assert_hopfield_delivered()` raises for a never-fired probe, an
  unreached step, a missing (step, block) cell, and a call at a zero-weight
  step; passes with the census on a full loop.
- **H.** Probe statistics against a float64 reference; the priced
  displacement for a grid α equals the displacement of a processor actually
  run at that α; the priced clamp rate at (α, β) equals what a processor run
  at (α, β) applies.
- **I.** Adaptive: `L_skew = 0` gives η̄ = 1 and an output `torch.equal` to
  dense; random logits give η̄ ∈ (−1, 1) and `β_eff = β(1−η̄)` as recorded.

On the checkpoint (GPU): `verify_base0_parity.py` gains the β=0 self case and
asserts the processor raises on both cross cases; eval_arm's warm-up prints
the Hopfield census before the first logged episode.

## 9. What would falsify the row

- **Displacement below the floor at every grid α** → the decomposition is
  numerically inert on this DiT (the self-attention is already nearly
  symmetric, or the softmax saturates); §6 stops the launch and the
  measurement is the result.
- **η̄ ≈ 1 everywhere** → adaptive control is a no-op here; drop it, keep the
  static arms.
- **`hop-dense − vanilla` comparable to the effects** → the odd-block eager
  term dominates; only `hop − hop-dense` is readable.
- **hop ≈ hop-temp at matched displacement** → nothing structure-specific;
  report as a generic self-block perturbation.
- **Both signs null on every axis** → the self-block structure is not a
  test-time lever; the deck's composite hypothesis loses its second half.
