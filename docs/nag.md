# NAG (Normalized Attention Guidance) on the PLADIS blend

How the normalization/refinement stages of NAG (arXiv:2505.21179v3, *Normalized
Attention Guidance: Universal Negative Guidance for Diffusion Models*) map onto
this repo's dense/sparse attention blend, what is algebraically new versus what
is a reparametrization of the dose ladder we already ran, and the arm design
that separates the two.

**Why we want it.** The all×text sharp-softmax family is the strongest
intervention on four of six LIBERO-plus axes, but its best λ differs per axis —
one rung of dose swings 3.8pp across axes and flips sign. The campaign asks
whether the per-query cap **widens the λ plateau**, turning "tune λ per axis"
into "run one strong λ and let NAG clip the activations that would make it
harmful". §5 states that objective, its endpoints, and the two ways it can fail
while looking like a success.

**Provenance.** The mapping in §1 is the operator's, from *260821 New
Reinforcements in Our Research* §2 ("NAG Normalization in Ours", slides
*NAG Normalization* / *Normalized PLADIS* / *NAG Refinement in PLADIS*); this
document follows its notation (`Z_PL`, `Z_NPL`, `ρ`, `ε`) and adds only the
consequences that constrain the arm design (§2), the placement decisions inside
the hook (§3), and the measurement that has to precede a launch (§5).

Status: **implemented, gated, not launched** (2026-08-26). What landed:
`pladis/attn_gr00t_n17.py` (the two stages + a census keyed by step/block/query
group), `--pladis-nag-tau` / `--pladis-nag-rho` on `eval_arm.py` (signature clause
appended only when on), `experiments/verify_nag.py` (8 CPU gates, all PASS),
`experiments/diag_nag.py` (Phase 0), the arms of §7 in the `original` / `language`
/ `robot` drivers, and their contrasts plus the dose-step interaction in
`analysis/analyze.py`. Launch is the operator's.

## 0. NAG as published

NAG runs two attention branches (positive/negative prompt), extrapolates in
attention-OUTPUT space, then stabilizes the result in two stages. Paper §4.1,
Eq. 7-10, with the reference implementation in Appendix C, Algorithm 1:

```
Z_e   = Z⁺ + φ·(Z⁺ − Z⁻)                                   # (7) extrapolation
R[i]  = ‖Z_e[i]‖₁ / ‖Z⁺[i]‖₁                                # (8) per-query L1 ratio
Ẑ[i] = min(R[i], τ)/R[i] · Z_e[i]                          # (9) NORMALIZATION
Z^NAG = α·Ẑ + (1−α)·Z⁺                                      # (10) REFINEMENT
```

Algorithm 1 takes the norms on the SDPA output with `p=1, dim=-1, keepdim=True`,
i.e. **per (batch, head, query token) over the head dimension** — not over the
merged head axis, and not over the token axis. Note the published form of (9) is
`where(R > τ, τ, R)/R`, with no epsilon anywhere.

Published defaults (Table 5): DiT families φ=4, τ=2.5, α∈{0.125, 0.25, 0.375};
UNet families φ=2, τ=2.5, α=0.5. The ablation (Fig. 8) removes the two stages in
exactly the order this document uses: full NAG → w/o refine (α=1) → w/o refine
and norm (α=1, τ=∞), the last of which is plain attention-space extrapolation.

Two properties of the published defaults matter for us more than the values:

1. **αφ ≈ 0.5–1.5 in every family** (Flux 4×0.25=1.0, SD3.5 4×0.125=0.5,
   SANA 4×0.375=1.5, SDXL 2×0.5=1.0). NAG is not a large-dose method. It is a
   *large raw extrapolation, hard-capped per query, then shrunk back* — the
   effective linear dose stays near 1 and the cap is what does the work.
2. **τ is the only nonlinearity.** Everything else in Eq. 7-10 is affine in Z_e.

## 1. The mapping onto our blend (operator's, deck §2)

Our intervention (`pladis/attn_gr00t_n17.py`) blends attention MAPS; `Z = W·V`
is linear in `W`, so in output space the arm we already run IS Eq. 7:

```
W_PLADIS = W_dense + λ·(W_sparse − W_dense)
Z_PL     = Z_d + λ·(Z_s − Z_d),        Z_d = W_d·V,  Z_s = W_s·V
```

Normalization and refinement then take the **dense branch as the baseline**:

```
R[i]      = ‖Z_PL[i]‖₁ / (‖Z_d[i]‖₁ + ε)              # NORMALIZATION
Z_NPL[i]  = min(R[i], τ)/R[i] · Z_PL[i]
Z_final   = ρ·Z_NPL + (1 − ρ)·Z_d                      # REFINEMENT
```

| stage | controls | reference point |
|---|---|---|
| Normalization | feature magnitude | origin |
| Refinement | deviation from the baseline | baseline feature `Z_d` (NAG's `Z⁺`) |

`Z⁺ = Z_d` is also what the fixed point forces: NAG's un-guided setting is φ=0 →
`Z⁺`, ours is λ=0 → `Z_d` (vanilla). The alternative reading (`Z⁺ = Z_s`, φ=λ−1)
would put the fixed point at λ=1 and calibrate the guardrail against the very
magnitude it exists to bound; it is recorded here only so it is not
re-litigated.

**What kind of conservation this is.** The blend needs no probability-mass fix on
this architecture — GR00T N1.7's cross blocks are dedicated cross-attention, so
both branches' rows already sum to 1 (`docs/loci.md` §0, SDXL pattern). NAG adds
a *different* conservation on top: not probability/modality mass, but **hidden
feature magnitude**. On a joint-attention track (π0.5, FLUX pattern) the hook
already renormalizes a key sub-block by its dense row mass, so a future port
there stacks two normalizations of different kinds and must be re-derived, not
copied.

## 2. What is algebraically new, and what is not

Three consequences, all of which constrain the arm design in §6.

**(a) NAG is a strict superset of the current arm.** τ=∞, ρ=1 gives
`min(R,∞)/R = 1.0` exactly and `1·Z_NPL = Z_PL`, so the NAG path reproduces the
current path — bit-identically, if implemented as §3 specifies. Every existing
eplog stays reproducible and the flag defaults to off.

**(b) Refinement alone is a dose rescale, not a new arm.** On any query row where
the cap is inactive (R ≤ τ):

```
Z_final = ρ·[Z_d + λ(Z_s − Z_d)] + (1−ρ)·Z_d = Z_d + (ρ·λ)·(Z_s − Z_d)
```

which is the plain arm at dose λ_eff = ρ·λ. So a "refinement-only" arm (τ=∞,
ρ<1) is `--pladis-scale (ρ·λ)` under another name, and this axis has already run
the λ ∈ {1, 1.5, 2} ladder at this locus. **We will not run it.** Refinement has
independent content only jointly with a raised λ and an active cap — which is
precisely the published recipe (§0, property 1).

**(c) Therefore the entire experimental content of NAG here lives in the clipped
branch**, and the correct control for any NAG arm is the *already-collected flat
arm at its unclipped-equivalent dose* `λ_eff = ρ·λ`. Two arms differing only in
whether queries above τ are capped is a surgical contrast with no dose confound —
the same iso-dose discipline the 08-16 step-schedule row used.

This also gives the campaign's first prediction at a cost of zero GPU-hours: if
the measurement in §5 finds that R exceeds every plausible τ on a negligible
fraction of queries, then NAG at our operating point **is** the arm we already
ran, and the row must not be launched.

## 3. Placement inside the hook

Per-cell decisions, each mirroring the deck, Algorithm 1, or an existing
invariant:

- **Norm axis.** `dim=-1` over `head_dim`, per (batch, head, query row) — the
  paper's own reduction. Our `Z` is `(B, H, Lq, D_head)`, identical in shape to
  Algorithm 1's SDPA output, so this transfers without reinterpretation.
- **Epsilon.** The deck writes ε in both denominators. We keep it in the ratio
  (`‖Z_d‖₁ + ε`, guarding a zero-magnitude baseline row) but implement the cap in
  the published `where(R > τ, τ/R, 1)` form rather than `min(R,τ)/(R + ε)`. The
  second ε would multiply every *uncapped* row by `R/(R+ε) < 1` — a silent
  shrink applied to rows the method says pass through unchanged, and enough to
  break both the τ-off nesting of §2(a) and the untouched-row bit-parity of the
  qgroup split. ε = 1e-6, in float32.
- **Dtype.** The blend runs in float32 for the softmax/entmax and casts to
  `value.dtype` (bf16) for the matmul. Ratios and the cap are computed in
  float32 and applied to the bf16 `Z_PL`; an L1 norm over a head's bf16 channels
  is not worth the precision risk.
- **Query groups.** NAG is applied to the full `Z`, and the qgroup row-select
  happens AFTER it, taking rows outside the group from `Z_d`. Row i of a matmul
  depends only on row i of the left operand, so the selected rows are
  bit-identical to today's slicing. Doing the ρ-blend on non-selected rows
  instead would return `ρ·x + (1−ρ)·x`, which is not bit-exactly `x` for general
  ρ — a silent 1-ulp perturbation of rows the arm claims not to touch.
- **Step schedule.** `λ_i = 0` steps keep the fused-SDPA path, untouched: NAG is
  computed only where the blend is. `late`/`inc` shapes compose with NAG
  unchanged.
- **λ=0.** NAG with `--pladis-scale 0` is rejected at the argument layer
  (`base0` must stay the fused-SDPA parity arm, and R≡1 makes NAG a no-op there).
- **Cost.** One extra `(Lq×Lk)·(Lk×D)` matmul per installed block per step
  (`Z_d`, needed as both the norm reference and the refinement target). At this
  DiT's shapes it is far below the 40 ms/step scale the sweep cost model is
  built on.

The NAG-off code path is left exactly as it is today — not refactored into a
τ=∞ special case — because λ>0 arms are not on a bit-parity gate and any change
to the order of floating-point operations would make already-collected arms
non-reproducible.

## 4. Interface

New flags on `eval_arm.py`, gr00t_n17-only (the pi05/smolvla hooks are untouched
until this lands on n17 and reads positive):

```
--pladis-nag-tau  <float>   default: off (inf).  τ ≥ 1 required.
--pladis-nag-rho  <float>   default: 1.0.        0 < ρ ≤ 1.
```

`rho`, not `alpha`: the deck's notation, and `alpha` is already the entmax family
parameter (entmax-**1.5**) on the sparse branch of this same processor.

- `--pladis-nag-rho` without `--pladis-nag-tau` is rejected: by §2(b) it is
  `--pladis-scale (ρ·λ)` wearing another arm's name.
- Signature clause is APPENDED ONLY WHEN ON — `,nagt{τ:g},nagr{ρ:g}` — so every
  eplog written before NAG existed keeps a byte-identical signature and still
  resumes (`harness/eplog.py:62-80` aborts on any change).
- Video/arm tag gains ` nag τ=… ρ=…`, on the same "what a reviewer watching the
  video is judging" rule as the schedule tag.

## 5. The objective: widen the λ plateau

The campaign question is not "does NAG beat the parent at one λ". It is:

> all×text with the sharp-softmax branch (β=2) is the dominant family on
> `original`, `language`, `robot` and `noise`, but **its best λ differs per
> axis**. Can the cap remove the per-axis λ tuning — not by making every axis
> agree on one λ*, but by widening the plateau, so that a single reasonably
> strong λ is near-optimal everywhere and NAG clips only the activations that
> would make it harmful?

**The premise, measured at HEAD** (`python3 analysis/analyze.py --<axis>`;
all×text, `--pladis-method softmax --pladis-beta 2.0`). Every plain rung below is
collected; the design input is where each axis's ladder peaks and how steep the
walk from λ=1 to λ=2 is, paired McNemar within the axis:

| axis | n | best λ | λ=1 → λ=2 | shape |
|---|---|---|---|---|
| original | 400 | **1.0** | −2.25pp (z=−1.80, p=.072) | falls from λ=1 |
| noise | 1601 | 1.5 | −1.37pp (z=−1.81, p=.071) | peaks at 1.5, −1.81 by λ=2 |
| robot | 1550 | 2.0 | +1.55pp (z=+1.62, p=.106) | climbing |
| language | 1537 | 2.0 | **+1.95pp (z=+2.72, p=.007)** | climbing |
| camera | 1599 | 1.0 | −1.06pp | family never beats vanilla |

The same dose step swings **4.2pp** across axes and flips sign: `original`+`noise`
want λ≈1–1.5, `language`+`robot` want λ=2. That is a real transfer failure, not
ladder noise, and it is what makes the objective below worth GPU-hours.

### 5.1 Endpoints

Let `S_f(axis, λ)` be pooled SR for family f ∈ {plain, nag} at dose λ. All four
arms of a comparison share one episode set (the repo asserts this), so every
statistic below is computed on per-episode paired contrasts.

1. **Primary — gap reduction (difference-in-differences).** Per axis, with
   `λ₀` = that axis's own best plain rung (1.0 on original, 1.5 on noise, 2.0 on
   language/robot) and `λ₁ = 2` the shared setting:
   `d_i = [nag_λ₁ − nag_λ₀] − [plain_λ₁ − plain_λ₀]` per episode `i`,
   `d_i ∈ {−2,…,2}`; report `mean(d) ± sd(d)/√n` and z. Success on an axis whose
   plain step is negative (original, noise) is `d > 0`; the *joint* claim is that
   `|step|` shrinks on **both** signs at once. McNemar does not apply
   here — it tests one 2×2 table — so `analyze.py` needs a 4-arm interaction
   contrast next to the existing pair contrasts.
   **Power, measured rather than assumed** (the DiD's SE computed on four real
   arms that already share an episode set, `analyze.py` machinery): **1.04pp at
   n=1537** (language) and **1.94pp at n=400** (original). So `language`, `robot`
   and `noise` can resolve a ~2pp change in the dose step at z≈2, while
   `original` at its default 100 episodes/suite is **directional only** — a full
   flattening of its −2.25pp step would read z≈1.2. Confirmation there would need
   `ORIG_EPISODES=500` (2,000 eps, SE ≈0.87pp) at 5× the cost, which is why
   `original` is scoped as the cheap shape probe and the confirmatory DiD belongs
   on the big-n axes. Episode sets are disjoint across axes, so the per-episode
   contrasts can also be pooled.
2. **Secondary — equivalence inside the NAG family.** TOST on
   `nag_λ2 − nag_λ1.5` against a pre-registered margin δ = 1.5pp. **Power note,
   stated in advance:** the paired McNemar SE on these axes is ≈0.7pp, so
   declaring equivalence at δ=1.5 needs `|Δ̂| ≲ 0.3pp`. This endpoint can fail for
   lack of power while endpoint 1 succeeds; it is reported, not relied on.
3. **Tertiary — worst-case regret of one fixed λ.**
   `regret_f(λ) = max_axis [ max_λ' S_f(axis,λ') − S_f(axis,λ) ]`. "No per-axis
   tuning" IS a claim about this number. Free to compute from the same arms.

### 5.2 The three outcomes, named in advance

- **Success (selective cap).** noise λ=2 recovers toward its λ=1.5 level AND
  language keeps its λ=2 gain. This requires the cap to be *selective*: the harm
  at high λ is carried by magnitude blow-up, the gain by the direction change.
- **Failure A (flattening both ways).** The plateau widens because the language
  gain is capped away too. A flat curve at a lower ceiling is not success — it
  trades tuning for performance, and the regret statistic (5.1.3) will show it.
- **Failure B (inert).** Clip rate ≈ 0 at every plausible τ; NAG reproduces its
  parent arm by construction (§2(a)) and nothing moves.

**A plateau that ρ bought is not success.** By §2(b), refinement compresses the
dose axis by ρ, so `S_nag(λ) = S_plain(ρλ)` wherever the cap is inactive: any
ρ<1 arm shows a narrower λ1.5→λ2 step for free, having only relabelled the
x-axis, with the optimum moved to λ*/ρ. **The plateau claim is therefore made
with ρ=1 (normalization only).** Refinement is tested afterwards, at matched
effective dose, as "does ρ<1 add anything the dose ladder does not".

## 6. Phase 0 — measure R, then decide whether to spend the hours

τ=2.5 is a Flux/SD3.5 number; our two branches differ by a sharpening operator,
not by a prompt, so nothing licenses transferring it. `experiments/diag_nag.py`
runs a handful of episodes per axis with the standard hook installed and the NAG
census armed (a module-level census in the hook, the `SCHED` pattern — not a
parallel implementation of the blend), and reports the distribution of

```
R = ‖Z_d + λ(Z_s − Z_d)‖₁ / ‖Z_d‖₁
```

by denoising step, block, and query-row group, at λ ∈ {1, 1.5, 2, 3} on the
all×text β=2 locus, for `original`, `language`, `noise` (and `robot` if cheap).
λ=1 is in the set because it is `original`'s own optimum: the cap has to be
provably near-inert there.

**τ* rule, differential (this is what the plateau needs).** The cap must be
*dose-selective*: near-inert where the ladder is still climbing and active where
it turns over. Pick the smallest τ ≥ 1 from {1.0, 1.1, 1.25, 1.5, 2.0, 2.5} with

```
clip-rate(λ=1) ≤ 5%   and   clip-rate(λ=1.5) ≤ 10%   and   clip-rate(λ=3) ≥ 20%
```

over (head, query row, block, step) slots. The first two conditions are the
inertness requirement at the two axes' own optima (`original` λ=1, `noise`
λ=1.5): whatever the cap does at λ=2 must not come from having already bitten
where the ladder was healthy. A τ that clips heavily at λ≤1.5 only
rescales the dose axis (Failure A by construction); a τ that never clips at λ=3
is Failure B. If no candidate satisfies all three, **the row is not launched** — the
arms would be re-runs of collected ones, and this repo does not run
interventions that cannot differ from their control (CLAUDE.md).

### 6.1 The measurement (libero_10, 3 episodes per axis)

Run at HEAD, all×text β=2, trajectory λ=2. Per axis, 3.9-11.3M query-row slots.
The `language` ladder first:

| λ | mean R | p90 | p99 | max | clip @τ=1.5 | @τ=2 | @τ=2.5 |
|---|---|---|---|---|---|---|---|
| 1.0 | 1.250 | 1.70 | 2.45 | 6.6 | 17.7% | 3.6% | 0.8% |
| 1.5 | 1.468 | 2.15 | 3.30 | 9.7 | 37.2% | 13.2% | 4.7% |
| 2.0 | 1.717 | 2.65 | 4.20 | 12.9 | 53.8% | 26.1% | 12.1% |
| 3.0 | 2.261 | 3.70 | 6.05 | 19.2 | 73.8% | 51.1% | 32.3% |

Three readings, all of which shaped §7:

* **Failure B is ruled out.** Sharpening this DiT's text cross-attention does
  inflate per-query output magnitude — already at λ=1 the mean row is 1.24× the
  dense branch, and the tail reaches 6×. There is something for a cap to hold.
* **τ\* = 2.5** is the only candidate satisfying the rule (0.8% / 4.7% / 32.3%
  against ≤5% / ≤10% / ≥20%); τ=2 misses on the λ=1.5 leg (13.2% vs 10%). At the
  shared operating point λ=2 it caps 12.1% of query rows — a real intervention,
  which is what the delivery assertion demands. It is
  also the paper's own default for every family — arrived at here from our own
  distribution, which is the only reason to trust it.
* **The magnitude story is not the late-step story.** Clip rate is nearly flat
  across the four denoising steps (step 0 hottest by ~2pp at every τ), so the
  campaign's late-step gain is not explained by magnitude blow-up being
  concentrated late. The gradient that does exist is across BLOCKS: at λ=2, τ=2
  the early text blocks clip 40/55/38% (blocks 0/4/8) against 6% at block 28 —
  the cap is mostly an early-block intervention, which is a locus prediction the
  campaign has never tested directly.

### 6.2 Cross-axis: the pre-registered check came out FLAT

The check was: at matched (λ, τ) the clip rate should rank **noise > language**,
because the axis where the dose turns over should be the axis whose activations
run hot. Measured, at λ=2, τ=2.5, next to each axis's plain λ=1 → λ=2 step:

| axis | plain λ=1 → λ=2 | mean R @λ=2 | clip @τ=2.5 | τ* by the §6 rule |
|---|---|---|---|---|
| original | **−2.25pp** (falls) | 1.711 | 12.0% | 2.5 |
| noise | **−1.37pp** (peaks at 1.5) | 1.748 | 12.5% | 2.5 |
| language | **+1.95pp** (climbs) | 1.717 | 12.1% | 2.5 |
| robot | **+1.55pp** (climbs) | 1.786 | **14.3%** | 2.5 |

Literally the prediction holds — noise (12.5%) does exceed language (12.1%) — by
0.4pp, while the hottest axis of all is `robot`, which *wants* λ=2. **The ordering
does not track the sign of the dose response.** The two axes with the most
opposite ladders (original falling, language climbing) have statistically
indistinguishable magnitude distributions (12.0% vs 12.1%, mean R 1.711 vs 1.717).

**What this kills.** Per-query magnitude inflation does not explain why the best
λ differs by axis. The simplest mechanistic story for a selective cap — "the
harmed axis is the one whose activations run hot" — is contradicted, and with it
the stated basis for funding `noise` (20 h/arm) ahead of the cheap axes.

**What it does not kill.** The plateau hypothesis needs selectivity in DOSE, not
across axes, and that is exactly what the ladder shows: 0.8% of rows capped at
λ=1 against 32% at λ=3, on every axis. The dose gradient is ~40× the cross-axis
spread (0.8→32 points versus 12.0→14.3). Whether the capped 12% of rows are the
ones carrying the harm — selectivity *within* an axis, across query rows — is
untouched by this measurement and has no cheap proxy: it is what Tiers 1-2 test.

**τ\* = 2.5 on all four axes independently.** The rule picks the same threshold
everywhere, which is the empirical precondition the "one shared setting" design
needs — a τ that had to be retuned per axis would only have moved the tuning
problem, not solved it.

## 7. Arms

Family and locus are the ones the question is about: all×text sharp-softmax,
`--pladis-method softmax --pladis-beta 2.0 --pladis-qgroup all --pladis-kind text`
(the entmax ladder stays alongside on every axis as the branch-swap control).
Every NAG arm below is **ρ=1** — normalization only, condition ①, per §5.2.

**The setting is shared, not tuned per axis.** One λ AND one τ* for all axes:
choosing τ per axis would only replace λ-tuning with τ-tuning, and the claim
would be empty. λ=2 is the shared operating point — the dose two axes already
want, and the one the other two fall off at.

**Tier 1 — "one setting, four axes" (~32 h, 4 arms).** One arm per axis:
`allxt-temp20-nagn-l20` (λ=2, **τ=2.5**, ρ=1) on `original`, `language`, `robot`,
`noise`. Three are wired (`original` carries it inside Tier 2); `noise` stays
OUT of its driver — §6.2's ranking check came out flat, which removed the stated
reason to spend 20 h/arm there before the cheap axes have said anything. Every
comparator is already collected, so the whole cross-axis claim costs four arms:

| axis | new arm reads against | what it has to do | ~cost |
|---|---|---|---|
| original | plain λ=1 (this axis's best) | recover +2.25pp | 1.3 h |
| noise | plain λ=1.5 (this axis's best) | recover +1.81pp | 20 h |
| language | plain λ=2 (its own best) | not lose the +2.54pp | 5.1 h |
| robot | plain λ=2 (its own best) | not lose the +1.48pp | 5.2 h |

Primary endpoint per axis: paired `nag(λ=2) − plain(best λ)`, i.e. the regret of
using the shared setting instead of that axis's tuned one (§5.1.3). Success is a
non-inferiority pattern across all four at once — recovery on the two falling
axes without giving back the gain on the two climbing ones.

**Tier 2 — the plateau curve, on the cheap axis first (~6.5 h, 4 arms).**
`original` is 400 episodes (≈1.3 h/arm) and falls fastest, so the whole shape
story fits in one afternoon: NAG at λ ∈ {1, 1.5, 3} (λ=2 is the Tier-1 arm) plus
the one missing plain rung `allxt-temp20l30`. This is where the DiD of §5.1.1 is
computed, and where λ=1 doubles as the **inertness control**: a selective τ* must
leave the axis's own optimum essentially unchanged.
Power is honest here: n=400 gives SE ≈ 1.5pp, so `original` alone cannot settle a
2pp question — it can show whether the cap does anything at all, cheaply, before
`noise` is funded.

**Tier 3 — the same curve on the climbing side (~15 h, 3 arms).** `language` at
λ ∈ {1.5, 3} plus plain `allxt-temp20l30`. Run once Tiers 1–2 give a direction.

Ordering: Phase 0 (**done**, §6.1-6.2) → Tier 2 + Tier 1's cheap axes (original,
language, robot ≈ 17 h) → `noise` (20 h) on what those 17 h show, not on the
ranking gate (which is spent) → Tier 3.
`camera` and `layout` stay out: the family never beats vanilla on either, so a
guardrail has no gain to protect there.

Refinement (condition ②) enters after a direction exists, as ρ<1 at raised λ with
the iso-dose flat arm as its control (§2(b),(c)) — never as a plateau claim.

## 8. Gates (CPU, no checkpoint) — `experiments/verify_nag.py`

Mirroring `verify_step_schedule.py`'s structure; all must print PASS:

- **A.** τ=off, ρ=1 is `torch.equal` to the current processor at the same λ, for
  entmax-1.5 and the sharp-softmax branch, with and without a qgroup split.
- **B.** Closed form on synthetic tensors: R matches a per-(head, row) reference
  computed independently; the cap fires exactly on `R > τ`; on uncapped rows the
  output equals `Z_d + ρλ(Z_s − Z_d)` to fp tolerance (the §2(b) identity).
- **C.** τ=1 with an inflating blend reduces every capped row's L1 norm to
  `‖Z_d‖₁`, and leaves rows with R ≤ 1 untouched.
- **D.** qgroup: rows outside the selected group are `torch.equal` to dense —
  bit-exact, for ρ ∉ {0, 1} where the naive implementation would not be.
- **E.** Step schedule composition: zero-weight steps stay bit-identical to
  vanilla with NAG armed; a weighted step carries the cap.
- **F.** Defenses raise: τ < 1, ρ outside (0,1], ρ<1 with τ=off, NAG with
  `--pladis-scale 0`, NAG requested on the pi05/smolvla tracks.
- **G.** Census/delivery: `assert_nag_delivered()` reports clip rate per step and
  raises when the census shows the cap never fired — the arm would be a silent
  re-run of its own control, and would burn 1,537 episodes proving nothing.

On-checkpoint gates stay as they are: `verify_base0_parity.py` (λ=0 bit-parity,
unaffected — NAG is rejected at λ=0) plus eval_arm's own delivery warm-up, which
gains the NAG census line next to the schedule census.

## 9. What would falsify the port

Named in advance, and mapped onto §5.2:

- **R never approaches any τ ≥ 1** (Failure B) → the cap is inert at this locus;
  §6 stops the row before it is launched, and the measurement itself is the
  result: sharpening this DiT's cross-attention does not inflate per-query
  output magnitude, so a magnitude guardrail has nothing to hold.
- ~~**Clip rate does not rank noise > language**~~ → **measured 2026-08-26, and it
  came out flat** (§6.2): 12.0/12.1/12.5/14.3% across original/language/noise/robot,
  with a climbing axis on top. Magnitude does not explain the axis-dependent
  optimum. The mechanism now rests entirely on dose selectivity (0.8% → 32% from
  λ=1 to λ=3), which the same measurement supports strongly.
- **NAG flattens both signs** (Failure A) → the plateau widened because the
  language gain was capped away. The regret statistic (§5.1.3) is what exposes
  this; a flat curve at a lower ceiling must be reported as a trade, not a fix.
- **Plain λ=3 does not fall off** on any axis → there was no overdose to rescue;
  the plateau is already wide and the premise was a two-point artifact.
- **Phase 1 arms land on their parents within noise** → the cap changes nothing
  where the intervention already helps, i.e. our regime sits below the
  instability NAG was built for.

A result that survives all of these — noise's λ=2 recovering while language's
λ=2 gain stands — is the claim the campaign is for: **one λ, no per-axis
tuning**, with the cap doing the axis-dependent work automatically.
