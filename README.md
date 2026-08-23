# pladis-vla

**Where should test-time attention sparsification act in a VLA policy?**
This repository applies PLADIS-style sparse cross-attention interventions to
the action-head DiT of a vision-language-action model (GR00T N1.7), factorized
over **query groups (state / action tokens) × key modalities (text / image)**,
and evaluates each intervention *locus* on the LIBERO-plus robustness
benchmark with paired, episode-level statistics.

The codebase is a self-contained evaluation harness: scheduling, perturbation
delivery, seeding, rollout, and logging are all owned by this repository, and
every delivery/parity claim is backed by an executable verification gate.

**Design space.** Interventions are evaluated over a grid of models × axes:
models {GR00T N1.7 (implemented); π0.5 (openpi track, implemented — π0 dropped);
SmolVLA and GR00T N1.5 (planned)} × perturbation axes {language, layout,
robot, original}. Campaigns are distributed across machines at
(model × axis) granularity — all arms of one comparison share one machine
and stack (docs/SETUP.md §0).

The two models expose **different intervention geometry**, and that difference is
itself a result: GR00T N1.7 has a separate cross-attention module, so the locus
factorizes as query group × key modality (§1.2); π0.5 has none — its action
tokens attend jointly over `[image | language | suffix]` in one softmax — so the
query axis collapses and the locus is the key sub-block alone (§1.3).

---

## 1. Method

### 1.1 PLADIS blend

PLADIS ([Kim & Sim, ICCV 2025](https://arxiv.org/abs/2503.07677)) is a
training-free, inference-time intervention that replaces the cross-attention
map with a dense/sparse extrapolation:

```
attn = dense + λ · (sparse − dense),   dense = softmax(z),  sparse = f(β · z)
```

`λ = 0` recovers the vanilla model; `λ = 1` substitutes the sparse map.
The sparse transform `f` (`entmax15` / `sparsemax` / `softmax`), the blend
strength `λ`, and the sparse-branch inverse temperature `β` (paper suppl.
G.1: `softmax(β·z)` with `β > 1` is the temperature-sharpened control,
τ = 1/β) are all exposed as flags (§6.1).

### 1.2 Intervention loci in GR00T N1.7

The GR00T N1.7 action head is an alternating DiT (`AlternateVLDiT`): odd
blocks self-attend over the `[state; action]` token sequence; even blocks
cross-attend to vision-language tokens, alternating between **text-key** and
**image-key** blocks. The hook (`pladis/attn_gr00t_n17.py`) restricts the blend
along two axes:

| axis | values | mechanism |
|---|---|---|
| query group (`--pladis-qgroup`) | `state` / `action` / `all` | row slice of the attention map (`[state(0:n); action(n:)]`) |
| key modality (`--pladis-kind`) | `text` / `image` / `all` | selection of even cross-blocks by the alternation rule |

Cells compose: `--pladis-cells actionxtext,stateximage` installs a different
query group per key kind in one pass (kinds must be disjoint).

A third axis is **time**. The head is a flow-matching policy integrated with
N=4 forward-Euler steps at t ∈ {0, .25, .5, .75}
(`num_inference_timesteps`; `dt = 1/N`, no noise re-injection), and every block
runs once per step, so a locus also has a position in the denoising trajectory:

| axis | values | mechanism |
|---|---|---|
| denoising schedule (`--pladis-schedule`) | `all` / one weight per step, e.g. `1,1,0,0` | per-step multiplier on λ: **λ_i = `--pladis-scale` · w_i**. A DiT forward pre-hook publishes the step index (recovered exactly from the discretized timestep bucket); a zero-weight step takes the vanilla fused-SDPA path |

The weights are read against a λ **base**, which stays the dose knob
(`--pladis-scale`). The shape row, and its instantiation at the λ=2 base the
language campaign uses:

| shape | weights `w` | Σw | effective λ per step at base 2 |
|---|---|---|---|
| vanilla | `[0,0,0,0]` | 0 | — (no hook) |
| all | `[1,1,1,1]` | 4 | `[2,2,2,2]` |
| early | `[1,1,0,0]` | 2 | `[2,2,0,0]` |
| late | `[0,0,1,1]` | 2 | `[0,0,2,2]` |
| increasing | `[0,0.5,1,1.5]` | 3 | `[0,1,2,3]` |
| decreasing | `[1.5,1,0.5,0]` | 3 | `[3,2,1,0]` |

The two same-Σw pairs (early/late, increasing/decreasing) hold total dose fixed
and vary only *where in the trajectory* it is spent, so shape is separable from
dose; the all-steps parent (Σw=4) is the reference for both. `[1,1,1,1]` is
bit-identical to the unscheduled arm at the same scale and needs no separate run
(`verify_step_schedule.py` gate F). Training-time t is drawn as
`(1 − Beta(1.5, 1))·0.999`, which puts 35.1 / 29.6 / 22.9 / 12.4 % of the
training mass in the four intervals the inference grid spans — the halves are not
equally well trained, and that asymmetry is part of what the contrast measures.

### 1.3 Intervention loci in π0.5

π0.5 has **no cross-attention module**. Its action ("suffix") tokens attend
jointly to a concatenated key sequence through one softmax in the Gemma expert
(a FLUX/MMDiT-style joint attention), so the blend is applied to a **column
sub-block** of the attention map rather than to a whole module. The layout at a
suffix denoising step, measured on the official checkpoint by
`verify_pi05_delivery.py` — `(query_len, key_len) = (10, 978)`:

```
                        KEY  (978 columns)
        ┌─────────────────────┬──────────────┬─────────────┐
        │  image  [0:768]     │ language     │ suffix      │
        │  3 slots × 256 tok  │ [768:968]    │ [968:978]   │
 QUERY  ├─────────────────────┼──────────────┼─────────────┤
 action │   --pladis-kind     │ --pladis-    │             │
 10 rows│      image          │  kind text   │             │
        └─────────────────────┴──────────────┴─────────────┘
                └────────── --pladis-kind prefix ──────────┘ (excl. suffix)
                └──────────── --pladis-kind all ───────────────────────────┘
```

Those are the **allocated** widths — the slice bounds. The **attendable** widths
are smaller, because both blocks are padded and the padding is masked out
(measured 2026-07-28):

| block | allocated | attendable | why the rest is masked |
|---|---|---|---|
| image | 768 | **512** | LIBERO has two cameras; `libero_policy.py:62-70` fills `right_wrist_0_rgb` with zeros and sets `image_mask=False` (it is `True` only for `PI0_FAST`) |
| language | 200 | **~9–20** | `PaligemmaTokenizer` pads to `max_token_len=200` with `mask=False`; real LIBERO instructions tokenize to 9–20 |

The blend is correct either way — masked columns carry `dense ≈ 0`, so the
mass term `m = dense[…, lo:hi].sum()` integrates only attendable mass — but any
**dose** quoted as a fraction of the allocated width understates the
intervention, ~10× on language. See gate 5c in §5.

Two consequences:

1. **The query axis collapses.** `pi05_libero` sets `discrete_state_input=False`,
   so proprio state never enters the transformer at all — the suffix is
   action-only. Every π0.5 arm is implicitly `action × <keys>`, and
   `--pladis-qgroup state` is rejected outright. This also makes the language
   block **pure instruction**: the paper's π0.5 discretizes proprio state into
   text tokens and openpi makes that the default (`pi0_config.py:38-39`), so on
   a stock π0.5 the `text` locus would mean "language + proprioception". The
   LIBERO config is the one that opts out, which is what lets `kind=text` be
   read as a clean language locus here.
2. **Sub-blocks are sharpened mass-preservingly.** For `text` / `image` /
   `prefix` the block's total softmax mass `m` is preserved and only
   redistributed within it — the pattern of the official FLUX code
   (`PLADIS/pipeline/pipeline_flux.py:104-113`).

| π0.5 arm | columns | relation to the original PLADIS |
|---|---|---|
| `kind=text` | `[768:968]` | **direct port** of the FLUX intervention: generative queries × conditioning keys, one block, mass-preserving |
| `kind=image` | `[0:768]` | **no upstream precedent** — in FLUX the image tokens are the *queries*, never the keys. This is the contrast that makes "locus matters" testable |
| `kind=prefix` | `[0:968]` | all conditioning as ONE block, so mass may migrate between image and language while the suffix columns stay dense. *Not* the same as text+image applied together, which would preserve each modality's mass separately |
| `kind=all` | `[0:978]` | **off-method.** This is the SDXL operation (`pipeline_sdxl.py:93-99`, a whole-map blend where `attn2` genuinely *is* cross-attention) applied to a joint-attention row, so it also sparsifies action↔action self-attention. PLADIS is defined "within all cross-attention modules"; treat this as a reference arm, not the "both modalities" arm |

The GR00T `state × …` cells have no π0.5 counterpart, so the π0.5 track
corresponds to the **action row** of the 2×2 grid of §1.2, extended along λ.

### 1.4 Arm vocabulary

| arm | flags | role |
|---|---|---|
| `vanilla` | (none) | stock model, fused-SDPA attention |
| `base0` | `--pladis-install --pladis-scale 0` | hook installed, λ=0 → delegates to the same fused SDPA; bit-identical to vanilla (install-plumbing control) |
| eager-dense control | `--pladis-install --pladis-scale 1.0 --pladis-method softmax` | dense softmax computed on the hook's eager path; numeric-path-matched baseline for the λ>0 arms |
| locus cells | `--pladis-scale λ --pladis-qgroup {state,action,all} --pladis-kind {text,image,all}` | the interventions under study |
| mixed cells | `--pladis-scale λ --pladis-cells <cell,cell>` | per-kind query groups |
| temperature control | `--pladis-scale 1.0 --pladis-method softmax --pladis-beta β` | sharpened-softmax counterpart to a sparse cell |
| step-scheduled cell | `--pladis-scale λ --pladis-qgroup … --pladis-kind … --pladis-schedule 1,1,0,0` | a locus cell with a per-step λ profile, λ_i = λ·w_i (§1.2); its all-steps parent and its same-Σw mirror are the paired references |

**The control arms differ between tracks, and not arbitrarily.** GR00T's vanilla
runs fused SDPA while λ>0 must materialize weights on an eager path, so `base0`
and the eager-dense arm exist to bracket a real kernel-level numeric difference
(§7). π0.5's vanilla is *already* on the eager path (openpi forces
`attn_implementation="eager"` on the expert every suffix step), so:

- `base0` is bit-identical to vanilla (`verify_pi05_hook.py` gate A, re-confirmed
  end-to-end by `verify_pi05_parity.py` check (b): eplogs equal over 10 episodes)
  and is verified there instead of consuming a 1,537-episode arm — the same call
  the n17 robot axis already made;
- the **eager-dense control arm is kept anyway**, for a different reason than on
  n17. With `method=softmax, β=1` the identity `m·p == dense[sub]` makes the
  blend collapse to dense for any λ, so the arm should be redundant — and at
  module level it is: check (c) finds **0.0000%** of bf16 attention elements
  differing, `max|dw| = 0`. But end-to-end it is not. Over 10 libero_10
  episodes, dense diverged from vanilla in **every one** (n_steps 197→186,
  246→241, 270→259, …; SR 0.900 → 1.000). The residual is float32
  reassociation inside the λ>0 branch — below bf16 resolution per call, and
  amplified by ~45k closed-loop attention calls per episode.

So π0.5 *does* carry a numeric term, just not GR00T's: not a fused-vs-eager
**kernel** difference, but the reassociation the λ>0 code path itself
introduces. It was measured once at sweep scale — `base_dense − vanilla` =
**−0.91 pp** over 1,537 episodes, the same sign as in the 10-episode gate.

**Reporting reference: vanilla** (operator decision, 2026-07-29). The λ=1
`base_dense` eplogs are kept and reported as an extra arm — they are the one
direct measurement of the term's size — but the ladder arms (λ=1.5, 2.0) do not
carry their own control. Read the two contrast families accordingly:

- `text − vanilla`, `image − vanilla` carry the numeric term **alongside** the
  intervention, so they understate a beneficial locus effect. At λ=1 that gap is
  visible: `text − vanilla` = +0.33 pp against `text − base_dense` = +1.24 pp.
- **`text − image` is unaffected** — both arms run the identical λ>0 path, so the
  term cancels *within* the pair. This is the study's primary contrast (§6.3),
  and it is the one that survives Bonferroni.

The eager-dense arm was written out of the design, reinstated (2026-07-28) by
the measurement intended to retire it, then demoted back to a reference
measurement once its size was known. The `if it turns out to matter, the arm
goes back in` clause is why check (c) measures rather than assumes.

## 2. Benchmark and protocol

- **Model**: [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)
  (one fine-tuned checkpoint per suite), served through the **official
  `Gr00tPolicy`** path (`harness/model_gr00t.py`).
- **Benchmark**: [LIBERO-plus](https://github.com/RLinf/LIBERO-plus)
  curated perturbation suites over the four LIBERO task suites
  (libero_10 / goal / object / spatial). Supported axes:

| axis | episodes/arm | perturbation |
|---|---|---|
| `language` | 1,537 | instruction rephrasing only (gate-verified: bddl differs in the `(:language)` line alone) |
| `layout` | 1,525 | scene changes — added distractors, moved objects/fixtures (BDDL placement resampling) |
| `robot` | 1,550 | robot init-state offsets, 5 strength levels 0.1–0.5 rad (runtime `Panda{k}` swap) |
| `noise` | 1,601 | obs-side corruption of the agentview stream, 5 families × 10 severities: motion blur 336, gaussian blur 341, zoom blur 288, fog 272, glass blur 364 |
| `camera` | 1,599 | agentview re-posing from the runtime `_view_` tail: orbit (yaw ±75°) 443, orbit_up (yaw + 15° elevation) 549, zoom (pivot-ray push-out 115–200 %) 313, reaim (bearing-only ±10°) 294 |
| `none` (original) | 400 | unperturbed per-task baselines, init states 0–9 |

- **π0.5 model**: the official `pi05_libero` checkpoint
  (`gs://openpi-assets/checkpoints/pi05_libero`, converted to PyTorch), served
  through the **official `openpi.policies.policy.Policy`** path
  (`harness/model_pi05.py`). RLinf is deliberately not on that path — see §7.
- **Rollout protocol.** Each track uses *its own model's* official protocol,
  because the anchor gate has to reproduce a published number; a shared protocol
  would make both anchors unreproducible by construction.

| | GR00T N1.7 (official Isaac-GR00T) | π0.5 (official openpi) |
|---|---|---|
| env-step cap | 720 | per suite: spatial 220 / object 280 / goal 300 / libero_10 520 |
| decoded chunk | 16 | 10 |
| executed (receding horizon) | 8 | 5 |
| settle steps after reset | 10 | 10 |
| denoising steps | — | 10 (openpi default) |

  Both terminate on first contact with success. Primary metric: `success_once`
  per episode. The cost of this choice is that cross-*model* absolute SR
  comparisons carry a protocol term; within-model arm contrasts do not.
- **Pairing**: all arms of an axis share the same seed-0 schedule, so
  episodes are paired across arms by construction (asserted at load time).

## 3. Repository layout

```
pladis/        attention hooks
  attn_gr00t_n17.py        weight-space hook (faithful to the official PLADIS
                       code path: eager blend at λ>0, native fused SDPA at
                       λ=0); qgroup/kind/cells gating
  attn_pi05.py         π0.5 joint-attention hook (FLUX-style: sharpen ONE key
                       sub-block, mass-preserving); explicit-flag
                       install_pladis() + assert_delivered() (§1.3)
harness/       evaluation loop, fully owned
  env.py               curated schedules, per-axis delivery, deterministic
                       per-episode env seeding
  rollout.py           obs→policy→step loop, per-chunk noise pinning; the
                       model-specific obs/action conversion is delegated to the
                       adapters' wrap_obs / predict_chunk / to_env_actions
                       (model_base.py), so all tracks share ONE loop
  model_base.py        ModelAdapter contract + conversions shared by adapters
  registry.py          one ModelSpec per VLA: venv, loader, hook module,
                       checkpoint root, protocol defaults (--model resolves here)
  model_gr00t.py       official Gr00tPolicy adapter
  model_pi05.py        official openpi Policy adapter — un-compiles
                       sample_actions, and to_env_actions is the identity
                       (π0.5 already emits LIBERO's [-1,+1] gripper)
  model_smolvla.py     SmolVLA adapter (lerobot track)
  eplog.py             per-episode TSV ledger (crash-safe, resume source,
                       arm-signature guarded)
  video.py             per-episode mp4 of the model's two camera views
experiments/   entry points
  run.sh               environment wrapper (all commands go through it);
                       --venv selects the model track's interpreter
  load_machine_env.sh  machine config loader (defaults + machine.env override)
  machine.env.example  per-machine config template (copy to machine.env)
  eval_arm.py          single-arm evaluator, ALL tracks (--model) — anchors,
                       parity checks, and sweeps share this one code path
  sweep_n17_*.sh       n17 sweep drivers (language / original / layout / robot /
                       noise / camera); arm-outer, suite-inner
  sweep_pi05_*.sh      π0.5 sweep drivers (+ sweep_pi05_common.sh); SUITE-outer,
                       one suite per GPU (§6.2)
  verify_*.py          verification gates (§5)
  diag_pi05_support.py entmax support-size measurement (§5 gate 5c)
  smoke_model.py       GPU smoke test (registry-driven, --model)
  smoke_pi05.py        GPU smoke + instruction delivery asserted at the tokenizer
scripts/       externals.lock (pinned sibling-checkout SHAs) + clone_externals.sh
analysis/      analyze.py [--model n17|pi05] --language|--layout|--robot|
               --noise|--camera (paired McNemar)
docs/          benchmark.md — cross-checked benchmark facts
results/       (gitignored) eplogs, videos, driver logs
```

## 4. Installation

- 1× CUDA GPU (bf16); ~13–17 s/episode. ~30 GB for checkpoints; ~5–10 GB per
  sweep if video recording is enabled.
- Machine setup is fully described in **`docs/SETUP.md`**: external checkouts
  pinned by SHA (`scripts/externals.lock` + `scripts/clone_externals.sh`),
  per-track venv recipes and version pins (`requirements.txt` holds the
  reference set), checkpoint downloads, and the per-machine config file
  (`experiments/machine.env`, gitignored — copy from `machine.env.example`).
- The attention hooks are line-for-line ports against pinned library
  versions (diffusers 0.35.1 for GR00T; transformers 4.53.2 for the openpi
  track) — re-run the parity gates of §5 after any upgrade.

### 4.1 Execution wrapper

Every Python entry point is invoked through `experiments/run.sh`, which
selects the model track's venv, and sets EGL rendering, the ImageMagick
library path, `PYTHONPATH`, and the HF token:

```bash
bash experiments/run.sh experiments/smoke_model.py                 # default venv: gr00t
bash experiments/run.sh --venv openpi experiments/verify_externals.py
```

Bypassing the wrapper (inline env prefixes, direct `python`) is unsupported.

## 5. Verification gates

The harness treats delivery and parity claims as testable artifacts. On a new
machine or after dependency changes, run in order:

0. **Externals** — `verify_externals.py`: sibling checkouts match
   `scripts/externals.lock` and the active venv matches the critical version
   pins.
1. **Anchor** — unperturbed LIBERO-10 reproduces the published model-card
   success rate within sampling error: `eval_arm.py --axis none --episodes 100`.
2. **Instruction delivery** — `smoke_model.py` asserts a language-variant
   episode reaches the model with the rephrased instruction (also logged per
   episode in the eplog `instruction` column).
3. **λ=0 parity** — `verify_base0_parity.py`: hook-installed λ=0 is
   bit-identical to the uninstalled model (module-level `torch.equal` on the
   N1.7 attention configuration + full-rollout eplog equality).
4. **Per-axis delivery gates** —
   `verify_language_axis.py` (variant bddl ≡ base outside the `(:language)`
   line for all 1,537 variants; neutral runtime tail; bit-identical paired
   scenes), `verify_layout_axis.py` (determinism, perturbation delivery,
   silent-nullification regression, cross-process pairing),
   `verify_robot_axis.py` (wiring, delivery mechanism, determinism, level
   scaling), `verify_noise_axis.py` (wiring, agentview-only corruption,
   per-episode reseed determinism), `verify_camera_axis.py` (wiring +
   non-neutral/unconfounded tails, delivered pose ≡ closed-form prediction
   from the tail, agentview-only isolation, determinism, all four suites;
   `--mode video` renders the unperturbed episode and all four viewpoint
   families side by side under identical scripted actions, re-asserting the
   isolation invariants per frame before it writes).
4b. **Step-schedule gate (CPU)** — `verify_step_schedule.py`: the timestep-bucket
   ↔ step-index map is exact for N ∈ {1,2,3,4,8,16}; a zero-weight step is
   `torch.equal` to diffusers' `AttnProcessor2_0` while a weighted step equals a
   plain processor at λ = scale·w; `[1,1,1,1]` reproduces the unscheduled arm
   bit-for-bit and the default (`all`) path is unchanged; the DiT pre-hook
   publishes the right index and rejects a mixed-timestep batch; wrong-length,
   all-zero, conflicting and unresolvable schedules raise; `assert_delivered()`
   catches a never-fired probe and a weighted step the loop never reaches. Needs
   no checkpoint and no GPU.
5. **π0.5 hook smoke (CPU)** — `verify_pi05_hook.py`: λ=0 and prefix passes
   bit-identical to stock gemma eager attention; kind blend ≡ the official
   FLUX mass-preserving formulation; row/block-mass preservation; β=1
   softmax collapse; geometry/qgroup/`assert_delivered` defenses; real
   `GemmaAttention` dispatch interception. Needs no openpi model, no GPU and no
   external checkout — on a fresh machine this is the cheapest first milestone.
5b. **π0.5 on-model delivery** — `verify_pi05_delivery.py`: the blend actually
   fires on a real chunk (`assert_delivered`), at exactly
   `(query_len, key_len) = (10, 978)`; the large-query PaliGemma prefix pass
   stays dense; the expert config reads `eager` *after* a forward;
   `sample_actions` is not a `torch.compile` wrapper. This is the gate that
   catches the π0.5-specific silent no-op — a λ>0 arm that runs as vanilla
   while the eplog and `.arm` sidecar both claim an intervention.
5c. **π0.5 support size (measurement, not pass/fail)** — `diag_pi05_support.py`:
   how many columns entmax15 actually keeps per block. Read it before choosing
   λ: an intervention that keeps 190 of 200 language columns cannot produce an
   interpretable null result, and it is also what decides whether `kind=prefix`
   earns an arm (does image's 768 columns crowd language's 200 out entirely?).
   Measured 2026-07-28, λ ∈ {1, 1.5, 2}, libero_10 language variants:

   | kind | allocated | kept (median) | of **attendable** |
   |---|---|---|---|
   | text | 200 | 1 | ~7 % (of ~15) |
   | image | 768 | 8 | 1.6 % (of 512) |
   | prefix | 968 | 2 | — |
   | all | 978 | 3 | — |

   Two readings that the raw output does not give you. **(i) Read the dose
   against the attendable width, not the allocated one** (§1.3): the tool
   divides by block width, so its "0.5 % of 200" for `text` is really ~7 % of
   the ~15 real tokens. The dose is aggressive, not marginal — the opposite of
   the failure mode this diagnostic was written to catch. **(ii) Support is
   λ-independent**, and the tool says so: entmax15 acts on `β·logits`, and λ
   only mixes its output with dense, so λ scales the *strength* of the
   intervention while leaving *which* columns survive untouched. β is the only
   sparsity knob, and it is 1.0 on every phase-1 arm.

   The same tool also measures the **λ>1 extrapolation** (added 2026-07-29).
   PLADIS is parameterized `λ·sparse + (1−λ)·dense`, so above λ=1 the dense
   coefficient is negative and every column entmax dropped lands at
   `(1−λ)·dense < 0`:

   | λ | kind | block cols negative | min w (med) | negative mass | row sum |
   |---|---|---|---|---|---|
   | 1.0 | both | 0.0 % | +0.0000 | 0.0000 | 1.000000 |
   | 1.5 | `text` | 6.5 % | −0.0046 | 0.0206 | 1.000000 |
   | 1.5 | `image` | 65.9 % | −0.0021 | 0.0710 | 1.000000 |
   | 2.0 | `text` | 6.5 % | −0.0092 | 0.0415 | 1.000000 |
   | 2.0 | `image` | 66.0 % | −0.0043 | 0.1460 | 1.000000 |

   Normalization holds exactly (max row-sum deviation 2.7e−06), magnitudes stay
   small, and negative mass is linear in λ−1 — the closed-form prediction, so
   this doubles as a check that the port follows PLADIS's parameterization.
   The asymmetry is what matters for interpretation: `image` puts **66 %** of
   its block below zero against `text`'s 6.5 %, because `image` has 512
   attendable columns of which ~8 survive, while `text`'s ~185 masked columns
   carry `dense ≈ 0` and so contribute ~0 when negated. **At λ>1 the
   `text − image` contrast therefore carries a term λ=1 did not have** — and no
   dense control can absorb it, since an arm that never sparsifies has no
   negative lobe to match. It is a property of the pair, to be weighed when the
   ladder's locus contrasts are read.

   For `prefix` the result inverts the hypothesis. The question was whether
   image's 768 columns crowd language's 200 out; instead language survives
   (`zero-language rows` 9.9 %, median language mass share 1.0000) and **image**
   is what gets eliminated. So the planned ground for dropping `prefix` from
   phase 2 does not hold — the open question is now whether it is distinguishable
   from `text`. `all` behaves as designed for a reference arm: 23 % of rows lose
   language entirely and median language mass share falls to 0.83, i.e. mass
   migrates into the action↔action columns.

π0.5's counterparts to gates 1–3 are `sweep_pi05_original.sh` (anchor, must
reproduce openpi's published 98.8 / 98.2 / 98.0 / **92.4**), `smoke_pi05.py`
(instruction delivery, asserted at openpi's tokenizer rather than at our own
adapter boundary), and `verify_pi05_parity.py` (λ=0 bit parity at the real
shapes + `base0` ≡ vanilla end-to-end + the dense-collapse measurement — which
was written to retire the eager-dense arm and instead reinstated it, §1.4).

**π0.5 anchor, measured** (2026-07-27, A6000, 100 episodes/suite, `--axis none`,
seed 0, `exec_horizon=5`, `max_steps` 520 for libero_10 / 300 elsewhere):

| suite | this harness | openpi published (pi0.5 @ 30k) |
|---|---|---|
| libero_object | 100.0 | 98.8 |
| libero_spatial | 99.0 | 98.2 |
| libero_goal | 97.0 | 98.0 |
| libero_10 | **95.0** | **92.4** |

Every suite is within sampling error of the model card at n=100 (±~2–3 pts,
1σ), so the serving path of §7 reproduces the published table. libero_10 is the
load-bearing one — it is the long-horizon suite, the one the serving-route
bisect depressed to 45% on the GR00T track, and the gate the sweep is blocked
on: it must land in [87, 98].

Gates 3–4, 5b and the anchors need the GPU + simulator stack of §4; gate 5 is
CPU-only. All gates print `PASS` / `ALL GATES PASSED` and exit 0.

## 6. Running experiments

### 6.1 Single arm

```bash
bash experiments/run.sh experiments/eval_arm.py \
  --suite libero_10 --axis language --episodes 0 --seed 0 \
  --model-path ../models/GR00T-N1.7-LIBERO/libero_10 \
  --out results/my_arm_eplog.tsv \
  [--video-dir results/videos/my_arm] \
  [--pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text]
```

| flag | meaning |
|---|---|
| `--episodes` | `0` = every curated variant exactly once (seed-0 schedule); `N>0` = first N |
| `--out` | eplog TSV; doubles as the **resume ledger** — episodes already logged are skipped (a fully-logged arm exits before the model loads) |
| `--pladis-install` | hooks are installed only via explicit flags (never environment variables) |
| `--pladis-scale` / `--pladis-method` / `--pladis-beta` | λ, sparse transform, sparse-branch inverse temperature (§1.1) |
| `--pladis-qgroup` / `--pladis-kind` / `--pladis-cells` | intervention locus (§1.2) |
| `--pladis-n-state-tokens` | leading state query rows (N1.7: 1); defines the `state`/`action` split |
| `--pladis-schedule` | N1.7 only: per-denoising-step multiplier on λ (`all` default, or one weight per step, e.g. `1,1,0,0` / `0,0.5,1,1.5`). Zero-weight steps run vanilla, so this is the **time** coordinate of the locus (§1.2) |
| `--model` | `gr00t_n17` (default) or `pi05`; selects the loader **and** the hook |

The same evaluator runs the π0.5 track:

```bash
bash experiments/run.sh --venv openpi experiments/eval_arm.py --model pi05 \
  --suite libero_10 --axis language --episodes 0 --seed 0 \
  --max-steps 520 --exec-horizon 5 \
  --model-path ../models/pi05_libero --out results/my_arm_eplog.tsv \
  [--pladis-install --pladis-scale 1.0 --pladis-kind text]
```

π0.5-only flags: `--pladis-n-img-prefix` (768), `--pladis-n-lang` (200),
`--pladis-max-suffix-query` (100) — the key-axis geometry of §1.3, re-validated
against the live `key_len` at run time. `--pladis-qgroup`, `--pladis-cells` and
`--pladis-n-state-tokens` are rejected for `pi05` (no query axis), and
`--pladis-kind prefix` is rejected for `gr00t_n17` (it names a column span, not a
block set). `--pladis-method` defaults per track (`ent15max` / `entmax15`) and
accepts either spelling.

Eplog schema (TSV): `episode, task_name, base_task, init_state_id,
instruction, success_once, success_at_end, n_steps, wall_s`.

**Resume safety.** The arm's full configuration is written alongside the
eplog as `<out>.arm`. Resuming a run whose flags differ from that signature
aborts rather than appending a second arm's episodes into one file — the TSV
itself carries no arm identity, so such a mix would be invisible to
`analyze.py`. Eplogs written before this repository added signatures resume
with a warning.

### 6.2 Sweeps

```bash
nohup bash experiments/sweep_n17_<axis>.sh > results/sweep/driver_<axis>.out 2>&1 &
```

One driver per axis (`language` / `original` / `layout` / `robot` / `noise` /
`camera`). Each
driver enumerates its arm list explicitly — the script is the source of truth
for which arms an axis carries. Drivers refuse to start from a dirty working
tree (results must be attributable to a commit; each run appends
`code <git-describe>` to the eplog's `.arm` sidecar). All drivers are
resume-safe at episode granularity, so re-running a driver skips completed
arms and executes only what is new. Outputs follow
`results/sweep/n17_{axis}_{arm}_{suite}_eplog.tsv` (+ a same-named `.out` log
and, when enabled, `videos/n17_{axis}_{arm}_{suite}/ep#####_{S|F}_{task}.mp4`).

The `noise` driver additionally accepts **arm names as arguments**, which filter
its arm list (an unknown name aborts; no argument runs all eleven). That axis is
CPU-bound rather than GPU-bound — its per-frame corruption costs 448 ms in the
motion family against a 40 ms/step base, so 21 % of the episodes carry 77 % of
the wall time and one arm projects to 25–52 h depending on mean episode length
(driver header for the measurements) — and two drivers over disjoint arms nearly
halve the wall clock:

```bash
nohup bash experiments/sweep_n17_noise.sh vanilla actionxtext actionxtext15 actionxtext20 > results/sweep/driver_noise.out   2>&1 &
nohup bash experiments/sweep_n17_noise.sh allxtext allxtext15 allxtext20                  > results/sweep/driver_noise_b.out 2>&1 &
nohup bash experiments/sweep_n17_noise.sh allxt-late-l2                                   > results/sweep/driver_noise_late.out 2>&1 &
nohup bash experiments/sweep_n17_noise.sh allxt-inc-l2                                    > results/sweep/driver_noise_inc.out  2>&1 &
nohup bash experiments/sweep_n17_noise.sh allxt-temp20-late-l2                            > results/sweep/driver_noise_late_temp.out 2>&1 &
nohup bash experiments/sweep_n17_noise.sh allxt-temp20-late-l15                           > results/sweep/driver_noise_late_temp_l15.out 2>&1 &
```

(the 08-18 step-schedule pair, the 08-21 sharp-softmax mirror of its `late` arm
and that mirror's λ=1.5 rung are one-arm drivers; run them after the six above
are collected — two concurrent processes is still the ceiling.)

Two concurrent eval processes is the ceiling on a 31 GB box (~10.3 GB RSS each).
Interleaving cannot break pairing — every RNG source is pinned per episode, not
per process (§8) — and selecting from a file that is never edited mid-run avoids
the stale-byte-offset hazard of appending arms to a script bash is executing.

The π0.5 drivers are **suite-outer**: they take the suite as `$1` and walk that
suite's whole arm list, so one suite pins to one GPU.

```bash
CUDA_VISIBLE_DEVICES=4 nohup bash experiments/sweep_pi05_language.sh libero_10      > results/sweep/driver_lang_libero_10.out 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup bash experiments/sweep_pi05_language.sh libero_goal    > results/sweep/driver_lang_libero_goal.out 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup bash experiments/sweep_pi05_language.sh libero_object  > results/sweep/driver_lang_libero_object.out 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup bash experiments/sweep_pi05_language.sh libero_spatial > results/sweep/driver_lang_libero_spatial.out 2>&1 &
```

**Why per-suite GPU sharding is legitimate** under the one-machine-one-stack rule
(docs/SETUP.md §0): the analysis pairs episodes by `(suite, episode)`, so a
McNemar pair never crosses suites, and pinning a *suite* to a device keeps every
pair on one numeric stack. Pinning an *arm* to a device would put a numeric-path
difference **inside** a pair — that is what §0 forbids. The suite→GPU map must
therefore stay fixed across every arm of a campaign; each run appends
`dev <n> <name> <uuid>` to the `.arm` sidecar so it is auditable after the fact.

### 6.3 Analysis

```bash
python3 analysis/analyze.py --language
python3 analysis/analyze.py --original  # in-distribution control (no perturbation)
python3 analysis/analyze.py --layout    # + perturbation-category breakdown
python3 analysis/analyze.py --robot     # + strength-level (L1–L5) breakdown
python3 analysis/analyze.py --noise     # + corruption-family breakdown
python3 analysis/analyze.py --camera    # + viewpoint-family breakdown
python3 analysis/analyze.py --model pi05 --language
```

**Statistical conventions.** Primary test: paired McNemar over the pooled
episode pairing (`z = (n01 − n10)/√(n01+n10)`, no continuity correction),
over `success_once`, reported per contrast with discordant counts. Pooled
contrasts are primary; single-suite contrasts are interpreted conservatively
(closed-loop rollouts amplify numeric noise at the single-suite scale — §7).
`analyze.py` prints a Bonferroni-adjusted p over the pooled contrast family
and marks which contrasts survive it. The reference arm is **vanilla** on both
tracks. n17 additionally contrasts each λ>0 arm against the eager-dense control,
because there the numeric term is a fused-vs-eager *kernel* difference; on π0.5
that control is a reference measurement rather than a per-arm baseline (§1.4).
The primary contrast per track is the **locus** pair — `actionxtext −
actionximage` for n17, `text − image` for π0.5 — and it is the contrast the
numeric term cannot reach, since both arms of the pair run the identical path.

**π0.5 language axis, λ=1 (phase 1 complete, 2026-07-29).** 1,537 curated
variants per arm, seed-0 schedule, paired.

| arm | pooled | 10 | goal | object | spatial |
|---|---|---|---|---|---|
| `vanilla` (reference) | 87.12 | 89.0 | 72.0 | 92.4 | 96.4 |
| **`text`** | **87.44** | 89.8 | 71.2 | 92.4 | 97.7 |
| `image` | 85.82 | 88.8 | 69.5 | 91.2 | 95.1 |
| `base_dense` *(numeric-term reference)* | 86.21 | 86.9 | 71.5 | 91.8 | 95.9 |

| contrast | Δ | discordant | z | p | p<sub>bonf</sub> |
|---|---|---|---|---|---|
| **`text − image`** | **+1.63 pp** | 57 : 32 | +2.65 | 0.0080 | **0.048 ✱** |
| `text − vanilla` | +0.33 pp | 38 : 33 | +0.59 | 0.553 | 1 |
| `image − vanilla` | −1.30 pp | 39 : 59 | −2.02 | 0.043 | 0.260 |
| `base_dense − vanilla` | −0.91 pp | 24 : 38 | −1.78 | 0.075 | 0.452 |
| `text − base_dense` | +1.24 pp | 46 : 27 | +2.22 | 0.026 | 0.157 |
| `image − base_dense` | −0.39 pp | 48 : 54 | −0.59 | 0.553 | 1 |

The locus contrast is the only one of six to survive Bonferroni. Both arms run
the identical operation at the identical λ with the same mass preservation —
they differ *only* in which modality's keys are sharpened, which is also why
this is the contrast the numeric term of §1.4 cannot reach.

**How much the reference choice moves the reading.** Against vanilla the
intervention looks inert: `text − vanilla` = +0.33 pp. `base_dense` sits 0.91 pp
*below* vanilla, so measured against that path instead, `text` is +1.24 pp. The
two numbers bracket the same effect; the locus contrast avoids the question
entirely by cancelling the term within the pair.

Note also the severity spread on the baseline itself: language perturbation
costs libero_goal **−24.7 pp** against its task-matched original, versus −2.5
to −7.6 pp elsewhere. Pooled numbers average over a very uneven axis.

**λ dose ladder (complete, 2026-07-31).** All 8 arms × 4 suites = 12,296
episodes. λ ∈ {1.0, 1.5, 2.0} over the locus pair.

| arm | λ=1.0 | λ=1.5 | λ=2.0 |
|---|---|---|---|
| `text…` | 87.44 | 87.51 | 87.57 |
| `image…` | 85.82 | **84.84** | 85.04 |
| locus Δ | +1.63 | **+2.67 ✱** | **+2.54 ✱** |

| contrast | Δ | discordant | z | p | p<sub>bonf</sub> |
|---|---|---|---|---|---|
| **`text15 − image15`** | +2.67 pp | 70 : 29 | +4.12 | 3.8e−05 | **0.0006 ✱** |
| **`text20 − image20`** | +2.54 pp | 77 : 38 | +3.64 | 2.8e−04 | **0.0044 ✱** |
| **`image15 − vanilla`** | −2.28 pp | 36 : 71 | −3.38 | 7.2e−04 | **0.0115 ✱** |
| **`image20 − vanilla`** | −2.08 pp | 40 : 72 | −3.02 | 0.0025 | **0.0400 ✱** |
| `text15 − vanilla` | +0.39 pp | 38 : 32 | +0.72 | 0.473 | 1 |
| `text20 − vanilla` | +0.46 pp | 44 : 37 | +0.78 | 0.437 | 1 |
| `text15 − text` | +0.07 pp | 31 : 30 | +0.13 | 0.898 | 1 |
| `text20 − text15` | +0.07 pp | 35 : 34 | +0.12 | 0.904 | 1 |

**Read the direction, not just the significance.** The locus contrast is the
strongest signal in the campaign (z = +4.12), but it is not driven by `text`.
The `text` arm does not move with λ at all — 87.44 → 87.51 → 87.57, with every
step and every comparison against vanilla at p > 0.43. Of the four contrasts
that survive Bonferroni, **three are `image` getting worse and none is `text`
getting better**.

So the honest statement is *"sharpening the image keys hurts, and hurts more as
λ rises"*, not *"sharpening the language keys helps"*. Locus matters — but
asymmetrically, and the effect lives on the arm this repo included as the
contrast rather than the one motivated by the FLUX port.

**The λ>1 gap is partly mechanical.** §5 gate 5c measured the negative lobe:
at λ>1, `image` puts 66 % of its block below zero (negative mass 0.0710 at
λ=1.5) against `text`'s 6.5 % (0.0206). λ=1.0 is the only rung where both arms
carry *zero* negative weight, and it is also the rung where the locus contrast
is weakest (+1.63 pp, not surviving Bonferroni). That ordering is what a
negative-lobe explanation predicts, so the ladder's widening contrast should not
be read as a purely dose-dependent locus effect.

The layout axis (`sweep_pi05_layout.sh`) is the discriminating test: it perturbs
the *scene* and leaves the instruction intact, reversing which modality carries
the story. A locus effect that flips with the perturbed modality would be much
stronger evidence than either axis alone.

## 7. Determinism and numerical-path conventions

**Determinism.** Three seeding layers make runs bit-reproducible on a fixed
software/hardware stack: (i) the episode schedule is a seeded permutation;
(ii) the environment is reseeded before every reset from
`seed·1,000,003 + episode`; (iii) the flow-matching init noise is pinned
before every chunk inference from `episode_seed·100,003 + step`. Identical
noise streams across arms mean arms differ only through the intervention.
Recording videos does not perturb the RNG path (verified).

**Numerical paths.** The vanilla model computes attention with fused SDPA;
the λ>0 blend requires materializing attention weights and therefore runs on
an eager path — in the official PLADIS code exactly as here
(`attn_gr00t_n17.py` follows the official convention: native fused path at λ=0,
eager weight-space blend at λ>0). Closed-loop rollouts chaotically amplify
the rounding-floor difference between the two paths, so vanilla-vs-λ>0
contrasts carry a numeric-path term alongside the intervention. The harness
controls for it with the **eager-dense control arm** (§1.4), which runs the
identical eager path with a plain softmax.

**π0.5's term is different in kind, and had to be measured to be found.** openpi
forces the Gemma expert onto transformers' eager attention on every suffix step,
so vanilla, `base0` and the λ>0 arms all run one kernel — there is no
fused-vs-eager difference to bracket. λ=0 is indeed bit-identical to vanilla,
and the softmax/β=1 "control" does collapse to dense by an algebraic identity,
leaving only floating-point reassociation. The design predicted that residual
would be negligible; `verify_pi05_parity.py` check (c) measured it instead of
assuming it, and the prediction was wrong. At module level the residual is
literally zero in bf16 (0.0000% of elements, `max|dw| = 0`), but a closed-loop
episode issues ~45k of those calls (18 expert layers × 10 denoise steps × ~250
chunks), and over 10 libero_10 episodes dense diverged from vanilla in **all
ten** (SR 0.900 → 1.000). So the eager-dense arm stays — not to absorb a kernel
difference, but the reassociation the λ>0 code path introduces on its own.

The general lesson the two tracks share: **a λ>0 arm never runs on the same
numeric path as vanilla, whatever the reason**, so both tracks carry two
baselines.

**Serving path.** Both tracks are served through their model's *official* policy
object, not through RLinf's RL wrappers. This is a finding, not a preference: a
controlled bisect (2026-07-14; stock LIBERO env, official protocol, same
GPU/ckpt, 20 eps each) showed RLinf's
`GR00T_N1_7_ForRLActionPrediction.predict_action_batch` depressing libero_10 from
80% to 45%, matching a depressed 46% anchor at the time. Those wrappers exist to
serve *training* rollouts. `harness/model_pi05.py` therefore builds the openpi
`Policy` from `openpi.*` alone; RLinf stays importable only as the reference
implementation for that bisect.

## 8. Acknowledgements

This repository builds on:
[PLADIS](https://github.com/cubeyoung/PLADIS) (method),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) /
[LIBERO-plus](https://github.com/RLinf/LIBERO-plus) (benchmark),
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (model and serving),
[openpi](https://github.com/Physical-Intelligence/openpi) (π0.5 model, PyTorch
port and serving),
[entmax](https://github.com/deep-spin/entmax) (sparse transformations).

## License

Code in this repository is released under the Apache-2.0 license (see SPDX
headers). Model checkpoints and benchmark assets retain their upstream
licenses.
