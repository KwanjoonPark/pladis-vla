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
**image-key** blocks. The hook (`pladis/attn_gr00t.py`) restricts the blend
along two axes:

| axis | values | mechanism |
|---|---|---|
| query group (`--pladis-qgroup`) | `state` / `action` / `all` | row slice of the attention map (`[state(0:n); action(n:)]`) |
| key modality (`--pladis-kind`) | `text` / `image` / `all` | selection of even cross-blocks by the alternation rule |

Cells compose: `--pladis-cells actionxtext,stateximage` installs a different
query group per key kind in one pass (kinds must be disjoint).

### 1.3 Arm vocabulary

| arm | flags | role |
|---|---|---|
| `vanilla` | (none) | stock model, fused-SDPA attention |
| `base0` | `--pladis-install --pladis-scale 0` | hook installed, λ=0 → delegates to the same fused SDPA; bit-identical to vanilla (install-plumbing control) |
| eager-dense control | `--pladis-install --pladis-scale 1.0 --pladis-method softmax` | dense softmax computed on the hook's eager path; numeric-path-matched baseline for the λ>0 arms |
| locus cells | `--pladis-scale λ --pladis-qgroup {state,action,all} --pladis-kind {text,image,all}` | the interventions under study |
| mixed cells | `--pladis-scale λ --pladis-cells <cell,cell>` | per-kind query groups |
| temperature control | `--pladis-scale 1.0 --pladis-method softmax --pladis-beta β` | sharpened-softmax counterpart to a sparse cell |

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
| `none` (original) | 400 | unperturbed per-task baselines, init states 0–9 |

- **Rollout protocol** (official Isaac-GR00T LIBERO evaluation): 720 env-step
  cap, 16-step decoded action chunk with the first 8 executed (receding
  horizon), success-on-first-contact termination. Primary metric:
  `success_once` per episode.
- **Pairing**: all arms of an axis share the same seed-0 schedule, so
  episodes are paired across arms by construction (asserted at load time).

## 3. Repository layout

```
pladis/        attention hooks
  attn_gr00t.py        weight-space hook (faithful to the official PLADIS
                       code path: eager blend at λ>0, native fused SDPA at
                       λ=0); qgroup/kind/cells gating
  attn_gr00t_fused.py  STAGED fused-anchored variant (§7); not imported by
                       any entry point
  attn_pi0.py          π0/π0.5 (Gemma joint-attention) variant; STAGED, not
                       wired to any entry point and not covered by §5 gates
harness/       evaluation loop, fully owned
  env.py               curated schedules, per-axis delivery, deterministic
                       per-episode env seeding
  rollout.py           obs→policy→step loop, per-chunk noise pinning,
                       train-convention observation formatting
  model_gr00t.py       official Gr00tPolicy adapter
  eplog.py             per-episode TSV ledger (crash-safe, resume source,
                       arm-signature guarded)
  video.py             per-episode mp4 of the model's two camera views
experiments/   entry points
  run.sh               environment wrapper (all commands go through it)
  eval_arm.py          single-arm evaluator — anchors, parity checks, and
                       sweeps share this one code path
  sweep_n17_*.sh       sweep drivers (language / original / layout / robot);
                       the arm list of each axis lives in the script itself
  verify_*.py          verification gates (§5)
  smoke_gr00t.py       GPU smoke test
analysis/      analyze.py --language|--layout|--robot  (paired McNemar)
docs/          benchmark.md — cross-checked benchmark facts
results/       (gitignored) eplogs, videos, driver logs
```

## 4. Installation

### 4.1 Requirements

- 1× CUDA GPU (developed on a single H100, bf16); ~13–17 s/episode
- ~30 GB for checkpoints; ~5–10 GB per sweep if video recording is enabled

### 4.2 External checkouts (sibling directories)

| path | content |
|---|---|
| `../LIBERO-plus` | benchmark checkout (bddl/init files, curated `task_classification.json`, bundled ImageMagick under `.magick`); `pip install -e` into the venv |
| `../models/GR00T-N1.7-LIBERO/` | `huggingface-cli download nvidia/GR00T-N1.7-LIBERO --local-dir ../models/GR00T-N1.7-LIBERO` |
| `~/.cache/huggingface` | Cosmos-Reason2-2B backbone (auto-downloaded on first load) |
| `~/.hf_user_token` | plain-text HF token, read at runtime by `run.sh` |

### 4.3 Python environment

Python 3.11 (uv-managed). Pinned versions (validated; the attention hook is a
line-for-line port of `diffusers` `AttnProcessor2_0` — re-run the parity gates
of §5 after any `diffusers`/`torch` upgrade):

| package | version | note |
|---|---|---|
| torch / torchvision | 2.6.0 / 0.21.0 | flash-attn 2.7.4.post1 |
| diffusers | 0.35.1 | attention processor base |
| entmax | 1.3 | sparse branch |
| robosuite | 1.4.1 | pip install (not editable) |
| mujoco | 3.6.0 | EGL rendering |
| transformers | 4.57.3 | numpy 2.4.6 |
| gr00t | 0.1.0 | official Isaac-GR00T checkout, `pip install -e` |
| liberoplus | 0.1.0 | `pip install -e ../LIBERO-plus` |

The exact set is recorded in `requirements.txt` (reference pins, not a lock
file — `gr00t` and `liberoplus` are editable sibling checkouts).

**Porting note.** Absolute paths that must be adapted on a new machine:

| location | value |
|---|---|
| `experiments/run.sh` | venv interpreter `PY=...`, HF token path, `MAGICK_HOME` |
| `experiments/sweep_n17_*.sh` | `MODEL_ROOT=...` |
| `experiments/eval_arm.py` | default `MODEL` — overridable with the `GR00T_MODEL_PATH` environment variable or `--model-path` |

### 4.4 Execution wrapper

Every Python entry point is invoked through `experiments/run.sh`, which sets
EGL rendering, the ImageMagick library path, `PYTHONPATH`, the HF token, and
the pinned interpreter:

```bash
bash experiments/run.sh experiments/smoke_gr00t.py    # model + env smoke test
```

Bypassing the wrapper (inline env prefixes, direct `python`) is unsupported.

## 5. Verification gates

The harness treats delivery and parity claims as testable artifacts. On a new
machine or after dependency changes, run in order:

1. **Anchor** — unperturbed LIBERO-10 reproduces the published model-card
   success rate within sampling error: `eval_arm.py --axis none --episodes 100`.
2. **Instruction delivery** — `smoke_gr00t.py` asserts a language-variant
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
   scaling).
5. **Fused-anchored equivalence** (only if adopting `attn_gr00t_fused.py`,
   §7) — `verify_fused_anchor.py cpu|cuda`.

Gates 3–5 need the GPU + simulator stack of §4 (gate 5 has a CPU pre-check);
there is no CPU-only test suite. All gates print `PASS` / `ALL GATES PASSED`
and exit 0.

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

One driver per axis (`language` / `original` / `layout` / `robot`). Each
driver enumerates its arm list explicitly — the script is the source of truth
for which arms an axis carries. All drivers are resume-safe at episode
granularity, so re-running a driver skips completed arms and executes only
what is new. Outputs follow
`results/sweep/n17_{axis}_{arm}_{suite}_eplog.tsv` (+ a same-named `.out` log
and, when enabled, `videos/n17_{axis}_{arm}_{suite}/ep#####_{S|F}_{task}.mp4`).

### 6.3 Analysis

```bash
python3 analysis/analyze.py --language
python3 analysis/analyze.py --layout    # + perturbation-category breakdown
python3 analysis/analyze.py --robot     # + strength-level (L1–L5) breakdown
```

**Statistical conventions.** Primary test: paired McNemar over the pooled
episode pairing (`z = (n01 − n10)/√(n01+n10)`, no continuity correction),
over `success_once`, reported per contrast with discordant counts. Pooled
contrasts are primary; single-suite contrasts are interpreted conservatively
(closed-loop rollouts amplify numeric noise at the single-suite scale — §7).
`analyze.py` prints a Bonferroni-adjusted p over the pooled contrast family
and marks which contrasts survive it. Each λ>0 arm is contrasted against
**both** baselines (vanilla and the eager-dense control).

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
(`attn_gr00t.py` follows the official convention: native fused path at λ=0,
eager weight-space blend at λ>0). Closed-loop rollouts chaotically amplify
the rounding-floor difference between the two paths, so vanilla-vs-λ>0
contrasts carry a numeric-path term alongside the intervention. The harness
controls for it with the **eager-dense control arm** (§1.3), which runs the
identical eager path with a plain softmax.

`pladis/attn_gr00t_fused.py` stages an alternative convention using the
algebraic identity `(d + λ(s−d))·V = SDPA + λ·(s−d)·V`: the dense
contribution of every arm is the same fused SDPA call, the correction term is
always computed (no λ gate), and λ=0 is bit-identical to vanilla by an IEEE
identity rather than by branching — removing the fused↔eager term by
construction. It is verified by `verify_fused_anchor.py` (bit-parity at λ=0,
row-level parity outside the query group, rounding-floor equivalence to the
weight-space hook) but is **not wired to any entry point**; adopting it means
copying it over `attn_gr00t.py` and re-running the §5 gates.

## 8. Acknowledgements

This repository builds on:
[PLADIS](https://github.com/cubeyoung/PLADIS) (method),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) /
[LIBERO-plus](https://github.com/RLinf/LIBERO-plus) (benchmark),
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (model and serving),
[entmax](https://github.com/deep-spin/entmax) (sparse transformations).

## License

Code in this repository is released under the Apache-2.0 license (see SPDX
headers). Model checkpoints and benchmark assets retain their upstream
licenses.
