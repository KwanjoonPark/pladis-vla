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
models {GR00T N1.7 (implemented); π0.5 (openpi track, hook staged — π0 dropped);
SmolVLA and GR00T N1.5 (planned)} × perturbation axes {language, layout,
robot, original}. Campaigns are distributed across machines at
(model × axis) granularity — all arms of one comparison share one machine
and stack (docs/SETUP.md §0).

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
  attn_gr00t_n17.py        weight-space hook (faithful to the official PLADIS
                       code path: eager blend at λ>0, native fused SDPA at
                       λ=0); qgroup/kind/cells gating
  attn_pi05.py         π0.5 (Gemma joint-attention, FLUX-style: sparsify the
                       language/image key sub-block, mass-preserving) variant;
                       explicit-flag install_pladis(); STAGED, not wired to any
                       entry point and not covered by §5 gates
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
  run.sh               environment wrapper (all commands go through it);
                       --venv selects the model track's interpreter
  load_machine_env.sh  machine config loader (defaults + machine.env override)
  machine.env.example  per-machine config template (copy to machine.env)
  eval_arm.py          single-arm evaluator — anchors, parity checks, and
                       sweeps share this one code path
  sweep_n17_*.sh       sweep drivers (language / original / layout / robot);
                       the arm list of each axis lives in the script itself
  verify_*.py          verification gates (§5)
  smoke_gr00t.py       GPU smoke test
scripts/       externals.lock (pinned sibling-checkout SHAs) + clone_externals.sh
analysis/      analyze.py --language|--layout|--robot  (paired McNemar)
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
bash experiments/run.sh experiments/smoke_gr00t.py                 # default venv: gr00t
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

Gates 3–4 need the GPU + simulator stack of §4; there is no CPU-only test
suite. All gates print `PASS` / `ALL GATES PASSED` and exit 0.

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
for which arms an axis carries. Drivers refuse to start from a dirty working
tree (results must be attributable to a commit; each run appends
`code <git-describe>` to the eplog's `.arm` sidecar). All drivers are
resume-safe at episode granularity, so re-running a driver skips completed
arms and executes only what is new. Outputs follow
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
(`attn_gr00t_n17.py` follows the official convention: native fused path at λ=0,
eager weight-space blend at λ>0). Closed-loop rollouts chaotically amplify
the rounding-floor difference between the two paths, so vanilla-vs-λ>0
contrasts carry a numeric-path term alongside the intervention. The harness
controls for it with the **eager-dense control arm** (§1.3), which runs the
identical eager path with a plain softmax.

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
