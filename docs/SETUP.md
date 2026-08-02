# Setting up pladis-vla on a new machine

The repository is the single distribution channel: everything version-shaped
(code, external-checkout SHAs, package pins, checkpoint sources) is tracked in
git; everything machine-shaped lives in one gitignored file
(`experiments/machine.env`); everything heavy (venvs, checkpoints, assets) is
built or downloaded per machine and validated by gates, not carried by git.

## 0. The one statistical rule for multi-server work

**All arms of one (model × axis) comparison must run on ONE machine and one
software stack.** Closed-loop rollouts chaotically amplify numeric differences
between GPUs/kernels, which breaks the episode pairing the analysis depends
on. The unit of cross-server parallelism is a whole (model × axis) campaign —
never individual arms of the same comparison. Every eplog records the commit
that produced it (`<out>.arm` sidecar, `code <git-describe>` lines).

## 1. Clone and pin

```bash
git clone https://github.com/KwanjoonPark/pladis-vla.git
cd pladis-vla
bash scripts/clone_externals.sh        # sibling checkouts at pinned SHAs
```

`scripts/externals.lock` pins LIBERO-plus (the benchmark itself — a different
SHA is a different benchmark), RLinf, the official Isaac-GR00T checkout the
gr00t venv installs editable, the **openpi fork** the π0.5 track installs
editable, and the official **PLADIS** release (reference only, never installed:
it is the source of record for the blend formulas both hooks port, and
`verify_pi05_hook.py` gate B asserts against it).
`clone_externals.sh` never mutates an existing checkout; SHA mismatches are
reported for the operator to resolve.

Expected workspace layout (`$WS` = parent of this repo):

```
$WS/pladis-vla        this repo
$WS/LIBERO-plus       benchmark (pinned) + assets/ (§5) + .magick/ (§4)
$WS/RLinf             gr00t venv host; standalone eval reference (pinned)
$WS/openpi            RLinf's openpi fork (pinned) — the π0.5 model code
$WS/PLADIS            official PLADIS release (pinned, reference only)
$WS/venvs/...         per-track venvs (§3)
$WS/models/...        checkpoints (downloaded, §4)
```

Use the **RLinf fork** of openpi, not upstream `Physical-Intelligence/openpi`:
`attn_pi05.py`'s delivery mechanism is verified against that fork's
`transformers_replace` gemma, whose dispatch site resolves
`eager_attention_forward` as a module global at call time
(`transformers_replace/models/gemma/modeling_gemma.py:312-314`) — which is
exactly what makes the monkeypatch fire.

## 2. Machine config

```bash
cp experiments/machine.env.example experiments/machine.env   # gitignored
```

Edit only what differs on this machine (venv locations, checkpoint roots, HF
token path). Unset values fall back to workspace-derived defaults
(`experiments/load_machine_env.sh`).

## 3. Virtual environments (one per model track)

Each track has its own venv; `experiments/run.sh --venv {gr00t|openpi|lerobot}`
selects the interpreter. The authoritative definition of a correctly-built
venv is the gate, not the recipe: `verify_externals.py` checks the
numerical-path-critical pins for the active venv.

**gr00t** (GR00T N1.7; default venv) — Python 3.11, built with uv:
- editable install of the pinned Isaac-GR00T checkout
  (`$PLADIS_VENV_GR00T/gr00t`, see externals.lock)
- `pip install -e $WS/LIBERO-plus` (package `liberoplus`)
- pins: torch 2.6.0 / diffusers 0.35.1 / entmax 1.3 / robosuite 1.4.1
  (pip, NOT editable) / mujoco 3.6.0 / transformers 4.57.3 — full reference
  set in `requirements.txt`
- ImageMagick runtime for LIBERO-plus at `$WS/LIBERO-plus/.magick`
  (`MAGICK_HOME`), shared by every track — `wand` needs MagickWand or
  `liberoplus.liberoplus.envs` will not import. Two ways:
  - source build: `./configure --prefix=$WS/LIBERO-plus/.magick --with-modules`
    — needs libpng/libjpeg **dev headers**; without them you get a Magick that
    cannot read the benchmark's PNG textures, which fails late and confusingly.
  - conda-forge binary (no root, delegates included, what A6000node-1 uses):
    `conda create -p $WS/LIBERO-plus/.magick -c conda-forge imagemagick`.
    Same `MAGICK_HOME` layout (`lib/libMagickWand-7.Q16HDRI.so`), so `run.sh`
    needs no change.
- LIBERO-plus assets: `assets.zip` (6.4 GB, 9.5 GB / 457k files unpacked) from
  the HF dataset `Sylvest/LIBERO-plus`. The zip carries a deep internal prefix
  (`inspire/hdd/.../LIBERO-plus-0/assets/`); extract it and move that `assets/`
  directory to **`$WS/LIBERO-plus/liberoplus/liberoplus/assets`** — the package
  is `liberoplus`, not the `libero/libero` path the upstream README names.
  Assets are untracked files inside the checkout — expected.

**openpi** (π0.5; π0 is dropped) — Python 3.11, built with uv from openpi's own
`uv.lock` so the resolution is reproducible. Do **not** run RLinf's
`requirements/install.sh` openpi recipe: it `uv sync`s RLinf's entire training
stack (ray/vllm/megatron), clones plain LIBERO, and builds flash-attn. The
minimal recipe, in this order:

```bash
UV_PROJECT_ENVIRONMENT=$WS/venvs/openpi uv sync --frozen --no-dev   # in $WS/openpi
# 2) the load-bearing step — see the warning below
cp -r $WS/openpi/src/openpi/models_pytorch/transformers_replace/* \
      $WS/venvs/openpi/lib/python3.11/site-packages/transformers/
uv pip install --python $WS/venvs/openpi/bin/python entmax==1.3
# Triton entmax backend (--pladis-sparse-backend adasplash) — editable from the
# lock-pinned checkout, NOT from PyPI: the checkout SHA is the reproducibility pin
uv pip install --python $WS/venvs/openpi/bin/python --no-deps -e $WS/adasplash
uv pip install --python $WS/venvs/openpi/bin/python --no-deps -e $WS/LIBERO-plus
# LIBERO-plus's own deps, added by hand because --no-deps skips them (see below)
uv pip install --python $WS/venvs/openpi/bin/python \
      robosuite==1.4.1 bddl==1.0.1 easydict thop future cloudpickle \
      matplotlib wand scikit-image gym
```

- **`-e $WS/LIBERO-plus --no-deps` is deliberate.** Its `requirements.txt` pins
  numpy 1.22.4 / transformers 4.21.1 / robosuite 1.4.0, which would tear down
  openpi's `numpy<2.0.0` and `transformers==4.53.2` pins. Install it without
  deps and add only what the import graph actually needs.
- **flash-attn is not needed.** The gemma expert is forced onto transformers'
  eager path at run time (`pi0_pytorch.py:447`) and the PaliGemma tower runs
  sdpa; a source build against gcc 9.4 costs hours for nothing.
- pins checked by the gate: **torch 2.7.1** / transformers 4.53.2 / entmax 1.3 /
  jax 0.5.3 / orbax-checkpoint 0.11.13. (torch is 2.7.1, not 2.6.0 —
  `$WS/openpi/pyproject.toml` pins it and the lock resolves to it.) jax and
  orbax are pinned because openpi imports jax even on the PyTorch path, and
  orbax 0.11.13 breaks on jax ≥ 0.7.
- `attn_pi05.py` replicates that gemma `eager_attention_forward` line-for-line —
  re-run the π0.5 λ=0 parity gate after ANY transformers change.

> **`transformers_replace` is a pip-invisible overwrite.** openpi copies its own
> `modeling_gemma.py` / `modeling_siglip.py` / `modeling_paligemma.py` over
> site-packages (they add the adaRMS timestep conditioning π0.5 needs). Any later
> `pip install transformers` silently reverts it *without changing a single
> version pin* — the model quietly becomes a different model while gate B still
> reads OK. `verify_externals.py` check C hashes the tree for exactly this
> reason, and `pi0_pytorch.py:117-124` raises if the siglip half is missing.

**lerobot** (SmolVLA) — on this machine the track runs inside the gr00t venv
(lerobot 0.4.4 is installed there; `PLADIS_VENV_LEROBOT` defaults to it in
`load_machine_env.sh`). Extra pin: `num2words` (SmolVLM processor hard
dependency). Checkpoint: HF `lerobot/smolvla_libero` (org-official, datasets:
lerobot/libero) → `$WS/models/smolvla_libero_official` (single checkpoint for
all four suites; anchor libero_10 64% vs paper-0.45B Long 71%). The
`HuggingFaceVLA/smolvla_libero` copy in `$WS/models/smolvla_libero` anchored
14pp lower (paired z=2.27) and is kept only as a reference. If HF requests
401, clear the stale token (`HF_TOKEN=`) — the repos are public.

## 4. Checkpoints (per machine, never in git)

Download into `$WS/models/` (roots configurable in machine.env). HF repos can
be updated in place — record/pin the revision you fetch; final identity
validation is the anchor gate (§5), which must reproduce the published
success rate.

| model | source | anchor it must reproduce |
|---|---|---|
| GR00T N1.7 | `huggingface-cli download nvidia/GR00T-N1.7-LIBERO --local-dir $WS/models/GR00T-N1.7-LIBERO` (per-suite subdirs) | model card SR |
| π0.5 (LIBERO) | official `gs://openpi-assets/checkpoints/pi05_libero`, **converted to PyTorch** (below) | spatial 98.8 / object 98.2 / goal 98.0 / **libero_10 92.4** |
| Cosmos-Reason2-2B backbone | auto-downloaded to `~/.cache/huggingface` on first N1.7 load | — |

π0.5, end to end (~12 GB download, 6.8 GB converted):

```bash
# fetch (public GCS; openpi's own helper caches to ~/.cache/openpi)
python -c "from openpi.shared import download; \
  print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"
# convert JAX/orbax params -> model.safetensors
bash experiments/run.sh --venv openpi $WS/openpi/examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
  --config_name pi05_libero --output_path $WS/models/pi05_libero --precision bfloat16
# the converter does NOT copy assets/, and Policy construction needs the norm stats:
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets $WS/models/pi05_libero/
```

Two conversion gotchas: the script infers π0.5-ness from the string `"pi05"`
being present in `--checkpoint_dir`, so do not rename the source directory; and
the final layout must be `model.safetensors` + `config.json` +
`assets/physical-intelligence/libero/norm_stats.json`.

> **Do not use `RLinf/RLinf-Pi05-LIBERO-SFT`** as the experiment checkpoint. It is
> PyTorch-native and therefore tempting, but it is a 40-trajectory few-shot SFT
> with libero_10 at **43.9%** (arXiv:2510.25889). A robustness study on a model
> whose baseline has already collapsed measures the floor, not the intervention.
> Acceptable for plumbing bring-up only.

HF token: plain-text file at `$PLADIS_HF_TOKEN_FILE` (default
`~/.hf_user_token`), read at runtime by run.sh.

## 5. New-machine gate ladder (run in order, per track)

No sweep starts until every gate below prints PASS/OK for the track you will
run. All commands go through `experiments/run.sh`.

```bash
# 0. externals + pins (any track; run under the track's venv)
bash experiments/run.sh [--venv openpi] experiments/verify_externals.py

# gr00t track
bash experiments/run.sh experiments/smoke_model.py          # model+env+instruction smoke
bash experiments/run.sh experiments/eval_arm.py \
  --axis none --episodes 100 --seed 0 --out results/anchor_eplog.tsv
#   -> anchor: must reproduce the model-card success rate within sampling error
bash experiments/run.sh experiments/verify_base0_parity.py  # λ=0 bit parity
bash experiments/run.sh experiments/verify_language_axis.py # axis delivery gates
bash experiments/run.sh experiments/verify_layout_axis.py --mode gates
bash experiments/run.sh experiments/verify_robot_axis.py
```

**openpi track (π0.5).** Gate 5 needs neither a GPU nor a checkpoint, so run it
first — on a bare machine it is the cheapest possible proof that the hook is
sane here (torch + transformers 4.53.2 + entmax is the whole dependency).

```bash
bash experiments/run.sh --venv openpi experiments/verify_pi05_hook.py      # 5  (CPU)
bash experiments/run.sh --venv openpi experiments/verify_externals.py      # 0
bash experiments/run.sh --venv openpi experiments/verify_language_axis.py  # model-independent
bash experiments/run.sh --venv openpi experiments/verify_pi05_delivery.py  # 5b (on-model)
bash experiments/run.sh --venv openpi experiments/smoke_pi05.py            # 2
CUDA_VISIBLE_DEVICES=4 bash experiments/sweep_pi05_original.sh libero_10   # 1  (anchor)
bash experiments/run.sh --venv openpi experiments/verify_pi05_parity.py    # 3
bash experiments/run.sh --venv openpi experiments/diag_pi05_support.py     # 5c (measurement)
```

Gate 5b is the one that catches the π0.5-specific silent failure: it runs one
real chunk and asserts the blend fired, at exactly the expected key geometry.
On the official checkpoint that is 180 blended forwards per chunk (18 expert
layers × 10 denoise steps) at `(query_len, key_len) = (10, 978)`, i.e.
`[image 768 | language 200 | suffix 10]`.

Gate 5c is a measurement, not a pass/fail: it reports how many columns entmax15
actually keeps. Read it before choosing λ — an intervention that keeps 190 of
200 language columns cannot produce an interpretable null result.

## 6. Git conventions

- `main` is the executable source of truth; every server pulls it. Feature
  work goes through short-lived branches merged into main. No long-lived
  per-server branches.
- **Sweeps run from committed code.** Sweep drivers abort on a dirty working
  tree (`PLADIS_ALLOW_DIRTY=1` overrides deliberately). Each run appends
  `code <git-describe>` to the eplog's `.arm` sidecar, so results are
  attributable to exact commits across servers.
- Results (`results/`) are gitignored and collected manually by the operator.

## 7. Runtime environment the openpi track needs

`experiments/run.sh`'s openpi branch exports two things that are not optional:

- **`JAX_PLATFORMS=cpu`** (+ `XLA_PYTHON_CLIENT_PREALLOCATE=false`). jax is a hard
  import dependency of openpi *even on the PyTorch path*
  (`openpi.transforms`, `openpi.policies.policy`, `openpi.shared.download` all
  import it). Left on GPU it preallocates ~75% of a device the moment openpi is
  imported — which OOMs the rollout or, on a shared box, takes a GPU that belongs
  to somebody else.
- **`TORCH_COMPILE_DISABLE=1`**. `pi0_pytorch.py:112` unconditionally does
  `torch.compile(self.sample_actions, mode="max-autotune")`. A compiled graph
  freezes whichever `eager_attention_forward` module global it traced, so it can
  un-install the PLADIS patch and hide the flow-noise RNG at the same time.
  `harness/model_pi05.py` also un-compiles explicitly and asserts; this is the belt.

## 8. Known machine notes

- The development machine's RLinf checkout carries tracked modifications from
  an earlier project phase (ManiSkill-era env patches, an env-gated hook
  install). None of them are on this repo's import surface — a clean RLinf
  at the pinned SHA is sufficient on new machines (`verify_externals.py`
  prints them as a WARN on the dev machine only).
- Untracked files inside external checkouts (LIBERO-plus assets, `.magick`,
  `*.egg-info`, venv dirs under RLinf) are build artifacts — expected, and
  ignored by the externals gates.
- **A6000node-1** (π0.5 track, provisioned 2026-07-27): 8× RTX A6000 48 GB, all
  available to this project (2026-07-30; an earlier note reserved 0–3 for
  another group, which was temporary). `CUDA_VISIBLE_DEVICES` is still mandatory
  on every driver — it is the suite→device pin of §0, not a permission check —
  and the box is shared, so check `nvidia-smi` for a neighbour's process before
  claiming a device. No `curl`, no `git-lfs`, no `uv`, and no system Python 3.11:
  `uv` came from `pip install uv` into the conda base, and ImageMagick from
  conda-forge (§3). π0.5 at bf16 uses ~36 GB with the 968-token prefix, so one
  process per GPU.
- The suite→GPU map for a campaign must stay **fixed across every arm**, and each
  run records its device in the eplog's `.arm` sidecar (`dev <n> <name> <uuid>`,
  appended as a provenance line, never line 1). Pinning a *suite* to a device is
  sound under §0 because the analysis pairs episodes by `(suite, episode)` and a
  McNemar pair never crosses suites; pinning an *arm* to a device would put a
  numeric-path difference inside a pair, which §0 forbids.
