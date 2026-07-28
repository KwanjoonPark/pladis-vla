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
SHA is a different benchmark), RLinf (import surface for the openpi track),
and the official Isaac-GR00T checkout the gr00t venv installs editable.
`clone_externals.sh` never mutates an existing checkout; SHA mismatches are
reported for the operator to resolve.

Expected workspace layout (`$WS` = parent of this repo):

```
$WS/pladis-vla        this repo
$WS/LIBERO-plus       benchmark (pinned)
$WS/RLinf             venv host + openpi serving reference (pinned)
$WS/models/...        checkpoints (downloaded, §4)
```

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
  (`MAGICK_HOME`; build ImageMagick with `--prefix=$WS/LIBERO-plus/.magick`)
- LIBERO-plus assets: follow the LIBERO-plus README download step
  (assets are untracked files inside the checkout — expected)

**openpi** (π0 / π0.5) — Python 3.11:
- openpi installed per RLinf's `requirements/install.sh` openpi recipe
  (the venv conventionally lives at `$WS/RLinf/openpi`)
- plus: `pip install -e $WS/LIBERO-plus`, `pip install entmax`
- pins checked by the gate: torch 2.6.0 / transformers 4.53.2 / entmax 1.3
  (the π0 attention hook replicates transformers 4.53.2's
  `eager_attention_forward` line-for-line — re-run the π0 λ=0 parity gate
  after ANY transformers upgrade)

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

| model | source |
|---|---|
| GR00T N1.7 | `huggingface-cli download nvidia/GR00T-N1.7-LIBERO --local-dir $WS/models/GR00T-N1.7-LIBERO` (per-suite subdirs) |
| π0 (LIBERO-long SFT) | `$WS/models/RLinf-Pi0-LIBERO-Long-SFT` (openpi format: `model.safetensors` + norm_stats) |
| π0.5 (LIBERO) | `gs://openpi-assets/checkpoints/pi05_libero` (convert to PyTorch safetensors if JAX-only) or HF `lerobot/pi05_libero_finetuned` → `$WS/models/pi05_libero` |
| Cosmos-Reason2-2B backbone | auto-downloaded to `~/.cache/huggingface` on first N1.7 load |

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

# openpi track (π0/π0.5): anchor → noise-pin → λ=0 parity → smoke
# (gate scripts land with the π0 track; same ladder shape)
```

## 6. Git conventions

- `main` is the executable source of truth; every server pulls it. Feature
  work goes through short-lived branches merged into main. No long-lived
  per-server branches.
- **Sweeps run from committed code.** Sweep drivers abort on a dirty working
  tree (`PLADIS_ALLOW_DIRTY=1` overrides deliberately). Each run appends
  `code <git-describe>` to the eplog's `.arm` sidecar, so results are
  attributable to exact commits across servers.
- Results (`results/`) are gitignored and collected manually by the operator.

## 7. Known machine notes

- The development machine's RLinf checkout carries tracked modifications from
  an earlier project phase (ManiSkill-era env patches, an env-gated hook
  install). None of them are on this repo's import surface — a clean RLinf
  at the pinned SHA is sufficient on new machines (`verify_externals.py`
  prints them as a WARN on the dev machine only).
- Untracked files inside external checkouts (LIBERO-plus assets, `.magick`,
  `*.egg-info`, venv dirs under RLinf) are build artifacts — expected, and
  ignored by the externals gates.
