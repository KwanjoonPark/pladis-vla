# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research evaluation harness, not a library or service. It applies PLADIS-style
test-time sparse-attention interventions to the action-head DiT of GR00T N1.7 and
measures each *intervention locus* (query group × key modality) on the LIBERO-plus
robustness benchmark with paired, episode-level statistics.

`README.md` is the scientific spec (method, arm vocabulary, protocol, statistics).
`docs/SETUP.md` is the machine-provisioning spec. `docs/benchmark.md` (Korean) holds
cross-verified LIBERO-plus facts. Read the relevant one before changing behavior that
either document asserts — the documents are treated as contracts, and several source
comments cite them by section.

## Environment

The repo assumes a workspace layout with sibling checkouts pinned by
`scripts/externals.lock`:

```
$WS/pladis-vla     this repo
$WS/LIBERO-plus    benchmark (package `liberoplus`, pip -e installed into the venv)
$WS/RLinf          venv host; `openpi` track import surface
$WS/models/...     checkpoints
```

`$WS` = parent of the repo. Machine-specific overrides go in `experiments/machine.env`
(gitignored; copy from `machine.env.example`). Defaults are derived in
[load_machine_env.sh](experiments/load_machine_env.sh).

There is no CPU-only path: everything except `analysis/analyze.py` needs a CUDA GPU
plus the MuJoCo/robosuite/ImageMagick simulator stack.

## Commands

Every Python entry point runs through the wrapper — it selects the track venv and
exports `MUJOCO_GL=egl`, `MAGICK_HOME`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `HF_TOKEN`.
Inline `VAR=... python ...` prefixes are unsupported (they were silently dropped by
backgrounded shells; that's why the wrapper exists).

```bash
bash experiments/run.sh [--venv gr00t|openpi|lerobot] <script.py> [args...]   # default: gr00t
```

Verification gates (there is no unit-test suite; these gates *are* the tests, and all
print `PASS` / `ALL GATES PASSED` and exit 0):

```bash
bash experiments/run.sh experiments/verify_externals.py       # checkout SHAs + venv pins
bash experiments/run.sh experiments/smoke_gr00t.py            # 2-episode GPU smoke
bash experiments/run.sh experiments/verify_base0_parity.py    # λ=0 bit-parity vs AttnProcessor2_0
bash experiments/run.sh experiments/verify_language_axis.py
bash experiments/run.sh experiments/verify_layout_axis.py --mode gates
bash experiments/run.sh experiments/verify_robot_axis.py
```

Run one arm (see README §6.1 for the full flag table):

```bash
bash experiments/run.sh experiments/eval_arm.py \
  --suite libero_10 --axis language --episodes 0 --seed 0 \
  --model-path $WS/models/GR00T-N1.7-LIBERO/libero_10 \
  --out results/my_arm_eplog.tsv \
  --pladis-install --pladis-scale 1.0 --pladis-qgroup action --pladis-kind text
```

Sweeps and analysis:

```bash
nohup bash experiments/sweep_n17_<language|original|layout|robot>.sh > results/sweep/driver_<axis>.out 2>&1 &
python3 analysis/analyze.py --language|--layout|--robot    # read-only, no venv/GPU needed
```

## Architecture

One linear data path, each layer owning exactly one concern:

```
LiberoPlusTaskSet   curated schedule (task_classification.json) → [EpisodeSpec]
LiberoPlusSession   one live sim; reset policy per axis; yields (obs, instruction)
run_episode         obs→policy→step loop; noise pinning; train-convention formatting
OfficialGr00tPolicy official Gr00tPolicy behind predict_action_batch()
EpisodeLogger       one flushed TSV row per episode + `<out>.arm` sidecar
```

`eval_arm.py` is the single evaluator: anchors, parity runs, and every sweep arm go
through it, so no result can come from a divergent code path. Sweep drivers are thin
shell loops over `eval_arm.py`; each driver's arm list is the source of truth for what
its axis carries.

`pladis/attn_gr00t_n17.py` is installed onto the model *after* loading by swapping
diffusers `Attention.processor` on selected DiT blocks. It is a port of the official
PLADIS processor, extended with the query-group (row-slice) axis. `pladis/attn_pi05.py`
is a staged π0.5 variant — not wired to any entry point and not covered by any gate.

## Invariants that constrain changes

These are the reason the code looks the way it does. Breaking one invalidates results
silently, which is why several of them are enforced as hard errors rather than defaults.

- **Arms must be paired.** All arms of an axis share the seed-0 schedule; `analyze.py`
  asserts episode-set and `task_name` equality across arms. Anything that changes the
  schedule for one arm but not others breaks the McNemar pairing.
- **Three seeding layers must stay deterministic**: the schedule permutation
  (`seed`), the env reseed before every reset (`seed·1_000_003 + episode`), and the
  flow-matching noise pin before every chunk inference
  (`episode_seed·100_003 + step`). Anything that consumes RNG on the model path must
  not be added between those pins. Video recording is verified not to perturb them.
- **λ=0 must stay on the fused SDPA path.** `base0` is asserted bit-identical to
  vanilla (`verify_base0_parity.py`). λ>0 necessarily runs an eager path; that numeric
  difference is controlled for by the eager-dense control arm, not ignored.
- **A no-op intervention must never run.** An install that selects zero blocks, a
  `kind` split on a DiT without text/image alternation, or an `n_state_tokens` that
  doesn't split the query rows all raise instead of degrading — otherwise the arm would
  burn a full sweep while being logged as an intervention.
- **The instruction is data, not assumption.** It always comes from
  `env.language_instruction` (liberoplus's own BDDL parse) and is written to the eplog
  per episode. Never source it from task-suite metadata — that was the upstream bug
  that silently evaluated original instructions on the language axis.
- **Layout is scene-altering** (`SCENE_ALTERING_AXES`): base-task init states must not
  be applied there, or the perturbation is silently reverted. Fixtures live in
  `model.body_pos`, outside `sim.get_state()`, so determinism checks must compare
  `body_xpos` too.
- **Eplogs carry no arm identity**, so the `<out>.arm` sidecar does: resuming with
  different flags aborts rather than mixing two arms into one file. The eplog is also
  the resume ledger — a fully-logged arm exits before the model loads.
- **Sweeps run from committed code.** Drivers call `pladis_require_clean_tree`
  (override: `PLADIS_ALLOW_DIRTY=1`), and each run appends `code <git-describe>` to the
  sidecar.
- **One (model × axis) campaign runs on one machine and one stack.** Cross-machine
  numeric differences break pairing; parallelize at campaign granularity only.

## Working conventions

- **Adding an arm** touches three places: the axis's `sweep_n17_*.sh` (the `run <tag>
  <flags>` line, with a dated comment explaining the arm's purpose), and in
  `analysis/analyze.py` the axis's `extra_arms` plus the `extra_contrasts` it should be
  tested in. Arms listed in `extra_arms` are skipped until all four suite eplogs exist,
  so appending to a running sweep is safe.
- **Re-running a sweep is the resume mechanism** — completed arms and episodes are
  skipped at episode granularity. Prefer re-invoking the driver over hand-running arms.
- **After any dependency bump** (especially diffusers on the gr00t track, transformers
  on the openpi track), re-run the parity gates: both hooks are line-for-line ports
  against pinned library versions.
- **Results are gitignored** (`results/`, `*.mp4`, `*.out`, `*.log`) and collected
  manually. Never commit them.
- Comments here carry provenance — upstream `file:line` citations, dated verification
  findings, and the failure mode a guard defends against. Match that style; a bare
  guard with no stated failure mode reads as removable.
