# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research evaluation harness, not a library or service. It applies PLADIS-style
test-time sparse-attention interventions to the action-head DiT of GR00T N1.7 and
measures each *intervention locus* (query group × key modality) on the LIBERO-plus
robustness benchmark with paired, episode-level statistics.

`README.md` is the scientific spec (method, arm vocabulary, protocol, statistics).
`docs/SETUP.md` is the machine-provisioning spec. `docs/benchmark.md` (Korean) holds
cross-verified LIBERO-plus facts. `docs/nag.md` is the design doc for the NAG
normalization row — its §2 algebra is what decides which NAG arms are worth running
and which are re-runs of the dose ladder under another name. Read the relevant one before changing behavior that
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
bash experiments/run.sh experiments/smoke_model.py            # 2-episode GPU smoke (--model)
bash experiments/run.sh experiments/verify_base0_parity.py    # λ=0 bit-parity vs AttnProcessor2_0
bash experiments/run.sh experiments/verify_language_axis.py
bash experiments/run.sh experiments/verify_layout_axis.py --mode gates
bash experiments/run.sh experiments/verify_robot_axis.py
bash experiments/run.sh experiments/verify_noise_axis.py
bash experiments/run.sh experiments/verify_camera_axis.py            # --mode video
bash experiments/run.sh experiments/verify_step_schedule.py   # --pladis-steps gate
bash experiments/run.sh experiments/verify_nag.py             # --pladis-nag-* gate
bash experiments/run.sh experiments/verify_eplog_host.py      # cross-machine guard
```

`experiments/diag_nag.py` is a measurement, not a gate: it runs a few episodes with
the NAG census armed and reports the L1-ratio distribution that `--pladis-nag-tau`
has to be chosen from (`docs/nag.md` §6). Choosing tau from the paper instead is
how a NAG arm ends up bit-identical to its own control.

`verify_camera_axis.py --mode video` writes one mp4 with the unperturbed
episode and all four viewpoint families side by side under identical scripted
actions — the same invariants as the gates, in a form a reviewer can check by
eye. It asserts them per frame before writing, so a passing video is evidence,
not illustration.

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
nohup bash experiments/sweep_n17_<language|original|layout|robot|noise>.sh > results/sweep/driver_<axis>.out 2>&1 &
python3 analysis/analyze.py --language|--layout|--robot    # read-only, no venv/GPU needed
```

## Architecture

One linear data path, each layer owning exactly one concern:

```
LiberoPlusTaskSet   curated schedule (task_classification.json) → [EpisodeSpec]
LiberoPlusSession   one live sim; reset policy per axis; yields (obs, instruction)
run_episode         obs→policy→step loop; noise pinning; train-convention formatting
ModelAdapter        wrap_obs / predict_chunk / to_env_actions seam (model_base.py);
                    adapters: OfficialGr00tPolicy, OfficialPi05Policy, SmolVLA —
                    one ModelSpec each in harness/registry.py (--model selects it)
EpisodeLogger       one flushed TSV row per episode + `<out>.arm` sidecar
```

`eval_arm.py` is the single evaluator: anchors, parity runs, and every sweep arm go
through it, so no result can come from a divergent code path. Sweep drivers are thin
shell loops over `eval_arm.py`; each driver's arm list is the source of truth for what
its axis carries.

`pladis/attn_gr00t_n17.py` is installed onto the model *after* loading by swapping
diffusers `Attention.processor` on selected DiT blocks. It is a port of the official
PLADIS processor, extended with the query-group (row-slice) axis. `pladis/attn_pi05.py`
monkeypatches transformers' gemma `eager_attention_forward` for the π0.5 track (gates:
verify_pi05_hook.py CPU, verify_pi05_parity.py / verify_pi05_delivery.py GPU);
`pladis/attn_smolvla.py` is the SmolVLA counterpart (gate: verify_smolvla_hook.py).

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
  that silently evaluated original instructions on the language axis. One carve-out
  (2026-08-02): `eval_arm.py --instruction-source task-meta` substitutes the
  training-distribution filename parse, for checkpoints (smolvla) fine-tuned on those
  strings where the BDDL parse is OOD phrasing that voids paper comparability. It is
  hard-restricted to `--axis none` (eval_arm raises otherwise), recorded in the arm
  signature, and the eplog still logs the string actually delivered.
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
  Enforced since 2026-08-26: the `.arm` sidecar records `host <name>` per run, an
  eplog refuses to be extended on a different machine (`PLADIS_ALLOW_HOST_MIX=1`
  overrides), and `analyze.py` prints a `[HOSTS]` block plus a `!host` marker on
  any contrast whose two arms came from different machines. A finished arm still
  resumes as a no-op anywhere, so drivers stay re-invokable.

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
