# SPDX-License-Identifier: Apache-2.0
"""π0.5 instruction-delivery gate (README §5 gate 2) — the perturbation must reach the model.

This is the π0.5 counterpart of smoke_gr00t.py, made stronger in one specific way: it
does not merely check that the eplog records a variant instruction, it intercepts the
string at the point where openpi TOKENIZES it and asserts on that.

Why that is the right place: the upstream bug this repo exists downstream of
(harness/env.py:22-24, docs/benchmark.md:38) was serving `task.language` — the ORIGINAL
task metadata — while believing the rephrased instruction had been delivered. Every
symptom of that bug is invisible: the rollout runs, the eplog looks fine, and the
language axis silently measures nothing. Asserting at the tokenizer closes the whole
path from `env.language_instruction` to the model's key tokens.

Passes when:
  1. an axis=none episode delivers the base task's original instruction,
  2. a language-variant episode delivers a DIFFERENT (rephrased) instruction,
  3. in both cases the string openpi tokenized is EXACTLY env.language_instruction
     (liberoplus's own bddl parse), and
  4. both episodes run to completion with a sane wall time.

Run: bash experiments/run.sh --venv openpi experiments/smoke_pi05.py
"""

from __future__ import annotations

import argparse
import os

from harness.env import LiberoPlusSession, LiberoPlusTaskSet
from harness.model_pi05 import load_pi05, preload_sim_stack
from harness.rollout import run_episode

# `_install_prompt_probe()` imports openpi.transforms, which poisons MagickWand's dlopen
# for the rest of the process — load_pi05 would then die at the liberoplus import.
preload_sim_stack()

TOKENIZED: list[str] = []


def _install_prompt_probe() -> None:
    """Record the prompt every TokenizePrompt call actually receives.

    Wrapping the transform (rather than reading our own adapter's `last_prompt`) is the
    point: it observes the string AFTER openpi's own input pipeline
    (LiberoInputs -> Normalize -> TokenizePrompt), so a transform that dropped or
    replaced the prompt would be caught, not papered over.
    """
    import openpi.transforms as _t

    orig = _t.TokenizePrompt.__call__

    def probe(self, data):
        if "prompt" in data:
            TOKENIZED.append(str(data["prompt"]))
        return orig(self, data)

    _t.TokenizePrompt.__call__ = probe


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get(
        "MODEL_ROOT_PI05", "/home/reallab/parkkwanjoon/workspace/models/pi05_libero"))
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--max-steps", type=int, default=520)   # openpi libero_10 protocol
    p.add_argument("--exec-horizon", type=int, default=5)
    args = p.parse_args()

    _install_prompt_probe()
    model = load_pi05(args.model_path)
    sess = LiberoPlusSession(seed=0)
    ok = True
    seen = {}

    for axis in (None, "language"):
        ts = LiberoPlusTaskSet(args.suite, axis)
        spec = ts.schedule(1, seed=0)[0]
        TOKENIZED.clear()
        r = run_episode(
            sess, spec, ts.init_states_of(spec.task_name), model,
            episode_seed=0 * 1_000_003 + spec.episode,
            max_steps=args.max_steps, exec_horizon=args.exec_horizon,
        )
        tag = "original" if axis is None else "language"
        seen[tag] = r.instruction

        if not TOKENIZED:
            print(f"[{tag}] FAIL openpi never tokenized a prompt — the instruction did "
                  f"not reach the model at all")
            ok = False
        elif len(set(TOKENIZED)) != 1:
            print(f"[{tag}] FAIL prompt changed mid-episode: {sorted(set(TOKENIZED))}")
            ok = False
        elif TOKENIZED[0] != r.instruction:
            print(f"[{tag}] FAIL tokenized {TOKENIZED[0]!r} != env.language_instruction "
                  f"{r.instruction!r} — something in the transform chain replaced it")
            ok = False
        else:
            print(f"[{tag}] OK   tokenized == env.language_instruction over "
                  f"{len(TOKENIZED)} chunk(s), {r.n_steps} steps, {r.wall_s:.1f}s")
            print(f"[{tag}]      {r.instruction!r}")

        if r.n_steps <= 0:
            print(f"[{tag}] FAIL episode ran 0 steps")
            ok = False

    # The whole point of the language axis: the two instructions must differ.
    if seen.get("original") == seen.get("language"):
        print("[delivery] FAIL the language-variant episode delivered the SAME string as "
              "the unperturbed one — the perturbation is not being delivered "
              "(this is exactly the upstream bug: harness/env.py:22-24)")
        ok = False
    else:
        print("[delivery] OK   variant instruction differs from the original")

    sess.close()
    print("ALL GATES PASSED" if ok else "GATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
