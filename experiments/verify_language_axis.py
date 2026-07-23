# SPDX-License-Identifier: Apache-2.0
"""Language-axis verification gates (env-only, no model, no GPU load).

The language axis claims to be an INSTRUCTION-ONLY perturbation: a variant
differs from its base task by the `(:language ...)` line and nothing else —
same scene, same pruned_init states, neutral runtime tail. Delivery itself is
checked continuously (rollout logs the instruction per episode; smoke_gr00t
asserts variant != original), but the "nothing else differs" claim was only
ever verified ad hoc (2026-07-16, in-session). This script codifies it:

  gates  ① wiring/schedule: curated counts (383/410/354/390), each variant
            exactly once, init_state_id 0, variant resolves to the base
            task's pruned_init array (same object, not a copy)
         ② neutral tail: every curated name carries exactly
            `_view_0_0_100_0_0_initstate_0` — no runtime perturbation rides
            along with the language axis
         ③ bddl diff: variant bddl == base bddl after removing the
            `(:language ...)` block, for EVERY variant; and the instruction
            text itself differs from the base (rephrasings listed if not)
         ④ scene pairing probe: base and variant reset with the same
            episode seed + init row -> bit-identical post-settle sim state,
            different instruction

  bash experiments/run.sh experiments/verify_language_axis.py [--probes N]
"""

import argparse
import os
import re

import numpy as np

from harness.env import EpisodeSpec, LiberoPlusSession, LiberoPlusTaskSet

SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
CURATED = {"libero_10": 383, "libero_goal": 410, "libero_object": 354, "libero_spatial": 390}
NEUTRAL_TAIL = "_view_0_0_100_0_0_initstate_0"
LANG_BLOCK = re.compile(r"\(:language[^)]*\)")


def spec_of(ts, name, episode=0):
    return EpisodeSpec(
        episode=episode,
        task_name=name,
        base_task=ts.base_task_of(name),
        bddl_path=ts._bddl[name],
        init_state_id=0,
    )


def split_language(bddl_path):
    """(language line, remainder) of a bddl file; fails loudly if absent."""
    with open(bddl_path) as f:
        text = f.read()
    m = LANG_BLOCK.search(text)
    assert m, f"no (:language ...) block in {bddl_path}"
    return m.group(0), text[: m.start()] + text[m.end() :]


def sim_state(sess):
    return np.asarray(sess._env.env.sim.get_state().flatten(), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=int, default=2, help="reset probes per suite (gate ④)")
    args = ap.parse_args()

    for suite in SUITES:
        ts = LiberoPlusTaskSet(suite, "language")
        base_ts = LiberoPlusTaskSet(suite, None)

        # ① wiring/schedule
        assert len(ts.task_names) == CURATED[suite], (
            f"{suite}: curated count {len(ts.task_names)} != {CURATED[suite]}"
        )
        sched = ts.schedule(len(ts.task_names), seed=0)
        assert sorted(s.task_name for s in sched) == sorted(ts.task_names)
        assert all(s.init_state_id == 0 for s in sched)
        assert not ts.scene_altering
        for name in ts.task_names:
            assert ts.init_states_of(name) is ts.init_states_of(ts.base_task_of(name)), (
                f"{name}: init states are not the base task's array"
            )
        print(f"[①] {suite}: {len(sched)} variants exactly once, id=0, base init shared  OK")

        # ② neutral tail
        for name in ts.task_names:
            assert name.endswith(NEUTRAL_TAIL), f"{name}: non-neutral runtime tail"
        print(f"[②] {suite}: all tails == {NEUTRAL_TAIL}  OK")

        # ③ bddl diff = language line only
        same_instr = []
        for name in ts.task_names:
            real = name[: -len(NEUTRAL_TAIL)]
            base = ts.base_task_of(name)
            v_lang, v_rest = split_language(os.path.join(ts.bddl_dir, real + ".bddl"))
            b_lang, b_rest = split_language(os.path.join(ts.bddl_dir, base + ".bddl"))
            assert v_rest == b_rest, f"{name}: bddl differs beyond the language line"
            if v_lang == b_lang:
                same_instr.append(name)
        print(f"[③] {suite}: {len(ts.task_names)} bddls == base outside (:language)  OK"
              + (f"  ⚠ {len(same_instr)} variants share the base wording: "
                 f"{[n[:60] for n in same_instr[:3]]}" if same_instr else ""))

        # ④ scene pairing probe
        sess = LiberoPlusSession(seed=0)
        probes = [sched[i * len(sched) // max(args.probes, 1)] for i in range(args.probes)]
        for spec in probes:
            base_spec = spec_of(base_ts, spec.base_task, episode=spec.episode)
            states = ts.init_states_of(spec.task_name)
            _, base_instr = sess.reset(base_spec, states)
            base_st = sim_state(sess)
            _, instr = sess.reset(
                spec_of(ts, spec.task_name, episode=spec.episode), states
            )
            var_st = sim_state(sess)
            assert np.array_equal(base_st, var_st), (
                f"{spec.task_name}: scene not bit-identical to base"
            )
            assert instr != base_instr, f"{spec.task_name}: instruction == base"
            print(f"[④] {suite}: ep{spec.episode} scene bit-identical, "
                  f"instr '{base_instr[:30]}...' -> '{instr[:30]}...'  OK")
        sess.close()

    print("[language-axis] ALL GATES PASSED")


if __name__ == "__main__":
    main()
