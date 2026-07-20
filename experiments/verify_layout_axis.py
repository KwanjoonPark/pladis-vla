# SPDX-License-Identifier: Apache-2.0
"""Layout-axis verification gates (env-only, no model, no GPU load).

The layout axis is the first scene-altering axis: the scene comes from BDDL
placement sampling under a per-episode np.random reseed instead of base-task
pruned_init states (harness/env.py). Before any sweep, this script must pass:

  gates  ① in-process determinism: same spec reset twice -> bit-identical
            mujoco state after settle
         ② perturbation delivered: `_level_sample` object positions differ
            from the base scene; AND applying the base pruned_init on the
            variant (the old code path) restores base positions exactly =
            the silent-nullification failure mode is real, not hypothetical
         ③ `_add` loads and carries extra bodies (old path would dim-crash)
         ④ instruction unchanged vs base (layout is a same-instruction axis)
         ⑤ schedule: full enumeration visits each variant once, id 0
  dump   writes a scene fingerprint (body name -> xyz) per probe episode;
         run twice in fresh processes and diff -> cross-process / cross-arm
         pairing (arms only share run seed + schedule, exactly what a fresh
         process reproduces)

  bash experiments/run.sh experiments/verify_layout_axis.py --mode gates
  bash experiments/run.sh experiments/verify_layout_axis.py --mode dump --out A.json
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from harness.env import LiberoPlusSession, LiberoPlusTaskSet


def body_positions(sess) -> dict[str, list[float]]:
    sim = sess._env.env.sim
    return {
        name: [round(float(v), 10) for v in sim.data.get_body_xpos(name)]
        for name in sim.model.body_names
    }


def sim_state(sess) -> np.ndarray:
    return np.asarray(sess._env.env.sim.get_state().flatten(), dtype=np.float64)


def find_variant(ts: LiberoPlusTaskSet, pattern: str) -> str:
    rx = re.compile(pattern)
    hits = [n for n in ts.task_names if rx.search(n)]
    assert hits, f"no curated task matches {pattern!r}"
    return hits[0]


def spec_of(ts: LiberoPlusTaskSet, name: str):
    sched = ts.schedule(len(ts.task_names), seed=0)
    return next(s for s in sched if s.task_name == name)


def probe_specs(suite: str):
    """(taskset, [(label, spec)]) — one _level/_moved and one _add variant."""
    ts = LiberoPlusTaskSet(suite, "layout")
    moved_pat = r"_moved_level\d+" if suite == "libero_goal" else r"(?<!_moved)_level\d+"
    return ts, [
        ("moved", spec_of(ts, find_variant(ts, moved_pat))),
        ("add", spec_of(ts, find_variant(ts, r"_add_\d+$"))),
    ]


def run_gates():
    for suite in ["libero_10", "libero_goal"]:
        print(f"\n=== {suite} ===", flush=True)
        ts, probes = probe_specs(suite)

        # gate ⑤ schedule shape
        assert ts.scene_altering and ts.init_states_of(ts.task_names[0]) is None
        sched = ts.schedule(len(ts.task_names), seed=0)
        assert sorted(s.task_name for s in sched) == sorted(ts.task_names)
        assert all(s.init_state_id == 0 for s in sched)
        print(f"[⑤] schedule: {len(sched)} variants exactly once, id=0  OK")

        base_ts = LiberoPlusTaskSet(suite, None)
        sess = LiberoPlusSession(seed=0)

        for label, spec in probes:
            base = spec.base_task
            base_spec = spec_of(base_ts, base)
            base_states = base_ts.init_states_of(base)

            # base scene via the unchanged fixed-init path
            _, base_instr = sess.reset(base_spec, base_states)
            base_pos = body_positions(sess)
            base_dim = len(sim_state(sess))

            # variant scene via the new seeded-reset path, twice (gate ①).
            # Compare BOTH the sim state and body world positions: fixtures
            # are repositioned via model.body_pos, invisible to get_state.
            _, instr = sess.reset(spec, None)
            pos1, st1 = body_positions(sess), sim_state(sess)
            _, _ = sess.reset(spec, None)
            pos2, st2 = body_positions(sess), sim_state(sess)
            assert np.array_equal(st1, st2), f"{spec.task_name}: reset not deterministic"
            assert pos1 == pos2, f"{spec.task_name}: fixture placement not deterministic"
            print(
                f"[①] {label}: double reset bit-identical "
                f"({len(st1)} state dims + {len(pos1)} body xpos)  OK"
            )

            # gate ④ instruction
            assert instr == base_instr, f"{label}: instruction changed: {instr!r}"
            print(f"[④] {label}: instruction == base  OK")

            common = sorted(
                n for n in pos1 if n in base_pos and ("_main" in n or "site" not in n)
            )
            delta = {
                n: float(np.linalg.norm(np.array(pos1[n]) - np.array(base_pos[n])))
                for n in common
            }
            moved_bodies = {n: round(d, 4) for n, d in delta.items() if d > 1e-4}

            if label == "moved":
                # gate ② perturbation delivered + old-path nullification demo
                assert len(sim_state(sess)) == base_dim, "level variant dim changed?"
                assert moved_bodies, f"{spec.task_name}: no body moved vs base!"
                print(f"[②] {label}: {spec.task_name}")
                print(f"     moved bodies vs base: {moved_bodies}")
                # literally the pre-change code path on the variant spec:
                # reset -> set_init_state(base) -> settle. Object bodies must
                # land back on base positions (sub-mm; controller transients
                # keep it from being bit-identical, objects are the claim).
                sess.reset(spec, base_states)
                null_pos = body_positions(sess)
                # Old-path verdict per moved object: free-joint objects are
                # dragged back to base by set_init_state (nullified); fixtures
                # live in model.body_pos, outside the qpos state, so the old
                # path leaves them wherever the un-reseeded reset resampled
                # them — neither base nor variant. Either way the variant
                # layout is NOT delivered; a few mm solver drift is expected.
                objs = [
                    n for n in moved_bodies
                    if not n.startswith(("robot0", "gripper0", "mount0"))
                ]
                verdict = {
                    n: (
                        float(np.linalg.norm(np.array(null_pos[n]) - np.array(base_pos[n]))),
                        float(np.linalg.norm(np.array(null_pos[n]) - np.array(pos1[n]))),
                    )
                    for n in objs
                }
                nullified = [n for n, (db, dv) in verdict.items() if db < 5e-3 < dv]
                kept_variant = [n for n, (db, dv) in verdict.items() if dv < 5e-3 < db]
                uncontrolled = [
                    n for n in verdict if n not in nullified and n not in kept_variant
                ]
                assert not kept_variant, (
                    f"old path preserved the variant layout?! {verdict}"
                )
                assert nullified, f"no object dragged back to base: {verdict}"
                print(
                    f"     old path verdict: {len(nullified)} objects nullified "
                    f"back to base {nullified}, {len(uncontrolled)} fixture "
                    f"bodies uncontrolled (outside qpos) {sorted(set(n.rsplit('_', 1)[0] for n in uncontrolled))} "
                    f"-> variant layout NOT delivered by old path; new path required"
                )
            else:
                # gate ③ _add: extra bodies, larger state (old path would crash)
                extra = sorted(set(pos1) - set(base_pos))
                dim = len(sim_state(sess))
                assert extra and dim > base_dim, "_add variant has no extra bodies?"
                print(f"[③] {label}: {spec.task_name}")
                print(f"     +{len(extra)} bodies {extra[:4]}..., dim {base_dim}->{dim}")

        sess.close()
    print("\nALL GATES PASSED", flush=True)


def run_dump(out: str):
    fp = {}
    for suite in ["libero_10", "libero_goal"]:
        ts, probes = probe_specs(suite)
        sess = LiberoPlusSession(seed=0)
        for label, spec in probes:
            _, instr = sess.reset(spec, None)
            fp[f"{suite}/{spec.task_name}"] = {
                "episode": spec.episode,
                "instruction": instr,
                "bodies": body_positions(sess),
            }
        sess.close()
    with open(out, "w") as f:
        json.dump(fp, f, indent=1, sort_keys=True)
    print(f"dumped {len(fp)} scenes -> {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gates", "dump"], default="gates")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.mode == "gates":
        run_gates()
    else:
        assert args.out
        run_dump(args.out)
