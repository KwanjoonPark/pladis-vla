# SPDX-License-Identifier: Apache-2.0
"""Gates for the noise axis (Sensor Noise, obs-side corruption of agentview).

  A. Wiring — curated counts per suite match the classification file
     (spatial 351 / object 422 / goal 379 / 10 449 = 1,601), every variant
     carries a `_noise_<N>` tail with N in 1..50, and all five corruption
     families are present in every suite's schedule.
  B. Delivery + pairing (libero_10, one variant per family) — after
     reset+settle, the agentview the model would see differs substantially
     from the neutral counterpart episode (same base task, same init state),
     while the WRIST image is bit-identical and the post-settle sim state is
     bit-identical: the corruption reaches exactly one camera and never
     touches the scene.
  C. Determinism — the same noise episode run twice (reset + settle + 3 noop
     steps) yields bit-identical agentview streams for a motion-family and a
     fog-family variant (the two families that draw per-frame from global
     np.random; covered by the per-episode reseed in Session.reset).
  D. Original-axis sanity — an axis=None episode run twice stays
     bit-identical under the (now unconditional) per-episode reseed.

Full-path MODEL regression for the reseed change (2-ep language vanilla
eplog bit-compare vs the stored sweep reference) runs via eval_arm, not
here — see the prep notes / commit message.

Run: bash experiments/run.sh experiments/verify_noise_axis.py
"""

import numpy as np

from harness.env import LiberoPlusSession, LiberoPlusTaskSet

EXPECT = {"libero_spatial": 351, "libero_object": 422, "libero_goal": 379, "libero_10": 449}
FAMILIES = {"motion": range(1, 11), "gauss": range(11, 21), "zoom": range(21, 31),
            "fog": range(31, 41), "glass": range(41, 51)}


def noise_n(name: str) -> int:
    return int(name.rsplit("_noise_", 1)[1])


def gate_A():
    for suite, expect in EXPECT.items():
        ts = LiberoPlusTaskSet(suite, "noise")
        assert len(ts.task_names) == expect, f"A: {suite} count {len(ts.task_names)} != {expect}"
        sched = ts.schedule(len(ts.task_names), seed=0)
        ns = [noise_n(s.task_name) for s in sched]
        assert all(1 <= n <= 50 for n in ns), f"A: {suite} has N outside 1..50"
        present = {fam for n in ns for fam, rng in FAMILIES.items() if n in rng}
        assert present == set(FAMILIES), f"A: {suite} missing families {set(FAMILIES) - present}"
    print("PASS gate A: wiring — 1,601 curated variants, tails valid, all 5 families per suite")


def _run_episode(sess, spec, init_states, n_steps=3):
    """reset+settle then n_steps noops; returns (agentviews, wrists, state)."""
    obs, _ = sess.reset(spec, init_states)
    avs, wrs = [obs["agentview_image"].copy()], [obs["robot0_eye_in_hand_image"].copy()]
    for _ in range(n_steps):
        obs, _, _, _ = sess.step(LiberoPlusSession.NOOP)
        avs.append(obs["agentview_image"].copy())
        wrs.append(obs["robot0_eye_in_hand_image"].copy())
    state = sess._env.env.sim.get_state().flatten().copy()
    return avs, wrs, state


def gate_B():
    """Delivery + pairing, with the measured fixture exception: same sim
    state, corruption present on agentview, and the ONLY model difference
    between the pseudo-name and base-name builds is fixture body_pos
    (fixtures are placed by a name-dependent deterministic draw at build —
    see the 08-02 investigation; scenes with no fixtures pair bit-exactly
    including the wrist camera)."""
    ts = LiberoPlusTaskSet("libero_10", "noise")
    ts0 = LiberoPlusTaskSet("libero_10", None)
    sched = ts.schedule(len(ts.task_names), seed=0)
    sched0 = ts0.schedule(len(ts0.task_names) * 10, seed=0)
    sess = LiberoPlusSession(seed=0, per_episode_np_seed=True)
    fixture_free_seen = False
    for fam, rng in FAMILIES.items():
        spec = next(s for s in sched if noise_n(s.task_name) in rng)
        base = next(s for s in sched0 if s.base_task == spec.base_task
                    and s.init_state_id == spec.init_state_id)
        av_n, wr_n, st_n = _run_episode(sess, spec, ts.init_states_of(spec.task_name))
        fixtures = set(sess._env.env.fixtures_dict)
        m = sess._env.env.sim.model
        bp_n = {m.body_id2name(i): m.body_pos[i].copy() for i in range(m.nbody)}
        av_b, wr_b, st_b = _run_episode(sess, base, ts0.init_states_of(base.task_name))
        m = sess._env.env.sim.model
        bp_b = {m.body_id2name(i): m.body_pos[i].copy() for i in range(m.nbody)}
        d = np.mean(np.abs(av_n[-1].astype(np.float32) - av_b[-1].astype(np.float32)))
        assert d > 1.0, f"B: {fam} agentview not corrupted (mean|d|={d:.3f}) — silent nullification"
        assert np.array_equal(st_n, st_b), f"B: {fam} sim state differs — scene not paired"
        moved = {k for k in bp_n if not np.array_equal(bp_n[k], bp_b.get(k))}
        stray = {b for b in moved if not any(b.startswith(f) for f in fixtures)}
        assert not stray, f"B: {fam} NON-fixture bodies moved across name types: {stray}"
        if not fixtures:
            assert all(np.array_equal(a, b) for a, b in zip(wr_n, wr_b)), \
                f"B: {fam} wrist differs in a fixture-free scene"
            fixture_free_seen = True
        print(f"  [{fam}] N={noise_n(spec.task_name)} agentview mean|d|={d:.1f}, "
              f"state paired, moved bodies subset of fixtures ({sorted(moved) or 'none'})")
    sess.close()
    print("PASS gate B: corruption on agentview; scene paired (fixture-pos exception only)"
          + ("" if fixture_free_seen else " [no fixture-free scene among sampled tasks]"))


def gate_C():
    ts = LiberoPlusTaskSet("libero_10", "noise")
    sched = ts.schedule(len(ts.task_names), seed=0)
    for fam in ("motion", "fog"):  # the per-frame np.random consumers
        spec = next(s for s in sched if noise_n(s.task_name) in FAMILIES[fam])
        runs = []
        for _ in range(2):
            sess = LiberoPlusSession(seed=0, per_episode_np_seed=True)
            runs.append(_run_episode(sess, spec, ts.init_states_of(spec.task_name)))
            sess.close()
        (av1, _, _), (av2, _, _) = runs
        assert all(np.array_equal(a, b) for a, b in zip(av1, av2)), \
            f"C: {fam} corruption stream not deterministic — per-episode reseed broken"
    print("PASS gate C: random-draw families (motion, fog) bit-deterministic across reruns")


def gate_D():
    ts = LiberoPlusTaskSet("libero_10", None)
    spec = ts.schedule(2, seed=0)[0]
    runs = []
    for _ in range(2):
        sess = LiberoPlusSession(seed=0)
        runs.append(_run_episode(sess, spec, ts.init_states_of(spec.task_name)))
        sess.close()
    (av1, wr1, st1), (av2, wr2, st2) = runs
    assert all(np.array_equal(a, b) for a, b in zip(av1, av2)) and np.array_equal(st1, st2), \
        "D: original-axis episode not bit-identical under the unconditional reseed"
    print("PASS gate D: original axis bit-identical under the per-episode reseed")


def main():
    gate_A(); gate_B(); gate_C(); gate_D()
    print("ALL GATES PASSED (env-level; run the 2-ep model regression before sweeping)")


if __name__ == "__main__":
    main()
