# SPDX-License-Identifier: Apache-2.0
"""Gates for the LIBERO-plus ROBOT-init axis (runtime axis, `_initstate_<k>`).

Mechanism under the official protocol (reset -> set_init_state(base pruned
init) -> settle; LIBERO-plus README: "identical to LIBERO, num_trials=1"):
  * env.reset() puts the arm at the Panda{k} class's perturbed init_qpos AND
    fixes the OSC nullspace reference (controller.initial_joint) there.
  * set_init_state snaps the arm back to the base task's stored pose but does
    NOT touch the controller -> the nullspace bias toward the perturbed config
    persists for the whole episode; settle steps recover part of the offset.
So the delivered perturbation = partial pose offset + persistent controller
bias, NOT the raw 0.1-0.5 rad offset. These gates pin that down:

  A wiring    — axis="robot" schedules the curated 393/350/398/409 variants,
                base/bddl/init resolution all consistent.
  B delivery  — vs the base task under the FULL harness reset path: nullspace
                ref = perturbed init_qpos (!= stock), arm pose differs after
                settle, object scene stays paired, instruction unchanged.
  C determin. — same episode twice (fresh session) -> identical sim state.
  D levels    — Panda{k} init_qpos offset norms match the documented
                0.1/0.2/0.3/0.4/0.5 schedule; curated k distribution printed.

Run: bash experiments/run.sh experiments/verify_robot_axis.py
"""

import re

import numpy as np

from harness.env import LiberoPlusSession, LiberoPlusTaskSet

STOCK = np.array([0.0, -1.61037389e-01, 0.0, -2.44459747e00, 0.0, 2.22675220e00, np.pi / 4])
EXPECT = {"libero_10": 393, "libero_spatial": 350, "libero_object": 398, "libero_goal": 409}
FAIL = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"[gate] {'PASS' if ok else 'FAIL'} {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def arm_state(sess):
    env = sess._env.env
    robot = env.robots[0]
    jidx = robot._ref_joint_pos_indexes
    return (
        np.array(env.sim.data.qpos[jidx]),
        np.array(robot.controller.initial_joint),
        np.asarray(robot.robot_model.init_qpos, dtype=float),
        np.array(env.sim.data.get_site_xpos("gripper0_grip_site")),
        np.array(env.sim.get_state().flatten()),
        np.array(jidx),
    )


def main():
    # --- A: wiring / schedule ---------------------------------------------
    for suite, n in EXPECT.items():
        ts = LiberoPlusTaskSet(suite, "robot")
        check(f"A count {suite}", len(ts.task_names) == n, f"{len(ts.task_names)} (expect {n})")
    ts = LiberoPlusTaskSet("libero_10", "robot")
    ks = [int(re.search(r"_initstate_(\d+)", t).group(1)) for t in ts.task_names]
    check("A initstate>0 everywhere", all(k > 0 for k in ks), f"k range {min(ks)}..{max(ks)}")
    lv = {f"0.{c + 1}": sum(1 for k in ks if c * 100 < k <= (c + 1) * 100) for c in range(5)}
    print(f"[info] libero_10 severity mix (||d||->count): {lv}")
    check("A not scene-altering", not ts.scene_altering and ts.init_states_of(ts.task_names[0]) is not None)

    sched = ts.schedule(0 or len(ts.task_names), seed=0)
    check("A single-visit init id 0", all(s.init_state_id == 0 for s in sched),
          "official num_trials=1 protocol")

    # --- B: delivery vs base under the full harness reset path ------------
    var_name = ts.task_names[0]
    base_ts = LiberoPlusTaskSet("libero_10", None)
    base_name = ts.base_task_of(var_name)
    var_spec = [s for s in sched if s.task_name == var_name][0]
    base_sched = base_ts.schedule(len(base_ts.task_names) * 10, seed=0)
    base_spec = [s for s in base_sched if s.task_name == base_name and s.init_state_id == 0][0]

    sess = LiberoPlusSession(seed=0)
    _, instr_var = sess.reset(var_spec, ts.init_states_of(var_name))
    q_var, nullref_var, model_iq_var, eef_var, sim_var, jidx = arm_state(sess)
    _, instr_base = sess.reset(base_spec, base_ts.init_states_of(base_name))
    q_base, nullref_base, model_iq_base, eef_base, sim_base, _ = arm_state(sess)

    check("B variant robot class perturbed", np.linalg.norm(model_iq_var - STOCK) > 0.05,
          f"||init_qpos-stock||={np.linalg.norm(model_iq_var - STOCK):.4f}")
    check("B base robot class stock", np.allclose(model_iq_base, STOCK, atol=1e-8))
    check("B nullspace ref = perturbed pose", np.allclose(nullref_var, model_iq_var, atol=1e-10),
          "controller.initial_joint carries the perturbation past set_init_state")
    d_arm = np.linalg.norm(q_var - q_base)
    d_eef = np.linalg.norm(eef_var - eef_base)
    check("B arm pose delivered after settle", d_arm > 0.01,
          f"|dq|={d_arm:.4f} rad, |d_eef|={d_eef * 1000:.1f} mm")
    # scene pairing: same base init state -> non-robot state identical up to
    # solver coupling (no contact between arm and objects during settle)
    mask = np.ones(len(sim_var), dtype=bool)
    nq = len(sess._env.env.sim.data.qpos)
    mask[1 + jidx] = False  # flattened state = [time, qpos, qvel]
    mask[1 + nq + jidx] = False
    d_scene = np.abs(sim_var[mask] - sim_base[mask]).max()
    check("B object scene paired with base", d_scene < 1e-3, f"max|d|={d_scene:.2e}")
    check("B instruction unchanged", instr_var == instr_base, repr(instr_var))
    sess.close()

    # --- C: determinism (fresh session, same spec) -------------------------
    sess2 = LiberoPlusSession(seed=0)
    sess2.reset(var_spec, ts.init_states_of(var_name))
    q2, nullref2, _, _, sim2, _ = arm_state(sess2)
    sess2.close()
    check("C settle state reproducible", np.array_equal(sim2, sim_var),
          f"max|d|={np.abs(sim2 - sim_var).max():.2e}")
    check("C nullspace ref reproducible", np.array_equal(nullref2, nullref_var))

    # --- D: severity levels (no sim) ---------------------------------------
    import liberoplus.liberoplus.envs.robots as R

    for c, mag in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        k = c * 100 + 1
        d = np.linalg.norm(np.asarray(getattr(R, f"OnTheGroundPanda{k}")(idn=0).init_qpos) - STOCK)
        check(f"D level k={k}", abs(d - mag) < 1e-6, f"||d||={d:.6f} (expect {mag})")

    print(f"\n[gate] {'ALL PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}", flush=True)
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
