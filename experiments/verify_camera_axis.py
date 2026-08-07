# SPDX-License-Identifier: Apache-2.0
"""Gates for the camera axis (Camera Viewpoints, agentview re-posing).

  A. Wiring — curated counts per suite (10/goal/object/spatial =
     419/408/396/376 = 1,599) match the classification file, every variant
     carries a `_view_` tail that is NON-neutral (a neutral tuple would be a
     silently no-op arm), no variant carries `_initstate_<k>` (k>0) or a
     `_noise_` tail (those would confound this axis with robot/noise), and
     all four families are present in every suite's schedule.
  B. Delivery mechanism (libero_10, one variant per family) — the perturbed
     `sim.model.cam_pos/cam_quat` for `agentview` after reset+settle equals
     the CLOSED-FORM prediction from the tail, recomputed here from the
     paired base episode's own camera pose. This is the strong form of the
     delivery check: it proves not just "something moved" but that the
     camera moved by exactly the documented geometry, and that
     set_init_state did not revert it (cam_pos/cam_quat live in the model,
     outside `sim.get_state()`). The base pose is READ from the unperturbed
     env rather than hardcoded because it is scene-dependent (Kitchen
     [0.6586,0,1.6104], Living-Room [0.6066,0,0.96], Study [0.4586,0,1.6104]
     — libero_*_manipulation.py `_setup_camera`).
  C. Pairing + isolation (same variants) — the agentview the model would see
     differs substantially from the neutral counterpart episode (same base
     task, same init state), while the WRIST image, the post-settle sim
     state, the fixture `body_pos` and the instruction are all identical:
     the perturbation reaches exactly one camera and never touches the
     scene, the arm, or the language.
  D. Determinism — the same camera episode run in two fresh sessions yields
     bit-identical agentview streams and sim states. The axis draws no RNG
     (the pose is a pure function of the filename tail), so it is NOT in
     RUNTIME_RNG_AXES; this gate is what makes that claim testable rather
     than assumed.
  E. Cross-suite construction — one camera episode in each of the four
     suites builds and renders. libero_10/goal/object/spatial route through
     the Kitchen/Living-Room/Study problem classes; the Background-Textures
     subclasses in the same package take a SHORTER `_setup_camera` signature
     (libero_floor_manipulation.py:412), so "it works on libero_10" does not
     generalize by inspection.

`--mode video` renders the same evidence for a human reviewer: one mp4 with
the unperturbed episode and all four families side by side, driven through an
IDENTICAL scripted action sequence, so every difference between panels is the
camera and nothing else. It re-asserts the gate-C invariants per FRAME (not
just after settle) and burns the running max|d| into each panel, so the video
is a gated artifact rather than an illustration.

Run: bash experiments/run.sh experiments/verify_camera_axis.py [--mode gates|video]
"""

import argparse
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation

from harness.env import EpisodeSpec, LiberoPlusSession, LiberoPlusTaskSet, _RUNTIME_TAIL
from harness.rollout import variant_marker_of

EXPECT = {"libero_spatial": 376, "libero_object": 396, "libero_goal": 408, "libero_10": 419}
FAMILIES = ("orbit", "orbit_up", "zoom", "reaim")
NEUTRAL = (0, 0, 100, 0, 0)

_VIEW = re.compile(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_(\d+)(_noise_(\d+))?$")

PIVOT = np.array([0.0, 0.0, 0.8])  # rotate/scale pivot, shared by all helpers

# The reference pose is read from the paired base episode, and upstream rounds
# pos/quat to 4 decimals after EVERY stage — so a prediction started from the
# already-rounded base can differ from the fork's (started from the unrounded
# literal) by a few 1e-5 per stage. 5e-4 bounds that chain and is still ~300x
# below the smallest real perturbation on this axis (zoom 115% moves the
# camera ~0.15 m; the 2 deg re-aim moves the quat ~0.017).
POSE_TOL = 5e-4


def view_params(name: str) -> tuple[int, int, int, int, int]:
    m = _VIEW.search(name)
    return tuple(int(x) for x in m.groups()[:5])


def family(name: str) -> str:
    """Same partition analysis/analyze.py:camera_cat uses — kept in sync by
    gate A asserting the four families are exhaustive over the curated set."""
    h, v, s, r, e = view_params(name)
    if s != 100:
        return "zoom"
    if r or e:
        return "reaim"
    return "orbit_up" if v else "orbit"


def _rot(axis: str, degrees: int, pos=None, quat=None):
    """Replica of the fork's rotate_around_{y,z}
    (libero_tabletop_manipulation.py:47-120). Both pre-multiply the world
    rotation onto the camera orientation and rotate the position about the
    axis through PIVOT. Two upstream details this must keep:
      * rotate_around_y builds `from_rotvec(radians(-degrees) * [0,1,0])`
        (line 68) — NEGATIVE, i.e. +v elevates the camera. Reading it as a
        plain +y euler rotation mirrors the view through the table and every
        orbit_up prediction misses.
      * rotate_around_z rotates the position about the WORLD origin (line
        117, no translation). PIVOT is on the z-axis, so that is identical to
        rotating about the pivot axis — done here the pivot way for both."""
    rot = Rotation.from_euler(axis, -degrees if axis == "y" else degrees, degrees=True)
    out = {}
    if pos is not None:
        out["pos"] = np.round(rot.apply(np.asarray(pos) - PIVOT) + PIVOT, 4)
    if quat is not None:
        w, x, y, z = quat
        new = (rot * Rotation.from_quat([x, y, z, w])).as_quat()  # (x,y,z,w)
        out["quat"] = np.round(np.array([new[3], new[0], new[1], new[2]]), 4)
    return out


def predict_camera(name: str, pos_av, quat_av) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form (pos, quat) the fork's `_setup_camera` must produce for
    this variant's tail, given the scene's own unperturbed pose. The ORDER
    matters and mirrors the source (`_setup_camera`, lines 319-345):
    elevation (y) before yaw (z), then the pivot-ray scale (orientation
    untouched), then the two end-point re-aims (orientation only)."""
    h, v, s, r, e = view_params(name)
    pos, quat = np.asarray(pos_av, dtype=float), np.asarray(quat_av, dtype=float)
    if v:
        out = _rot("y", v, pos, quat)
        pos, quat = out["pos"], out["quat"]
    out = _rot("z", h, pos, quat)     # runs even at h=0 (upstream does too)
    pos, quat = out["pos"], out["quat"]
    if s != 100:
        pos = np.round(PIVOT + (pos - PIVOT) * (s / 100), 4)
    if r:
        quat = _rot("z", r, quat=quat)["quat"]
    if e:
        quat = _rot("y", e, quat=quat)["quat"]
    return pos, quat


def _spec(ts, name, init_state_id=0):
    return EpisodeSpec(0, name, ts.base_task_of(name),
                       ts._bddl[name] if name in ts._bddl
                       else f"{ts.bddl_dir}/{name}.bddl", init_state_id)


def _run(sess, spec, init_states, n_steps=3):
    """reset+settle then n_steps noops; returns everything the gates compare."""
    obs, instruction = sess.reset(spec, init_states)
    avs = [obs["agentview_image"].copy()]
    for _ in range(n_steps):
        obs, _, _, _ = sess.step(LiberoPlusSession.NOOP)
        avs.append(obs["agentview_image"].copy())
    env = sess._env.env
    m = env.sim.model
    cid = m.camera_name2id("agentview")
    return {
        "avs": avs,
        "wrist": obs["robot0_eye_in_hand_image"].copy(),
        "state": env.sim.get_state().flatten().copy(),
        "cam_pos": m.cam_pos[cid].copy(),
        "cam_quat": m.cam_quat[cid].copy(),
        "body_pos": {m.body_id2name(i): m.body_pos[i].copy() for i in range(m.nbody)},
        "instruction": instruction,
    }


def gate_A():
    total = 0
    for suite, expect in EXPECT.items():
        ts = LiberoPlusTaskSet(suite, "camera")
        assert len(ts.task_names) == expect, \
            f"A: {suite} count {len(ts.task_names)} != {expect}"
        sched = ts.schedule(len(ts.task_names), seed=0)
        fams = set()
        for s in sched:
            m = _VIEW.search(s.task_name)
            assert m, f"A: {s.task_name} has no parseable _view_ tail"
            p = view_params(s.task_name)
            assert p != NEUTRAL, f"A: {s.task_name} is a NEUTRAL view — no-op arm"
            assert int(m.group(6)) == 0, \
                f"A: {s.task_name} carries initstate>0 — confounded with the robot axis"
            assert m.group(7) is None, \
                f"A: {s.task_name} carries a _noise_ tail — confounded with the noise axis"
            # the variant name must be the BASE task plus the tail alone: a
            # content marker (_language_/_light_/_add_) would mean this axis
            # is silently carrying a second perturbation
            assert _RUNTIME_TAIL.sub("", s.task_name) == s.base_task, \
                f"A: {s.task_name} carries a content marker on top of the view tail"
            fams.add(family(s.task_name))
            assert variant_marker_of(s) == f"view_{'_'.join(str(x) for x in p)}", \
                f"A: {s.task_name} mislabels as '{variant_marker_of(s)}'"
        assert fams == set(FAMILIES), f"A: {suite} missing families {set(FAMILIES) - fams}"
        total += len(ts.task_names)
    assert total == 1599, f"A: curated total {total} != 1,599"
    print(f"PASS gate A: wiring — {total} curated variants, all non-neutral, "
          f"initstate=0, no noise tail, 4 families per suite, markers surface")


def _picks(ts, sched):
    """One variant per family, all sharing ONE base task where possible so the
    env rebuild count stays low and the families are compared like-for-like."""
    by_base = {}
    for s in sched:
        by_base.setdefault(s.base_task, {}).setdefault(family(s.task_name), s)
    full = [b for b, d in by_base.items() if len(d) == len(FAMILIES)]
    assert full, "no base task carries all four camera families"
    return by_base[sorted(full)[0]]


def gates_BC():
    ts = LiberoPlusTaskSet("libero_10", "camera")
    ts0 = LiberoPlusTaskSet("libero_10", None)
    picks = _picks(ts, ts.schedule(len(ts.task_names), seed=0))
    sess = LiberoPlusSession(seed=0)
    base_name = next(iter(picks.values())).base_task
    base = _spec(ts0, base_name, init_state_id=0)
    ref = _run(sess, base, ts0.init_states_of(base_name))
    print(f"  base task {base_name}")
    print(f"  unperturbed agentview pos={np.round(ref['cam_pos'], 4)} "
          f"quat={np.round(ref['cam_quat'], 4)}")
    for fam in FAMILIES:
        spec = picks[fam]
        got = _run(sess, spec, ts.init_states_of(spec.task_name))
        exp_pos, exp_quat = predict_camera(spec.task_name, ref["cam_pos"], ref["cam_quat"])
        # MuJoCo stores the quat as written; sign is not normalized, so compare
        # up to the q ~ -q double cover.
        dq = min(np.abs(got["cam_quat"] - exp_quat).max(),
                 np.abs(got["cam_quat"] + exp_quat).max())
        assert np.abs(got["cam_pos"] - exp_pos).max() < POSE_TOL, \
            f"B: {fam} cam_pos {got['cam_pos']} != predicted {exp_pos}"
        assert dq < POSE_TOL, f"B: {fam} cam_quat {got['cam_quat']} != predicted {exp_quat}"
        moved = np.abs(got["cam_pos"] - ref["cam_pos"]).max()
        turned = min(np.abs(got["cam_quat"] - ref["cam_quat"]).max(),
                     np.abs(got["cam_quat"] + ref["cam_quat"]).max())
        assert moved > 1e-3 or turned > 1e-3, \
            f"B: {fam} camera did not move — silent nullification"
        d_av = np.mean(np.abs(got["avs"][-1].astype(np.float32)
                              - ref["avs"][-1].astype(np.float32)))
        d_wr = np.abs(got["wrist"].astype(np.int16) - ref["wrist"].astype(np.int16)).max()
        assert d_av > 1.0, f"C: {fam} agentview unchanged (mean|d|={d_av:.3f})"
        assert d_wr == 0, f"C: {fam} WRIST changed (max|d|={d_wr}) — not agentview-only"
        assert np.array_equal(got["state"], ref["state"]), \
            f"C: {fam} sim state differs — scene not paired"
        # Fixtures live in model.body_pos, outside sim.get_state() (the layout
        # lesson, docs/benchmark.md) — a name-dependent placement draw at build
        # would move them without moving the state vector, so compare them too.
        strays = {k for k in got["body_pos"]
                  if not np.array_equal(got["body_pos"][k], ref["body_pos"].get(k))}
        assert not strays, f"C: {fam} bodies moved across name types: {sorted(strays)}"
        assert got["instruction"] == ref["instruction"], \
            f"C: {fam} instruction differs — language confound"
        print(f"  [{fam:8s}] view={view_params(spec.task_name)} "
              f"|dpos|={moved:.4f} |dquat|={turned:.4f} "
              f"agentview mean|d|={d_av:5.1f} wrist/state/body_pos/instruction identical")
    sess.close()
    print("PASS gate B: agentview pose == closed-form prediction from the tail "
          "(survives set_init_state)")
    print("PASS gate C: perturbation is agentview-only — wrist, sim state, "
          "body_pos and instruction all bit-identical to the paired base episode")


def gate_D():
    ts = LiberoPlusTaskSet("libero_10", "camera")
    picks = _picks(ts, ts.schedule(len(ts.task_names), seed=0))
    for fam in FAMILIES:
        spec = picks[fam]
        runs = []
        for _ in range(2):
            sess = LiberoPlusSession(seed=0)
            runs.append(_run(sess, spec, ts.init_states_of(spec.task_name)))
            sess.close()
        a, b = runs
        assert all(np.array_equal(x, y) for x, y in zip(a["avs"], b["avs"])), \
            f"D: {fam} agentview stream not deterministic across fresh sessions"
        assert np.array_equal(a["state"], b["state"]), f"D: {fam} sim state not deterministic"
        assert np.array_equal(a["cam_pos"], b["cam_pos"]) and \
            np.array_equal(a["cam_quat"], b["cam_quat"]), f"D: {fam} camera pose not deterministic"
    print("PASS gate D: all four families bit-deterministic across fresh sessions "
          "(no RNG on this axis -> not in RUNTIME_RNG_AXES)")


def gate_E():
    for suite in EXPECT:
        ts = LiberoPlusTaskSet(suite, "camera")
        ts0 = LiberoPlusTaskSet(suite, None)
        spec = ts.schedule(1, seed=0)[0]
        sess = LiberoPlusSession(seed=0)
        ref = _run(sess, _spec(ts0, spec.base_task), ts0.init_states_of(spec.base_task),
                   n_steps=1)
        got = _run(sess, spec, ts.init_states_of(spec.task_name), n_steps=1)
        sess.close()
        exp_pos, exp_quat = predict_camera(spec.task_name, ref["cam_pos"], ref["cam_quat"])
        dq = min(np.abs(got["cam_quat"] - exp_quat).max(),
                 np.abs(got["cam_quat"] + exp_quat).max())
        assert np.abs(got["cam_pos"] - exp_pos).max() < POSE_TOL and dq < POSE_TOL, \
            f"E: {suite} pose {got['cam_pos']}/{got['cam_quat']} != predicted"
        print(f"  [{suite:14s}] {family(spec.task_name):8s} "
              f"view={view_params(spec.task_name)} base pos={np.round(ref['cam_pos'], 4)} "
              f"-> matches prediction")
    print("PASS gate E: all four suites construct and deliver the predicted pose")


# ---------------------------------------------------------------- video mode

# Scripted OSC_POSE deltas, identical for every panel. A noop sequence would
# render five near-static images and prove nothing about whether the scene
# tracks together, so the arm is driven through a circle in the xy-plane with a
# slow descent and a gripper toggle. Deterministic and model-free on purpose:
# a policy in the loop would make the panels diverge for a reason that has
# nothing to do with the camera.
def scripted_action(t: int, period: int = 40) -> np.ndarray:
    phase = 2 * np.pi * t / period
    return np.array([0.35 * np.sin(phase), 0.35 * np.cos(phase), -0.12,
                     0.0, 0.0, 0.0,
                     1.0 if (t // (period // 2)) % 2 else -1.0], dtype=np.float32)


def _panel(av, wrist, caption, sub, d_av, d_wr, font, size=256):
    """One column: agentview over wrist, both rotated 180 deg exactly as
    OfficialGr00tPolicy.wrap_obs (and harness/video.py) does, so the panel
    shows what the model would receive rather than the raw render.

    The two cameras carry DIFFERENT statistics on purpose. Agentview reports
    mean|d|, because max|d| saturates at 255 the moment any pixel differs and
    would read identically for a 2 deg re-aim and a 75 deg orbit. Wrist reports
    max|d|, because the claim there is bit-identity and only the max can
    witness it."""
    import cv2

    def _cell(img, tag, stat, val, colour):
        cell = np.ascontiguousarray(img[::-1, ::-1])
        if cell.shape[0] != size:
            cell = cv2.resize(cell, (size, size), interpolation=cv2.INTER_AREA)
        bar = np.full((20, size, 3), 18, np.uint8)
        cv2.putText(bar, f"{tag} {stat}|d| vs base {val:6.1f}", (5, 14),
                    font, 0.40, colour, 1, cv2.LINE_AA)
        return np.vstack([cell, bar])

    head = np.full((36, size, 3), 30, np.uint8)
    cv2.putText(head, caption, (5, 15), font, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(head, sub, (5, 30), font, 0.38, (150, 200, 255), 1, cv2.LINE_AA)
    return np.vstack([
        head,
        _cell(av, "AGENTVIEW", "mean", d_av,
              (120, 200, 255) if d_av > 1 else (110, 110, 110)),
        _cell(wrist, "WRIST", " max", d_wr,
              (120, 255, 140) if d_wr == 0 else (255, 110, 110)),
    ])


def render_video(out_path: str, steps: int, suite: str):
    """One mp4: unperturbed + the four families, lockstep on identical actions.

    Every panel is an independent run of the SAME base task and init state, so
    the sim must evolve identically in all five; the per-frame asserts below
    are gate C re-applied at every step instead of once after settle."""
    import cv2
    import imageio.v2 as imageio

    ts = LiberoPlusTaskSet(suite, "camera")
    ts0 = LiberoPlusTaskSet(suite, None)
    picks = _picks(ts, ts.schedule(len(ts.task_names), seed=0))
    base_task = next(iter(picks.values())).base_task
    columns = [("base", _spec(ts0, base_task), ts0.init_states_of(base_task))] + [
        (fam, picks[fam], ts.init_states_of(picks[fam].task_name)) for fam in FAMILIES
    ]

    sess = LiberoPlusSession(seed=0)
    streams, states, poses = {}, {}, {}
    for name, spec, init_states in columns:
        obs, instruction = sess.reset(spec, init_states)
        frames = [(obs["agentview_image"].copy(), obs["robot0_eye_in_hand_image"].copy())]
        for t in range(steps):
            obs, _, _, _ = sess.step(scripted_action(t))
            frames.append((obs["agentview_image"].copy(),
                           obs["robot0_eye_in_hand_image"].copy()))
        env = sess._env.env
        cid = env.sim.model.camera_name2id("agentview")
        streams[name] = frames
        states[name] = env.sim.get_state().flatten().copy()
        poses[name] = (env.sim.model.cam_pos[cid].copy(),
                       env.sim.model.cam_quat[cid].copy())
        print(f"  [{name:8s}] {steps + 1} frames  "
              f"cam_pos={np.round(poses[name][0], 4)}", flush=True)
    sess.close()

    # The video claims "same scene, different camera" — assert it before writing
    # one, on the FULL trajectory rather than the post-settle frame alone.
    for name, _, _ in columns[1:]:
        assert np.array_equal(states[name], states["base"]), \
            f"video: {name} sim state diverged from base under identical actions"
        assert all(np.array_equal(w, wb) for (_, w), (_, wb)
                   in zip(streams[name], streams["base"])), \
            f"video: {name} wrist stream differs from base — not agentview-only"

    font = cv2.FONT_HERSHEY_SIMPLEX
    caps = {"base": ("base (unperturbed)", "view 0_0_100_0_0")}
    for fam, spec, _ in columns[1:]:
        h, v, s, r, e = view_params(spec.task_name)
        caps[fam] = (fam, f"h={h} v={v} scale={s}% rot={r} vert={e}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=20, codec="libx264", quality=7,
                                pixelformat="yuv420p")
    peak = {}
    for i in range(steps + 1):
        cols = []
        for name, _, _ in columns:
            av, wr = streams[name][i]
            avb, wrb = streams["base"][i]
            d_av = float(np.abs(av.astype(np.int16) - avb.astype(np.int16)).mean())
            d_wr = float(np.abs(wr.astype(np.int16) - wrb.astype(np.int16)).max())
            peak[name] = max(peak.get(name, 0.0), d_av)
            cols.append(_panel(av, wr, caps[name][0], caps[name][1], d_av, d_wr, font))
        grid = np.hstack(cols)
        banner = np.full((26, grid.shape[1], 3), 20, np.uint8)
        cv2.putText(banner, f"LIBERO-plus camera axis | {suite} | {base_task[:70]} "
                            f"| identical scripted actions | step {i}",
                    (6, 18), font, 0.44, (235, 235, 235), 1, cv2.LINE_AA)
        frame = np.vstack([banner, grid])
        # Pad to a multiple of the ffmpeg macro block instead of letting the
        # writer resize: a resize resamples every panel and would blur the very
        # detail the video exists to show (harness/video.py does the same, via
        # its header height).
        pad = -frame.shape[0] % 16
        if pad:
            frame = np.vstack([frame, np.full((pad, frame.shape[1], 3), 20, np.uint8)])
        writer.append_data(frame)
    writer.close()
    print(f"\n  per-family peak agentview mean|d| vs base: "
          + ", ".join(f"{k}={v:.1f}" for k, v in peak.items() if k != "base"))
    print(f"  wrist max|d| vs base: 0 on every frame of every family (asserted)")
    print(f"PASS video: {out_path} ({steps + 1} frames, {len(columns)} panels)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gates", "video"], default="gates")
    p.add_argument("--out", default="results/camera_axis_check.mp4",
                   help="mp4 path for --mode video")
    p.add_argument("--steps", type=int, default=120, help="control steps per panel")
    p.add_argument("--suite", default="libero_10")
    args = p.parse_args()
    if args.mode == "video":
        render_video(args.out, args.steps, args.suite)
        return
    gate_A(); gates_BC(); gate_D(); gate_E()
    # smoke_model.py is language-specific by construction (its pass condition
    # is "the variant instruction differs from the anchor's"), so the model
    # smoke for this axis is a 2-episode eval_arm run on the real sweep path.
    print("ALL GATES PASSED (env-level; run the 2-ep eval_arm --axis camera "
          "smoke before sweeping)")


if __name__ == "__main__":
    main()
