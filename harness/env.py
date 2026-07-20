# SPDX-License-Identifier: Apache-2.0
"""LIBERO-plus episode provider — the harness owns every input the model sees.

Contract (docs/benchmark_facts.md):
  * Schedule comes from the CURATED benchmark list (task_classification.json),
    not from `ls bddl_files` — the on-disk 500/suite include pre-filter easy
    tasks the paper removed (libero_10 language: 500 on disk, 383 curated).
  * The instruction handed to the model is `env.language_instruction`, i.e.
    liberoplus's own parse of the loaded bddl. Never task-suite metadata
    (that was the RLinf bug that silently evaluated original instructions).
  * Fixed init states: the base task's `.pruned_init` (50, D) applied via
    set_init_state, with a hard dim assert (catches scene-mismatched axes
    like `_add` where the state vector grows).
  * EXCEPT scene-altering axes (layout): the BDDL itself moves the placement
    regions (`_level_sample`) or adds objects (`_add`), so applying the base
    task's init states would silently restore the original layout (the
    perturbation-not-delivered failure mode) or dim-crash on `_add`. There
    the scene comes from the BDDL's own placement sampling, made
    deterministic and arm-paired by reseeding right before EVERY reset:
    bddl_base_domain.seed(s) is np.random.seed(s) and robosuite placement
    samplers draw from that global stream.
  * The schedule is an explicit, seeded, logged list of EpisodeSpec — no
    implicit `(seed+idx) % len` scattered in env code.
  * Plain bddl paths carry no hidden runtime randomizers (env_wrapper only
    activates camera/robot/noise perturbations for `_view_..._initstate_...`
    pseudo-filenames; see benchmark_facts.md).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

AXIS_TO_CATEGORY = {
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "layout": "Objects Layout",
    # robot is a runtime axis: the curated name's `_initstate_<k>` (k>0) makes
    # env_wrapper swap in a Panda{k} robot class whose init_qpos is perturbed
    # (levels ||d||=0.1..0.5 by k-century, envs/robots/new_init.py). Delivery
    # under the official protocol (reset -> set_init_state(base) -> settle,
    # LIBERO-plus README: "identical to LIBERO, num_trials=1") is INDIRECT:
    # set_init_state snaps the arm back to the base pose, but the OSC
    # controller's nullspace reference (initial_joint, captured at reset =
    # perturbed pose; set_init_state never updates it) pulls toward the
    # perturbed config through the whole episode, and the settle steps recover
    # part of the pose offset. Same base init states as axis=None -> scene
    # stays paired with the original arm. Gates: verify_robot_axis.py.
    "robot": "Robot Initial States",
    # camera / noise runtime axes: supported by env_wrapper, not wired yet.
}

# Axes whose BDDL changes the scene itself (moved regions / added objects):
# base-task init states must NOT be applied (see module docstring). The
# curated layout names carry no `_view_..._initstate_` runtime tail.
SCENE_ALTERING_AXES = frozenset({"layout"})

# Content-variant markers, exhaustively enumerated from task_classification.json
# (libero_10: language 383, light 274, add 173, table 168, level+sample 139,
# tb 121; the remaining 1,261 have no marker = runtime-only axes). Note the
# layout combo form "_level3_sample7" — digits attach directly. libero_goal
# layout uses "_moved_level3_sample7" (185 entries, goal-only; no bare _moved).
_VARIANT_MARKER = re.compile(
    r"(_(language|light|table|tb|add)_\d+|(_moved)?_level\d+_sample\d+)$"
)

# Curated task names canonically carry a runtime-axis pseudo-suffix, e.g.
# "..._language_10_view_0_0_100_0_0_initstate_0[_noise_1]". env_wrapper parses
# camera/robot/noise params from it (all-neutral = no perturbation) and strips
# it to find the real bddl. We pass the full pseudo path through, but check
# existence of (and load init states for) the stripped real file.
_RUNTIME_TAIL = re.compile(r"_view_\d+_\d+_\d+_\d+_\d+_initstate_\d+(_noise_\d+)?$")


def _pkg_root() -> str:
    import liberoplus.liberoplus as lp

    return os.path.dirname(os.path.abspath(lp.__file__))


@dataclass(frozen=True)
class EpisodeSpec:
    episode: int  # position in the schedule (pairing key across arms)
    task_name: str  # curated variant name == bddl stem (or base name for axis=None)
    base_task: str  # variant name with the axis marker stripped
    bddl_path: str
    # row of the base task's pruned_init; scene-altering axes have no fixed
    # states, there it is the per-variant visit counter (bookkeeping only —
    # the reset seed keys on `episode`)
    init_state_id: int


class LiberoPlusTaskSet:
    """Curated task list + fixed init states for one (suite, axis)."""

    def __init__(self, suite: str = "libero_10", axis: Optional[str] = "language"):
        self.suite = suite
        self.axis = axis
        root = _pkg_root()
        self.bddl_dir = os.path.join(root, "bddl_files", suite)
        self.init_dir = os.path.join(root, "init_files", suite)

        with open(os.path.join(root, "benchmark", "task_classification.json")) as f:
            classification = json.load(f)
        if axis is None:
            # anchor mode: base tasks derived from the curated list itself
            # (never `ls` the bddl dir — it contains stray files, e.g. "* copy.bddl")
            names = sorted({self.base_task_of(t["name"]) for t in classification[suite]})
        else:
            category = AXIS_TO_CATEGORY[axis]
            names = sorted(
                t["name"] for t in classification[suite] if t["category"] == category
            )
        self.task_names = names

        self._bddl = {}
        for name in names:
            real = _RUNTIME_TAIL.sub("", name)
            assert os.path.exists(os.path.join(self.bddl_dir, real + ".bddl")), (
                f"curated task has no bddl on disk: {real}.bddl"
            )
            # full pseudo path: env_wrapper parses (neutral) runtime params from it
            self._bddl[name] = os.path.join(self.bddl_dir, name + ".bddl")

        # base-task pruned init states, loaded eagerly (10 small files);
        # scene-altering axes use none (BDDL placement sampling instead)
        self.scene_altering = axis in SCENE_ALTERING_AXES
        self._init_states: dict[str, np.ndarray] = {}
        if not self.scene_altering:
            for name in names:
                base = self.base_task_of(name)
                if base not in self._init_states:
                    p = os.path.join(self.init_dir, base + ".pruned_init")
                    assert os.path.exists(p), f"missing init states: {p}"
                    states = torch.load(p, map_location="cpu", weights_only=False)
                    self._init_states[base] = np.asarray(states, dtype=np.float64)

    def base_task_of(self, name: str) -> str:
        return _VARIANT_MARKER.sub("", _RUNTIME_TAIL.sub("", name))

    def init_states_of(self, name: str) -> Optional[np.ndarray]:
        if self.scene_altering:
            return None
        return self._init_states[self.base_task_of(name)]

    def schedule(self, n_episodes: int, seed: int) -> list[EpisodeSpec]:
        """Explicit paired schedule: same (suite, axis, n, seed) => same list.

        Variants are visited in a seeded permutation, cycling as needed; the
        init state id advances deterministically per visit of each variant.
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.task_names))
        visits: dict[str, int] = {}
        specs = []
        for i in range(n_episodes):
            name = self.task_names[order[i % len(order)]]
            k = visits.get(name, 0)
            visits[name] = k + 1
            if self.scene_altering:
                init_state_id = seed + k  # visit counter, no fixed-state list
            else:
                init_state_id = (seed + k) % len(self.init_states_of(name))
            specs.append(
                EpisodeSpec(
                    episode=i,
                    task_name=name,
                    base_task=self.base_task_of(name),
                    bddl_path=self._bddl[name],
                    init_state_id=init_state_id,
                )
            )
        return specs


class LiberoPlusSession:
    """Owns one live simulator; reloads only when the bddl changes."""

    # standard LIBERO practice: after set_init_state, let physics settle with
    # no-op actions (gripper open) before handing control to the policy
    SETTLE_STEPS = 10
    NOOP = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)

    def __init__(self, camera_height: int = 256, camera_width: int = 256, seed: int = 0):
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.seed = seed
        self._env = None
        self._loaded_bddl = None

    def _load(self, bddl_path: str):
        from liberoplus.liberoplus.envs import OffScreenRenderEnv

        if self._env is not None:
            self._env.close()
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
        )
        self._env.seed(self.seed)
        self._loaded_bddl = bddl_path

    def reset(self, spec: EpisodeSpec, init_states: Optional[np.ndarray]):
        """Returns (obs, instruction). Instruction is liberoplus's bddl parse.

        init_states None = scene-altering axis: the episode scene IS the
        BDDL's placement sampling. Reseed the global np.random stream (what
        robosuite samplers draw from) right before reset — same value in
        every arm (keys on run seed + schedule position only), so arms see
        identical initial scenes; same convention as the flow-noise pin."""
        if spec.bddl_path != self._loaded_bddl:
            self._load(spec.bddl_path)

        if init_states is None:
            self._env.seed(self.seed * 1_000_003 + spec.episode)
            obs = self._env.reset()
        else:
            self._env.reset()
            state = init_states[spec.init_state_id]
            sim_dim = len(self._env.env.sim.get_state().flatten())
            assert state.shape[-1] == sim_dim, (
                f"init state dim {state.shape[-1]} != sim dim {sim_dim} for "
                f"{spec.task_name} — scene-altering axis? (see benchmark_facts.md)"
            )
            obs = self._env.set_init_state(state)
        for _ in range(self.SETTLE_STEPS):
            obs, _, _, _ = self._env.step(self.NOOP)

        instruction = self._env.language_instruction
        return obs, instruction

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    def check_success(self) -> bool:
        return bool(self._env.check_success())

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._loaded_bddl = None
