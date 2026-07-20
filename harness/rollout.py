# SPDX-License-Identifier: Apache-2.0
"""Sequential rollout loop: obs → policy → step, with per-episode noise pinning.

Owns the model-facing data path end-to-end (docs/benchmark_facts.md):
  * Observation formatting replicates RLinf's LIBERO conventions exactly —
    both camera images rotated 180° ("to match train preprocessing",
    rlinf/envs/libero/utils.py:90), state = [eef_pos(3), axisangle(3),
    gripper_qpos(2)] (rlinf/envs/libero/libero_env.py:609-620).
  * The instruction string is passed in explicitly by the caller (from
    LiberoPlusSession.reset) and recorded per episode — delivery is data,
    not assumption.
  * Gripper mapping for LIBERO replicates rlinf/envs/action_utils.py:77-78:
    g -> sign(2g-1) * -1.
  * Flow init noise (torch.randn in the action head, no generator arg) is
    pinned by reseeding torch before EVERY chunk inference with a value
    derived from (episode_seed, control_step). Identical across arms =>
    arms differ only through the intervention, not RNG stream drift.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .env import _RUNTIME_TAIL, EpisodeSpec, LiberoPlusSession
from .video import EpisodeVideo


def variant_marker_of(spec: EpisodeSpec) -> str:
    """Perturbation marker of this episode, e.g. "language_29",
    "moved_level3_sample7", "initstate_316"; unperturbed base -> "original".

    Runtime-axis perturbations live in the pseudo-filename tail, not a
    content marker: a non-zero `_initstate_<k>` (robot axis) must surface
    here or robot-axis videos would all be labeled "original"."""
    stripped = _RUNTIME_TAIL.sub("", spec.task_name)
    marker = stripped[len(spec.base_task):].strip("_")
    tail = _RUNTIME_TAIL.search(spec.task_name)
    if tail:
        k = int(re.search(r"_initstate_(\d+)", tail.group(0)).group(1))
        if k:
            marker = f"{marker}_initstate_{k}".strip("_")
    return marker or "original"


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x,y,z,w) quaternion -> axis-angle. Copied verbatim from robosuite via
    rlinf/envs/libero/utils.py:112 (train-time state convention)."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def wrap_obs_gr00t(raw_obs: dict, instruction: str) -> dict:
    """Raw robosuite obs -> the env_obs dict GR00T's predict_action_batch expects
    (B=1). Keys/dtypes mirror rlinf libero env's _wrap_obs output."""
    main = raw_obs["agentview_image"][::-1, ::-1].copy()  # 180° rotation
    wrist = raw_obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
    state = np.concatenate(
        [
            raw_obs["robot0_eef_pos"],
            quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)
    return {
        "main_images": torch.from_numpy(main[None]),  # (1, H, W, 3) uint8
        "wrist_images": torch.from_numpy(wrist[None]),
        "states": torch.from_numpy(state[None]),  # (1, 8) float32
        "task_descriptions": [instruction],
    }


def libero_gripper_transform(chunk: np.ndarray) -> np.ndarray:
    """Model gripper in [0,1] -> LIBERO {-1,+1}; rlinf/envs/action_utils.py:77."""
    chunk = chunk.copy()
    chunk[..., -1] = 2 * chunk[..., -1] - 1
    chunk[..., -1] = np.sign(chunk[..., -1]) * -1.0
    return chunk


@dataclass
class EpisodeResult:
    episode: int
    task_name: str
    base_task: str
    init_state_id: int
    instruction: str
    success_once: int
    success_at_end: int
    n_steps: int
    wall_s: float


def run_episode(
    sess: LiberoPlusSession,
    spec: EpisodeSpec,
    init_states: Optional[np.ndarray],
    model,
    episode_seed: int,
    max_steps: int = 512,
    stop_on_success: bool = True,
    exec_horizon: Optional[int] = None,
    video_dir: Optional[str] = None,
    video_label: str = "",
    video_suite: str = "",
) -> EpisodeResult:
    """exec_horizon: execute only the first k actions of each predicted chunk
    (re-plan every k steps). The validated Isaac-GR00T LIBERO protocol uses 8
    of 16; None executes the full chunk.
    video_dir: when set, record agentview+wrist (model's view) to one mp4 per
    episode — observation consumer only, never perturbs the model/RNG path.
    video_label: model/arm tag burned into the video header (ASCII)."""
    t0 = time.time()
    raw_obs, instruction = sess.reset(spec, init_states)
    video = None
    if video_dir is not None:
        # "(suite - marker)" prefixes the DISPLAYED instruction only; the
        # model still receives the untouched instruction string
        marker = variant_marker_of(spec)
        prefix = f"({video_suite} - {marker})" if video_suite else f"({marker})"
        video = EpisodeVideo(
            video_dir, spec.episode, spec.task_name, f"{prefix} {instruction}", video_label
        )
    if video is not None:
        video.add(raw_obs)

    chunk_len = int(model.output_action_chunks)
    if exec_horizon is not None:
        chunk_len = min(chunk_len, int(exec_horizon))
    success_once = False
    steps = 0
    while steps < max_steps:
        env_obs = wrap_obs_gr00t(raw_obs, instruction)
        # pin the flow init noise for this inference; same schedule in every arm
        torch.manual_seed(episode_seed * 100_003 + steps)
        with torch.no_grad():
            raw_action, _ = model.predict_action_batch(env_obs, mode="eval")
        actions = libero_gripper_transform(np.asarray(raw_action))
        if actions.ndim == 3:  # (B=1, chunk, 7)
            actions = actions[0]

        for a in actions[:chunk_len]:
            raw_obs, _, _, _ = sess.step(a.astype(np.float32))
            steps += 1
            if video is not None:
                video.add(raw_obs)
            if sess.check_success():
                success_once = True
                break
            if steps >= max_steps:
                break
        if success_once and stop_on_success:
            break

    if video is not None:
        video.close(bool(success_once))

    return EpisodeResult(
        episode=spec.episode,
        task_name=spec.task_name,
        base_task=spec.base_task,
        init_state_id=spec.init_state_id,
        instruction=instruction,
        success_once=int(success_once),
        success_at_end=int(sess.check_success()),
        n_steps=steps,
        wall_s=round(time.time() - t0, 2),
    )
