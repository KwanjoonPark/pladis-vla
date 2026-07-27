# SPDX-License-Identifier: Apache-2.0
"""Sequential rollout loop: obs → policy → step, with per-episode noise pinning.

The loop is model-agnostic; every model-specific convention (observation
formatting, action-space decode) lives behind the ModelAdapter interface
(harness/model_base.py): wrap_obs → predict_chunk → to_env_actions.

Owned here (docs/benchmark.md):
  * The instruction string is passed in explicitly by the caller (from
    LiberoPlusSession.reset), handed to the adapter, and recorded per
    episode — delivery is data, not assumption.
  * Flow init noise is pinned by reseeding the GLOBAL torch RNG before EVERY
    chunk inference with a value derived from (episode_seed, control_step).
    Identical across arms => arms differ only through the intervention, not
    RNG stream drift. Valid only for models that draw their init noise from
    the global stream (torch.randn/torch.normal without a generator) —
    certified per model by its noise-pin gate before any sweep.
"""

from __future__ import annotations

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
    """model: a ModelAdapter (harness/model_base.py).
    exec_horizon: execute only the first k actions of each predicted chunk
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
        env_obs = model.wrap_obs(raw_obs, instruction)
        # pin the flow init noise for this inference; same schedule in every arm
        torch.manual_seed(episode_seed * 100_003 + steps)
        with torch.no_grad():
            raw_chunk = model.predict_chunk(env_obs)
        actions = model.to_env_actions(np.asarray(raw_chunk))

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
