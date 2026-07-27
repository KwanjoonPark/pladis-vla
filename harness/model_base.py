# SPDX-License-Identifier: Apache-2.0
"""Model-adapter contract for the rollout loop, plus conversions shared by
adapters.

The loop (harness/rollout.py) is model-agnostic; everything a model needs to
control lives behind this interface:

    class ModelAdapter(Protocol):
        name: str                    # registry key, e.g. "gr00t_n17"
        output_action_chunks: int    # decoded chunk length
        def wrap_obs(raw_obs, instruction) -> Any
            # raw robosuite obs + instruction -> whatever predict_chunk eats.
            # Owns the model's TRAIN-TIME observation conventions (camera
            # rotation, state layout, key names).
        def predict_chunk(env_obs) -> np.ndarray      # (chunk, 7), MODEL space
        def to_env_actions(chunk) -> np.ndarray       # -> LIBERO env space

Noise-pinning contract: the loop reseeds the GLOBAL torch RNG before every
predict_chunk call. An adapter is only admissible if its model draws the flow
init noise from the global stream (torch.randn/torch.normal without a
generator) — verified per model by its noise-pin gate before any sweep.
"""

from __future__ import annotations

import math

import numpy as np


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
