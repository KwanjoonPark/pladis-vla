# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 loader — OFFICIAL Gr00tPolicy serving path.

History: this harness originally served the model through RLinf's
GR00T_N1_7_ForRLActionPrediction.predict_action_batch. A controlled bisect
(2026-07-14; stock LIBERO env, official protocol, same GPU/ckpt, 20 eps each)
showed that wrapper depresses libero_10 from 80% (official Gr00tPolicy;
README-reported 94.35% @ n=200) to 45% — matching the harness's depressed 46%
anchor. The RLinf dependency is gone: we now wrap the official policy behind
the same predict_action_batch interface the rollout loop already uses.

Invariants kept:
  * Noise pinning still works — official get_action samples the flow init
    noise via global torch.randn (gr00t/model/gr00t_n1d7/gr00t_n1d7.py:335),
    no private generator.
  * PLADIS hooks install unchanged — .action_head exposes the same
    AlternateVLDiT modules (pladis/attn_gr00t.py's _find_alternate_dit
    resolves action_head.model on this adapter).
  * Official protocol: Gr00tPolicy(embodiment_tag="LIBERO_PANDA") -> value
    "libero_sim" -> projector id 2; processor applies the checkpoint's
    albumentations eval transform + q01/q99 min-max state norm + relative->
    absolute action decode; decoded chunk = 16 steps (delta_indices).
"""

from __future__ import annotations

import numpy as np
import torch

from .model_base import quat2axisangle

_ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


class OfficialGr00tPolicy:
    """Official Gr00tPolicy behind the harness's ModelAdapter interface
    (harness/model_base.py): wrap_obs / predict_chunk / to_env_actions own
    every GR00T-specific observation and action convention.

    predict_action_batch(env_obs, mode="eval") -> (actions (B, 16, 7), None)
    is kept as the historical interface (verify_* scripts use it), with the
    gripper in model space [0, 1] (to_env_actions applies the LIBERO gripper
    transform, mathematically identical to the official env's
    normalize+binarize+invert).
    """

    name = "gr00t_n17"

    def __init__(self, model_path: str, device: str = "cuda"):
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

        self._policy = Gr00tPolicy(
            embodiment_tag="LIBERO_PANDA", model_path=model_path, device=device
        )
        self._wrapper = Gr00tSimPolicyWrapper(self._policy)
        # PLADIS hook point + query layout, same modules as the model exposes
        self.model = self._policy.model
        self.action_head = self._policy.model.action_head
        self.output_action_chunks = len(
            self._policy.modality_configs["action"].delta_indices
        )

    def eval(self):
        return None  # keep the old loader contract (do not chain)

    # ---- ModelAdapter interface (harness/model_base.py) ----

    @staticmethod
    def wrap_obs(raw_obs: dict, instruction: str) -> dict:
        """Raw robosuite obs -> the env_obs dict predict_chunk expects (B=1).
        Replicates RLinf's LIBERO train preprocessing exactly: both camera
        images rotated 180° (rlinf/envs/libero/utils.py:90), state =
        [eef_pos(3), axisangle(3), gripper_qpos(2)]
        (rlinf/envs/libero/libero_env.py:609-620)."""
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

    def predict_chunk(self, env_obs: dict) -> np.ndarray:
        chunk, _ = self.predict_action_batch(env_obs, mode="eval")
        arr = np.asarray(chunk)
        return arr[0] if arr.ndim == 3 else arr  # (16, 7), model action space

    @staticmethod
    def to_env_actions(chunk: np.ndarray) -> np.ndarray:
        """Model gripper in [0,1] -> LIBERO {-1,+1}; rlinf/envs/action_utils.py:77."""
        chunk = chunk.copy()
        chunk[..., -1] = 2 * chunk[..., -1] - 1
        chunk[..., -1] = np.sign(chunk[..., -1]) * -1.0
        return chunk

    # ---- historical batch interface (verify_* scripts) ----

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "eval"):
        assert mode == "eval", "harness serves eval rollouts only"
        states = env_obs["states"].float().numpy()[:, None]  # (B, 1, 8)
        flat = {
            "video.image": env_obs["main_images"].numpy()[:, None],  # (B,1,H,W,C)
            "video.wrist_image": env_obs["wrist_images"].numpy()[:, None],
            "state.x": states[..., 0:1],
            "state.y": states[..., 1:2],
            "state.z": states[..., 2:3],
            "state.roll": states[..., 3:4],
            "state.pitch": states[..., 4:5],
            "state.yaw": states[..., 5:6],
            "state.gripper": states[..., 6:8],
            "annotation.human.action.task_description": tuple(
                env_obs["task_descriptions"]
            ),
        }
        action, _ = self._wrapper.get_action(flat)
        chunk = np.concatenate(
            [np.asarray(action[f"action.{k}"]) for k in _ACTION_KEYS], axis=-1
        )  # (B, 16, 7)
        return chunk, None


def load_gr00t_n1d7(model_path: str, device: str = "cuda") -> OfficialGr00tPolicy:
    # ORDER MATTERS: MagickWand's dlopen fails if torch/cv2 have already
    # loaded conflicting shared libraries into this process (verified
    # 2026-07-14). Preload the sim stack before the policy pulls in the
    # full model/processor import graph.
    import liberoplus.liberoplus.envs  # noqa: F401  (pulls in wand)

    return OfficialGr00tPolicy(model_path, device=device)
