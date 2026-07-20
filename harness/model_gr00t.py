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

_ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


class OfficialGr00tPolicy:
    """Official Gr00tPolicy behind the harness's model-facing interface.

    predict_action_batch(env_obs, mode="eval") -> (actions (B, 16, 7), None)
    with the gripper in model space [0, 1] (rollout applies the LIBERO
    gripper transform, mathematically identical to the official env's
    normalize+binarize+invert).
    """

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
