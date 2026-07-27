# SPDX-License-Identifier: Apache-2.0
"""Official lerobot SmolVLAPolicy adapter (HuggingFaceVLA/smolvla_libero).

Observation conventions replicate lerobot's own LIBERO eval stack — the
train-time convention source for this checkpoint:
  * camera mapping ``agentview_image -> observation.images.image``,
    ``robot0_eye_in_hand_image -> observation.images.image2``
    (lerobot/envs/libero.py:140-141);
  * both images rotated 180° — the HuggingFaceVLA/libero orientation
    convention (lerobot/processor/env_processor.py libero_processor);
  * state = [eef_pos(3), eef axis-angle(3), gripper_qpos(2)] (same step);
  * images float32 CHW in [0,1] (lerobot/envs/utils.py:80-83); the policy
    preprocessor pipeline stored with the checkpoint (to_batch, newline,
    tokenizer padding="longest", device, normalizer) handles the rest.

Action path: ``predict_action_chunk`` -> the checkpoint's postprocessor
(unnormalizer) -> the LIBERO env's native 7-dim OSC-delta space; lerobot's
LIBERO env steps that action raw, so ``to_env_actions`` is the identity.

Noise-pin admissible: SmolVLA ``sample_noise`` draws ``torch.normal`` from
the GLOBAL stream (modeling_smolvla.py:609) — the rollout loop's per-chunk
reseed pins the flow init noise across arms.
"""

from __future__ import annotations

import numpy as np

from harness.model_base import quat2axisangle


class SmolVLAAdapter:
    name = "smolvla"

    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.policy = SmolVLAPolicy.from_pretrained(model_path)
        self.policy.to(device)
        self.policy.eval()
        self.policy.reset()  # ACTION queue only; predict_action_chunk is stateless
        self.pre, self.post = make_pre_post_processors(self.policy.config, model_path)
        self.device = device
        self.output_action_chunks = int(self.policy.config.chunk_size)  # 50

    def eval(self):
        self.policy.eval()
        return self

    @staticmethod
    def wrap_obs(raw_obs: dict, instruction: str) -> dict:
        """Raw robosuite obs -> the env_obs dict predict_chunk expects."""
        main = raw_obs["agentview_image"][::-1, ::-1].copy()  # 180° rotation
        wrist = raw_obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        state = np.concatenate(
            [
                raw_obs["robot0_eef_pos"],
                quat2axisangle(raw_obs["robot0_eef_quat"]),
                raw_obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32)
        return {"image": main, "image2": wrist, "state": state, "task": instruction}

    def predict_chunk(self, env_obs: dict) -> np.ndarray:
        torch = self.torch

        def img(x):
            return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255.0

        batch = {
            "observation.images.image": img(env_obs["image"]),
            "observation.images.image2": img(env_obs["image2"]),
            "observation.state": torch.from_numpy(env_obs["state"]).float(),
            "task": env_obs["task"],
        }
        batch = self.pre(batch)
        chunk = self.policy.predict_action_chunk(batch)  # (1, chunk, 7), model space
        chunk = self.post(chunk)
        return chunk.squeeze(0).float().cpu().numpy()

    @staticmethod
    def to_env_actions(chunk: np.ndarray) -> np.ndarray:
        return chunk


def load_smolvla(model_path: str, device: str = "cuda") -> SmolVLAAdapter:
    # ORDER MATTERS: MagickWand's dlopen fails if torch/cv2 have already
    # loaded conflicting shared libraries into this process (verified
    # 2026-07-14, same guard as load_gr00t_n1d7). Preload the sim stack
    # before the policy pulls in the full model/processor import graph.
    import liberoplus.liberoplus.envs  # noqa: F401  (pulls in wand)

    return SmolVLAAdapter(model_path, device)
