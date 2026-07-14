# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 loader — the one place the harness leans on RLinf (read-only).

The model wrapper (obs transforms, unnormalization, chunked action output)
is battle-tested RLinf code; we import it rather than re-owning it. The cfg
below replicates RLinf's examples/embodiment/config/model/gr00t_n1d7.yaml
defaults verbatim — if RLinf changes those defaults, this stays pinned.

PLADIS hooks are NOT installed here and NOT via env vars: arms install them
explicitly through pladis.attn_gr00t.install_pladis(model, ...) so every
experiment's configuration is visible Python data.
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf

RLINF_PATH = os.environ.get(
    "RLINF_PATH", "/home/reallab/parkkwanjoon/workspace/RLinf"
)

# pinned copy of examples/embodiment/config/model/gr00t_n1d7.yaml (eval-relevant
# fields; value head kept for from_pretrained compatibility, unused in eval)
GR00T_N1D7_CFG = {
    "model_type": "gr00t_n1d7",
    "precision": "bf16",
    "action_dim": 7,
    "num_action_chunks": 16,
    "denoising_steps": 4,
    "obs_converter_type": "libero",
    "embodiment_tag": "libero_sim",
    "add_value_head": True,
    "rl_head_config": {
        "joint_logprob": False,
        "noise_method": "flow_sde",
        "ignore_last": False,
        "safe_get_logprob": False,
        "noise_anneal": False,
        "noise_params": [0.7, 0.3, 400],
        "noise_level": 0.5,
        "action_noise_scale": 0.1,
        "add_value_head": True,
        "chunk_critic_input": False,
        "detach_critic_input": True,
        "disable_dropout": True,
        "use_vlm_value": False,
        "value_vlm_mode": "mean_token",
        "padding_value": 570,
    },
}


def load_gr00t_n1d7(
    model_path: str,
    backbone_model_path: str,
    device: str = "cuda",
):
    # ORDER MATTERS: MagickWand's dlopen fails if torch/cv2 have already
    # loaded conflicting shared libraries into this process (verified
    # 2026-07-14: wand imports fine standalone, fails after model load).
    # RLinf never saw this because env and rollout were separate Ray
    # processes. Preload the sim stack before any model imports.
    import liberoplus.liberoplus.envs  # noqa: F401  (pulls in wand)

    if RLINF_PATH not in sys.path:
        sys.path.insert(0, RLINF_PATH)

    cfg = OmegaConf.create(
        {**GR00T_N1D7_CFG, "model_path": model_path, "backbone_model_path": backbone_model_path}
    )

    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    assert os.environ.get("PLADIS_ENABLE", "0") == "0", (
        "PLADIS_ENABLE is set — this harness installs hooks explicitly via "
        "pladis.attn_gr00t.install_pladis(), not via RLinf's env-var gate."
    )
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()  # RLinf's eval() override returns None — do not chain
    return model
