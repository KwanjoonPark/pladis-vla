# SPDX-License-Identifier: Apache-2.0
"""π0.5 loader — OFFICIAL openpi ``Policy`` serving path.

Route choice: the model is built with ``openpi.*`` only. RLinf's
``OpenPi0ForRLActionPrediction`` is deliberately NOT on this path, even though its
``predict_action_batch`` has the same shape as the interface below.

Why: the GR00T track already paid for that lesson. A controlled bisect
(harness/model_gr00t.py:4-11) showed RLinf's RL wrapper depressing libero_10 from 80%
(official Gr00tPolicy) to 45%. RLinf's openpi wrapper carries the same risk surface —
its own Euler loop, its own de-normalization, its own image pipeline — and it exists to
serve *training* rollouts, not to reproduce published eval numbers. ``Policy.infer`` is
the exact object that produced openpi's published LIBERO table
(openpi/examples/libero/README.md: pi0.5 @ 30k = 98.8 / 98.2 / 98.0 / 92.4).
``run.sh`` still puts RLinf on PYTHONPATH, but only for the serving-route bisect
reference (toolkits/standalone_eval_scripts/openpi/libero_eval.py); no ``rlinf.`` symbol
is imported here.

Invariants kept (same three as the GR00T adapter):
  * Noise pinning still works — pi0_pytorch.py:172-179 ``sample_noise`` uses
    ``torch.normal`` with NO generator, i.e. the global torch RNG, exactly like GR00T's
    ``torch.randn``. harness/rollout.py's per-chunk ``torch.manual_seed`` transfers
    unchanged. (openpi's Euler loop draws the initial noise once; RLinf's re-draws every
    step. Another reason to be on this route: the RNG stream here is the simple one.)
  * PLADIS hooks install unchanged — ``pladis/attn_pi05.py`` monkeypatches
    transformers' gemma ``eager_attention_forward``, and the expert's dispatch site
    resolves that name as a module global at call time
    (openpi transformers_replace/models/gemma/modeling_gemma.py:312-314), so the patch
    fires without needing a handle on any module.
  * Official protocol: TrainConfig ``pi05_libero`` -> Pi0Config(pi05=True,
    action_horizon=10, discrete_state_input=False); quantile norm from the checkpoint's
    own ``assets/physical-intelligence/libero/norm_stats.json``; ResizeImages to 224 with
    pad happens inside ``Policy.infer``, not here.
"""

from __future__ import annotations

import numpy as np
import torch

from .rollout import wrap_obs_gr00t

# pi05_libero decodes a 10-step chunk (Pi0Config.action_horizon); the eval protocol
# executes the first 5 (standalone libero_eval.py's `action_chunk`). The harness applies
# the second number via --exec-horizon, so it is NOT baked in here.
_TRAIN_CONFIG_NAME = "pi05_libero"


def _identity_chunk(chunk: np.ndarray) -> np.ndarray:
    """π0.5 actions are ALREADY in LIBERO's native gripper convention.

    harness/rollout.py:86-91 ``libero_gripper_transform`` (``g -> sign(2g-1) * -1``) is
    GR00T-only: rlinf/envs/action_utils.py:66-79 applies that remap solely for
    OPENVLA / OPENVLA_OFT / GR00T_N1D6 / GR00T_N1D7, and RLinf's own π0.5 eval steps the
    env with the raw action. Corroborated by this checkpoint's own norm_stats:
    ``actions.q01[6] = -1.0``, ``actions.q99[6] = 0.9996`` — the gripper dimension is
    already [-1, +1]. Applying the GR00T transform here would INVERT the gripper, a
    wrong-absolute-SR failure that arm-vs-arm contrasts cannot see.
    """
    return chunk


class OfficialPi05Policy:
    """Official openpi ``Policy`` behind the harness's model-facing interface.

    predict_action_batch(env_obs, mode="eval") -> (actions (1, 10, 7), None)
    """

    # --- the rollout seam (harness/rollout.py). Mandatory attributes, never defaulted:
    # a model that forgets postprocess_chunk must crash, not silently inherit GR00T's
    # gripper remap. Reusing wrap_obs_gr00t verbatim is the point, not laziness — it
    # guarantees π0.5 and GR00T see bit-identical pixels and the same 8-D state, and it
    # already matches what RLinf's standalone libero_eval.py builds for π0.5.
    wrap_obs = staticmethod(wrap_obs_gr00t)
    postprocess_chunk = staticmethod(_identity_chunk)

    def __init__(self, model_path: str, device: str = "cuda"):
        # ORDER MATTERS: MagickWand's dlopen fails if torch/cv2 have already loaded
        # conflicting shared libraries into this process (same hazard documented at
        # harness/model_gr00t.py:86-90). Preload the sim stack first.
        import liberoplus.liberoplus.envs  # noqa: F401  (pulls in wand)

        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        train_config = _config.get_config(_TRAIN_CONFIG_NAME)
        if not train_config.model.pi05:
            raise RuntimeError(
                f"{_TRAIN_CONFIG_NAME} is not a π0.5 config — this adapter is π0.5-only "
                "(π0's suffix carries a state query row, so the pladis/attn_pi05.py "
                "geometry and its qgroup rejection would both be wrong)."
            )
        self._policy = _policy_config.create_trained_policy(
            train_config, model_path, pytorch_device=device
        )
        self._uncompile_sample_actions()

        # PLADIS install target + geometry source (the PI0Pytorch module).
        self.model = self._policy._model
        self.output_action_chunks = int(train_config.model.action_horizon)
        # The prompt actually handed to Policy.infer on the last call. Delivery is data,
        # not assumption (harness/env.py:22-24) — experiments/smoke_pi05.py asserts on it.
        self.last_prompt: str | None = None

    def _uncompile_sample_actions(self) -> None:
        """Undo pi0_pytorch.py:112's ``torch.compile(self.sample_actions, "max-autotune")``.

        A compiled graph is traced ONCE and freezes whichever
        ``eager_attention_forward`` module global was live at trace time. Trace it before
        install_pladis and the arm silently runs vanilla; trace it after and every arm
        re-triggers a max-autotune compile. It would also hide the flow-noise RNG from
        the per-chunk pin. RLinf hits the same wall and works around it the same way
        (rlinf/models/embodiment/openpi/openpi_action_model.py:154-157).

        ``sample_actions`` is assigned as an INSTANCE attribute, so deleting it restores
        the plain bound method from the class.
        """
        model = self._policy._model
        vars(model).pop("sample_actions", None)
        self._policy._sample_actions = model.sample_actions
        fn = self._policy._sample_actions
        if hasattr(fn, "_torchdynamo_orig_callable") or hasattr(fn, "_torchdynamo_inline"):
            raise RuntimeError(
                "π0.5 sample_actions is still a torch.compile wrapper after un-compiling; "
                "the PLADIS patch and the flow-noise pin would both be unreliable."
            )

    def eval(self):
        return None  # keep the loader contract of model_gr00t.py (do not chain)

    @torch.no_grad()
    def predict_action_batch(self, env_obs: dict, mode: str = "eval"):
        assert mode == "eval", "harness serves eval rollouts only"
        # Policy.infer is B=1 by construction (policy.py:78 adds the batch axis itself,
        # :98 strips it). The harness rollout loop is strictly sequential, so this is not
        # a limitation — but a silent B>1 truncation would be, hence the assert.
        n = len(env_obs["task_descriptions"])
        if n != 1:
            raise ValueError(f"openpi Policy.infer is batch-size-1; got B={n}")

        instruction = env_obs["task_descriptions"][0]
        self.last_prompt = instruction
        obs = {
            # uint8 HWC, already 180°-rotated by wrap_obs_gr00t. Do NOT resize to 224
            # here — openpi's ResizeImages does aspect-preserving resize_with_pad inside
            # infer, and pre-resizing would change the padding geometry.
            "observation/image": env_obs["main_images"].numpy()[0],
            "observation/wrist_image": env_obs["wrist_images"].numpy()[0],
            "observation/state": env_obs["states"].numpy()[0],
            "prompt": instruction,
        }
        out = self._policy.infer(obs)
        chunk = np.asarray(out["actions"])  # (action_horizon, 7)
        if chunk.ndim != 2 or chunk.shape[-1] != 7:
            raise RuntimeError(f"unexpected π0.5 action shape {chunk.shape}, want (H, 7)")
        return chunk[None], None  # (1, H, 7) — the interface rollout.py expects


def load_pi05(model_path: str, device: str = "cuda") -> OfficialPi05Policy:
    return OfficialPi05Policy(model_path, device=device)
