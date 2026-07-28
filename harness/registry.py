# SPDX-License-Identifier: Apache-2.0
"""Model registry: one ModelSpec per supported VLA.

The registry is import-light on purpose — loaders and hook modules are
resolved lazily by dotted path, so the gr00t venv never imports openpi code
and vice versa. Running a model under the wrong venv fails with a message
naming the right `run.sh --venv`.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    venv: str                 # run.sh --venv value this model runs under
    loader: str               # "module:function" -> callable(model_path) -> adapter
    hook_module: str          # pladis.attn_<model>, install_pladis[_cells] inside
    model_root_env: str       # env var holding the checkpoint root (machine.env)
    model_root_default: str   # fallback when the env var is unset
    per_suite: bool           # True: checkpoint root has one subdir per suite
    default_exec_horizon: int
    default_max_steps: int
    default_n_state_tokens: int
    path_env_override: str | None = None  # legacy whole-path override env var

    def default_model_path(self, suite: str) -> str:
        if self.path_env_override and os.environ.get(self.path_env_override):
            return os.environ[self.path_env_override]
        root = os.environ.get(self.model_root_env) or self.model_root_default
        return os.path.join(root, suite) if self.per_suite else root


MODELS: dict[str, ModelSpec] = {
    "gr00t_n17": ModelSpec(
        name="gr00t_n17",
        venv="gr00t",
        loader="harness.model_gr00t:load_gr00t_n1d7",
        hook_module="pladis.attn_gr00t_n17",
        model_root_env="MODEL_ROOT_GR00T_N17",
        model_root_default="/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO",
        per_suite=True,
        default_exec_horizon=8,
        default_max_steps=720,
        default_n_state_tokens=1,
        path_env_override="GR00T_MODEL_PATH",
    ),
    "smolvla": ModelSpec(
        name="smolvla",
        venv="lerobot",
        loader="harness.model_smolvla:load_smolvla",
        hook_module="pladis.attn_smolvla",
        model_root_env="MODEL_ROOT_SMOLVLA",
        # lerobot/smolvla_libero (org-official, datasets: lerobot/libero). Anchor
        # libero_10 n=100: 64% vs paper-0.45B Long 71 (z~1.4, consistent); beats
        # HuggingFaceVLA/smolvla_libero 50% paired +14pp z=2.27 (models/smolvla_libero).
        model_root_default="/home/reallab/parkkwanjoon/workspace/models/smolvla_libero_official",
        per_suite=False,  # one checkpoint for all suites
        # exec-horizon insensitivity measured on libero_10: h1 vs h10 paired n=20
        # discordant 3:2 (z~0.4) -> keep 10 (8x cheaper than the paper's per-step sim protocol)
        default_exec_horizon=10,
        default_max_steps=720,
        default_n_state_tokens=1,
    ),
}


def _resolve(dotted: str, what: str, spec: ModelSpec):
    mod_name, _, attr = dotted.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise SystemExit(
            f"[registry] cannot import {mod_name!r} ({what} for model {spec.name!r}): {exc}\n"
            f"[registry] this model runs under the {spec.venv!r} venv — invoke via\n"
            f"[registry]   bash experiments/run.sh --venv {spec.venv} ..."
        )
    return getattr(mod, attr) if attr else mod


def resolve_loader(spec: ModelSpec):
    return _resolve(spec.loader, "loader", spec)


def resolve_hooks(spec: ModelSpec):
    return _resolve(spec.hook_module, "hook module", spec)
