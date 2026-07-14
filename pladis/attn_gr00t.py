# SPDX-License-Identifier: Apache-2.0
"""PLADIS test-time sparse-attention guidance for the GR00T N1.7 action-head DiT,
with query-group (state vs action) gating.

Ported from Isaac-GR00T-latest/gr00t/model/modules/pladis_attn.py so the RLinf
rollout worker needs no modified gr00t package. PLADIS (arXiv:2503.07677) blends
the dense softmax map with a sparse (entmax15/sparsemax) map:

    attn = dense + lambda * (sparse - dense)

``pladis_scale == 0`` -> exact vanilla softmax (parity check).

New here vs the Isaac-GR00T original: ``qgroup`` restricts the blend to a QUERY
row group. The N1.7 DiT query sequence is ``[state (n_state_tokens); action (H)]``
(gr00t_n1d7.py builds ``sa_embs = cat(state_features, action_features)``), so
    qgroup="state"  -> blend only query row(s) 0:n_state_tokens
    qgroup="action" -> blend only query rows n_state_tokens:
    qgroup="all"    -> blend every query row (original behavior)
Key-group selection stays per-block: AlternateVLDiT's even (cross) blocks attend
to text keys when ``idx % (2*attend_text_every_n_blocks) == 0``, else image keys,
so ``kind`` picks blocks and the {state,action}x{image,text} cells compose as
(qgroup, kind) pairs.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional

import torch

try:
    from entmax import entmax15, sparsemax
except Exception as exc:  # pragma: no cover - surfaced only if entmax missing
    raise ImportError(
        "PLADIS needs the `entmax` package (pip install entmax) for the sparse branch."
    ) from exc

_VALID_QGROUPS = ("all", "state", "action")


class PLADISAttnProcessor:
    """Dense/sparse-extrapolation attention processor (single forward pass)."""

    def __init__(
        self,
        pladis_scale: float = 1.5,
        method: str = "ent15max",
        beta: float = 1.0,
        qgroup: str = "all",
        n_state_tokens: int = 1,
    ) -> None:
        self.pladis_scale = float(pladis_scale)
        self.method = method
        # beta scales the logits of the sparse branch ONLY (a temperature on the
        # entmax/sparsemax reference); the dense softmax branch is left untouched.
        self.beta = float(beta)
        if qgroup not in _VALID_QGROUPS:
            raise ValueError(f"qgroup must be one of {_VALID_QGROUPS}, got {qgroup!r}")
        self.qgroup = qgroup
        self.n_state_tokens = int(n_state_tokens)

    def _sparse(self, logits: torch.Tensor) -> torch.Tensor:
        z = self.beta * logits
        if self.method == "sparsemax":
            return sparsemax(z, dim=-1)
        elif self.method == "ent15max":
            return entmax15(z, dim=-1)
        elif self.method == "softmax":
            # alpha=1 entmax == softmax; with beta=1 the sparse branch equals the dense
            # branch so PLADIS collapses to vanilla for ANY scale -> integration sanity check.
            return torch.softmax(z, dim=-1)
        else:
            raise ValueError(f"Unknown PLADIS method: {self.method}")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # --- PLADIS: dense/sparse extrapolation in place of scaled_dot_product_attention ---
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        logits = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (B, H, Lq, Lk)
        # softmax/entmax in float32 for numerical stability under bf16 autocast.
        logits = logits.float()
        if attention_mask is not None:
            # SDPA treats a BOOL mask as True=attend / False=-inf. Adding a bool
            # tensor would add 0/1 instead and silently disable masking (this is
            # what the AlternateVLDiT text/image key masks are), so convert to an
            # additive float mask. Large-finite instead of -inf keeps entmax on
            # the sparse branch NaN-free (matches the pi05 hook's clamp).
            neg = torch.finfo(torch.float32).min / 4
            if attention_mask.dtype == torch.bool:
                attention_mask = (~attention_mask).to(torch.float32) * neg
            logits = (logits + attention_mask.float()).clamp_min(neg)
        dense = torch.softmax(logits, dim=-1)
        if self.pladis_scale == 0.0:
            attn_weight = dense  # exact vanilla path (no entmax) for the parity check
        else:
            sparse = self._sparse(logits)
            attn_weight = dense + self.pladis_scale * (sparse - dense)
            if self.qgroup != "all":
                # Query rows are [state(0:n_state_tokens); action(n_state_tokens:)].
                # Keep the blend only on the selected group; all other rows stay dense.
                ns = self.n_state_tokens
                if self.qgroup == "state":
                    attn_weight = torch.cat(
                        [attn_weight[..., :ns, :], dense[..., ns:, :]], dim=-2
                    )
                else:  # action
                    attn_weight = torch.cat(
                        [dense[..., :ns, :], attn_weight[..., ns:, :]], dim=-2
                    )
        attn_weight = attn_weight.to(value.dtype)
        hidden_states = torch.matmul(attn_weight, value)
        # -------------------------------------------------------------------------------

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def _find_alternate_dit(model):
    """Return the AlternateVLDiT/DiT module holding ``transformer_blocks``.

    Accepts either the top GR00T model, the action head, or the DiT itself.
    """
    # top model -> action_head -> model (the DiT)
    for path in ("action_head.model", "model", ""):
        obj = model
        ok = True
        for attr in filter(None, path.split(".")):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and hasattr(obj, "transformer_blocks"):
            return obj
    raise AttributeError("Could not locate a DiT with `transformer_blocks` on the given model.")


def cross_block_indices(dit, kind: str = "text") -> List[int]:
    """Even (cross-attention) block indices of the DiT, optionally split by target.

    kind: "all" (every even/cross block), "text" (even blocks that cross-attend to
    language tokens), or "image" (even blocks that cross-attend to image tokens).
    Text/image split follows AlternateVLDiT.forward: a cross block attends to text
    when ``idx % (2 * attend_text_every_n_blocks) == 0``, else to image.
    """
    n = len(dit.transformer_blocks)
    every = getattr(dit, "attend_text_every_n_blocks", None) or 1
    even = [i for i in range(n) if i % 2 == 0]
    if kind == "all":
        return even
    text = [i for i in even if i % (2 * every) == 0]
    if kind == "text":
        return text
    if kind == "image":
        return [i for i in even if i not in set(text)]
    raise ValueError(f"kind must be all|text|image, got {kind}")


def install_pladis(
    model,
    pladis_scale: float = 1.5,
    method: str = "ent15max",
    beta: float = 1.0,
    blocks: Optional[List[int]] = None,
    kind: str = "text",
    qgroup: str = "all",
    n_state_tokens: int = 1,
) -> List[int]:
    """Install PLADISAttnProcessor on selected cross blocks; returns the block idxs used.

    If ``blocks`` is given it is used verbatim (must be even/cross indices). Otherwise
    all cross blocks of ``kind`` (text|image|all) are targeted. ``qgroup`` restricts
    the blend to state/action query rows (see module docstring).
    """
    dit = _find_alternate_dit(model)
    if blocks is None:
        blocks = cross_block_indices(dit, kind=kind)
    targets = set(blocks)
    installed = []
    for idx, block in enumerate(dit.transformer_blocks):
        if idx in targets:
            block.attn1.set_processor(
                PLADISAttnProcessor(
                    pladis_scale=pladis_scale,
                    method=method,
                    beta=beta,
                    qgroup=qgroup,
                    n_state_tokens=n_state_tokens,
                )
            )
            installed.append(idx)

    msg = (
        f"[PLADIS] installed on blocks {installed} "
        f"(scale={pladis_scale}, method={method}, beta={beta}, kind={kind}, "
        f"qgroup={qgroup}, n_state_tokens={n_state_tokens}, "
        f"n_layers={len(dit.transformer_blocks)})"
    )
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)  # survives SIGTERM before stdout buffer flush
    return installed


def install_pladis_from_env(model) -> str:
    """Env-gated install for the RLinf rollout worker (mirrors the openpi hook).

    Env vars:
      PLADIS_ENABLE=1        master switch (else no-op)
      PLADIS_SCALE=1.5       lambda blend; 0 = exact-vanilla parity check
      PLADIS_METHOD=ent15max ent15max|sparsemax|softmax
      PLADIS_BETA=1.0        sparse-branch temperature
      PLADIS_KIND=text       text|image|all  (key group via block selection)
      PLADIS_QGROUP=all      all|state|action (query row group)
      PLADIS_BLOCKS=8,12     explicit block indices (overrides KIND)
      PLADIS_NSTATE=1        number of leading state query tokens
    """
    if os.environ.get("PLADIS_ENABLE", "").lower() not in ("1", "true", "yes"):
        return "[PLADIS-gr00t] disabled (PLADIS_ENABLE unset)"
    blocks_env = os.environ.get("PLADIS_BLOCKS", "").strip()
    blocks = [int(x) for x in blocks_env.split(",") if x.strip()] if blocks_env else None
    installed = install_pladis(
        model,
        pladis_scale=float(os.environ.get("PLADIS_SCALE", "1.5")),
        method=os.environ.get("PLADIS_METHOD", "ent15max"),
        beta=float(os.environ.get("PLADIS_BETA", "1.0")),
        blocks=blocks,
        kind=os.environ.get("PLADIS_KIND", "text"),
        qgroup=os.environ.get("PLADIS_QGROUP", "all"),
        n_state_tokens=int(os.environ.get("PLADIS_NSTATE", "1")),
    )
    return f"[PLADIS-gr00t] active on blocks {installed}"
