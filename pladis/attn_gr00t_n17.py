# SPDX-License-Identifier: Apache-2.0
"""PLADIS test-time sparse-attention guidance for the GR00T N1.7 action-head DiT,
with query-group (state vs action) gating.

Ported from Isaac-GR00T-latest/gr00t/model/modules/pladis_attn.py so the RLinf
rollout worker needs no modified gr00t package. PLADIS (arXiv:2503.07677) blends
the dense softmax map with a sparse (entmax15/sparsemax) map:

    attn = dense + lambda * (sparse - dense)

``pladis_scale == 0`` -> delegate to the same fused ``F.scaled_dot_product_attention``
call diffusers' AttnProcessor2_0 makes, so base0 is BIT-identical to vanilla. This is
the official PLADIS semantics: lambda=0 never leaves the native SDPA path (the repo
gates the processor swap on ``do_sparse_guidance = pladis_scale > 0``,
PLADIS/pipeline/pipeline_sdxl.py:1215,1707); only the lambda>0 branch switches to the
manual torch softmax/entmax implementation (theirs: pipeline_sdxl.py:69-105).

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
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

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
        if self.pladis_scale == 0.0:
            # Official lambda=0 semantics: stay on the fused SDPA path, byte-for-byte
            # the call AttnProcessor2_0 makes (bool mask passed through untouched).
            # base0 == vanilla bit-exact; the hook only exercises install plumbing.
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
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
            sparse = self._sparse(logits)
            attn_weight = dense + self.pladis_scale * (sparse - dense)
            if self.qgroup != "all":
                # Query rows are [state(0:n_state_tokens); action(n_state_tokens:)].
                # Keep the blend only on the selected group; all other rows stay dense.
                ns = self.n_state_tokens
                # A wrong n_state_tokens mis-slices the two groups SILENTLY (no
                # shape error — cat() reassembles any split), so the whole
                # state/action contrast would be meaningless. Check the split is
                # non-degenerate against the live query length instead.
                n_query = attn_weight.shape[-2]
                if not 0 < ns < n_query:
                    raise ValueError(
                        f"n_state_tokens={ns} does not split a {n_query}-row query "
                        f"sequence into non-empty [state; action] groups — the "
                        f"qgroup={self.qgroup!r} arm would be degenerate."
                    )
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
    even = [i for i in range(n) if i % 2 == 0]
    if kind == "all":
        return even
    if kind not in ("text", "image"):
        raise ValueError(f"kind must be all|text|image, got {kind}")
    # NOT a soft default: with every==1 the rule collapses (every even block is
    # a text block) and kind="image" would silently select ZERO blocks — the
    # arm would then run as plain vanilla while being logged as an intervention.
    every = getattr(dit, "attend_text_every_n_blocks", None)
    if not every or every < 2:
        raise ValueError(
            f"attend_text_every_n_blocks={every!r} gives no text/image alternation "
            f"on this DiT, so kind={kind!r} is not a well-defined key group. "
            f"Use kind='all' or pass explicit `blocks=`."
        )
    text = [i for i in even if i % (2 * every) == 0]
    if kind == "text":
        return text
    return [i for i in even if i not in set(text)]


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

    # An empty install is indistinguishable from vanilla at rollout time: the
    # arm would consume a full sweep and be reported as an intervention while
    # having changed nothing. Never let it start.
    if not installed:
        raise RuntimeError(
            f"PLADIS install selected no blocks (kind={kind!r}, blocks={blocks!r}, "
            f"n_layers={len(dit.transformer_blocks)}) — this arm would be "
            f"bit-identical to vanilla."
        )

    msg = (
        f"[PLADIS] installed on blocks {installed} "
        f"(scale={pladis_scale}, method={method}, beta={beta}, kind={kind}, "
        f"qgroup={qgroup}, n_state_tokens={n_state_tokens}, "
        f"n_layers={len(dit.transformer_blocks)})"
    )
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)  # survives SIGTERM before stdout buffer flush
    return installed


def install_pladis_cells(
    model,
    cells: str,
    pladis_scale: float = 1.5,
    method: str = "ent15max",
    beta: float = 1.0,
    n_state_tokens: int = 1,
) -> List[int]:
    """Install a (possibly different) qgroup per key-kind block set.

    ``cells`` is a comma-separated list of ``{qgroup}x{kind}`` cells, e.g.
    ``"actionxtext,stateximage"`` = blend action rows on text blocks AND state
    rows on image blocks in the same pass. Kinds must be disjoint (one
    processor per block): express {action,state} on one kind as qgroup=all.
    """
    parsed = []
    for cell in cells.split(","):
        qgroup, sep, kind = cell.strip().partition("x")
        if not sep or qgroup not in _VALID_QGROUPS or kind not in ("all", "text", "image"):
            raise ValueError(f"bad cell {cell!r}: expected {{qgroup}}x{{kind}}")
        parsed.append((qgroup, kind))
    kinds = [k for _, k in parsed]
    if len(set(kinds)) != len(kinds) or ("all" in kinds and len(kinds) > 1):
        raise ValueError(f"cells must target disjoint kinds, got {kinds} "
                         "(use qgroup=all for both row groups on one kind)")
    installed: List[int] = []
    for qgroup, kind in parsed:
        installed += install_pladis(
            model,
            pladis_scale=pladis_scale,
            method=method,
            beta=beta,
            kind=kind,
            qgroup=qgroup,
            n_state_tokens=n_state_tokens,
        )
    return sorted(installed)
