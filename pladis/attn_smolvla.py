# SPDX-License-Identifier: Apache-2.0
"""PLADIS sparse-attention guidance for SmolVLA's action expert (lerobot track).

SmolVLA (lerobot ``SmolVLAPolicy``) interleaves two layer types in the expert
(``attention_mode="cross_attn"``, ``self_attn_every_n_layers=2``):

  * CA layers (odd ``layer_idx``): expert (action) queries attend to the VLM
    prefix ONLY — keys are the expert's re-projection of the cached VLM K/V
    (smolvlm_with_expert.py ``forward_cross_attn_layer``); key axis =
    ``[image | language | state]`` with NO suffix columns.
  * SA layers (``layer_idx % n == 0``): at denoise time the suffix K/V is
    CONCATENATED onto the cached prefix K/V (``forward_attn_layer``,
    ``fill_kv_cache=False``), so the key axis is ``[prefix | suffix]`` —
    a pi0.5-style joint row.

The blend is the PLADIS extrapolation ``attn = dense + λ(sparse − dense)``.
Key SUB-BLOCK interventions use the official FLUX mass-preserving form
(PLADIS/pipeline/pipeline_flux.py:105-113, same construction as
``attn_pi05.py``)::

    w[sub] = dense[sub] + λ·(m·p − dense[sub]),  m = Σ_sub dense,  p = f(β·z_sub)

which by the identity ``dense[sub] = m·softmax_block(sub)`` equals FLUX's
``m·(λ·f(z_sub) + (1−λ)·softmax(z_sub))`` and keeps every row normalized for
any λ. Whole-row interventions (``kind="prefix"`` on CA rows) use the plain
blend — a full row is already a normalized distribution.

``kind`` selects the locus (query axis collapses — the suffix is action-only):
  ``text``   CA layers, language columns          (a×t)
  ``image``  CA layers, image columns             (a×i)
  ``state``  CA layers, the state token column(s) (a×state — key-side state,
             the dual of GR00T's state-QUERY arms). ONLY meaningful for
             n_state >= 2: at width 1 the mass-preserving blend is a bit-exact
             no-op and install raises (SmolVLA embeds state as ONE token, so
             this locus does not exist on real checkpoints — 2026-08-02)
  ``prefix`` CA layers, whole row                 (a×prefix — the true
             analogue of GR00T's allxall: every cross column, no self.
             PLAIN blend: the ONE kind that lets mass cross block borders)
  ``self``   SA layers, suffix columns            (a×self — action↔action)
  ``cams``   CA layers, image split per camera    (a×cam — camera1 and
             camera2 each sharpened with its OWN mass fixed; no mass moves
             between cameras, unlike ``image``)
  ``text-image`` CA layers, text AND image blocks, each with its own mass
             fixed — the maximal mass-preserving prefix intervention
             (state is width 1, where preservation is the identity)

``qgroup`` is accepted for API symmetry with the other hooks: the suffix has
no state query row, so ``state`` raises and ``action`` == ``all``.

GEOMETRY IS SELF-CALIBRATING. Language padding differs per checkpoint: the
registry default ``lerobot/smolvla_libero`` pads to ``max_length`` 48
(``pad_language_to`` in its config → CONSTANT prefix 177 = 128 img + 48 lang
+ 1 state; 16 expert layers = 8 SA even-idx + 8 CA odd-idx), while
``HuggingFaceVLA/smolvla_libero`` pads ``longest`` (variable ≤48; 32 layers).
One static layout cannot serve both, so the hook derives the live width
per inference. Every inference begins with the VLM prefix pass
(``sample_actions`` → ``fill_kv_cache=True``), whose calls have
``q_len == key_len == prefix_len``: the hook records that length, then
classifies each denoise call as CA (``key == prefix_len``) or SA
(``key == prefix_len + q_len``) and derives the language width as
``prefix_len − n_img − n_state``. Anything that fits neither geometry, or a
derived language width outside ``[1, n_lang_max]``, raises — the exact-fit
discipline from the attn_pi05 review (F1), made robust to variable padding.

Further decisions carried over from that review:
  * β is applied BEFORE the finite clamp (F4), so large β cannot overflow
    masked logits to −inf inside entmax.
  * The patch is bound to ONE ``SmolVLMWithExpertModel`` INSTANCE (F2) — no
    transformers/global monkeypatch; other models in the process are
    untouched. ``get_attention_interface`` resolves
    ``self.eager_attention_forward`` per call, so the instance attribute
    shadows the class method.
  * ``assert_delivered()`` guards against the silent-vanilla failure mode.

λ=0 and non-intervened passes (the VLM prefix pass in particular) run an
op-for-op replication of lerobot's ``eager_attention_forward`` (fp32 matmul,
bool-mask ``torch.where`` with big_neg, fp32 softmax, cast, matmul) — the
λ=0-parity gate asserts ``torch.equal`` against the stock method.

Ported against lerobot 0.4.4 ``smolvlm_with_expert.py:504-548``. Eval-only
(no dropout in the stock path; training shapes are out of scope). Noise-pin
admissible: ``sample_noise`` draws via global ``torch.normal``
(modeling_smolvla.py:609).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from entmax import entmax15, sparsemax
except Exception as exc:  # pragma: no cover - surfaced only if entmax missing
    raise ImportError(
        "PLADIS needs the `entmax` package (pip install entmax) for the sparse branch."
    ) from exc

_VALID_METHODS = ("entmax15", "sparsemax", "softmax")
# `cams` and `text-image` are MULTI-BLOCK kinds (2026-08-02, operator request:
# every sweep arm must preserve modality mass): each listed block is sharpened
# within itself with its own dense mass held fixed, so no mass crosses a block
# boundary at any λ.
#   cams        CA image split per camera: [0:n_img/2 | n_img/2:n_img], each MP
#   text-image  CA image block + text block, each MP — the MAXIMAL mass-preserving
#               prefix intervention (the state block is width 1, where MP is the
#               identity, so nothing else can be sharpened under mass preservation)
_VALID_KINDS = ("text", "image", "state", "prefix", "self", "cams", "text-image")


@dataclass
class _Cfg:
    """Install-time config, set exclusively from install_pladis arguments."""

    scale: float = 0.0
    method: str = "entmax15"
    beta: float = 1.0
    kind: str = "text"
    n_img: int = 128        # image key block: n_cams * tokens_per_cam (2*64 for smolvla_libero)
    n_lang_max: int = 48    # tokenizer_max_length — upper bound; live width is derived
    n_state: int = 1        # state key token(s)
    prefix_len: Optional[int] = None  # recorded at the prefix pass of each inference
    installed: bool = False
    n_calls_ca: int = 0     # CA-layer forwards that applied the blend
    n_calls_sa: int = 0     # SA-layer forwards that applied the blend


CFG = _Cfg()


def _sparse(z: torch.Tensor, method: str) -> torch.Tensor:
    if method == "sparsemax":
        return sparsemax(z, dim=-1)
    if method == "entmax15":
        return entmax15(z, dim=-1)
    # beta=1 softmax -> sparse == dense -> collapses to vanilla for ANY scale
    # (the eager-dense / integration-sanity control, as in the other hooks).
    return torch.softmax(z, dim=-1)


def _blend_rows(scores_f: torch.Tensor, blocks, whole_row: bool) -> torch.Tensor:
    """``scores_f``: [B,H,Q,K] float32 post-mask logits; ``blocks``: disjoint
    (lo, hi) key spans. Returns blended weights.

    Each block is sharpened WITHIN itself with its own dense mass held fixed —
    the N1.5-doc / FLUX form applied per block — so no mass crosses a block
    boundary and the row stays normalized for any λ. Single-block kinds pass a
    one-element list; multi-block kinds (cams, text-image) simply iterate."""
    dense = torch.softmax(scores_f, dim=-1)
    scale, beta, method = CFG.scale, CFG.beta, CFG.method
    # finite floor AFTER the beta multiply (masked cols carry ~ -3.4e38): entmax
    # needs finite input, and clamping first then scaling could overflow to -inf.
    neg = torch.finfo(torch.float32).min / 4

    if whole_row:
        z = (beta * scores_f).clamp_min(neg)
        sparse = _sparse(z, method)
        return dense + scale * (sparse - dense)

    w = dense.clone()
    for lo, hi in blocks:
        z = (beta * scores_f[..., lo:hi]).clamp_min(neg)
        p = _sparse(z, method)                          # block distribution (sums to 1)
        m = dense[..., lo:hi].sum(dim=-1, keepdim=True)  # original block mass
        w[..., lo:hi] = dense[..., lo:hi] + scale * (m * p - dense[..., lo:hi])
    return w


def _geometry(kind: str, q_len: int, k_len: int):
    """Classify the call and return (layer_type, blocks, whole_row) or raise.

    layer_type: "prefix" (record + dense), "ca", or "sa"; blocks: list of
    disjoint (lo, hi) key spans the blend treats independently."""
    ni, ns = CFG.n_img, CFG.n_state

    if q_len == k_len:
        # VLM prefix pass (fill_kv_cache=True): record the live prefix length.
        if k_len <= ni + ns:
            raise RuntimeError(
                f"PLADIS smolvla: prefix pass of length {k_len} is not longer than "
                f"n_img+n_state={ni + ns} — n_img is wrong for this checkpoint."
            )
        CFG.prefix_len = k_len
        return "prefix", [], False

    P = CFG.prefix_len
    if P is None:
        raise RuntimeError(
            "PLADIS smolvla: denoise-shaped call before any prefix pass — cannot "
            "calibrate the key layout. (Each inference must start with the VLM "
            "prefix pass; check the serving path.)"
        )
    if k_len == P:
        layer = "ca"
    elif k_len == P + q_len:
        layer = "sa"
    else:
        # Exact-fit failure (attn_pi05 review F1): silently mis-sliced columns
        # would void the kind contrast — fail loud instead.
        raise RuntimeError(
            f"PLADIS smolvla geometry mismatch: q_len={q_len}, key_len={k_len}, "
            f"recorded prefix_len={P} — matches neither CA (key==prefix) nor "
            f"SA (key==prefix+q)."
        )

    n_lang = P - ni - ns
    if not 1 <= n_lang <= CFG.n_lang_max:
        raise RuntimeError(
            f"PLADIS smolvla: derived language width {n_lang} out of [1, "
            f"{CFG.n_lang_max}] (prefix_len={P}, n_img={ni}, n_state={ns}) — "
            f"n_img/n_state do not match this checkpoint's prefix layout."
        )

    if kind == "image":
        blocks, whole = [(0, ni)], False
    elif kind == "cams":
        # per-camera mass preservation: prefix image span = [camera1 | camera2] in the
        # checkpoint preprocessor's feature order (camera1 = agentview, camera2 = wrist
        # — model_smolvla.py wrap_obs + the ckpt's saved rename map), 64 connector
        # tokens each on the official ckpt. Install asserts n_img is even.
        blocks, whole = [(0, ni // 2), (ni // 2, ni)], False
    elif kind == "text":
        blocks, whole = [(ni, ni + n_lang)], False
    elif kind == "text-image":
        blocks, whole = [(0, ni), (ni, ni + n_lang)], False
    elif kind == "state":
        blocks, whole = [(ni + n_lang, P)], False
    elif kind == "prefix":
        blocks, whole = [(0, k_len)], True
    else:  # self: suffix columns of an SA row
        blocks, whole = [(P, k_len)], False
    return layer, blocks, whole


def _maybe_blend(masked_att_weights: torch.Tensor) -> torch.Tensor:
    """Dispatch on call geometry; dense (stock softmax) for every pass that is
    not the configured intervention locus."""
    q_len = masked_att_weights.shape[-2]
    k_len = masked_att_weights.shape[-1]

    layer, blocks, whole = _geometry(CFG.kind, q_len, k_len)

    intervene = (
        CFG.scale != 0.0
        and ((layer == "ca" and CFG.kind != "self") or (layer == "sa" and CFG.kind == "self"))
    )
    if not intervene:
        return torch.softmax(masked_att_weights, dim=-1)

    if layer == "ca":
        CFG.n_calls_ca += 1
    else:
        CFG.n_calls_sa += 1
    return _blend_rows(masked_att_weights, blocks, whole)


def make_pladis_eager_attention_forward(model):
    """Return a bound replacement for ``model.eager_attention_forward``
    (lerobot 0.4.4 smolvlm_with_expert.py:504-548, replicated op-for-op with
    the PLADIS blend inserted at the softmax)."""

    def eager_attention_forward(
        attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = model.num_attention_heads
        num_key_value_heads = model.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )
        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = _maybe_blend(masked_att_weights)

        probs = probs.to(dtype=value_states.dtype)
        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(
            batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
        )
        return att_output

    eager_attention_forward._pladis_wrapped = True
    return eager_attention_forward


def _find_vlm_with_expert(model):
    """Accept the harness SmolVLAAdapter (.policy), a SmolVLAPolicy, its .model
    (VLAFlowMatching), or the SmolVLMWithExpertModel itself.

    The .policy hop matches the other tracks' convention — eval_arm hands every
    hook the ADAPTER, not the bare policy (found 2026-08-02 by the first
    eval_arm-routed delivery smoke; the 07-28 smoke installed on the policy
    directly and never exercised this path)."""
    for outer in (model, getattr(model, "policy", None)):
        for obj in (outer, getattr(outer, "model", None)):
            if obj is None:
                continue
            if hasattr(obj, "eager_attention_forward") and hasattr(obj, "get_attention_interface"):
                return obj
            inner = getattr(obj, "vlm_with_expert", None)
            if inner is not None and hasattr(inner, "eager_attention_forward"):
                return inner
    raise TypeError(
        "install_pladis(model): could not locate a SmolVLMWithExpertModel "
        "(expected a SmolVLAAdapter, a SmolVLAPolicy, its .model, or the "
        "vlm_with_expert module)."
    )


def install_pladis(
    model,
    pladis_scale: float = 1.5,
    method: str = "entmax15",
    beta: float = 1.0,
    kind: str = "text",
    n_img: Optional[int] = None,
    n_lang_max: Optional[int] = None,
    n_state_tokens: Optional[int] = None,
    qgroup: str = "all",
) -> str:
    """Install the PLADIS blend on ONE SmolVLA instance's eager attention.

    Instance-bound (review F2): overrides ``eager_attention_forward`` on the
    located SmolVLMWithExpertModel object only. Configuration comes from
    ARGUMENTS, never environment variables. Returns a status string; run one
    denoise inference and call :func:`assert_delivered` to prove the hook
    fired.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    if qgroup == "state":
        raise ValueError(
            "qgroup='state' is invalid for SmolVLA: the suffix is action-only "
            "(state is a prefix KEY token — intervene on it with kind='state'). "
            "Use qgroup in {all, action}."
        )
    if qgroup not in ("all", "action"):
        raise ValueError(f"qgroup must be all|action for SmolVLA, got {qgroup!r}")
    if kind == "state" and int(n_state_tokens if n_state_tokens is not None else CFG.n_state) < 2:
        # Width-1 block => mass-preserving blend is a BIT-EXACT identity: the block
        # conditional is the constant 1, and m IS the single dense entry, so
        # dense + λ(m·1 − dense) == dense for every λ (found 2026-08-02 by the Phase-D
        # diagnostic). Same contract as the other hooks' empty-install guards: an arm
        # that cannot differ from vanilla must never burn a sweep.
        raise ValueError(
            "kind='state' with a single state key token is a bit-exact no-op under the "
            "mass-preserving sub-block blend (entmax over one column is 1; the blend "
            "reduces to dense for every scale). SmolVLA embeds state as ONE prefix "
            "token, so this locus does not exist — drop the arm."
        )
    if kind == "cams" and int(n_img if n_img is not None else CFG.n_img) % 2:
        raise ValueError(
            "kind='cams' splits the image span into two equal per-camera blocks and "
            f"needs an even n_img (got {n_img if n_img is not None else CFG.n_img})."
        )

    target = _find_vlm_with_expert(model)

    if n_img is not None:
        CFG.n_img = int(n_img)
    if n_lang_max is not None:
        CFG.n_lang_max = int(n_lang_max)
    if n_state_tokens is not None:
        CFG.n_state = int(n_state_tokens)
    CFG.scale = float(pladis_scale)
    CFG.method = method
    CFG.beta = float(beta)
    CFG.kind = kind
    CFG.prefix_len = None
    CFG.n_calls_ca = 0
    CFG.n_calls_sa = 0

    target.eager_attention_forward = make_pladis_eager_attention_forward(target)
    CFG.installed = True

    msg = (
        f"[PLADIS-smolvla] installed scale={CFG.scale} method={CFG.method} beta={CFG.beta} "
        f"kind={CFG.kind} n_img={CFG.n_img} n_lang_max={CFG.n_lang_max} "
        f"n_state={CFG.n_state} on {type(target).__name__}"
    )
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)  # survives SIGTERM before stdout flushes
    return msg


def assert_delivered() -> None:
    """Raise unless the blend actually fired since install. ``kind='self'``
    must have hit SA layers, every other kind CA layers; λ=0 is exempt (its
    no-op is intentional)."""
    if not CFG.installed:
        raise RuntimeError("PLADIS not installed; call install_pladis(model, ...) first.")
    if CFG.scale == 0.0:
        return
    fired = CFG.n_calls_sa if CFG.kind == "self" else CFG.n_calls_ca
    if fired == 0:
        raise RuntimeError(
            f"PLADIS fired ZERO times on the {'SA' if CFG.kind == 'self' else 'CA'} "
            f"path after a forward pass (kind={CFG.kind!r}) — the arm is silently "
            f"identical to vanilla. Check that install_pladis targeted the served "
            f"instance and that n_img/n_state match the checkpoint's prefix layout."
        )
