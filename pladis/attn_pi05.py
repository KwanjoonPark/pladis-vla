# SPDX-License-Identifier: Apache-2.0
"""PLADIS FLUX-style sparse-attention guidance for the π0.5 (openpi) action expert.

π0.5 has NO separate cross-attention module: the action "suffix" tokens attend jointly to
``[image | language | suffix]`` keys through ONE softmax in the Gemma expert (a FLUX/MMDiT-style
joint attention). PLADIS blends the dense softmax map with a sparse (entmax15/sparsemax) one::

    attn = dense + scale * (sparse - dense)

and, exactly as the original PLADIS FLUX code does (PLADIS/pipeline/pipeline_flux.py), sharpens ONLY
a key sub-block (language or image) while preserving that block's total row-mass, so every attention
row stays normalized::

    w[sub] = dense[sub] + scale * (m * p - dense[sub]),   m = dense[sub].sum(-1),  p = method(beta*sub)

(Since ``dense[sub]`` is the *full-row* softmax restricted to the block, it already sums to ``m`` and
equals ``m * softmax(sub)``; the identity makes the line above algebraically identical to FLUX's
``m * (lambda*entmax(sub) + (1-lambda)*softmax(sub))``.)

Only the SUFFIX denoising steps (small query length) are touched; the large-query prefix
self-attention pass is left as stock softmax. ``scale == 0`` is numerically identical to stock
softmax (λ=0 parity).

**π0.5-only.** π0 is dropped from this harness. Unlike GR00T's ``[state; action]`` suffix, π0.5's
suffix is action-only, so the query-row (``qgroup``) axis collapses and the design axis is ``kind``
(which key modality sub-block to sparsify). ``qgroup`` is accepted only for API symmetry with
``attn_gr00t_n17.py``: ``state`` is invalid (raises) and ``action`` == ``all``.

On the ``pi05_libero`` checkpoint the proprioceptive state is not an input AT ALL, by either route
(verified 2026-07-28 against the π0.5 paper §IV-A / Appendix E and openpi's own config). The paper's
π0.5 discretizes state into TEXT tokens in the prefix, and openpi makes that the default
(``pi0_config.py:38-39``: ``discrete_state_input = pi05``) — but ``config.py:736`` overrides it to
``False`` for LIBERO, so ``TokenizePrompt`` passes ``state=None``; and ``embed_suffix``
(``pi0_pytorch.py:237-261``) only builds a state token when NOT ``pi05``. Consequence for this hook:
the ``n_lang`` key block is PURE INSTRUCTION, with no state digits mixed in, so ``kind="text"`` is a
clean language locus rather than "language + proprioception".

Ported against transformers **4.53.2**'s gemma ``eager_attention_forward`` (the openpi-track pin).
Because PLADIS needs materialized attention weights, this hook forces the eager gemma path — the
model MUST be loaded with ``attn_implementation="eager"`` and the expert must be a ``gemma`` module,
or the patch never fires and the arm is silently identical to vanilla. ``assert_delivered()`` is the
defense against that silent no-op.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from entmax import entmax15, sparsemax
except Exception as exc:  # pragma: no cover - surfaced only if entmax missing
    raise ImportError(
        "PLADIS needs the `entmax` package (pip install entmax) for the sparse branch."
    ) from exc

# Optional Triton backend for the sparse branch (deep-spin/adasplash). entmax 1.3's
# transforms are exact but sorting-based; adasplash solves the same threshold with a
# Triton kernel and is ~5x faster at THIS hook's shapes ([1,8,10,{200,768,968}] fp32:
# 423us -> 84us, measured 2026-07-30 on an A6000).
#
# Substituting it is only legitimate because the SUPPORT is identical — which columns
# survive is the identity of the intervention, and a backend that kept a different set
# would be a different arm wearing the same flags. Verified at all three block widths:
# support sets equal elementwise, max|dp| 6.6e-07..1.7e-06 (fp32 rounding). The gate
# `verify_pi05_hook.py` re-checks this on the machine; `--pladis-sparse-backend` selects
# it, defaulting to `entmax` so previously collected arms remain byte-reproducible.
#
# NOTE this is the entmax FUNCTION, not adasplash's flash-attention path. That path
# cannot serve this hook: PLADIS blends `dense + λ(m·p − dense)`, so it needs the dense
# map materialized, which is exactly what flash attention refuses to produce.
try:
    from adasplash import triton_entmax15, triton_sparsemax
    _HAVE_ADASPLASH = True
except Exception:  # pragma: no cover - optional dependency
    triton_entmax15 = triton_sparsemax = None
    _HAVE_ADASPLASH = False

_VALID_BACKENDS = ("entmax", "adasplash")

_VALID_METHODS = ("entmax15", "sparsemax", "softmax")
# text/image/prefix are single-block, mass-preserving sharpenings — the shape of the
# operation the official FLUX code performs (PLADIS/pipeline/pipeline_flux.py:104-113
# sharpens exactly one (generative-query x conditioning-key) block and leaves the rest
# of the row dense). `all` blends the WHOLE row, which additionally sparsifies the
# action<->action self-attention columns and lets mass migrate across modalities: that
# is the SDXL operation (pipeline_sdxl.py:93-99), where attn2 really is cross-attention
# so the whole row IS conditioning. On pi0.5's joint-attention row it is off-method
# (PLADIS README: "within all cross-attention modules"), so `all` is a reference arm,
# not the "both modalities" arm. The mass-preserving "all conditioning" arm is `prefix`.
_VALID_KINDS = ("all", "text", "image", "prefix")


@dataclass
class _Cfg:
    """Install-time config read by the (module-global) patched attention function.

    Set exclusively from ``install_pladis`` arguments — never from environment variables — so the
    hook follows the explicit-flag convention of ``attn_gr00t_n17.py``.
    """

    scale: float = 0.0          # λ; 0 == vanilla (stock softmax, parity)
    method: str = "entmax15"    # entmax15 | sparsemax | softmax (softmax == eager-dense control)
    # Which implementation computes `method`. Identical support either way (see the
    # adasplash import note); `entmax` stays the default so an arm re-run without the
    # flag reproduces earlier eplogs exactly.
    sparse_backend: str = "entmax"   # entmax | adasplash
    beta: float = 1.0           # inverse temperature on the SPARSE branch ONLY
    kind: str = "text"          # all | text | image  (which key modality sub-block to sparsify)
    # ALLOCATED block widths — the slice bounds. The ATTENDABLE widths are smaller, because
    # both blocks are padded and the padding is masked out (measured 2026-07-28):
    #   image  768 -> 512: LIBERO has two cameras, so libero_policy.py:62-70 fills
    #                      `right_wrist_0_rgb` with zeros and masks it (image_mask False for
    #                      every non-PI0_FAST model).
    #   lang   200 -> ~9-20: PaligemmaTokenizer pads to max_token_len=200 with mask=False.
    # The blend is correct either way — masked columns carry dense~0, so the mass term
    # `m = dense[..., lo:hi].sum()` integrates only attendable mass, and `clamp_min` keeps
    # entmax finite on them. But any DOSE reported as a fraction of these widths understates
    # the intervention (~10x on language); see experiments/diag_pi05_support.py.
    n_img_prefix: int = 768     # image key block width: 256 tokens * 3 openpi image slots
    n_lang: int = 200           # language key block width (π0.5 max_token_len; overridable)
    max_suffix_query: int = 100  # gate: only patch steps whose query length is <= this
    installed: bool = False
    n_calls: int = 0            # forwards that actually applied the sparse blend (delivery counter)
    # (query_len, key_len) of every forward the blend actually fired on. Pure
    # instrumentation, read by experiments/verify_pi05_delivery.py.
    #
    # Why it is worth carrying: PaliGemma's language model is ALSO a transformers gemma
    # module and is ALSO forced onto the eager path (pi0_pytorch.py:447), so this
    # module-global patch fires on the VLM pass too. Today the only thing keeping that
    # pass dense is the `query_len > max_suffix_query` gate below. A config that
    # autoregressively decodes tokens (query_len == 1) would slip UNDER that gate and be
    # silently blended — an intervention we never intended, invisible in the eplog.
    # Asserting seen_shapes == {(action_horizon, prefix+action_horizon)} catches it.
    seen_shapes: set = field(default_factory=set)


CFG = _Cfg()


def _sparse(z: torch.Tensor, method: str) -> torch.Tensor:
    # adasplash's kernels act on the LAST dim (no `dim` argument) — which is the axis we
    # slice on anyway, so no permute is needed. CPU tensors have no Triton kernel, so the
    # CPU gates fall back to entmax regardless of the selected backend.
    if CFG.sparse_backend == "adasplash" and z.is_cuda:
        if method == "sparsemax":
            return triton_sparsemax(z)
        if method == "entmax15":
            return triton_entmax15(z)
    if method == "sparsemax":
        return sparsemax(z, dim=-1)
    if method == "entmax15":
        return entmax15(z, dim=-1)
    # alpha=1 entmax == softmax; with beta=1 the sparse branch equals the dense branch, so PLADIS
    # collapses to plain softmax for ANY scale -> the eager-dense (numeric-path-matched) control arm.
    return torch.softmax(z, dim=-1)


def _pladis_blend(scores: torch.Tensor, query_len: Optional[int], key_len: int) -> torch.Tensor:
    """``scores``: ``[B, H, Q, K]`` post-mask pre-softmax logits. Returns blended attention weights
    (same shape), in float32."""
    scores_f = scores.to(torch.float32)
    dense = torch.softmax(scores_f, dim=-1)

    # Correct dense-returns (NOT silent failures): λ=0 parity, and the large-query PREFIX pass
    # (only the small-query suffix denoising steps are sparsified).
    if CFG.scale == 0.0 or query_len is None or query_len > CFG.max_suffix_query:
        return dense

    CFG.n_calls += 1
    CFG.seen_shapes.add((int(query_len), int(key_len)))
    scale, beta, method = CFG.scale, CFG.beta, CFG.method
    # finite scores for entmax (a causal/pad mask uses ~ -inf; entmax would NaN on it).
    neg = torch.finfo(torch.float32).min / 4
    scores_f = scores_f.clamp_min(neg)

    if CFG.kind == "all":
        sparse = _sparse(beta * scores_f, method)
        return dense + scale * (sparse - dense)

    # region-preserving sub-block sharpening (the FLUX pattern).
    # key axis = [ image(0:n_img) | language(n_img:n_img+n_lang) | suffix ]
    if CFG.kind == "text":
        lo, hi = CFG.n_img_prefix, CFG.n_img_prefix + CFG.n_lang
    elif CFG.kind == "prefix":
        # The whole conditioning prefix as ONE block: entmax normalizes across image AND
        # language together, so mass may migrate between the two modalities, while the
        # suffix (action<->action) columns stay exactly dense. This is the
        # "did not choose a locus" counterpart to text/image — and it is NOT the same as
        # text+image applied together, which would preserve each modality's mass
        # separately. Still one FLUX-shaped single-block operation, unlike kind="all".
        lo, hi = 0, CFG.n_img_prefix + CFG.n_lang
    else:  # image
        lo, hi = 0, CFG.n_img_prefix
    if hi > key_len:
        # We are on a suffix step (query gate passed) so key_len is the full sequence; a block that
        # does not fit means n_img_prefix/n_lang are wrong for this model. Mis-slicing would make the
        # whole kind={text,image} contrast meaningless, so fail loud instead of returning dense.
        raise RuntimeError(
            f"PLADIS kind={CFG.kind!r} needs key columns [{lo}:{hi}] but key_len={key_len}. "
            f"n_img_prefix/n_lang ({CFG.n_img_prefix}/{CFG.n_lang}) do not match this π0.5 model — "
            f"probe the real [image|language|suffix] layout and pass n_img_prefix/n_lang."
        )
    sub = scores_f[..., lo:hi]
    p = _sparse(beta * sub, method)                    # sharpened dist over the block (sums to 1)
    m = dense[..., lo:hi].sum(dim=-1, keepdim=True)    # original softmax mass on the block
    w = dense.clone()
    w[..., lo:hi] = dense[..., lo:hi] + scale * (m * p - dense[..., lo:hi])
    return w


def make_pladis_eager_attention_forward(orig_fn):
    """Wrap transformers 4.53.2 gemma ``eager_attention_forward``, replacing its internal softmax
    with the PLADIS blend. We re-derive attn from ``(query, key, value)`` rather than call
    ``orig_fn`` so we can intercept the softmax."""

    def eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        # replicate transformers' gemma eager path (repeat_kv + scaled scores + mask add)
        from transformers.models.gemma.modeling_gemma import repeat_kv

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        q_len = attn_weights.shape[-2]
        k_len = attn_weights.shape[-1]
        attn_weights = _pladis_blend(attn_weights, q_len, k_len).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    eager_attention_forward._pladis_wrapped = True
    eager_attention_forward._orig = orig_fn
    return eager_attention_forward


def _uses_eager_gemma(model) -> Optional[bool]:
    """``True`` if the model's attention implementation is transformers ``eager``, ``False`` if
    provably not, ``None`` if undeterminable. The hook patches the gemma *eager* path, so an
    SDPA/Flash/custom expert attention would make it a silent no-op."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for attr in ("_attn_implementation", "attn_implementation"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return v == "eager"
    return None


def _derive_geometry(model, n_img_prefix, n_lang):
    """Best-effort key-axis geometry: explicit args win, else the π0.5 defaults (768 / 200).

    Kept explicit for now — when the openpi config surface is known on the eval machine, derive
    ``n_lang`` from ``max_token_len`` and ``n_img_prefix`` from the embedded image-slot count and
    drop the defaults. The geometry is re-validated at run time against the live ``key_len`` inside
    ``_pladis_blend`` (raises on mismatch)."""
    ni = CFG.n_img_prefix if n_img_prefix is None else int(n_img_prefix)
    nl = CFG.n_lang if n_lang is None else int(n_lang)
    return ni, nl


def install_pladis(
    model,
    pladis_scale: float = 1.5,
    method: str = "entmax15",
    beta: float = 1.0,
    kind: str = "text",
    n_img_prefix: Optional[int] = None,
    n_lang: Optional[int] = None,
    max_suffix_query: int = 100,
    qgroup: str = "all",
    sparse_backend: str = "entmax",
) -> str:
    """Install the PLADIS blend on the π0.5 Gemma-expert eager attention (explicit-flag entry).

    Mirrors :func:`pladis.attn_gr00t_n17.install_pladis`: configuration comes from ARGUMENTS (never
    environment variables), then transformers' gemma ``eager_attention_forward`` is monkeypatched.
    Returns a status string. Run one suffix inference and call :func:`assert_delivered` afterwards to
    prove the hook actually fired.

    ``qgroup`` is accepted only for API symmetry with the GR00T hook; π0.5's suffix is action-only
    (no state query row), so ``state`` is invalid and ``action`` == ``all``.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    if sparse_backend not in _VALID_BACKENDS:
        raise ValueError(
            f"sparse_backend must be one of {_VALID_BACKENDS}, got {sparse_backend!r}"
        )
    if sparse_backend == "adasplash" and not _HAVE_ADASPLASH:
        # Refuse rather than fall back: a silent fallback would log `adasplash` in the
        # arm signature while running entmax, so the sidecar would misdescribe the run.
        raise RuntimeError(
            "sparse_backend='adasplash' requested but the package is not importable "
            "(pip install adasplash). Refusing to fall back silently — the .arm sidecar "
            "would then name a backend that did not run."
        )
    if qgroup == "state":
        raise ValueError(
            "qgroup='state' is invalid for π0.5: its suffix is action-only (state is embedded as "
            "discrete language keys, not a query row). Use qgroup in {all, action}."
        )
    if qgroup not in ("all", "action"):
        raise ValueError(f"qgroup must be all|action for π0.5, got {qgroup!r}")

    ni, nl = _derive_geometry(model, n_img_prefix, n_lang)
    CFG.scale = float(pladis_scale)
    CFG.method = method
    CFG.beta = float(beta)
    CFG.kind = kind
    CFG.n_img_prefix = ni
    CFG.n_lang = nl
    CFG.max_suffix_query = int(max_suffix_query)
    CFG.sparse_backend = sparse_backend
    CFG.n_calls = 0
    CFG.seen_shapes = set()

    eager = _uses_eager_gemma(model)
    if eager is False:
        # A non-eager expert would never call the patched function -> the arm would run as vanilla
        # while being logged as an intervention. Refuse rather than silently no-op.
        raise RuntimeError(
            "π0.5 model is not on transformers' eager attention path (attn_implementation != "
            "'eager'); the PLADIS gemma patch would be a silent no-op. Load the model with "
            "attn_implementation='eager'."
        )
    # eager is True (good) or None (undeterminable -> the caller MUST run assert_delivered()).

    import transformers.models.gemma.modeling_gemma as mg

    if not getattr(mg.eager_attention_forward, "_pladis_wrapped", False):
        orig = mg.eager_attention_forward
        mg.eager_attention_forward = make_pladis_eager_attention_forward(orig)
        # some transformers versions dispatch via ALL_ATTENTION_FUNCTIONS["eager"]
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            ALL_ATTENTION_FUNCTIONS["eager"] = mg.eager_attention_forward
        except Exception:
            pass
    CFG.installed = True

    msg = (
        f"[PLADIS-pi05] installed scale={CFG.scale} method={CFG.method} beta={CFG.beta} "
        f"kind={CFG.kind} n_img={CFG.n_img_prefix} n_lang={CFG.n_lang} "
        f"max_suffix_query={CFG.max_suffix_query} backend={CFG.sparse_backend} "
        f"eager={eager}"
    )
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)  # survives SIGTERM before the stdout buffer flushes
    return msg


def assert_delivered() -> None:
    """Raise unless the PLADIS blend actually fired since install (``CFG.n_calls > 0``).

    A zero count after a real suffix inference means the π0.5 Gemma expert is NOT on transformers'
    eager path, so the arm is silently identical to vanilla — the single most important failure to
    catch. This ports the empty-install guard of ``attn_gr00t_n17.py`` to the monkeypatch case; a
    delivery gate calls it after one forward. λ=0 is exempt (its blend is an intentional no-op)."""
    if not CFG.installed:
        raise RuntimeError("PLADIS not installed; call install_pladis(model, ...) first.")
    if CFG.scale == 0.0:
        return  # base0 / λ=0: nothing to deliver by design.
    if CFG.n_calls == 0:
        raise RuntimeError(
            "PLADIS fired ZERO times after a forward pass — the π0.5 Gemma expert is not using "
            "transformers' eager attention, so this arm is silently identical to vanilla. Load the "
            "model with attn_implementation='eager' and confirm the action expert is a `gemma` "
            "module (not gemma2/custom SDPA)."
        )
