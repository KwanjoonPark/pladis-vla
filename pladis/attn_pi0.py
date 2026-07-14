# PLADIS on Pi0/Pi05 (openpi PaliGemma+expert joint attention).
#
# Diffusion PLADIS replaces the cross-attention softmax (Q=latent, K=text) with
#   attn = dense + scale * (sparse - dense),   sparse = entmax_a(scores)
# to sharpen instruction conditioning. Pi0/Pi05 has NO separate cross-attn
# module: the action "suffix" tokens attend jointly to [image | language |
# suffix] keys through ONE softmax in the Gemma expert. We therefore hook the
# eager attention and apply the PLADIS blend, with three region modes:
#   - "all"  : blend over the whole key row (action self+cross sharpening)
#   - "text" : sharpen ONLY the language-key sub-block, preserving its total
#              mass (image/suffix weights untouched, row stays normalized)
#   - "image": same but over the image-key sub-block
#
# Only the SUFFIX denoising steps are touched (query length is small); the
# prefix self-attention pass (Q == prefix length) is left as stock softmax.
#
# scale==0  ==>  mathematically identical to stock softmax (parity), for every
# mode. Env-gated so the baseline path is byte-for-byte unchanged when disabled.

import os

import torch
import torch.nn.functional as F

try:
    from entmax import entmax15, sparsemax
except Exception:  # entmax optional until PLADIS is enabled
    entmax15 = None
    sparsemax = None


class _Ctx:
    enable = False
    scale = 0.0
    method = "entmax15"      # entmax15 | sparsemax
    kind = "text"            # all | text | image
    qgroup = "all"           # all | state | action  (query-row group in the suffix)
    has_state_token = False  # pi0-base: suffix = [state@0; action 1:]; pi05: action-only
    n_img_prefix = 768       # 256 tokens * 3 image slots (openpi always embeds 3 slots)
    n_lang = 200             # language tokens (pi05: 200; pi0-base: 48 — derived from config)
    max_suffix_query = 100   # gate: only patch steps with query len below this
    _installed = False
    _n_calls = 0             # diagnostic


CTX = _Ctx()


def _sparse(scores_f):
    if CTX.method == "sparsemax":
        return sparsemax(scores_f, dim=-1)
    return entmax15(scores_f, dim=-1)


def _pladis_softmax(attn_weights, query_len, key_len):
    """attn_weights: [B,H,Q,K] post-mask pre-softmax scores (fp32).
    Returns blended attention weights, same shape/dtype."""
    dense = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    if not CTX.enable or CTX.scale == 0.0:
        return dense
    # gate: only the suffix denoising steps (small query), never the prefix pass
    if query_len is None or query_len > CTX.max_suffix_query:
        return dense
    if entmax15 is None:
        return dense

    CTX._n_calls += 1
    scale = float(CTX.scale)
    # finite scores for entmax (causal mask uses finfo.min ~ -inf)
    neg = torch.finfo(torch.float32).min / 4
    scores_f = attn_weights.to(torch.float32).clamp_min(neg)

    if CTX.kind == "all":
        sparse = _sparse(scores_f)
        w = dense + scale * (sparse - dense)
        return _apply_qgroup(dense, w)

    # region-preserving sub-block sharpening (text / image)
    # key axis = [ prefix(image[0:n_img] , language[n_img:n_img+n_lang]) | suffix ]
    n_img = CTX.n_img_prefix
    n_lang = CTX.n_lang
    if CTX.kind == "text":
        lo, hi = n_img, n_img + n_lang
    else:  # image
        lo, hi = 0, n_img
    # guard: region must fit in the key axis
    if hi > key_len:
        return dense

    sub = scores_f[..., lo:hi]                       # [B,H,Q,R]
    p = _sparse(sub)                                 # sharpened dist over region (sum=1)
    m = dense[..., lo:hi].sum(dim=-1, keepdim=True)  # original mass on region
    w = dense.clone()
    w[..., lo:hi] = dense[..., lo:hi] + scale * (m * p - dense[..., lo:hi])
    return _apply_qgroup(dense, w)


def _apply_qgroup(dense, w):
    """Restrict the blend to the selected query-row group of the suffix pass.

    Suffix query rows: pi0-base = [state@0; action 1:]; pi05 = action-only (no
    state row — install_pladis_from_env fails fast on qgroup="state" there, and
    qgroup="action" degenerates to all rows)."""
    if CTX.qgroup == "all":
        return w
    ns = 1 if CTX.has_state_token else 0
    if CTX.qgroup == "state":
        if ns == 0:
            return dense  # unreachable (fail-fast at install); defensive no-op
        out = dense.clone()
        out[..., :ns, :] = w[..., :ns, :]
        return out
    # action rows
    if ns == 0:
        return w
    out = w.clone()
    out[..., :ns, :] = dense[..., :ns, :]
    return out


def make_pladis_eager_attention_forward(orig_fn):
    """Wrap transformers' gemma eager_attention_forward, replacing its internal
    softmax with the PLADIS blend. We re-derive attn from (query,key,value)
    rather than call orig_fn so we can intercept the softmax."""

    def eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        # replicate transformers gemma eager path (repeat_kv + scaled scores)
        from transformers.models.gemma.modeling_gemma import repeat_kv

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        q_len = attn_weights.shape[-2]
        k_len = attn_weights.shape[-1]
        attn_weights = _pladis_softmax(attn_weights, q_len, k_len).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    eager_attention_forward._pladis_wrapped = True
    eager_attention_forward._orig = orig_fn
    return eager_attention_forward


def install_pladis_from_env(config_name=None):
    """Read PLADIS_* env vars and, if enabled, monkeypatch gemma eager attention.
    Returns a short status string. Safe to call once at model creation.

    ``config_name`` (the openpi train-config name, e.g. "pi05_maniskill",
    "pi0_libero") derives the model-family-dependent geometry:
      pi05_* -> no state query token, n_lang=200 (discrete-state prompt)
      pi0_*  -> suffix has a state token at row 0, n_lang=48
    Env vars PLADIS_NIMG / PLADIS_NLANG still override the derived values."""
    enable = os.environ.get("PLADIS_ENABLE", "0") == "1"
    CTX.enable = enable
    CTX.scale = float(os.environ.get("PLADIS_SCALE", "0.0"))
    CTX.method = os.environ.get("PLADIS_METHOD", "entmax15")
    CTX.kind = os.environ.get("PLADIS_KIND", "text")
    CTX.qgroup = os.environ.get("PLADIS_QGROUP", "all")
    if CTX.qgroup not in ("all", "state", "action"):
        raise ValueError(f"PLADIS_QGROUP must be all|state|action, got {CTX.qgroup!r}")
    if config_name is not None:
        name = str(config_name)
        if name.startswith("pi05"):
            CTX.has_state_token = False
            CTX.n_lang = 200
        elif name.startswith("pi0"):
            CTX.has_state_token = True
            CTX.n_lang = 48
    if "PLADIS_NIMG" in os.environ:
        CTX.n_img_prefix = int(os.environ["PLADIS_NIMG"])
    if "PLADIS_NLANG" in os.environ:
        CTX.n_lang = int(os.environ["PLADIS_NLANG"])
    if enable and CTX.qgroup == "state" and not CTX.has_state_token:
        # pi05 puts the state into the language prefix as discrete TOKENS (keys),
        # so there is no state query row to gate — refuse instead of silently no-op.
        raise RuntimeError(
            "PLADIS_QGROUP=state requires a state query token in the suffix; "
            f"config {config_name!r} (pi05-family) has none. Use pi0-base or "
            "qgroup in {all, action}."
        )

    # Always install the wrapper (it is a no-op unless CTX.enable & scale>0),
    # so the same code path runs for baseline and PLADIS -> clean parity.
    import transformers.models.gemma.modeling_gemma as mg

    if not getattr(mg.eager_attention_forward, "_pladis_wrapped", False):
        orig = mg.eager_attention_forward
        mg.eager_attention_forward = make_pladis_eager_attention_forward(orig)
        # some transformers versions resolve via ALL_ATTENTION_FUNCTIONS
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            ALL_ATTENTION_FUNCTIONS["eager"] = mg.eager_attention_forward
        except Exception:
            pass
        CTX._installed = True

    return (
        f"[PLADIS-pi05] enable={CTX.enable} scale={CTX.scale} method={CTX.method} "
        f"kind={CTX.kind} qgroup={CTX.qgroup} has_state_token={CTX.has_state_token} "
        f"n_img={CTX.n_img_prefix} n_lang={CTX.n_lang} "
        f"config={config_name!r} installed={CTX._installed}"
    )
