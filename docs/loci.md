# Intervention loci across architectures

How the PLADIS blend maps onto each model's attention structure, with the
official-implementation counterpart declared per architecture. Every
architectural claim below is measured from the model source named next to it
— not recalled. Hooks may not be written for an architecture until its
section here is approved.

## 0. Method and its two official implementation patterns

PLADIS (arXiv:2503.07677) blends dense and sparse attention maps:

```
w = dense + λ·(sparse − dense),   dense = softmax(z),  sparse = entmax15(β·z)
```

The official repository implements this in exactly two patterns, matching two
attention architectures:

| architecture | official reference | pattern |
|---|---|---|
| dedicated cross-attention (queries see one key set) | `PLADIS/pipeline/pipeline_sdxl.py:69-105` (`SparseAttnProcessor`) | blend the whole map; row mass is 1 in both branches, no renormalization needed |
| joint attention (one softmax over mixed key regions) | `PLADIS/pipeline/pipeline_flux.py:100-114` | blend a **key-column sub-block** only, then rescale it by the dense row-block mass (`:111-113`) so each row's total mass is preserved |

Rule: a hook for a dedicated-cross-attention model mirrors the SDXL
processor; a hook for a joint-attention model mirrors the FLUX processor.
The FLUX sub-block algebra, per query row:

```
m_r      = Σ_{k∈r} dense[k]                      # dense mass of region r
w[r]     = m_r · [ (1−λ)·softmax(z_r) + λ·entmax15(β·z_r) ]   # renormalized blend
w[¬r]    = dense[¬r]                              # complement untouched
```

so `Σ_k w[k] = 1` row-wise for any λ, and λ=0 gives back `dense` exactly.

## 1. GR00T N1.7 — dedicated cross-attention (SDXL pattern) [implemented]

Measured: the action-head DiT's query sequence is `[state; action]` only
(`sa_embs = cat(state_features, action_features)`,
`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:248` in the pinned Isaac-GR00T
checkout); `AlternateVLDiT.forward` runs odd blocks as self-attention and
even blocks as cross-attention whose key/value set is EITHER text or image
tokens by the alternation rule `idx % (2·attend_text_every_n_blocks) == 0`
(`gr00t/model/modules/dit.py:380-401`).

Mapping (`pladis/attn_gr00t_n17.py`): each even block is a dedicated
cross-attention with single-modality keys → SDXL pattern, whole-map blend;
`kind` = block selection (text-key vs image-key blocks); `qgroup` = query-row
slice `[state(0:n); action(n:)]`. No mass renormalization required — every
row's keys are all one modality, and rows outside the qgroup keep the plain
dense map (mass 1).

## 2. π0 / π0.5 — joint attention (FLUX pattern) [this track]

Measured from the installed openpi
(`RLinf/openpi/lib/python3.11/site-packages/openpi/models_pytorch/`):

- **Prefix** = `[image tokens | language tokens]`: `embed_prefix`
  (`pi0_pytorch.py:186-234`) concatenates SigLIP image embeddings (256
  tokens per image slot × 3 slots = 768) and language-token embeddings, with
  full bidirectional attention inside the prefix (`att_masks += [0]`).
  Language length = `max_token_len` (pi0_libero: 48; pi05_libero: 200).
- **Prefix KV computed once and cached** for the whole denoising loop:
  `sample_actions` (`pi0_pytorch.py:376-399`), `use_cache=True`; the prefix
  pass forces eager attention (`:391`).
- **Suffix** per denoise step: `embed_suffix` (`pi0_pytorch.py:237-315`).
  π0: `[state token (1) | action tokens (H=50)]` — the state branch exists
  only when `not self.pi05` (`:243-261`). π0.5: no state token; suffix =
  `[action tokens (H=10)]` and the timestep conditions via adaRMS instead
  (`:287-296`). Attention-mask boundaries: prefix cannot attend the suffix
  (`att_masks += [1]` at the state/action starts, `:261, :307`).
- **The denoise pass is JOINT attention**: `denoise_step`
  (`pi0_pytorch.py:421-470`) builds
  `full_att_2d_masks = cat([prefix_pad, suffix_att])` — every suffix query
  row attends `[image | language | (state) | suffix-causal]` keys in ONE
  softmax. There is **no cross-attention module to swap**; the expert pass
  also forces eager attention (`:447`), and the Gemma layer resolves
  `modeling_gemma.eager_attention_forward` at call time
  (`gemma_pytorch.py:201`; transformers 4.53.2 defines it module-level at
  `modeling_gemma.py:230`) — which is why the hook patches that module
  function.
- **Flow init noise** is drawn from the global torch RNG (`sample_noise`,
  `pi0_pytorch.py:172-179`: `torch.normal(...)` with no generator) → the
  harness noise-pin contract holds; `Policy.infer(obs, noise=...)` exists as
  an explicit-noise fallback.

Mapping (`pladis/attn_pi0.py`): **FLUX pattern, not SDXL** — the hook slices
key-column regions of the joint softmax row:

```
columns: [ img 0:768 | lang 768:768+L | state (768+L, π0 only) | suffix tail ]
```

`kind=text` blends the `lang` region, `kind=image` the `img` region, each
with the FLUX mass-preservation formula of §0; all other columns keep their
dense weights. `qgroup` = suffix query rows `[state(1, π0 only); action(H)]`
— the same row-split semantic as N1.7. Scope: only suffix (denoise) passes
are patched — the prefix VLM pass is left untouched (equivalent to processor
placement in the official repo, where only selected attention modules get
the sparse processor).

**π0.5 vs π0 differences that the hook must honor**: no state query row
(`qgroup=state` fails fast), L=200 vs 48, H=10 vs 50.

**Cross-model comparability notes**: `kind=all` on π0 covers the suffix
self-columns too (no N1.7 analogue — N1.7's cross blocks have no
action-to-action keys); flag when comparing `allxall` across models. π0's
vanilla path is already eager softmax — there is no fused↔eager numeric-path
term on this track, so base0 (scale 0) is bit-identical to vanilla by the
same code path and no eager-dense control arm is needed.

## 3. SmolVLA — joint-key cross-attention (FLUX pattern) [planned]

From lerobot's `modeling_smolvla.py` (to be re-measured with file:line when
that track starts): the action expert interleaves cross-attention layers
(action queries → cached prefix KV = `[64·frames visual | text | 1 state]`)
with causal self-attention every 2nd layer. The cross-attention key set mixes
modalities in one softmax → FLUX pattern again (region slice + mass renorm).
Queries are action tokens only → **no qgroup axis** (fail fast); `kind` can
gain a `state` key-region. Section must be completed and approved before the
hook is written.

## 4. GR00T N1.5 — dedicated cross-attention with fused-VL keys [planned]

N1.5's DiT cross-attends to a single fused vision-language sequence in every
cross block (to be measured on the N1.5 checkout when that track starts):
dedicated cross-attention module (SDXL pattern for the swap point) whose key
set mixes modalities → `kind` becomes a key-column region slice WITH mass
renormalization (FLUX formula) inside an SDXL-style processor; `qgroup` row
gating as in N1.7.

## 5. Deviations from official PLADIS (registry)

Anything not in this table is a bug, not a choice.

| deviation | where | rationale |
|---|---|---|
| `qgroup` query-row gating | all hooks | the research variable of this project (locus factorization); official PLADIS blends every query row |
| key-modality selection (`kind`) | all hooks | locus factorization; official selects layers, not key modalities |
| fp32 upcast of softmax/entmax logits | attn_gr00t_n17 | numerical stability under bf16 autocast (official runs fp16/fp32 ambient) |
| bool→additive mask with large-finite clamp | attn_gr00t_n17 | SDPA bool-mask convention + entmax NaN safety |
| `β` (inverse temperature) exposed as a flag | all hooks | temperature-control arm (paper suppl. G.1) |
| λ=0 delegates to the native attention path | attn_gr00t_n17 | official gates the processor swap on `pladis_scale > 0` (pipeline_sdxl.py:1215,1707); base0 ≡ vanilla bit-exact |
| suffix-pass-only patch scope (`max_suffix_query`) | attn_pi0 | equivalent of the official per-module processor placement, expressed as a call-site gate because the Gemma path has no processor registry |
| blend order `dense + λ(sparse−dense)` | all hooks | algebraically identical to the official `λ·sparse + (1−λ)·dense` |
