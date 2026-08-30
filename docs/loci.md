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

### 1.1 GR00T N1.7 — odd self-attention blocks: Hopfield sym/skew decomposition [implemented]

A second intervention family lives on the blocks §1 leaves alone. Measured:
`AlternateVLDiT.forward` runs every ODD block as self-attention through the
SAME `attn1` module, with `encoder_hidden_states=None` and
`attention_mask=None` (`gr00t/model/modules/dit.py:380-388`), over the
`[state(1); action(40)]` sequence of `gr00t_n1d7.py:248` — so its logits
`L = QKᵀ/√d` are SQUARE (41×41 per head, 32 heads × 48, 16 odd blocks of 32)
and queries and keys are the same tokens. That is the precondition of the
symmetric/skew decomposition of Cho, Han & Jin (ICML 2026, *Balancing
Fidelity and Diversity in Diffusion Models via Symmetric Attention
Decomposition: Hopfield Perspective*): with `L = L_sym + L_skew`,
`L_sym = (L+Lᵀ)/2` is a Hopfield energy landscape over the tokens and
`L_skew = (L−Lᵀ)/2` a circulation that leaves the energy untouched
(`ξᵀL_skewξ ≡ 0`). The even (cross) blocks have VLM keys — non-square, keys ≠
queries — so the decomposition is undefined there and they stay the PLADIS
locus of §1. The two block sets are disjoint, which is what lets both
processors be installed on one DiT (a combined arm is the union of the two
flag sets).

Mapping (`pladis/attn_gr00t_n17.py`, `HopfieldAttnProcessor`): per odd block
and head, `L_α = L + (α−1)·L_skew`, `Ξ = softmax(L)V`, `Ξ_α = softmax(L_α)V`,
`Ξ_b = Ξ + β(Ξ_α − Ξ)`, then the reference code's norm-match
`Ξ_out = Ξ_b · clamp(‖Ξ_α‖₂/‖Ξ_b‖₂, 0.25, 4)` per (batch, head, query row).
`qgroup` = the same `[state; action]` row slice as §1, rows outside it keep
`Ξ`; the denoising-step schedule multiplies β per step; `β=0` (and every
zero-weight step) takes the fused-SDPA path, bit-identical to vanilla; `α=1`
is bit-identical to the eager dense path (the self-block eager-dense
control). Full derivation, arms and gates: `docs/hopfield.md`.

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

## 3. SmolVLA — interleaved CA/SA over a joint-key prefix [implemented]

Measured against lerobot 0.4.4 (`smolvlm_with_expert.py`) and verified on the
REGISTRY-DEFAULT checkpoint `lerobot/smolvla_libero`
(`models/smolvla_libero_official`; delivery smoke through eval_arm,
2026-08-02). The earlier `HuggingFaceVLA/smolvla_libero` numbers
(2026-07-27) are kept below as the secondary geometry the self-calibration
also covers.

- **Checkpoint lineage** (train_config.json): `lerobot/smolvla_libero` is the
  org-official RETRAIN by pepijn223 on `lerobot/libero` (the port of the
  paper's PI dataset, 1,693 eps) at 25k steps / bs32 — 1/8 the paper's
  compute (100k / bs64). The paper's own Table-2 weights were never released
  (community non-repro on record: lerobot#1369). Its config sets
  `n_action_steps=50` (open-loop full chunk — the paper's own k=50 ablation
  collapses to 51.8); our protocol overrides to exec 10. A camera3 exists in
  the config but no third camera is in the batch, so only 2 are embedded —
  the P==177 assert below re-verifies that coincidence on every inference.
- All attention flows through ONE function, `eager_attention_forward`
  (:504-548; `get_attention_interface` :500 resolves
  `self.eager_attention_forward`, so an INSTANCE attribute shadows it —
  the hook patches one served instance, nothing global; the finder accepts
  the harness adapter via its `.policy` hop, found 2026-08-02).
- Layer types (`attention_mode="cross_attn"`, `self_attn_every_n_layers=2`):
  official ckpt = 16 expert layers = **8 SA (even `layer_idx`) + 8 CA (odd)**,
  census per 10-step chunk: 16 prefix-pass calls + **80 CA + 80 SA** denoise
  calls = 176 (delivery smoke: kind=text fires CA=80/SA=0, kind=self
  CA=0/SA=80). (HFVLA ckpt: 32 layers = 16+16, 160 fires per chunk.)
  * CA (odd `layer_idx`): expert re-projects the cached VLM prefix K/V
    (:342-365) — key axis `[image | language | state]`, NO suffix columns.
  * SA (even): suffix K/V concatenated onto cached prefix
    K/V (:264-266) — key axis `[prefix | suffix]`, a π0.5-style joint row.
- Prefix layout: `n_img = 128` (2 cameras × 64 connector tokens),
  `n_state = 1`. Official ckpt pads language to `max_length` 48
  (`pad_language_to`) → **prefix_len is the CONSTANT 177 = 128 + 48 + 1**,
  with n_lang=48 sitting exactly at the n_lang_max bound (delivery smoke
  asserts 177 every inference). HFVLA pads `longest` → per-episode width
  (probed: 141 = 128 + 12 + 1). The hook self-calibrates from the prefix
  pass (q == k) of each inference and exact-fits every denoise call
  (`pladis/attn_smolvla.py`), so both paddings are served by one code path.
- Queries are action tokens only (suffix = action+timestep MLP; state is a
  prefix KEY) → **no qgroup axis** (`state` raises). `kind` gains `prefix`
  (whole CA row = GR00T allxall analogue), `self` (SA suffix block), and two
  MULTI-BLOCK mass-preserving kinds (2026-08-02, operator decision that every
  sweep arm must preserve modality mass): `cams` = the image span split
  per camera ([camera1 64 | camera2 64] = [agentview | wrist] in the ckpt
  preprocessor's feature order), each half sharpened with its OWN mass fixed;
  `text-image` = text and image blocks each mass-fixed — the maximal
  mass-preserving prefix intervention (the sweep replaces the plain
  whole-row `prefix` arm with it; `prefix` stays available as the one kind
  that can move mass across block borders).
  `kind=state` exists in the vocabulary but is REFUSED at install for real
  SmolVLA checkpoints: state is ONE prefix token, and the mass-preserving
  blend on a width-1 block is a bit-exact identity (entmax over one column
  is 1 and m is the dense entry itself) — a no-op arm the Phase-D
  diagnostic caught before it burned a sweep (2026-08-02).
- Sub-block kinds use the FLUX mass-preserving blend; `prefix` uses the
  plain whole-row blend. Noise-pin admissible: `sample_noise` =
  global `torch.normal` (modeling_smolvla.py:609).
- Gates: `verify_smolvla_hook.py` (CPU, 7 gates incl. bit-parity vs stock
  and self-calibration) + on-checkpoint delivery smoke THROUGH eval_arm
  (`--pladis-install --pladis-kind text|self`, 2 eps each: census above,
  prefix_len=177, `_assert_smolvla_delivery` before any episode logs) —
  ALL PASS on the official ckpt, 2026-08-02.
- Instruction protocol (2026-08-02): anchors (`--axis none`) use
  `--instruction-source task-meta` — the filename-derived training strings
  (verified equal to `libero.benchmark task_maps[*].language` on all 40
  tasks); the BDDL parse is OOD phrasing for this checkpoint and collapsed
  the old anchors (spatial 41→87 on the instruction swap alone, disc 47:1,
  z=+6.6; stack parity vs lerobot-eval z=0.77 n.s.). LIBERO-plus sweep arms
  keep the BDDL parse — on the language axis it IS the perturbation.

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

Deviations from the Hopfield reference code (Cho et al. 2026, Appendix A.1,
Algorithm 2) in `HopfieldAttnProcessor` (§1.1):

| deviation | where | rationale |
|---|---|---|
| blend and norm-match on `Z = P·(X W_V)` (per-head 48-ch output) instead of on `Ξ = P·X` (1536-ch input features) with `W_V` applied afterwards | attn_gr00t_n17 | `Z = Ξ W_V^h` by associativity, so the blend is the paper's up to rounding; only the norm-match ratio is measured on 48 channels instead of 1536. Same placement as the NAG stage (docs/nag.md §3), one code convention for both output-space stages |
| `L_α = L + (α−1)·L_skew` instead of `L_sym + α·L_skew` | attn_gr00t_n17 | identical algebra (paper Eq. 39 form); makes `α=1` bit-identical to the dense path instead of 1-ulp off, so the eager-dense control is `torch.equal` |
| `qgroup` row gating, β step schedule | attn_gr00t_n17 | this project's locus factorization; the reference applies the blend to every row of every self-attention layer at every step |
| diagnostics (`E`, `r`, `Align`, `η`) computed on `Ξ = P·X` exactly as published | attn_gr00t_n17 (probe) | no deviation — listed so the two spaces are not confused: the *intervention* is on `Z`, the *measurement* on `Ξ` |
