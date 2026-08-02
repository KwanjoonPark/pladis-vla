# SPDX-License-Identifier: Apache-2.0
"""On-model delivery gate for pladis/attn_pi05.py (§5 gate 5b) — needs the real π0.5.

verify_pi05_hook.py proves the blend's ALGEBRA on synthetic tensors. It cannot prove the
two things that only the real model can answer, and that would each turn a λ>0 arm into
a silent no-op or a mis-sliced intervention:

  A. delivery      — the π0.5 Gemma expert really is a transformers `gemma` module on
                     the eager path, so the module-global monkeypatch fires.
  B. geometry      — the key axis really is [image 768 | language 200 | suffix 10], so
                     kind=text/image/prefix slice the blocks they claim to.

Also checked here, because each is a silent-failure mode rather than a crash:
  C. the expert config reads "eager" AFTER a forward (pi0_pytorch.py:447 sets it during
     the first suffix step, not at construction — so checking it earlier is meaningless).
  D. sample_actions is not a torch.compile wrapper (a compiled graph freezes whichever
     eager_attention_forward global it traced, un-installing the hook invisibly).
  E. the large-query PREFIX pass stays dense — PaliGemma's language model is ALSO a
     gemma module on the eager path, so the same patch fires there; only the
     max_suffix_query gate keeps it out of the blend.

Run: bash experiments/run.sh --venv openpi experiments/verify_pi05_delivery.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from harness.env import LiberoPlusSession, LiberoPlusTaskSet
from harness.model_pi05 import load_pi05
from pladis.attn_pi05 import CFG, assert_delivered, install_pladis

# pi05_libero: 3 openpi image slots x 256 tokens + max_token_len 200 = 968 conditioning
# keys; the suffix query is action_horizon = 10, so a suffix step sees 978 keys.
EXPECT_N_IMG, EXPECT_N_LANG, EXPECT_SUFFIX = 768, 200, 10
EXPECT_PREFIX = EXPECT_N_IMG + EXPECT_N_LANG
EXPECT_SHAPE = (EXPECT_SUFFIX, EXPECT_PREFIX + EXPECT_SUFFIX)


def _one_chunk(model, sess, ts, spec):
    raw_obs, instruction = sess.reset(spec, ts.init_states_of(spec.task_name))
    torch.manual_seed(0)
    with torch.no_grad():
        chunk, _ = model.predict_action_batch(model.wrap_obs(raw_obs, instruction), mode="eval")
    return np.asarray(chunk), instruction


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get(
        "MODEL_ROOT_PI05", "/home/reallab/parkkwanjoon/workspace/models/pi05_libero"))
    p.add_argument("--suite", default="libero_10")
    args = p.parse_args()

    ok = True
    ts = LiberoPlusTaskSet(args.suite, "language")
    spec = ts.schedule(1, seed=0)[0]
    sess = LiberoPlusSession(seed=0)
    model = load_pi05(args.model_path)

    # ---- D: no torch.compile wrapper on the sampling path -------------------------
    fn = model._policy._sample_actions
    if hasattr(fn, "_torchdynamo_orig_callable"):
        print("[D] FAIL sample_actions is still a torch.compile wrapper")
        ok = False
    else:
        print(f"[D] OK   sample_actions un-compiled ({type(fn).__name__})")

    # ---- A + B: install, run ONE real chunk, then assert --------------------------
    # Order matters: attn_pi05._uses_eager_gemma reads model.config._attn_implementation,
    # which the openpi policy object does not have -> returns None, install proceeds, and
    # the whole guard collapses onto assert_delivered(). Passing the gemma submodule
    # instead would RAISE here, because its config only becomes "eager" during the first
    # suffix step (pi0_pytorch.py:447). So: install -> forward -> assert.
    install_pladis(model, pladis_scale=1.0, method="entmax15", kind="text",
                   n_img_prefix=EXPECT_N_IMG, n_lang=EXPECT_N_LANG)
    chunk, instruction = _one_chunk(model, sess, ts, spec)

    try:
        assert_delivered()
        print(f"[A] OK   blend fired {CFG.n_calls} time(s) on a real chunk")
    except RuntimeError as exc:
        print(f"[A] FAIL {exc}")
        ok = False

    if CFG.seen_shapes == {EXPECT_SHAPE}:
        print(f"[B] OK   geometry: blended only at (q,k)={EXPECT_SHAPE} "
              f"[image {EXPECT_N_IMG} | language {EXPECT_N_LANG} | suffix {EXPECT_SUFFIX}]")
    else:
        print(f"[B] FAIL geometry: blended at {sorted(CFG.seen_shapes)}, want "
              f"{{{EXPECT_SHAPE}}}. A different key_len means n_img_prefix/n_lang do not "
              f"match this checkpoint and kind={{text,image,prefix}} mis-slice.")
        ok = False

    # ---- E: the prefix/VLM pass (query_len 968) must NOT have been blended --------
    # It shares the patched module global; only the max_suffix_query gate excludes it.
    leaked = [s for s in CFG.seen_shapes if s[0] > CFG.max_suffix_query]
    if leaked:
        print(f"[E] FAIL prefix pass was blended at {leaked} "
              f"(max_suffix_query={CFG.max_suffix_query})")
        ok = False
    else:
        print(f"[E] OK   no forward with query_len > {CFG.max_suffix_query} was blended "
              f"(the PaliGemma prefix pass stayed dense)")

    # ---- C: the expert really is on transformers' eager path AFTER a forward ------
    try:
        expert_cfg = model.model.paligemma_with_expert.gemma_expert.model.config
        impl = getattr(expert_cfg, "_attn_implementation", None)
        if impl == "eager":
            print("[C] OK   gemma_expert._attn_implementation == 'eager' after a forward")
        else:
            print(f"[C] FAIL gemma_expert._attn_implementation == {impl!r}, want 'eager'")
            ok = False
    except AttributeError as exc:
        print(f"[C] FAIL cannot reach the gemma expert config: {exc}")
        ok = False

    # sanity: the chunk is usable and in LIBERO's native gripper range
    if chunk.shape != (1, EXPECT_SUFFIX, 7) or not np.isfinite(chunk).all():
        print(f"[*] FAIL chunk shape/finiteness: {chunk.shape}")
        ok = False
    else:
        g = chunk[0, :, 6]
        print(f"[*] OK   chunk {chunk.shape} finite; gripper range "
              f"[{g.min():+.3f}, {g.max():+.3f}] (LIBERO-native [-1,+1], no remap)")
        print(f"[*] instruction delivered: {instruction!r}")

    sess.close()
    print("ALL GATES PASSED" if ok else "GATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
