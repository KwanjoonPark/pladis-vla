# SPDX-License-Identifier: Apache-2.0
"""π0.5 λ=0 parity gate (README §5 gate 3) + the measurement that retires two sweep arms.

Three checks:

(a) MODULE-LEVEL λ=0 parity, at the REAL π0.5 shapes. The hook's λ=0 branch returns
    `torch.softmax(scores.float(), -1)` and the caller casts to query.dtype — which is
    exactly what stock gemma's
    `nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)` computes.
    Asserted with torch.equal against the eager_attention_forward that openpi's
    transformers_replace actually installs (NOT stock 4.53.2 — that file adds adaRMS and
    is the one the model runs), at both the suffix shape (q=10, k=978) and the prefix
    shape (q=968), in bf16 and fp32.

(b) END-TO-END base0 parity: N episodes vanilla vs `--pladis-install --pladis-scale 0`,
    eplogs compared column-for-column. This is what lets sweep_pi05_language.sh OMIT
    base0 as a 1,537-episode arm — the same call sweep_n17_robot.sh:9-12 made for n17.

(c) DENSE-COLLAPSE MEASUREMENT (not a pass/fail on the published design, a decision
    input). With method=softmax and β=1, `p = softmax(sub)` and `m*p == dense[sub]` is an
    identity, so the blend collapses to dense for ANY λ and only floating-point
    reassociation remains. In GR00T the eager-dense arm absorbed a genuine
    fused-SDPA-vs-eager KERNEL difference; π0.5 is on the eager path already, so the
    claim is that vanilla IS the numeric control and the arm is unnecessary. This check
    quantifies it: the fraction of bf16 attention-weight elements that differ, plus an
    N-episode eplog comparison. If the divergence is material, the arm goes back in.

Run: bash experiments/run.sh --venv openpi experiments/verify_pi05_parity.py
"""

from __future__ import annotations

import argparse
import os

import torch

from harness.env import LiberoPlusSession, LiberoPlusTaskSet
from harness.model_pi05 import load_pi05, preload_sim_stack
from harness.rollout import run_episode
from pladis import attn_pi05
from pladis.attn_pi05 import CFG, install_pladis

# `_expert_shape()` below imports openpi at MODULE scope, which poisons MagickWand's
# dlopen for the rest of the process — check (b)'s load_pi05 would then die at
# liberoplus import, after check (a) has already run. Preload first.
preload_sim_stack()

# real pi05_libero geometry, confirmed on the checkpoint by verify_pi05_delivery.py
N_IMG, N_LANG, SUFFIX = 768, 200, 10
PREFIX = N_IMG + N_LANG

# Action-expert attention shape, READ from openpi's own config rather than hardcoded —
# gemma_300m is num_heads=8 / num_kv_heads=1 (i.e. GQA with 8 groups, not 1), and getting
# that wrong would make this gate compare a differently-shaped attention than the model
# actually runs. depth=18 also explains the 180 blended forwards/chunk the delivery gate
# observes (18 layers x 10 denoise steps).
def _expert_shape():
    from openpi.models import gemma as _gemma

    c = _gemma.get_config("gemma_300m")
    return c.num_heads, c.num_kv_heads, c.head_dim, c.depth


HEADS, KV_HEADS, HEAD_DIM, DEPTH = _expert_shape()


def _restore_stock():
    """Un-patch transformers' gemma eager attention.

    install_pladis monkeypatches a MODULE GLOBAL, and the hook has no uninstall by design
    (every sweep arm is its own process, so it never needs one). This gate is the one
    place that needs it: without restoring, the "vanilla" baseline of check (b) would run
    through the patched function with whatever CFG the previous check left behind, and the
    comparison would be circular — comparing base0 against base0.
    """
    import transformers.models.gemma.modeling_gemma as mg

    orig = getattr(mg.eager_attention_forward, "_orig", None)
    if orig is not None:
        mg.eager_attention_forward = orig
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            ALL_ATTENTION_FUNCTIONS["eager"] = orig
        except Exception:
            pass
    CFG.installed = False
    return mg.eager_attention_forward


class _FakeModule:
    """Stands in for a GemmaAttention module: eager_attention_forward only reads
    num_key_value_groups and training off it."""

    def __init__(self, groups: int):
        self.num_key_value_groups = groups
        self.training = False


def _qkv(q_len, k_len, dtype, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, HEADS, q_len, HEAD_DIM, generator=g).to(dev, dtype)
    k = torch.randn(1, KV_HEADS, k_len, HEAD_DIM, generator=g).to(dev, dtype)
    v = torch.randn(1, KV_HEADS, k_len, HEAD_DIM, generator=g).to(dev, dtype)
    return q, k, v


def check_module_parity(dev="cuda") -> bool:
    """(a) module-level λ=0 bit parity against openpi's installed gemma eager path."""
    import transformers.models.gemma.modeling_gemma as mg

    stock = getattr(mg.eager_attention_forward, "_orig", mg.eager_attention_forward)
    if getattr(stock, "_pladis_wrapped", False):
        print("[a] FAIL could not recover an unwrapped eager_attention_forward")
        return False

    ok = True
    scaling = HEAD_DIM ** -0.5
    # λ=0 must be bit-vanilla; and λ>0 at the PREFIX shape must also be bit-vanilla,
    # because the max_suffix_query gate has to keep the VLM pass off the blend.
    cases = [
        ("suffix q=10  k=978  λ=0",   SUFFIX, PREFIX + SUFFIX, 0.0),
        ("prefix q=968 k=968  λ=0",   PREFIX, PREFIX, 0.0),
        ("prefix q=968 k=968  λ=1.5", PREFIX, PREFIX, 1.5),
    ]
    for dtype in (torch.bfloat16, torch.float32):
        for name, q_len, k_len, lam in cases:
            install_pladis(model=None, pladis_scale=lam, method="entmax15", kind="text",
                           n_img_prefix=N_IMG, n_lang=N_LANG)
            q, k, v = _qkv(q_len, k_len, dtype, dev)
            mod = _FakeModule(HEADS // KV_HEADS)  # GQA groups (gemma_300m: 8 // 1 = 8)
            with torch.no_grad():
                o_ref, w_ref = stock(mod, q, k, v, None, scaling, dropout=0.0)
                o_new, w_new = mg.eager_attention_forward(mod, q, k, v, None, scaling,
                                                          dropout=0.0)
            bit = torch.equal(o_ref, o_new) and torch.equal(w_ref, w_new)
            tag = str(dtype).replace("torch.", "")
            if bit:
                print(f"[a] OK   {name:26s} {tag:9s} bit-identical to stock eager")
            else:
                d = (w_ref.float() - w_new.float()).abs().max().item()
                print(f"[a] FAIL {name:26s} {tag:9s} max|dw|={d:.3e}")
                ok = False
    return ok


def measure_dense_collapse(dev="cuda") -> None:
    """(c) how far does (λ, softmax, β=1) actually move the attention weights?"""
    import transformers.models.gemma.modeling_gemma as mg

    stock = getattr(mg.eager_attention_forward, "_orig", mg.eager_attention_forward)
    scaling = HEAD_DIM ** -0.5
    for kind in ("text", "image", "prefix"):
        install_pladis(model=None, pladis_scale=1.0, method="softmax", beta=1.0, kind=kind,
                       n_img_prefix=N_IMG, n_lang=N_LANG)
        q, k, v = _qkv(SUFFIX, PREFIX + SUFFIX, torch.bfloat16, dev, seed=7)
        mod = _FakeModule(HEADS // KV_HEADS)  # GQA groups (gemma_300m: 8 // 1 = 8)
        with torch.no_grad():
            _, w_ref = stock(mod, q, k, v, None, scaling, dropout=0.0)
            _, w_new = mg.eager_attention_forward(mod, q, k, v, None, scaling, dropout=0.0)
        neq = (w_ref != w_new)
        frac = neq.float().mean().item()
        worst = (w_ref.float() - w_new.float()).abs().max().item()
        print(f"[c] kind={kind:6s} bf16 elements differing: {frac * 100:.4f}%  "
              f"max|dw|={worst:.3e}  (identity says 0; residual is fp reassociation)")


def check_rollout_parity(model_path, suite, n_eps, dev="cuda") -> bool:
    """(b) vanilla vs base0 over real rollouts, and (c)'s end-to-end half."""
    ts = LiberoPlusTaskSet(suite, "language")
    specs = ts.schedule(n_eps, seed=0)
    sess = LiberoPlusSession(seed=0)
    model = load_pi05(model_path)
    # checks (a)/(c) ran install_pladis, which patched the module global for good. The
    # baseline below must be TRUE vanilla, not base0, or (b) proves nothing.
    _restore_stock()

    def rollout(label):
        rows = []
        for spec in specs:
            r = run_episode(sess, spec, ts.init_states_of(spec.task_name), model,
                            episode_seed=0 * 1_000_003 + spec.episode,
                            max_steps=520, exec_horizon=5)
            rows.append(r)
        print(f"[b] {label:22s} SR={sum(r.success_once for r in rows) / len(rows):.3f} "
              f"steps={[r.n_steps for r in rows]}")
        return rows

    # vanilla: the hook is never installed, so the module global is untouched
    base = rollout("vanilla")

    install_pladis(model=None, pladis_scale=0.0, method="entmax15", kind="text",
                   n_img_prefix=N_IMG, n_lang=N_LANG)
    zero = rollout("base0 (λ=0)")

    install_pladis(model=None, pladis_scale=1.0, method="softmax", beta=1.0, kind="text",
                   n_img_prefix=N_IMG, n_lang=N_LANG)
    dense = rollout("dense (λ=1,softmax)")
    sess.close()

    cols = ("episode", "task_name", "instruction", "success_once", "success_at_end",
            "n_steps")
    ok = True
    for label, rows in (("base0", zero), ("dense", dense)):
        diffs = [
            (a.episode, c, getattr(a, c), getattr(b, c))
            for a, b in zip(base, rows) for c in cols
            if getattr(a, c) != getattr(b, c)
        ]
        if not diffs:
            print(f"[{'b' if label == 'base0' else 'c'}] OK   {label}: eplogs identical to "
                  f"vanilla on {cols} over {len(rows)} episode(s)")
        elif label == "base0":
            print(f"[b] FAIL base0 diverged from vanilla: {diffs[:6]}")
            ok = False
        else:
            # not a failure of THIS gate — it is the finding that would put the
            # eager-dense arm back into the sweep
            print(f"[c] NOTE dense diverged from vanilla on {len(diffs)} field(s): "
                  f"{diffs[:6]}")
            print("[c]      -> chaotic amplification of fp reassociation is real at this "
                  "scale; reinstate the eager-dense control as a sweep arm")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=os.environ.get(
        "MODEL_ROOT_PI05", "/home/reallab/parkkwanjoon/workspace/models/pi05_libero"))
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--episodes", type=int, default=10,
                   help="rollout episodes per arm for check (b)")
    args = p.parse_args()

    ok = check_module_parity()
    measure_dense_collapse()
    ok &= check_rollout_parity(args.model_path, args.suite, args.episodes)

    print("ALL GATES PASSED" if ok else "GATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
