# SPDX-License-Identifier: Apache-2.0
"""GPU smoke: 1 anchor + 1 language-variant episode through the full harness,
for any registered model.

    bash experiments/run.sh [--venv <venv>] experiments/smoke_model.py \
        [--model gr00t_n17] [--model-path ...]

Passes when: the model loads through its registry adapter, both episodes run
to completion, and the eplog rows carry the delivered instructions (the
language row's instruction must differ from the anchor's) — the standing
guard against the silent instruction-delivery failure mode.
"""

import argparse

from harness.env import LiberoPlusTaskSet, LiberoPlusSession
from harness.eplog import EpisodeLogger
from harness.registry import MODELS, resolve_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gr00t_n17", choices=sorted(MODELS))
    p.add_argument("--model-path", default=None)
    args = p.parse_args()
    spec = MODELS[args.model]
    model_path = args.model_path or spec.default_model_path("libero_10")

    model = resolve_loader(spec)(model_path)
    print(f"[smoke] {args.model} loaded, chunk={model.output_action_chunks}")

    from harness.rollout import run_episode  # after the loader's import-order dance

    lang = LiberoPlusTaskSet("libero_10", "language")
    base = LiberoPlusTaskSet("libero_10", axis=None)
    lang_spec = lang.schedule(1, seed=0)[0]
    base_spec = [
        s for s in base.schedule(10, seed=0) if s.base_task == lang_spec.base_task
    ][0]

    sess = LiberoPlusSession(camera_height=256, camera_width=256, seed=0)
    out = f"results/smoke_{args.model}_eplog.tsv"
    log = EpisodeLogger(out, resume=False)
    rows = {}
    for tag, ts, ep_spec in (("anchor", base, base_spec), ("language", lang, lang_spec)):
        r = run_episode(
            sess, ep_spec, ts.init_states_of(ep_spec.task_name), model,
            episode_seed=0,
            max_steps=spec.default_max_steps,
            exec_horizon=spec.default_exec_horizon,
        )
        log.log(r)
        rows[tag] = r
        print(
            f"[smoke] {tag:8s} success={r.success_once} steps={r.n_steps} "
            f"{r.wall_s}s\n         instr: {r.instruction}"
        )
    sess.close()
    log.close()
    assert rows["language"].instruction != rows["anchor"].instruction, (
        "language-variant instruction identical to the anchor's — "
        "perturbed instruction did NOT reach the model"
    )
    print(f"[smoke] PASS — see {out}")


if __name__ == "__main__":
    main()
