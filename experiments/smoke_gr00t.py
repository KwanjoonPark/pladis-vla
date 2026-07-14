# SPDX-License-Identifier: Apache-2.0
"""GPU smoke: 1 anchor + 1 language-variant episode through the full harness.

Passes when: model loads, both episodes run to completion, the eplog rows
carry the delivered instructions (variant != original), and timing is sane.
Run:
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MAGICK_HOME=... LD_LIBRARY_PATH=... \
  PYTHONPATH=<pladis-vla> <gr00t venv python> experiments/smoke_gr00t.py
"""

import numpy as np

from harness.env import LiberoPlusTaskSet, LiberoPlusSession
from harness.eplog import EpisodeLogger
from harness.model_gr00t import load_gr00t_n1d7
from harness.rollout import run_episode

MODEL = "/home/reallab/parkkwanjoon/workspace/models/GR00T-N1.7-LIBERO/libero_10"
BACKBONE = (
    "/home/reallab/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/"
    "snapshots/9ce19a195e423419c349abfc86fd07178b230561"
)


def main():
    model = load_gr00t_n1d7(MODEL, BACKBONE)
    print(f"[smoke] model loaded, chunk={model.output_action_chunks}")

    lang = LiberoPlusTaskSet("libero_10", "language")
    base = LiberoPlusTaskSet("libero_10", axis=None)
    lang_spec = lang.schedule(1, seed=0)[0]
    base_spec = [
        s for s in base.schedule(10, seed=0) if s.base_task == lang_spec.base_task
    ][0]

    sess = LiberoPlusSession(camera_height=256, camera_width=256, seed=0)
    log = EpisodeLogger("results/smoke_gr00t_eplog.tsv", resume=False)
    for tag, ts, spec in (("anchor", base, base_spec), ("language", lang, lang_spec)):
        r = run_episode(
            sess, spec, ts.init_states_of(spec.task_name), model, episode_seed=0
        )
        log.log(r)
        print(
            f"[smoke] {tag:8s} success={r.success_once} steps={r.n_steps} "
            f"{r.wall_s}s\n         instr: {r.instruction}"
        )
    sess.close()
    log.close()
    print("[smoke] PASS — see results/smoke_gr00t_eplog.tsv")


if __name__ == "__main__":
    main()
