# SPDX-License-Identifier: Apache-2.0
"""Per-episode TSV logging. One row per episode, flushed immediately —
a crash costs only the running episode. The instruction column doubles as
the standing proof that perturbed language reached the model."""

from __future__ import annotations

import os
from dataclasses import asdict, fields

from .rollout import EpisodeResult

COLUMNS = [f.name for f in fields(EpisodeResult)]


class EpisodeLogger:
    def __init__(self, path: str, resume: bool = True):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.done_episodes: set[int] = set()
        if resume and os.path.exists(path):
            with open(path) as f:
                header = f.readline().rstrip("\n").split("\t")
                assert header == COLUMNS, f"eplog schema mismatch in {path}"
                for line in f:
                    self.done_episodes.add(int(line.split("\t", 1)[0]))
            self._fh = open(path, "a")
        else:
            self._fh = open(path, "w")
            self._fh.write("\t".join(COLUMNS) + "\n")
            self._fh.flush()

    def log(self, result: EpisodeResult):
        row = asdict(result)
        # instructions may contain tabs/newlines in principle — normalize
        row["instruction"] = " ".join(str(row["instruction"]).split())
        self._fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        self._fh.flush()
        self.done_episodes.add(result.episode)

    def close(self):
        self._fh.close()
