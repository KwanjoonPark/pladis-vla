# SPDX-License-Identifier: Apache-2.0
"""Per-episode TSV logging. One row per episode, flushed immediately —
a crash costs only the running episode. The instruction column doubles as
the standing proof that perturbed language reached the model.

Because the eplog IS the resume ledger, resuming into a file written by a
DIFFERENT arm would silently produce one file holding two arms' episodes —
undetectable downstream, since the TSV carries no arm identity. The arm
signature is therefore stored in a `<path>.arm` sidecar (not in the TSV, so
the schema every consumer parses is unchanged) and checked on resume."""

from __future__ import annotations

import os
from dataclasses import asdict, fields

from .rollout import EpisodeResult

COLUMNS = [f.name for f in fields(EpisodeResult)]


class EpisodeLogger:
    def __init__(self, path: str, resume: bool = True, arm_signature: str | None = None):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.done_episodes: set[int] = set()
        self._sig_path = path + ".arm"
        if resume and os.path.exists(path):
            self._check_arm(arm_signature)
            with open(path) as f:
                header = f.readline().rstrip("\n").split("\t")
                assert header == COLUMNS, f"eplog schema mismatch in {path}"
                for line in f:
                    # a hard crash can leave the final row truncated mid-write;
                    # a partial row is not a completed episode, so drop it and
                    # let the run redo that episode rather than failing to resume
                    fields_ = line.rstrip("\n").split("\t")
                    if len(fields_) != len(COLUMNS) or not fields_[0].isdigit():
                        print(f"[eplog] dropping partial trailing row in {path}", flush=True)
                        continue
                    self.done_episodes.add(int(fields_[0]))
            self._fh = open(path, "a")
        else:
            self._fh = open(path, "w")
            self._fh.write("\t".join(COLUMNS) + "\n")
            self._fh.flush()
        if arm_signature is not None:
            with open(self._sig_path, "w") as f:
                f.write(arm_signature + "\n")

    def _check_arm(self, arm_signature: str | None) -> None:
        """Refuse to append this arm's episodes to another arm's ledger."""
        if arm_signature is None:
            return
        if not os.path.exists(self._sig_path):
            print(
                f"[eplog] WARNING: {self.path} predates arm signatures — cannot "
                f"verify it was written by {arm_signature!r}",
                flush=True,
            )
            return
        with open(self._sig_path) as f:
            prev = f.read().strip()
        if prev != arm_signature:
            raise SystemExit(
                f"[eplog] REFUSING to resume {self.path}: it was written by arm "
                f"{prev!r} but this run is {arm_signature!r}. Resuming would mix "
                f"two arms in one eplog. Use a different --out."
            )

    def log(self, result: EpisodeResult):
        row = asdict(result)
        # instructions may contain tabs/newlines in principle — normalize
        row["instruction"] = " ".join(str(row["instruction"]).split())
        self._fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        self._fh.flush()
        self.done_episodes.add(result.episode)

    def close(self):
        self._fh.close()
