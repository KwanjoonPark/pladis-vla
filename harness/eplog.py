# SPDX-License-Identifier: Apache-2.0
"""Per-episode TSV logging. One row per episode, flushed immediately —
a crash costs only the running episode. The instruction column doubles as
the standing proof that perturbed language reached the model.

Because the eplog IS the resume ledger, resuming into a file written by a
DIFFERENT arm would silently produce one file holding two arms' episodes —
undetectable downstream, since the TSV carries no arm identity. The arm
signature is therefore stored in a `<path>.arm` sidecar (not in the TSV, so
the schema every consumer parses is unchanged) and checked on resume.

Sidecar format: line 1 = the arm signature (the identity that is checked);
optional further lines = provenance records ("code <git-describe> host <name>",
one per run that actually wrote episodes), appended so a multi-server campaign
can attribute every eplog to the commit AND the machine that produced it. Only
line 1 participates in the resume check.

The host is recorded because of the campaign's other pairing rule (SETUP.md §0):
one (model x axis) campaign runs on ONE machine and one stack, since closed-loop
rollouts amplify kernel-level numeric differences and that breaks the episode
pairing every contrast depends on. Before the host was recorded, extending one
machine's arm on another was invisible after the fact — the eplog looked normal
and the contrast looked significant. Now it is refused (override:
PLADIS_ALLOW_HOST_MIX=1) and analyze.py can see which machine each arm came
from."""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, fields

from .rollout import EpisodeResult

COLUMNS = [f.name for f in fields(EpisodeResult)]


def host_id() -> str:
    """Machine identity for the provenance record.

    PLADIS_HOST overrides it, for hosts whose name is not stable (containers,
    schedulers) — the identity has to be the MACHINE, not whatever the
    orchestrator called the process this time.
    """
    return os.environ.get("PLADIS_HOST") or socket.gethostname()


def parse_hosts(line: str) -> list[str]:
    """Hosts named by one provenance line. Records written before 2026-08-26 carry
    no host token, so they yield nothing rather than a fabricated identity."""
    toks = line.split()
    return [toks[i + 1] for i, t in enumerate(toks) if t == "host" and i + 1 < len(toks)]


class EpisodeLogger:
    def __init__(self, path: str, resume: bool = True, arm_signature: str | None = None,
                 provenance: str | None = None):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.done_episodes: set[int] = set()
        self._sig_path = path + ".arm"
        resuming = resume and os.path.exists(path)
        if resuming:
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
        self._host = host_id()
        # Written on the FIRST logged episode, not here: a provenance record is a
        # claim that this run produced episodes, and a driver re-invocation that
        # resumes a finished arm produces none.
        self._provenance = provenance
        self._provenance_written = provenance is None
        self._host_conflict: set[str] = set()
        if arm_signature is not None:
            keep_history = resuming and os.path.exists(self._sig_path)
            if keep_history and self.done_episodes:
                prior = self._prior_hosts()
                if prior and self._host not in prior:
                    self._host_conflict = prior
                    print(
                        f"[eplog] WARNING: {self.path} holds {len(self.done_episodes)} "
                        f"episodes from {sorted(prior)}; this machine is "
                        f"{self._host!r}. Extending it here would put two machines' "
                        f"numerics in one arm (SETUP.md S0).",
                        flush=True,
                    )
            elif not keep_history:
                with open(self._sig_path, "w") as f:
                    f.write(arm_signature + "\n")

    def _prior_hosts(self) -> set[str]:
        """Machines that have already written episodes into this eplog."""
        hosts: set[str] = set()
        if not os.path.exists(self._sig_path):
            return hosts
        with open(self._sig_path) as f:
            f.readline()  # line 1 is the signature, never a provenance record
            for line in f:
                hosts |= set(parse_hosts(line))
        return hosts

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
            prev = f.readline().strip()
        if prev != arm_signature:
            raise SystemExit(
                f"[eplog] REFUSING to resume {self.path}: it was written by arm "
                f"{prev!r} but this run is {arm_signature!r}. Resuming would mix "
                f"two arms in one eplog. Use a different --out."
            )

    def _record_provenance(self) -> None:
        """Append this run's provenance, refusing a cross-machine extension first."""
        if self._host_conflict and os.environ.get("PLADIS_ALLOW_HOST_MIX") == "1":
            print(f"[eplog] PLADIS_ALLOW_HOST_MIX=1: extending {sorted(self._host_conflict)} "
                  f"episodes on {self._host!r} anyway — this arm is no longer paired "
                  f"with the rest of its axis.", flush=True)
            self._host_conflict = set()
        if self._host_conflict:
            raise SystemExit(
                f"[eplog] REFUSING to extend {self.path} on {self._host!r}: its "
                f"{len(self.done_episodes)} logged episodes were produced on "
                f"{sorted(self._host_conflict)}. One (model x axis) campaign runs on "
                f"ONE machine (SETUP.md S0) — mixing them breaks the episode pairing "
                f"every contrast depends on, and nothing downstream could see it. "
                f"Run this arm on the original machine, or re-run the whole contrast "
                f"here under a different --out. Override: PLADIS_ALLOW_HOST_MIX=1."
            )
        if not self._provenance_written:
            with open(self._sig_path, "a") as f:
                f.write(f"code {self._provenance} host {self._host}\n")
            self._provenance_written = True

    def log(self, result: EpisodeResult):
        if not self._provenance_written or self._host_conflict:
            self._record_provenance()
        row = asdict(result)
        # instructions may contain tabs/newlines in principle — normalize
        row["instruction"] = " ".join(str(row["instruction"]).split())
        self._fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        self._fh.flush()
        self.done_episodes.add(result.episode)

    def close(self):
        self._fh.close()
