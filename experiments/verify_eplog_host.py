# SPDX-License-Identifier: Apache-2.0
"""CPU gates for the eplog's machine-provenance guard (no GPU, no checkpoint).

SETUP.md §0: one (model × axis) campaign runs on ONE machine and one stack —
closed-loop rollouts amplify kernel-level numeric differences, which breaks the
episode pairing every contrast depends on. Until 2026-08-26 the `.arm` sidecar
recorded the commit but not the machine, so extending one machine's arm on
another was invisible afterwards: the eplog looked normal and the contrast looked
significant. These gates pin the guard that closed that hole.

  A. A fresh eplog writes the signature immediately, and the provenance record
     ONLY once an episode is actually logged (a run that produces no episodes has
     nothing to attribute), as `code <version> host <name>`.
  B. Resuming on the same host appends a second record and logs normally.
  C. Resuming on a DIFFERENT host raises before the first row is written, naming
     both machines — the eplog is left exactly as the other machine had it.
  D. PLADIS_ALLOW_HOST_MIX=1 turns that into a loud warning, and both hosts end
     up recorded so analyze.py can still see the arm is compromised.
  E. A no-op resume (every episode already logged) on another host does NOT
     raise: sweep drivers re-invoke every arm on every run, and a finished arm
     must stay a seconds-long no-op on any machine.
  F. Records written before the host token existed yield no host rather than a
     fabricated one, so pre-2026-08-26 eplogs keep resuming.
  G. analyze.py READS it back: a guard nothing surfaces is a guard nobody acts on,
     and analyze.py has to stay importable with no venv/GPU/simulator for that.

Run: bash experiments/run.sh experiments/verify_eplog_host.py
"""

import os
import sys
import tempfile

from harness.eplog import EpisodeLogger, host_id, parse_hosts
from harness.rollout import EpisodeResult

SIG = "suite=libero_10|axis=language|seed=0|pladis=off"


def _result(ep: int) -> EpisodeResult:
    return EpisodeResult(episode=ep, task_name="t", base_task="b", init_state_id=0,
                         instruction="pick up the thing", success_once=1,
                         success_at_end=1, n_steps=42, wall_s=1.0)


def _sidecar(path: str) -> list[str]:
    with open(path + ".arm") as f:
        return [ln.rstrip("\n") for ln in f]


def _logger(path: str, host: str, version: str = "abc1234") -> EpisodeLogger:
    os.environ["PLADIS_HOST"] = host
    return EpisodeLogger(path, resume=True, arm_signature=SIG, provenance=version)


def _raises(fn, *, what: str) -> str:
    try:
        fn()
    except SystemExit as exc:
        return str(exc)
    raise AssertionError(f"expected a raise: {what}")


def gate_A(path):
    log = _logger(path, "machine-a")
    assert _sidecar(path) == [SIG], f"A: provenance written before any episode: {_sidecar(path)}"
    log.log(_result(0))
    log.close()
    lines = _sidecar(path)
    assert lines[0] == SIG and lines[1] == "code abc1234 host machine-a", lines
    print("PASS gate A: signature at open, provenance only once an episode is logged")


def gate_B(path):
    log = _logger(path, "machine-a", version="def5678")
    assert log.done_episodes == {0}
    log.log(_result(1))
    log.close()
    lines = _sidecar(path)
    assert lines[1:] == ["code abc1234 host machine-a", "code def5678 host machine-a"], lines
    print("PASS gate B: same-host resume appends a record and logs")


def gate_C(path):
    before = open(path).read()
    log = _logger(path, "machine-b")
    msg = _raises(lambda: log.log(_result(2)), what="cross-machine extension")
    assert "machine-a" in msg and "machine-b" in msg, msg
    assert open(path).read() == before, "C: the eplog was modified before the refusal"
    assert not any("machine-b" in ln for ln in _sidecar(path)), \
        "C: a refused run still stamped its host on the sidecar"
    print("PASS gate C: cross-machine extension refused before the first row")


def gate_D(path):
    os.environ["PLADIS_ALLOW_HOST_MIX"] = "1"
    try:
        log = _logger(path, "machine-b")
        log.log(_result(2))
        log.close()
    finally:
        del os.environ["PLADIS_ALLOW_HOST_MIX"]
    hosts = {h for ln in _sidecar(path)[1:] for h in parse_hosts(ln)}
    assert hosts == {"machine-a", "machine-b"}, hosts
    print("PASS gate D: the override logs, and leaves both machines on the record")


def gate_E(path):
    # every episode this run would write is already logged -> resume no-op
    log = _logger(path, "machine-c")
    before = _sidecar(path)
    assert log.done_episodes == {0, 1, 2}
    log.close()
    assert _sidecar(path) == before, "E: a no-op resume stamped a provenance record"
    print("PASS gate E: a finished arm stays a no-op on any machine")


def gate_F():
    assert parse_hosts("code a98f9ca") == []          # pre-2026-08-26 record
    assert parse_hosts("code a98f9ca host gpu-01") == ["gpu-01"]
    assert parse_hosts("code a98f9ca host") == []     # truncated write
    assert host_id(), "F: host_id() returned nothing"
    print("PASS gate F: host parsing ignores pre-host records and truncated lines")


def gate_G():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "analysis"))
    import analyze  # stdlib-only by design; this import IS part of the assertion

    with tempfile.TemporaryDirectory() as d:
        analyze.SWEEP = __import__("pathlib").Path(d)
        for suite, records in (("libero_10", ["code aaa host gpu-01"]),
                               ("libero_goal", ["code aaa host gpu-01",
                                                "code bbb host gpu-02"]),
                               ("libero_object", ["code aaa"]),  # predates the token
                               ("libero_spatial", [])):          # sidecar-less arm
            if suite == "libero_spatial":
                continue
            with open(os.path.join(d, f"n17_lang_someArm_{suite}_eplog.tsv.arm"), "w") as f:
                f.write(SIG + "\n" + "".join(r + "\n" for r in records))
        got = analyze.arm_hosts("n17_lang", "someArm")
        assert got == {"gpu-01", "gpu-02"}, got
        assert analyze.arm_hosts("n17_lang", "missingArm") == set()
    print("PASS gate G: analyze.py reads the hosts back (and stays stdlib-only)")


def main():
    prev_host = os.environ.get("PLADIS_HOST")
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate_eplog.tsv")
            gate_A(path); gate_B(path); gate_C(path); gate_D(path); gate_E(path)
        gate_F()
        gate_G()
    finally:
        os.environ.pop("PLADIS_HOST", None)
        if prev_host is not None:
            os.environ["PLADIS_HOST"] = prev_host
    print("ALL GATES PASSED (CPU; the cross-machine rule itself is SETUP.md S0)")


if __name__ == "__main__":
    main()
