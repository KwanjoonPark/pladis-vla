# SPDX-License-Identifier: Apache-2.0
"""Gate 0 on any machine: the external checkouts and the active venv match
what this repository's results were produced against.

Checks, in order:
  A. every entry of scripts/externals.lock: checkout exists, HEAD == pinned
     SHA (FAIL otherwise); tracked modifications are a WARN (untracked build
     artifacts — assets, .magick, egg-info, venvs — are expected and ignored).
  B. version pins of the ACTIVE venv's critical packages (the numerical-path
     surface): a mismatch is a FAIL because the attention hooks are
     line-for-line ports against specific library versions (README §4.3).

Run under the venv you are about to use:
    bash experiments/run.sh experiments/verify_externals.py            # gr00t
    bash experiments/run.sh --venv openpi experiments/verify_externals.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "scripts" / "externals.lock"

# numerical-path-critical pins per venv (subset of requirements; the hook code
# is a line-for-line port against these — see README §4.3 / docs/SETUP.md)
PINS = {
    "gr00t": {"torch": "2.6.0", "diffusers": "0.35.1", "entmax": "1.3",
              "robosuite": "1.4.1", "transformers": "4.57.3"},
    "openpi": {"torch": "2.6.0", "transformers": "4.53.2", "entmax": "1.3"},
}


def which_venv() -> str:
    exe = sys.executable
    for name, env in (("gr00t", "PLADIS_VENV_GR00T"), ("openpi", "PLADIS_VENV_OPENPI"),
                      ("lerobot", "PLADIS_VENV_LEROBOT")):
        root = os.environ.get(env, "")
        if root and exe.startswith(os.path.abspath(root) + os.sep):
            return name
    return "unknown"


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    ok = True

    for line in LOCK.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, dest, _url, sha = line.split("\t")
        path = Path(os.path.expandvars(dest.replace("$WS", os.environ.get("WS", ""))))
        if not (path / ".git").exists():
            print(f"[A] FAIL {name}: no checkout at {path}")
            ok = False
            continue
        head = git(["rev-parse", "HEAD"], path)
        if head != sha:
            print(f"[A] FAIL {name}: HEAD {head[:9]} != pinned {sha[:9]} ({path})")
            ok = False
            continue
        tracked_dirty = [
            l for l in git(["status", "--short"], path).splitlines()
            if l and not l.startswith("??")
        ]
        note = f"  WARN {len(tracked_dirty)} tracked modification(s)" if tracked_dirty else ""
        print(f"[A] OK   {name} @ {sha[:9]}{note}")

    venv = which_venv()
    pins = PINS.get(venv)
    if pins is None:
        print(f"[B] SKIP version pins: no pin table for venv {venv!r}")
    else:
        for pkg, want in pins.items():
            try:
                have = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                print(f"[B] FAIL {pkg}: not installed (want {want})")
                ok = False
                continue
            if have != want:
                print(f"[B] FAIL {pkg}: {have} != pinned {want}")
                ok = False
            else:
                print(f"[B] OK   {pkg} == {want}")

    print("[externals] PASS" if ok else "[externals] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


