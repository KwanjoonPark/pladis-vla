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
  C. openpi venv only: the `transformers_replace` overwrite is intact. openpi
     copies its own modeling_*.py over site-packages/transformers (adaRMS
     timestep conditioning, which pi0.5 needs). pip cannot see that overwrite,
     so ANY later `pip install transformers` reverts it while every version pin
     in B still reads correct — the model silently becomes a different model.
     Compared by content hash against the checkout.

Run under the venv you are about to use:
    bash experiments/run.sh experiments/verify_externals.py            # gr00t
    bash experiments/run.sh --venv openpi experiments/verify_externals.py
"""

from __future__ import annotations

import hashlib
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
    # 2026-07-27: torch is 2.7.1, NOT 2.6.0 — RLinf/openpi's pyproject.toml pins
    # `torch==2.7.1` and the uv.lock resolves to it, so the old 2.6.0 entry FAILed a
    # correctly-built venv. jax/orbax are pinned because openpi imports jax even on
    # the PyTorch path and orbax 0.11.13 breaks on jax>=0.7
    # (RLinf/requirements/embodied/models/openpi.txt).
    "openpi": {"torch": "2.7.1", "transformers": "4.53.2", "entmax": "1.3",
               "jax": "0.5.3", "orbax-checkpoint": "0.11.13"},
}

# openpi's transformers_replace tree: files it copies over site-packages/transformers.
# Checked by content hash (check C) — see module docstring for why version pins cannot
# catch a reverted overwrite.
TRANSFORMERS_REPLACE_SUBDIR = "src/openpi/models_pytorch/transformers_replace"


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_transformers_replace(openpi_checkout: Path | None) -> bool:
    """Every file openpi ships in transformers_replace must be byte-identical to the
    copy living in the active venv's site-packages/transformers."""
    if openpi_checkout is None:
        print("[C] FAIL transformers_replace: no `openpi` entry in externals.lock")
        return False
    src_root = openpi_checkout / TRANSFORMERS_REPLACE_SUBDIR
    if not src_root.is_dir():
        print(f"[C] FAIL transformers_replace: not found at {src_root}")
        return False
    try:
        import transformers

        dst_root = Path(transformers.__file__).resolve().parent
    except Exception as exc:
        print(f"[C] FAIL transformers_replace: cannot import transformers ({exc})")
        return False

    ok, n = True, 0
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        dst = dst_root / src.relative_to(src_root)
        if not dst.exists():
            print(f"[C] FAIL missing in site-packages: {src.relative_to(src_root)}")
            ok = False
            continue
        if _sha256(src) != _sha256(dst):
            print(f"[C] FAIL overwrite reverted/stale: {src.relative_to(src_root)}")
            ok = False
            continue
        n += 1
    if ok:
        print(f"[C] OK   transformers_replace: {n} file(s) intact in {dst_root}")
    else:
        print("[C]      fix: cp -r "
              f"{src_root}/* {dst_root}/")
    return ok


def main() -> int:
    ok = True
    dests: dict[str, Path] = {}

    for line in LOCK.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, dest, _url, sha = line.split("\t")
        path = Path(os.path.expandvars(dest.replace("$WS", os.environ.get("WS", ""))))
        dests[name] = path
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

    if venv == "openpi":
        ok &= check_transformers_replace(dests.get("openpi"))
    else:
        print(f"[C] SKIP transformers_replace: only applies to the openpi venv (got {venv!r})")

    print("[externals] PASS" if ok else "[externals] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


