"""Many worlds: try several patches at once, one copy of the tree per patch.

Each world is a copy of the files that would ship (copy-on-write clones on
APFS, plain copies elsewhere) with one patch applied. Worlds then run through
fleet.py like any job, so each leaves a receipt. Big ignored folders the
check needs locally (node_modules, .venv) are linked, not copied; they never
ship, because those names are excluded from every tarball.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import engine
from .config import STATE_DIR

WORLDS_DIR = STATE_DIR / "worlds"
SHARED_DIRS = ("node_modules", ".venv")


def _load_clonefile():
    if sys.platform != "darwin":
        return None
    try:
        fn = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True).clonefile
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
    fn.restype = ctypes.c_int
    return fn


_CLONEFILE = _load_clonefile()


def copy_file(src: Path, dst: Path) -> None:
    """Clone src to dst where the filesystem can, else copy; links stay links."""
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
        return
    if _CLONEFILE is not None and _CLONEFILE(os.fsencode(src), os.fsencode(dst), 0) == 0:
        return
    shutil.copy2(src, dst)


def copy_tree(root: Path, dest: Path, shared: tuple[str, ...] = SHARED_DIRS) -> int:
    """Copy the shipped files of root into dest; link shared top-level dirs."""
    dest.mkdir(parents=True)
    count = 0
    for full, arcname in engine.iter_shipped(root):
        target = dest / arcname
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_file(full, target)
        count += 1
    for name in shared:
        src = root / name
        if src.is_dir() and not (dest / name).exists():
            os.symlink(src.resolve(), dest / name)
    return count


def _git_env(dest: Path) -> dict:
    # Never let git find a repo above the world (a home folder under git
    # would make `git apply` patch paths relative to that repo instead).
    return dict(os.environ, GIT_CEILING_DIRECTORIES=str(dest.parent))


def apply_patch(dest: Path, patch: Path) -> tuple[bool, str]:
    """git apply patch inside dest (all or nothing). (ok, error text)."""
    proc = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch.resolve())],
                          cwd=dest, capture_output=True, text=True, env=_git_env(dest))
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def diff_stat(patch: Path, cwd: Path) -> str:
    proc = subprocess.run(["git", "apply", "--stat", str(patch.resolve())],
                          cwd=cwd, capture_output=True, text=True, env=_git_env(cwd))
    return proc.stdout.rstrip() if proc.returncode == 0 else ""


def assign(count: int, targets: list[str]) -> list[str]:
    """Spread count worlds round-robin over targets."""
    return [targets[i % len(targets)] for i in range(count)]


def verdict(worlds: list[dict]) -> dict:
    """Pick the winner: the first passing patch in the order given."""
    base = next((w for w in worlds if w["patch"] is None), None)
    passing = [w for w in worlds if w["patch"] is not None and w.get("state") == "succeeded"]
    winner = passing[0] if passing else None
    if winner is None:
        summary = "no patch passed"
    elif base is None:
        summary = f"{winner['label']} passed on {winner.get('target')}"
    elif base.get("state") == "succeeded":
        summary = f"{winner['label']} passed, but so does the unpatched tree"
    elif base.get("state") == "failed":
        summary = f"{winner['label']} passed and the unpatched tree fails, so the patch fixes it"
    else:
        summary = f"{winner['label']} passed (the baseline did not finish)"
    return {"winner": winner["label"] if winner else None, "also_passed": [w["label"] for w in passing[1:]],
            "summary": summary}
