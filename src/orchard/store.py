from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from .config import ORCHARD_HOME

BLOB_DIR = ORCHARD_HOME / "cache" / "blobs"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blob_path(sha: str) -> Path:
    return BLOB_DIR / sha[:2] / sha


def has_blob(sha: str) -> bool:
    return blob_path(sha).exists()


def add_blob(data: bytes, sha: str | None = None) -> str:
    sha = sha or sha256_bytes(data)
    path = blob_path(sha)
    if path.exists():
        return sha
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha


def add_blob_file(path: Path, sha: str | None = None) -> str:
    sha = sha or sha256_file(path)
    dest = blob_path(sha)
    if dest.exists():
        return sha
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    shutil.copyfile(path, tmp)
    os.replace(tmp, dest)
    return sha
