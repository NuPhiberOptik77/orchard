"""Receipts: sealed records of what a job ran on and what it produced.

At launch Orchard hashes the tree that ships into a tree digest and writes
inputs.json in the job's state folder. When the job finishes it seals
receipt.json (inputs, worker, outcome, log and out/ digests) with
HMAC-SHA256 under a per-install key, and indexes it by work key: the digest
of everything that decides the result. `orchard run --reuse` answers from a
verified, succeeded receipt for the same work key instead of running again.

The seal proves this install recorded these facts and nobody changed them
since; it is not third-party proof (whoever holds the key can mint one).
Like events, receipts must never break job control: record_inputs and seal
swallow errors and return None.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from pathlib import Path

from . import events, placement
from .config import ORCHARD_HOME, STATE_DIR
from .job import JobSpec

KEY_PATH = ORCHARD_HOME / "receipt.key"
# Evidence attached to a job at submit, and where it may run: these change with
# history or placement, not with the work.
EVIDENCE_KEYS = ("proven_os", "remote_only")
INDEX_DIR = STATE_DIR / "receipts"
VERSION = 1


def _key() -> bytes:
    try:
        return KEY_PATH.read_bytes()
    except FileNotFoundError:
        pass
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    try:
        fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return KEY_PATH.read_bytes()
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return key


def _mac(data: bytes) -> str:
    return hmac.new(_key(), data, hashlib.sha256).hexdigest()


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry_line(full: Path, arcname: str) -> bytes:
    st = full.lstat()
    if stat.S_ISLNK(st.st_mode):
        return f"L\0{arcname}\0{os.readlink(full)}\n".encode()
    if not stat.S_ISREG(st.st_mode):
        # A folder (a submodule or nested repo), a socket or a pipe: it ships
        # as nothing a digest can name, and opening a pipe would hang.
        return f"?\0{arcname}\n".encode()
    exe = "x" if st.st_mode & 0o111 else "-"
    return f"F\0{arcname}\0{exe}\0{_sha_file(full)}\n".encode()


def digest_entries(entries: list[tuple[Path, str]]) -> str:
    h = hashlib.sha256()
    seen = set()
    for full, arcname in sorted(entries, key=lambda e: e[1]):
        if arcname in seen:  # git lists a path once per stage while a merge is unresolved
            continue
        seen.add(arcname)
        try:
            h.update(_entry_line(full, arcname))
        except OSError:
            h.update(f"?\0{arcname}\n".encode())
    return h.hexdigest()


def digest_manifest(manifest: dict) -> str:
    """The digest tree_digest gives the tree a replica manifest describes:
    {arc: ["L", target, ...] | ["F", exe, sha, ...] | ["?", ...]}, "?" being
    an entry that could not be read as a file or a link."""
    h = hashlib.sha256()
    for arc in sorted(manifest):
        e = manifest[arc]
        if e[0] == "L":
            line = f"L\0{arc}\0{e[1]}\n"
        elif e[0] == "F":
            line = f"F\0{arc}\0{e[1]}\0{e[2]}\n"
        else:
            line = f"?\0{arc}\n"
        h.update(line.encode())
    return h.hexdigest()


def tree_digest(root: Path, tracked_only: bool = False) -> tuple[str, int]:
    """Digest of exactly the files that ship from root, and how many there are."""
    from .engine import iter_shipped  # engine imports this module

    entries = list(iter_shipped(root, tracked_only))
    return digest_entries(entries), len(entries)


def dir_digest(path: Path) -> str | None:
    """Digest of every file under path, or None when it is missing or empty."""
    if not path.is_dir():
        return None
    entries = []
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            full = Path(dirpath) / name
            entries.append((full, str(full.relative_to(path))))
    return digest_entries(entries) if entries else None


def work_fields(job: JobSpec) -> dict:
    """What decides a job's result, apart from the tree. Env values are folded
    into a keyed digest and never stored."""
    env_values = sorted((job.env_values or {}).items())
    return {
        "cmd": list(job.cmd),
        "kind": job.kind,
        "python": job.python,
        "requirements": hashlib.sha256((job.requirements or "").encode()).hexdigest(),
        "tags": sorted(job.tags or []),
        "requires": {k: v for k, v in (job.requires or {}).items() if k not in EVIDENCE_KEYS},
        "env": sorted(job.env or []),
        "env_mac": _mac(_canonical(env_values)) if env_values else None,
    }


def work_key(tree: str, fields: dict) -> str:
    return hashlib.sha256(_canonical({"v": VERSION, "tree": tree, **fields})).hexdigest()


def describe(root: Path, job: JobSpec) -> dict:
    tree, files = tree_digest(root, tracked_only=(job.kind == "build"))
    fields = work_fields(job)
    return {"tree": tree, "files": files, "work": fields, "work_key": work_key(tree, fields)}


def record_inputs(sdir: Path, cwd: Path, job: JobSpec, worker: str) -> dict | None:
    try:
        inputs = {"v": VERSION, "project": str(cwd), "worker": worker, "recorded_at": time.time()}
        inputs.update(describe(cwd, job))
        (sdir / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
        return inputs
    except Exception:
        return None


def sign(receipt: dict) -> dict:
    body = {k: v for k, v in receipt.items() if k != "seal"}
    return {**body, "seal": _mac(_canonical(body))}


def verify(receipt: dict) -> bool:
    seal = receipt.get("seal")
    if not isinstance(seal, str):
        return False
    body = {k: v for k, v in receipt.items() if k != "seal"}
    return hmac.compare_digest(seal, _mac(_canonical(body)))


def seal(sdir: Path, status: dict, worker: str, out_dir: Path | None) -> dict | None:
    """Seal and index the receipt for a finished job; None when it has no inputs record."""
    try:
        inputs = json.loads((sdir / "inputs.json").read_text())
        log = sdir / "log.txt"
        receipt = sign({
            "v": VERSION,
            "job": sdir.name,
            "project": inputs.get("project"),
            "tree": inputs["tree"],
            "files": inputs.get("files"),
            "work": inputs["work"],
            "work_key": inputs["work_key"],
            "worker": {
                "name": worker,
                "machine": status.get("machine"),
                "os_family": status.get("os_family"),
            },
            "state": status.get("state"),
            "exit_code": status.get("exit_code"),
            "attempt": status.get("attempt"),
            "started_at": status.get("started_at"),
            "ended_at": status.get("ended_at"),
            "elapsed_s": status.get("elapsed_s"),
            "log_sha256": _sha_file(log) if log.exists() else None,
            "out_sha256": dir_digest(out_dir) if out_dir is not None else None,
            "sealed_at": time.time(),
        })
        (sdir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        index = INDEX_DIR / receipt["work_key"]
        index.mkdir(parents=True, exist_ok=True)
        (index / f"{sdir.name}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        events.append("sealed", job=sdir.name, worker=worker, state=receipt["state"])
        return receipt
    except Exception:
        return None


def receipts_for(key: str) -> list[dict]:
    """Every readable receipt indexed under a work key, newest first."""
    found = []
    for path in (INDEX_DIR / key).glob("*.json"):
        try:
            found.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda r: r.get("sealed_at") or 0, reverse=True)


def find_reusable(key: str) -> dict | None:
    for receipt in receipts_for(key):
        if receipt.get("state") == "succeeded" and verify(receipt):
            return receipt
    return None


def all_receipts() -> list[dict]:
    found = []
    for path in INDEX_DIR.glob("*/*.json"):
        try:
            found.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return found


def proven_oses(cmd: list[str], project: Path | str) -> list[str]:
    """OSes where the latest finished run of cmd in project passed, by verified receipt."""
    latest: dict[str, dict] = {}
    for receipt in all_receipts():
        family = (receipt.get("worker") or {}).get("os_family")
        if (not family or receipt.get("project") != str(project)
                or (receipt.get("work") or {}).get("cmd") != list(cmd)
                or receipt.get("state") == "cancelled" or not verify(receipt)):
            continue
        if (receipt.get("sealed_at") or 0) >= (latest.get(family, {}).get("sealed_at") or 0):
            latest[family] = receipt
    return sorted(f for f, r in latest.items() if r.get("state") == "succeeded")


def attach_proof(job: JobSpec, cwd: Path) -> None:
    """Record, on an `os:proven` job, the OSes it has passed on. Done once at
    submit so the dispatcher's later checks read the same answer from spec.json."""
    if placement.OS_TAG_PREFIX + placement.PROVEN_OS in (job.tags or []):
        job.requires = {**(job.requires or {}), "proven_os": proven_oses(job.cmd, cwd)}


def deep_check(receipt: dict, sdir: Path, out_dir: Path | None) -> list[str]:
    """What no longer matches the receipt on disk (seal, log, out/)."""
    problems = []
    if not verify(receipt):
        problems.append("seal does not verify (receipt edited, or sealed under another key)")
    log = sdir / "log.txt"
    if receipt.get("log_sha256") and (not log.exists() or _sha_file(log) != receipt["log_sha256"]):
        problems.append("log.txt changed since sealing")
    if out_dir is not None and receipt.get("out_sha256") and dir_digest(out_dir) != receipt["out_sha256"]:
        problems.append(f"{out_dir} changed since sealing")
    return problems
