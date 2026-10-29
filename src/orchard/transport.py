from __future__ import annotations

import subprocess
from pathlib import Path

from .config import Worker

SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=5",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/orchard-ssh-%C",
    "-o", "ControlPersist=300",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
]


class RemoteError(Exception):
    pass


def run(
    worker: Worker,
    remote_cmd: str,
    stdin: bytes | None = None,
    timeout: float = 30,
    stdin_file: Path | None = None,
    stdout_file: Path | None = None,
) -> subprocess.CompletedProcess:
    argv = ["ssh", *SSH_OPTS, worker.target, remote_cmd]
    if stdin_file is not None or stdout_file is not None:
        fin = open(stdin_file, "rb") if stdin_file is not None else None
        fout = open(stdout_file, "wb") if stdout_file is not None else None
        try:
            return subprocess.run(
                argv,
                stdin=fin,
                stdout=fout if fout is not None else subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        finally:
            if fin is not None:
                fin.close()
            if fout is not None:
                fout.close()
    return subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)


def run_checked(
    worker: Worker,
    remote_cmd: str,
    stdin: bytes | None = None,
    timeout: float = 30,
    stdin_file: Path | None = None,
) -> str:
    proc = run(worker, remote_cmd, stdin=stdin, timeout=timeout, stdin_file=stdin_file)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RemoteError(f"{worker.name}: {remote_cmd!r} failed ({proc.returncode}): {stderr}")
    return proc.stdout.decode("utf-8", "replace")


def run_script(
    worker: Worker, script: bytes, script_args: list[str], timeout: float = 30
) -> subprocess.CompletedProcess:
    joined = " ".join(script_args)
    remote_cmd = f"/usr/bin/python3 - {joined}".strip()
    return run(worker, remote_cmd, stdin=script, timeout=timeout)


def push_bytes(worker: Worker, remote_path: str, data: bytes, timeout: float = 300) -> None:
    remote_dir = remote_path.rsplit("/", 1)[0]
    remote_cmd = f'mkdir -p "{remote_dir}" && cat > "{remote_path}"'
    run_checked(worker, remote_cmd, stdin=data, timeout=timeout)


def pull_bytes(worker: Worker, remote_path: str, timeout: float = 300) -> bytes:
    proc = run(worker, f'cat "{remote_path}"', timeout=timeout)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RemoteError(f"{worker.name}: cat {remote_path} failed: {stderr}")
    return proc.stdout


def file_sha256_remote(worker: Worker, remote_path: str) -> str | None:
    proc = run(worker, f'test -f "{remote_path}" && shasum -a 256 "{remote_path}"')
    if proc.returncode != 0:
        return None
    out = proc.stdout.decode("utf-8", "replace").strip()
    if not out:
        return None
    return out.split()[0]
