from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from . import transport
from .config import ORCHARD_HOME, Worker

REMOTE_EXPORTS = (
    'export PATH="$HOME/.orchard/tools:$PATH"; '
    'export UV_PYTHON_INSTALL_DIR="$HOME/.orchard/python"; '
    'export UV_CACHE_DIR="$HOME/.orchard/cache/uv"; '
    'export UV_TOOL_DIR="$HOME/.orchard/tools";'
)

AGENT_REMOTE_PATH = "$HOME/.orchard/bin/orchard-agent.py"


def require_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv not found on PATH (brew install uv)")
    return uv


def _uv_spec_major_minor(python_spec: str) -> str:
    parts = python_spec.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    return python_spec


def platform_tag(machine: str) -> str:
    mapping = {"arm64": "aarch64-apple-darwin", "x86_64": "x86_64-apple-darwin"}
    if machine not in mapping:
        raise ValueError(f"unsupported machine architecture: {machine}")
    return mapping[machine]


def capture_python(cwd: Path) -> str:
    version_file = cwd / ".python-version"
    if version_file.exists():
        text = version_file.read_text().strip().splitlines()
        if text and text[0]:
            return text[0]
    cfg = cwd / ".venv" / "pyvenv.cfg"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "version" and value.strip():
                return _uv_spec_major_minor(value.strip())
    return "3.12"


def _filter_requirements(lines: list[str]) -> tuple[list[str], int]:
    """Drop local-path requirements that cannot cross machines.

    `name @ file://…` and `-e file://…` (or any `-e `) refer to this Air's
    absolute paths — often with spaces — and would fail a remote uv sync.
    """
    kept, dropped = [], 0
    for line in lines:
        if " @ file://" in line or line.startswith("-e "):
            dropped += 1
            continue
        kept.append(line)
    return kept, dropped


def capture_requirements(cwd: Path) -> tuple[str | None, list[str]]:
    notes: list[str] = []
    venv_python = cwd / ".venv" / "bin" / "python"
    if venv_python.exists():
        uv = require_uv()
        proc = subprocess.run(
            [uv, "pip", "freeze", "--python", str(venv_python)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            notes.append(f"uv pip freeze failed: {proc.stderr.strip()}")
            return None, notes
        kept, dropped = _filter_requirements([ln for ln in proc.stdout.splitlines() if ln.strip()])
        if dropped:
            notes.append(f"dropped {dropped} local-path requirement(s)")
        return ("\n".join(kept) + "\n" if kept else None), notes
    req_file = cwd / "requirements.txt"
    if req_file.exists():
        return req_file.read_text(), notes
    return None, notes


def resolve_for_platform(requirements: str | None, python_spec: str, platform: str) -> tuple[bool, str]:
    if not requirements or not requirements.strip():
        return True, "no requirements"
    uv = require_uv()
    proc = subprocess.run(
        [uv, "pip", "compile", "-", "--python-platform", platform,
         "--python-version", _uv_spec_major_minor(python_spec), "--quiet"],
        input=requirements, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode == 0:
        return True, "resolve ok"
    detail = proc.stderr.strip().splitlines()
    return False, (detail[-1] if detail else "resolve failed")


def ensure_bench_python() -> str | None:
    base = ORCHARD_HOME / "python"
    matches = sorted(base.glob("cpython-3.12*/bin/python3"))
    if matches:
        return str(matches[-1])
    uv = require_uv()
    env_vars = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(base))
    try:
        subprocess.run(
            [uv, "python", "install", "3.12"],
            capture_output=True, text=True, timeout=600, env=env_vars,
        )
    except subprocess.TimeoutExpired:
        return None
    matches = sorted(base.glob("cpython-3.12*/bin/python3"))
    return str(matches[-1]) if matches else None


def env_key(python_spec: str, requirements: str | None) -> str:
    h = hashlib.sha256()
    h.update(python_spec.encode())
    h.update(b"\x00")
    h.update((requirements or "").encode())
    return h.hexdigest()[:16]


def ensure_env_local(key: str, python_spec: str, requirements: str | None) -> Path:
    uv = require_uv()
    venv = ORCHARD_HOME / "envs" / key
    if not (venv / "bin" / "python").exists():
        venv.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([uv, "venv", str(venv), "--python", python_spec],
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"uv venv failed: {proc.stderr.strip()}")
    if requirements:
        req_file = venv / "requirements.txt"
        req_file.write_text(requirements)
        proc = subprocess.run([uv, "pip", "sync", str(req_file), "--python", str(venv / "bin" / "python")],
                              capture_output=True, text=True, timeout=1200)
        if proc.returncode != 0:
            raise RuntimeError(f"uv pip sync failed: {proc.stderr.strip()}")
    return venv


def ensure_uv_remote(worker: Worker) -> None:
    cmd = (
        'test -x "$HOME/.orchard/tools/uv" || '
        '(curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.orchard/tools" sh)'
    )
    transport.run_checked(worker, cmd, timeout=300)
    transport.run_checked(worker, f'{REMOTE_EXPORTS} uv --version', timeout=60)


def sync_env_remote(worker: Worker, key: str, python_spec: str, requirements: str | None, rhome: str) -> str:
    venv = f"{rhome}/.orchard/envs/{key}"
    cmd = (
        f'{REMOTE_EXPORTS} mkdir -p "{rhome}/.orchard/envs"; '
        f'test -x "{venv}/bin/python" || uv venv "{venv}" --python {python_spec}'
    )
    transport.run_checked(worker, cmd, timeout=900)
    if requirements:
        req_path = f"{venv}/requirements.txt"
        transport.push_bytes(worker, req_path, requirements.encode())
        sync_cmd = f'{REMOTE_EXPORTS} VIRTUAL_ENV="{venv}" uv pip sync "{req_path}"'
        transport.run_checked(worker, sync_cmd, timeout=1800)
    return venv


def ensure_agent_uploaded(worker: Worker) -> None:
    from . import agent as agent_module
    script = Path(agent_module.__file__).read_bytes()
    from .store import sha256_bytes
    sha = sha256_bytes(script)
    remote_sha = transport.file_sha256_remote(worker, AGENT_REMOTE_PATH)
    if remote_sha != sha:
        transport.push_bytes(worker, AGENT_REMOTE_PATH, script)
    transport.run_checked(
        worker,
        'mkdir -p "$HOME/.orchard" && touch "$HOME/.orchard/.metadata_never_index"',
        timeout=30,
    )
