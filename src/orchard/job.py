from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class JobSpec:
    name: str
    cmd: list[str]
    python: str = "3.12"
    requirements: str | None = None
    requirements_file: str | None = None
    timeout_s: int | None = None
    requires: dict = field(default_factory=dict)
    kind: str = "run"
    tags: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    env_values: dict = field(default_factory=dict)


def load_job_file(path: Path) -> JobSpec:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("job file must contain a YAML mapping")
    cmd = data.get("cmd")
    if cmd is None:
        raise ValueError("job file must define 'cmd'")
    if isinstance(cmd, str):
        cmd = cmd.split()
    if not isinstance(cmd, list) or not all(isinstance(part, str) for part in cmd):
        raise ValueError("'cmd' must be a list of strings")
    requires = data.get("requires") or {}
    if not isinstance(requires, dict):
        raise ValueError("'requires' must be a mapping")
    tags = data.get("tags") or []
    if not isinstance(tags, list) or not all(isinstance(part, str) for part in tags):
        raise ValueError("'tags' must be a list of strings")
    env_names = data.get("env") or []
    if not isinstance(env_names, list) or not all(isinstance(part, str) for part in env_names):
        raise ValueError("'env' must be a list of environment variable names")
    return JobSpec(
        name=str(data.get("name") or "job"),
        cmd=cmd,
        python=str(data.get("python") or "3.12"),
        requirements=data.get("requirements"),
        requirements_file=data.get("requirements_file"),
        timeout_s=int(data["timeout_s"]) if data.get("timeout_s") else None,
        requires=requires,
        kind=str(data.get("kind") or "run"),
        tags=[str(part) for part in tags],
        env=[str(part) for part in env_names],
    )


def capture_env(names: list[str]) -> tuple[dict, list[str]]:
    """Read the named variables from this machine's environment.

    Returns (values, missing). Values ride along in the job spec so the
    dispatcher/worker can set them for the job; only declared names ever
    leave this machine.
    """
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value is None:
            missing.append(name)
        else:
            values[name] = value
    return values, missing


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:32] or "job"


def new_job_id(name: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{slugify(name)}-{stamp}-{secrets.token_hex(2)}"


def agent_job_payload(
    job: JobSpec, job_id: str, job_dir: str, workdir: str, venv: str | None, preempt: bool = False
) -> dict:
    return {
        "id": job_id,
        "cmd": job.cmd,
        "venv": venv,
        "workdir": workdir,
        "job_dir": job_dir,
        "timeout_s": job.timeout_s,
        "preempt": preempt,
        "env": dict(job.env_values or {}),
    }
