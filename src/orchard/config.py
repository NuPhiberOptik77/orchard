from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ORCHARD_HOME = Path.home() / ".orchard"
CONFIG_PATH = ORCHARD_HOME / "workers.toml"
STATE_DIR = ORCHARD_HOME / "state"
JOBS_DIR = STATE_DIR / "jobs"
QUEUE_DIR = STATE_DIR / "queue"
DEFAULT_WORKER_HOST = "pro.local"


@dataclass
class Worker:
    name: str
    host: str
    user: str | None = None
    weight: float = 0.2
    tags: list[str] = field(default_factory=list)

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


def load_workers() -> list[Worker]:
    if not CONFIG_PATH.exists():
        return []
    data = tomllib.loads(CONFIG_PATH.read_text())
    return [Worker(**entry) for entry in data.get("workers", [])]


def save_workers(workers: list[Worker]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for w in workers:
        lines.append("[[workers]]")
        lines.append(f'name = "{w.name}"')
        lines.append(f'host = "{w.host}"')
        if w.user:
            lines.append(f'user = "{w.user}"')
        lines.append(f"weight = {w.weight}")
        lines.append(f'tags = {json_list(w.tags)}')
        lines.append("")
    CONFIG_PATH.write_text("\n".join(lines))


def json_list(items: list[str]) -> str:
    inner = ", ".join(f'"{item}"' for item in items)
    return f"[{inner}]"


def init_config(worker: Worker, force: bool = False) -> Path:
    if CONFIG_PATH.exists() and not force:
        raise FileExistsError(f"{CONFIG_PATH} already exists (use --force to overwrite)")
    save_workers([worker])
    return CONFIG_PATH
