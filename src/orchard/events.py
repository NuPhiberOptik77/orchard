"""Append-only JSONL event log: the durable trace of what Orchard did.

Every scheduler decision and job lifecycle transition appends one JSON
object per line to ~/.orchard/state/events.jsonl (rotated once to
events.jsonl.1 past MAX_BYTES). Logging must never break job control,
so append() swallows all errors.
"""

from __future__ import annotations

import json
import time

from .config import STATE_DIR

EVENTS_PATH = STATE_DIR / "events.jsonl"
MAX_BYTES = 5 * 1024 * 1024


def append(event: str, job: str | None = None, worker: str | None = None, **fields: object) -> None:
    record = {"ts": time.time(), "event": event}
    if job:
        record["job"] = job
    if worker:
        record["worker"] = worker
    record.update(fields)
    try:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if EVENTS_PATH.exists() and EVENTS_PATH.stat().st_size > MAX_BYTES:
            EVENTS_PATH.replace(EVENTS_PATH.with_suffix(".jsonl.1"))
        with open(EVENTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def load() -> list[dict]:
    """All readable records, oldest generation first; bad lines are skipped."""
    records: list[dict] = []
    for path in (EVENTS_PATH.with_suffix(".jsonl.1"), EVENTS_PATH):
        try:
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
