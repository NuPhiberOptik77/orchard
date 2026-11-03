"""Machine consensus: one label for a failure, and a memory of past labels.

A failed check is rerun on the same machine and on every other capable one.
The runs are then classified (pure functions, no I/O):

- pass: the first run passed
- flaky: the same machine passed on a rerun
- real: no run passed anywhere
- load-sensitive: every failing run started on a busy machine and an idle
  one passed
- platform-only: every failing run was on a platform (architecture plus OS)
  that no passing run shared
- machine-only: it failed on one machine and passed on another of the same
  platform that was not busier

Each verdict is remembered with a failure signature (the last error-looking
log line, with numbers, paths and hex ids normalised), so a repeat failure
says how it was labelled before.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from .config import STATE_DIR

MEMORY_PATH = STATE_DIR / "consensus.jsonl"
# load average per core at the start of a run; at or above this it was busy
BUSY_LOAD = 0.75
LABELS = ("pass", "flaky", "real", "load-sensitive", "platform-only", "machine-only")

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ERRORISH = re.compile(r"error|fail|exception|assert|panic|traceback|fatal|✕|✗", re.I)
_NORMALISE = (
    (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
    (re.compile(r"\b[0-9a-f]{7,}\b"), "<hex>"),
    (re.compile(r"(?:[A-Za-z]:)?(?:[\w.@~-]*/)+[\w.@-]+"), "<path>"),
    (re.compile(r"\d+(?:\.\d+)?"), "<n>"),
    (re.compile(r"\s+"), " "),
)


def signature(log_text: str) -> str | None:
    """The normalised last error-looking line of a log, or None."""
    for line in reversed(log_text.splitlines()):
        line = _ANSI.sub("", line).strip()
        if line and _ERRORISH.search(line):
            for pattern, token in _NORMALISE:
                line = pattern.sub(token, line)
            return line.strip()[:200]
    return None


def signature_id(sig: str | None) -> str | None:
    return hashlib.sha256(sig.encode()).hexdigest()[:12] if sig else None


def passed(run: dict) -> bool:
    return run.get("state") == "succeeded"


def usable(run: dict) -> bool:
    return run.get("state") in ("succeeded", "failed")


def platform(run: dict) -> str:
    return f"{run.get('machine') or '?'}/{run.get('os_family') or '?'}"


def busy(run: dict) -> bool:
    load = run.get("load_per_core")
    return load is not None and load >= BUSY_LOAD


def classify(first: dict, reruns: list[dict]) -> tuple[str, str]:
    """(label, one-line reason) for a first run and its reruns."""
    if passed(first):
        return "pass", "the first run passed"
    reruns = [r for r in reruns if usable(r)]
    same = [r for r in reruns if r.get("target") == first.get("target")]
    if any(passed(r) for r in same):
        return "flaky", f"failed then passed on {first.get('target')}"
    runs = [first] + reruns
    good = [r for r in runs if passed(r)]
    bad = [r for r in runs if not passed(r)]
    if not good:
        where = sorted({str(r.get("target")) for r in runs})
        return "real", f"failed on every machine tried ({', '.join(where)})"
    good_targets = ", ".join(sorted({str(r.get("target")) for r in good}))
    if all(busy(r) for r in bad) and any(not busy(r) for r in good):
        return "load-sensitive", f"failed only on busy machines; passed on idle {good_targets}"
    bad_platforms = {platform(r) for r in bad}
    if not bad_platforms & {platform(r) for r in good}:
        return "platform-only", f"fails on {', '.join(sorted(bad_platforms))}; passes on {good_targets}"
    bad_targets = ", ".join(sorted({str(r.get("target")) for r in bad}))
    return "machine-only", f"fails on {bad_targets} but passes on {good_targets} (same platform)"


def exit_code(label: str) -> int:
    """0 pass, 1 real (fix the code), 2 anything that is not plainly the code."""
    return {"pass": 0, "real": 1}.get(label, 2)


def remember(verdict: dict, path: Path | None = None) -> None:
    path = path or MEMORY_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps({"at": time.time(), **verdict}) + "\n")
    except OSError:
        pass


def _memory(path: Path | None = None) -> list[dict]:
    path = path or MEMORY_PATH
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    found = []
    for line in lines:
        try:
            found.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return found


def recall(sig: str | None, path: Path | None = None) -> Counter:
    """How a failure signature was labelled before, label -> count."""
    if not sig:
        return Counter()
    return Counter(v.get("label") for v in _memory(path) if v.get("signature") == sig)


def known_noise(sig: str | None, path: Path | None = None) -> bool:
    """Was this signature labelled flaky or load-sensitive before?"""
    seen = recall(sig, path)
    return bool(seen["flaky"] or seen["load-sensitive"])


def describe_memory(seen: Counter) -> str | None:
    if not seen:
        return None
    total = sum(seen.values())
    parts = ", ".join(f"{label} {n}" for label, n in seen.most_common())
    return f"seen {total} time{'s' if total != 1 else ''} before: {parts}"


def offenders(path: Path | None = None, limit: int = 10) -> list[tuple[str, Counter]]:
    """Signatures most often labelled as noise, for fixing."""
    by_sig: dict[str, Counter] = {}
    for v in _memory(path):
        sig = v.get("signature")
        if sig and v.get("label") not in ("pass", "real"):
            by_sig.setdefault(sig, Counter())[v["label"]] += 1
    return sorted(by_sig.items(), key=lambda kv: -sum(kv[1].values()))[:limit]
