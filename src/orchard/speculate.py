"""Speculative runs: have the answer before anyone asks.

Each project keeps a short list of checks it is likely to be asked for: a
command plus the tags, env names and timeout it runs with. `run` works out each
check's work key for the tree as it is now and queues every check that has
neither a receipt nor a speculative job already on it. When the agent later
runs the same check with --reuse, the receipt answers at once.

Speculative jobs are ordinary background jobs named spec-<token>-<command>, so
they take the normal queue and a worker runs them only when it is idle. They
never run on this Mac (--no-local): a job run in place would write into the
tree it is meant to check, and the change would set off the next round. A
queued job ships the tree when it starts, so it is never stale; a running job
is stale once the tree has moved on. Ownership is the list of ids `submit`
recorded, never the name, so a user's job that happens to be called spec-
something is never cancelled. Check lists live in the Orchard state dir, never
in the project.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from . import dispatcher, events, fleet, receipts
from .config import STATE_DIR
from .job import slugify

SPEC_DIR = STATE_DIR / "speculate"
ENDED = ("done", "failed", "cancelled")
MAX_IN_FLIGHT = 3  # per project, so speculation never crowds out real work
OWN_CAP = 2000  # recorded ids kept per project; older ones only drop out of stats


def _path(project: Path) -> Path:
    return SPEC_DIR / (hashlib.sha256(str(project).encode()).hexdigest()[:16] + ".json")


def make_check(cmd: list[str], tags: list[str] | None = None, env: list[str] | None = None,
               timeout: int | None = None) -> dict:
    return {"cmd": list(cmd), "tags": sorted(dict.fromkeys(tags or [])),
            "env": sorted(dict.fromkeys(env or [])), "timeout": timeout}


def _same(a: dict, b: dict) -> bool:
    return (a["cmd"], a["tags"], a["env"]) == (b["cmd"], b["tags"], b["env"])


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def load(project: Path) -> list[dict]:
    try:
        return json.loads(_path(project).read_text()).get("checks", [])
    except (OSError, json.JSONDecodeError):
        return []


def save(project: Path, checks: list[dict]) -> None:
    _write(_path(project), {"project": str(project), "checks": checks})


def add(project: Path, check: dict) -> str:
    """Add a check, or update the timeout of the same one; says which."""
    checks = load(project)
    for i, old in enumerate(checks):
        if _same(old, check):
            checks[i] = check
            save(project, checks)
            return "updated"
    save(project, [*checks, check])
    return "added"


def remove(project: Path, cmd: list[str]) -> int:
    checks = load(project)
    kept = [c for c in checks if c["cmd"] != list(cmd)]
    save(project, kept)
    return len(checks) - len(kept)


def _jobs_path(project: Path) -> Path:
    return _path(project).with_suffix(".jobs.json")


def _recorded(project: Path) -> list[str]:
    try:
        ids = json.loads(_jobs_path(project).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [i for i in ids if isinstance(i, str)] if isinstance(ids, list) else []


def own_ids(project: Path) -> set[str]:
    """Ids of the jobs speculation queued for this project."""
    return set(_recorded(project))


def _record(project: Path, job_id: str) -> None:
    _write(_jobs_path(project), [*_recorded(project), job_id][-OWN_CAP:])


@contextlib.contextmanager
def _locked(project: Path):
    """One `run` at a time per project, so two watchers never double-queue."""
    path = _path(project).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def learn(project: Path, top: int = 3) -> list[dict]:
    """The checks run most often in this project, read from sealed receipts.
    Speculative runs do not count: they would only teach it about itself. Nor
    do runs a check cannot repeat, such as builds or job files with extra
    requirements: their work keys would never match."""
    own = own_ids(project)
    expected: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    for r in receipts.all_receipts():
        work = r.get("work") or {}
        if (r.get("project") != str(project) or r.get("job") in own
                or not work.get("cmd") or not receipts.verify(r)):
            continue
        group = json.dumps([work["cmd"], work.get("tags") or [], work.get("env") or []])
        if group not in expected:
            cmd, tags, env = json.loads(group)
            expected[group] = receipts.work_fields(fleet.job_like_run(cmd, project, tags, env))
        want = expected[group]
        if (work.get("kind"), work.get("requires") or {}) == (want["kind"], want["requires"]):
            counts[group] += 1
    found = []
    for key, runs in counts.most_common(top):
        cmd, tags, env = json.loads(key)
        found.append({**make_check(cmd, tags, env), "runs": runs})
    return found


def own_entries(project: Path) -> list[dict]:
    """Unfinished queue entries speculation started for this project."""
    own = own_ids(project)
    return [e for e in dispatcher.load_entries()
            if e["id"] in own and e.get("cwd") == str(project) and e.get("state") not in ENDED]


def _fields(entry: dict) -> dict | None:
    try:
        return receipts.work_fields(dispatcher.spec_of(entry))
    except Exception:
        return None


def _running_key(entry: dict) -> str | None:
    """The work key a running entry launched with (written at launch)."""
    try:
        return json.loads((Path(entry["sdir"]) / "inputs.json").read_text()).get("work_key")
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _failed(key: str) -> dict | None:
    """The newest verified failed receipt on a key, if no newer one passed."""
    for receipt in receipts.receipts_for(key):
        if receipt.get("state") == "failed" and receipts.verify(receipt):
            return receipt
    return None


def survey(project: Path, checks: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict]]:
    """(row, check) per check, and this project's speculative jobs no check needs."""
    entries = own_entries(project)
    tree, _files = receipts.tree_digest(project)
    rows, needed = [], set()
    for check in checks:
        job = fleet.job_like_run(check["cmd"], project, check["tags"], check["env"], check.get("timeout"))
        fields = receipts.work_fields(job)
        key = receipts.work_key(tree, fields)
        row = {"cmd": check["cmd"], "key": key, "state": "to queue", "job": None}
        hit = receipts.find_reusable(key)
        failed = None if hit else _failed(key)
        if hit:
            row.update(state="answered", job=hit["job"])
        elif failed:
            # Same tree, same work: running it again would fail again.
            row.update(state="failed", job=failed["job"])
        else:
            # A queued job will ship whatever the tree is when it starts; a
            # running one is current only if it launched with this key (unknown:
            # leave it). One is enough; the rest are duplicates.
            current = [e for e in entries if _fields(e) == fields
                       and (e.get("state") == "queued" or _running_key(e) in (key, None))]
            if current:
                needed.add(current[0]["id"])
                row.update(state="in flight", job=current[0]["id"])
        rows.append((row, check))
    return rows, [e for e in entries if e["id"] not in needed]


def submit(project: Path, check: dict) -> tuple[str | None, str | None]:
    """Queue one check in the background, never on this Mac; (job id, None) or
    (None, why)."""
    name = "-".join(["spec", fleet.new_token(), slugify(" ".join(check["cmd"]))])
    argv = fleet.child_argv("auto", name, check["cmd"], check["tags"], check["env"],
                            check.get("timeout"), background=True, no_local=True)
    proc = subprocess.run(argv, cwd=project, capture_output=True, text=True, env=fleet.child_env())
    sdir = fleet.find_job(name)
    # The folder exists before placement, so only a queue entry means queued.
    if sdir is not None and dispatcher.entry_path(sdir.name).exists():
        _record(project, sdir.name)
        return sdir.name, None
    if sdir is not None:
        shutil.rmtree(sdir, ignore_errors=True)  # ours: the name carries a fresh token
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    return None, lines[-1] if lines else f"orchard run exited {proc.returncode}"


def cancel(project: Path, job_id: str) -> bool:
    if job_id not in own_ids(project):
        return False
    proc = subprocess.run([sys.executable, "-m", "orchard", "cancel", job_id],
                          capture_output=True, text=True, env=fleet.child_env())
    return proc.returncode == 0


def run(project: Path, checks: list[dict] | None = None, limit: int = MAX_IN_FLIGHT) -> dict:
    """Bring speculation in line with the tree: cancel this project's stale
    speculative jobs, then queue every check with no receipt and no job on it,
    up to `limit` speculative jobs in flight for this project."""
    with _locked(project):
        checks = load(project) if checks is None else checks
        rows, stale = survey(project, checks)
        cancelled = [e["id"] for e in stale if cancel(project, e["id"])]
        in_flight = len([e for e in own_entries(project) if e["id"] not in cancelled])
        for row, check in rows:
            if row["state"] != "to queue":
                continue
            if in_flight >= limit:
                row.update(state="held", why=f"{limit} speculative jobs already in flight")
                continue
            job, why = submit(project, check)
            if job:
                in_flight += 1
                row.update(state="queued", job=job)
            else:
                row.update(state="not queued", why=why)
        queued = [r["job"] for r, _ in rows if r["state"] == "queued"]
        if queued or cancelled:
            events.append("speculated", queued=queued, cancelled=cancelled)
        return {"checks": [r for r, _ in rows], "cancelled": cancelled}


def fingerprint(project: Path) -> str:
    """Cheap change detector: names, modes, sizes and mtimes of the files that ship."""
    from .engine import iter_shipped  # engine imports receipts, which this imports

    h = hashlib.sha256()
    for full, arcname in sorted(iter_shipped(project), key=lambda e: e[1]):
        try:
            st = full.lstat()
            h.update(f"{arcname}\0{st.st_mode}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
        except OSError:
            h.update(f"?\0{arcname}\n".encode())
    return h.hexdigest()


def watch(project: Path, settle: float = 5.0, interval: float = 1.0, limit: int = MAX_IN_FLIGHT,
          rounds: int | None = None, sleep=time.sleep, say=print) -> None:
    """Run now, then again each time the tree changes and then holds still for
    `settle` seconds. While a run left checks held or failed with an error, try
    again after every settle period even if nothing changed. `rounds` bounds
    the polls (tests); None runs until killed."""
    steady_needed = max(1, math.ceil(settle / interval))

    def attempt() -> bool:
        """Run once; True if it should be tried again without a change."""
        try:
            report = run(project, limit=limit)
        except Exception as exc:  # a bad round must not end the watch
            say({"checks": [], "cancelled": [], "error": f"{type(exc).__name__}: {exc}"})
            return True
        say(report)
        return any(r["state"] == "held" for r in report["checks"])

    done = seen = fingerprint(project)
    retry = attempt()
    steady = polls = 0
    while rounds is None or polls < rounds:
        polls += 1
        sleep(interval)
        now = fingerprint(project)
        if now != seen:
            seen, steady = now, 0
            continue
        steady += 1
        if steady >= steady_needed and (now != done or retry):
            done, steady = now, 0
            retry = attempt()


def stats(project: Path) -> dict:
    """Hit rate: speculated checks later asked for. Waste: the rest."""
    own = own_ids(project)
    reused = Counter(e.get("job") for e in events.load() if e.get("event") == "reused")
    mine = [r for r in receipts.all_receipts()
            if r.get("job") in own and r.get("project") == str(project) and receipts.verify(r)]
    hits = [r for r in mine if reused[r["job"]]]
    misses = [r for r in mine if not reused[r["job"]]]
    return {
        "speculated": len(mine),
        "asked_for": len(hits),
        "hit_rate": round(len(hits) / len(mine), 2) if mine else None,
        "saved_s": round(sum((r.get("elapsed_s") or 0) * reused[r["job"]] for r in hits), 1),
        "wasted_s": round(sum(r.get("elapsed_s") or 0 for r in misses), 1),
    }
