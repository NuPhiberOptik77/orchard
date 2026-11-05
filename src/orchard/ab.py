"""A/B tests: does a setting really help, with the proof kept.

Runs one command under two or more settings ("arms") in mirrored order,
A B B A A B B A, because a machine drifts: caches warm up and other work comes
and goes, so later runs differ from earlier ones whatever the setting.

An arm counts as proven better or worse than the first arm only when their
ranges don't overlap at all and there were enough runs for that to be unlikely
by luck. That is blunt on purpose: a few more runs are cheap, a false "faster"
is not. A run that another Orchard job overlapped on a watched machine doesn't
count. The whole test lands in one record, sealed like a receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import signal
import statistics
import subprocess
import time
from pathlib import Path

from . import agent, dispatcher, receipts
from .config import JOBS_DIR, STATE_DIR
from .job import slugify

AB_DIR = STATE_DIR / "ab"
MIN_RUNS = 2
# The largest chance of two arms separating by luck alone that still counts as proven.
MAX_CHANCE = 0.10
HOOK_TIMEOUT = 300
AFTER_TRIES = 3

_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_NAME = re.compile(r"^[\w.-]+$")
# A comma starts a new setting only when KEY= follows, so a value may hold commas.
_NEXT_SETTING = re.compile(r",(?=[A-Za-z_][A-Za-z0-9_]*=)")


def parse_arm(spec: str) -> dict:
    """"fast:T=4,MODE=x" becomes a name and the environment that arm runs with."""
    name, _, rest = spec.partition(":")
    if not _NAME.match(name):
        raise ValueError(f"bad arm name in {spec!r}")
    env = {}
    if rest:
        for pair in _NEXT_SETTING.split(rest):
            key, eq, value = pair.partition("=")
            if not eq or not key:
                raise ValueError(f"arm {name}: expected KEY=VALUE, got {pair!r}")
            env[key] = value
    return {"name": name, "env": env}


def parse_arms(specs: list[str]) -> list[dict]:
    arms = [parse_arm(spec) for spec in specs]
    names = [arm["name"] for arm in arms]
    if len(names) < 2:
        raise ValueError("needs at least two --arm")
    if len(set(names)) != len(names):
        raise ValueError("arm names must differ")
    return arms


def parse_hooks(specs: list[str], arms: list[dict]) -> dict[str, str]:
    """"paused=sh pause.sh" becomes {arm: shell command}."""
    known = {arm["name"] for arm in arms}
    hooks = {}
    for spec in specs:
        name, eq, command = spec.partition("=")
        if not eq or not command or name not in known:
            raise ValueError(f"expected ARM=COMMAND for a named arm, got {spec!r}")
        hooks[name] = command
    return hooks


def run_order(names: list[str], rounds: int) -> list[str]:
    order: list[str] = []
    for i in range(rounds):
        order += names if i % 2 == 0 else names[::-1]
    return order


def last_number(pattern: str, text: str) -> float | None:
    """The last match in the output: its first group if it has one, else all of it."""
    matches = list(re.finditer(pattern, text, re.M))
    if not matches:
        return None
    match = matches[-1]
    try:
        value = float(match.group(1) if match.re.groups else match.group(0))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def same_digest(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.M)
    if not match:
        return None
    found = match.group(1) if match.re.groups else match.group(0)
    return hashlib.sha256((found or "").encode()).hexdigest()


def luck(n: int, m: int) -> float:
    """The chance n and m runs separate completely if the setting did nothing."""
    return 2 / math.comb(n + m, n)


def compare(base: list[float], other: list[float], lower_better: bool) -> dict:
    base_median, other_median = statistics.median(base), statistics.median(other)
    below, above = max(other) < min(base), min(other) > max(base)
    chance = luck(len(base), len(other))
    if not (below or above):
        verdict = "not separated"
    elif chance > MAX_CHANCE:
        verdict = "too few runs"
    else:
        verdict = "better" if below == lower_better else "worse"
    change = (other_median - base_median) / base_median * 100 if base_median else None
    return {"verdict": verdict, "median": other_median, "base_median": base_median,
            "change_pct": change, "chance": chance}


def judge(arms: list[dict], runs: list[dict], lower_better: bool) -> tuple[str, str, list[dict]]:
    values = {arm["name"]: [r["value"] for r in runs if r["arm"] == arm["name"] and r["counted"]]
              for arm in arms}
    short = [name for name, got in values.items() if len(got) < MIN_RUNS]
    if short:
        return "void", f"too few runs that count: {', '.join(short)}", []
    base = arms[0]["name"]
    comparisons = [{"arm": arm["name"], "baseline": base, **compare(values[base], values[arm["name"]], lower_better)}
                   for arm in arms[1:]]
    if any(c["verdict"] in ("better", "worse") for c in comparisons):
        return "proven", "", comparisons
    return "not separated", "", comparisons


def exit_code(state: str) -> int:
    return {"proven": 0, "not separated": 1}.get(state, 2)


def overlapping_jobs(workers: list[str], start: float, end: float) -> list[str]:
    """Orchard jobs on these workers that ran at any point between start and end."""
    hits = {entry["id"] for entry in dispatcher.load_entries()
            if entry.get("worker") in workers and entry.get("state") in ("running", "paused")}
    if JOBS_DIR.exists():
        for job in JOBS_DIR.iterdir():
            try:
                placed = json.loads((job / "placement.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if placed.get("worker") not in workers:
                continue
            status = agent.read_status(str(job)) or {}
            began, ended = status.get("started_at"), status.get("ended_at")
            if began and ended and began < end and ended > start:
                hits.add(job.name)
    return sorted(hits)


def _kill(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _hook(command: str, env: dict, log_path: Path, timeout: int = HOOK_TIMEOUT) -> int:
    # Its own session, so a Ctrl-C at the terminal can't kill the hook that puts things back.
    with log_path.open("ab") as log:
        log.write(f"$ {command}\n".encode())
        log.flush()
        try:
            return subprocess.run(["/bin/sh", "-c", command], stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, env=env, timeout=timeout,
                                  start_new_session=True).returncode
        except subprocess.TimeoutExpired:
            return 124


def _after(command: str, env: dict, log_path: Path, name: str, failures: list[str]) -> None:
    """Puts back what the before hook changed, so a signal waits until that is done."""
    pending = []
    saved = {sig: signal.signal(sig, lambda *_: pending.append(1)) for sig in _SIGNALS}
    try:
        for attempt in range(AFTER_TRIES):
            if attempt:
                time.sleep(2)
            if _hook(command, env, log_path) == 0:
                break
        else:
            failures.append(f"after hook failed for {name}: what it restores may NOT be back (see {log_path})")
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    if pending:
        raise KeyboardInterrupt


def _run_command(cmd: list[str], env: dict, log_path: Path, timeout: int | None,
                 stop_if: str | None, poll: int, hook_log: Path) -> tuple[str, int | None, float]:
    began = time.monotonic()
    next_check = began + poll
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    try:
        while True:
            try:
                code = proc.wait(timeout=1)
                return ("ok" if code == 0 else "failed"), code, time.monotonic() - began
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if timeout and now - began > timeout:
                _kill(proc)
                return "timeout", None, now - began
            if stop_if and now >= next_check:
                next_check = now + poll
                if _hook(stop_if, env, hook_log, timeout=60) == 0:
                    _kill(proc)
                    return "stopped", None, now - began
    except BaseException:
        _kill(proc)
        raise


def _save(record: dict) -> None:
    path = AB_DIR / f"{record['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)


def _one_run(cmd: list[str], arm: dict, index: int, folder: Path, reference: list[str], opts: dict) -> tuple[dict, str | None]:
    """One run with its hooks. Returns the run and, if the test must stop, why."""
    name = arm["name"]
    env = {**os.environ, **arm["env"], "ORCHARD_AB_ARM": name, "ORCHARD_AB_RUN": str(index)}
    log_path = folder / f"{index:02d}-{slugify(name)}.log"
    hook_log = folder / "hooks.log"
    run: dict = {"arm": name, "n": index, "state": "not started", "counted": False, "log": log_path.name}
    before, after = opts["before"].get(name), opts["after"].get(name)
    try:
        if before and _hook(before, env, hook_log) != 0:
            return run, f"before hook failed for {name} (see {hook_log})"
        run["started_at"] = time.time()
        run["state"], run["exit_code"], run["seconds"] = _run_command(
            cmd, env, log_path, opts["timeout"], opts["stop_if"], opts["poll"], hook_log)
        run["ended_at"] = time.time()
    finally:
        if after:
            _after(after, env, hook_log, name, opts["failures"])
    data = log_path.read_bytes()
    text = data.decode(errors="replace")
    run["log_sha256"] = hashlib.sha256(data).hexdigest()
    if run["state"] != "ok":
        return run, f"run {index} ({name}): {run['state']} (see {log_path})"
    value = run["seconds"] if opts["metric"] is None else last_number(opts["metric"], text)
    if value is None:
        return run, f"run {index} ({name}): --metric found no number (see {log_path})"
    run["value"] = value
    if opts["same"]:
        digest = same_digest(opts["same"], text)
        if digest is None:
            return run, f"run {index} ({name}): --same matched nothing (see {log_path})"
        run["same_sha256"] = digest
        if not reference:
            reference.append(digest)
        elif digest != reference[0]:
            return run, f"run {index} ({name}): output changed (see {log_path})"
    overlapped = overlapping_jobs(opts["watch"], run["started_at"], run["ended_at"]) if opts["watch"] else []
    run["overlapped"] = overlapped
    run["counted"] = not overlapped
    return run, None


def run_test(cmd: list[str], arms: list[dict], *, rounds: int = 4, metric: str | None = None,
             same: str | None = None, lower_better: bool = True, before: dict | None = None,
             after: dict | None = None, watch: list[str] | None = None, stop_if: str | None = None,
             poll: int = 30, timeout: int | None = None, say=print) -> dict:
    test_id = f"ab-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    folder = AB_DIR / test_id
    folder.mkdir(parents=True)
    order = run_order([arm["name"] for arm in arms], rounds)
    by_name = {arm["name"]: arm for arm in arms}
    opts = {"metric": metric, "same": same, "before": before or {}, "after": after or {},
            "watch": watch or [], "stop_if": stop_if, "poll": poll, "timeout": timeout, "failures": []}
    record: dict = {"v": 1, "id": test_id, "cmd": cmd, "cwd": os.getcwd(), "arms": arms, "order": order,
                    "metric": metric or "wall seconds", "lower_better": lower_better, "same": same,
                    "before": opts["before"], "after": opts["after"], "watch": opts["watch"],
                    "state": "running", "reason": "", "runs": [], "comparisons": [],
                    "started_at": time.time(), "ended_at": None}
    _save(record)
    reference: list[str] = []
    problem = None

    def interrupt(*_):
        raise KeyboardInterrupt

    saved = {sig: signal.signal(sig, interrupt) for sig in _SIGNALS}
    try:
        for index, name in enumerate(order, 1):
            say(f"[{index}/{len(order)}] {name}")
            run, problem = _one_run(cmd, by_name[name], index, folder, reference, opts)
            record["runs"].append(run)
            _save(record)
            if run.get("value") is not None:
                note = f"  DOES NOT COUNT: {', '.join(run['overlapped'])} overlapped" if run.get("overlapped") else ""
                say(f"        {run['value']:g}{note}")
            if problem or opts["failures"]:
                break
    except KeyboardInterrupt:
        problem = "interrupted"
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)

    record["ended_at"] = time.time()
    problem = "; ".join(opts["failures"] + ([problem] if problem else []))
    if problem:
        record["state"], record["reason"] = "void", problem
    else:
        record["state"], record["reason"], record["comparisons"] = judge(arms, record["runs"], lower_better)
    sealed = receipts.sign(record)
    _save(sealed)
    return sealed


def report(record: dict) -> list[str]:
    lines = [f"{record['id']}: {record['state']}" + (f" ({record['reason']})" if record.get("reason") else "")]
    comparisons = {c["arm"]: c for c in record.get("comparisons") or []}
    width = max(len(arm["name"]) for arm in record["arms"])
    direction = "lower is better" if record.get("lower_better") else "higher is better"
    lines.append(f"  metric: {record['metric']} ({direction})")
    for i, arm in enumerate(record["arms"]):
        runs = [r for r in record["runs"] if r["arm"] == arm["name"]]
        got = [r["value"] for r in runs if r["counted"]]
        if got:
            spread = f"median {statistics.median(got):.4g}, {min(got):.4g} to {max(got):.4g}"
        else:
            spread = "no runs that count"
        line = f"  {arm['name']:<{width}}  {spread}, {len(got)} of {len(runs)} runs count"
        if i == 0:
            line += "  [baseline]"
        comparison = comparisons.get(arm["name"])
        if comparison:
            change = comparison["change_pct"]
            line += f"  {comparison['verdict']}"
            if change is not None:
                line += f", {change:+.1f}%"
            line += f", luck 1 in {round(1 / comparison['chance'])}"
        lines.append(line)
    lines.append(f"  proof: {AB_DIR / (record['id'] + '.json')}")
    return lines


def find(prefix: str) -> dict | None:
    matches = sorted(AB_DIR.glob(f"{prefix}*.json")) if AB_DIR.exists() else []
    if not matches:
        return None
    return json.loads(matches[-1].read_text())


def check(record: dict) -> list[str]:
    """Why this record can't be trusted: an edited seal, or a log that no longer matches."""
    problems = [] if receipts.verify(record) else ["seal does not verify (record edited, or not finished)"]
    folder = AB_DIR / record["id"]
    for run in record.get("runs") or []:
        want = run.get("log_sha256")
        if not want:
            continue
        try:
            got = hashlib.sha256((folder / run["log"]).read_bytes()).hexdigest()
        except OSError:
            problems.append(f"log missing: {run['log']}")
            continue
        if got != want:
            problems.append(f"log changed: {run['log']}")
    return problems


def recent(limit: int = 20) -> list[dict]:
    if not AB_DIR.exists():
        return []
    records = []
    for path in sorted(AB_DIR.glob("ab-*.json"), reverse=True)[:limit]:
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return records
