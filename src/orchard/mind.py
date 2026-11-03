"""One mind: a goal planned from shared memory, run across every body.

`orchard mind run -- <cmd>` turns one or more checks on one tree into a goal.
Memory is a set of signed records (goal, fit, launch, result, lost, retry,
verdict, refute) kept in immutable segments under ~/.orchard/mind/segments/.
Any session, on any install, that holds the same records computes the same
plan and the same digest: the planner is a pure function of the records.

Each record is canonical JSON with an HMAC under the install's receipt key, so
this install can tell its own records (verified) from tampered ones (forged,
refuted at ingest) and from another install's (unverifiable: an HMAC cannot be
checked without the key). Trust is part of the goal itself; records under an
untrusted key are shown but never complete, close, launch or move a step.

Nothing here names a machine, path or person: records carry body ids (random
hex minted per body), digests and ints. Local-only files (via.jsonl,
bodies.json, claims) map those ids back to targets for display.

Importing this module has no side effects: no file, directory or key is
created until a function that needs one runs.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from . import (
    agent,
    config,
    consensus,
    dispatcher,
    engine,
    env,
    events,
    fleet,
    placement,
    receipts,
    speculate,
    transport,
)
from .config import Worker
from .job import JobSpec

# ---------------------------------------------------------------- layout

MIND_DIR = config.ORCHARD_HOME / "mind"
SEG_DIR = MIND_DIR / "segments"        # <sha256>.jsonl, write-once, replicated
VIA_PATH = MIND_DIR / "via.jsonl"      # provenance, one line per stored segment arrival
BODIES_PATH = MIND_DIR / "bodies.json"  # body id -> local target ("local" or a worker name)
CLAIM_DIR = MIND_DIR / "claims"        # <step>.<gen>[.h<k>|.run], write-once
SESSION_DIR = MIND_DIR / "sessions"    # <id>.json
ZOMBIES_PATH = MIND_DIR / "zombies.jsonl"
LOG_DIR = MIND_DIR / "logs"
BODY_ID_PATH = config.ORCHARD_HOME / "body.id"

# ---------------------------------------------------------------- constants

VERSION = 1
KINDS = ("goal", "fit", "launch", "result", "lost", "retry", "verdict", "refute")
FAIL_STATES = ("failed", "timed_out", "killed")
NOISE_LABELS = ("flaky", "load-sensitive")
STATUSES = ("verified", "forged", "unverifiable")
SOURCES = ("self", "sync", "import", "found")

SENSE_EVERY_S = 15
ORPHAN_CONFIRM = 2
FIT_EVERY_S = 300
SEAL_GRACE_S = 120
MAX_GEN = 6
DEFAULT_MS = 60000
DEFAULT_TIMEOUT_S = 3600
SEGMENT_MAX_BYTES = 1 << 20

# Field types the planner relies on, per kind. "str?" and "int?" also allow
# None; "strs" is a list of str. Extra fields are allowed and ignored. Ints
# exclude bool, and every number the planner reads is an int.
REQUIRED = {
    "goal": {"goal_id": "str", "author": "str", "project": "str", "tree": "str", "trust": "strs", "steps": "steps"},
    "fit": {"goal_id": "str", "body_id": "str", "caps": "caps", "fits": "strs", "problems": "problems"},
    "launch": {"goal_id": "str", "step_id": "str", "gen": "int", "body_id": "str", "token": "str", "at": "int"},
    "result": {"step_id": "str", "gen": "int", "body_id": "str?", "state": "str", "exit_code": "int?",
               "elapsed_ms": "int", "bench": "int", "work_key": "str", "tree": "str", "cmd_key": "str",
               "log_sha256": "str?", "out_sha256": "str?", "receipt_sha": "str", "os_family": "str?",
               "signature_id": "str?", "at": "int"},
    "lost": {"step_id": "str", "gen": "int", "body_id": "str", "why": "str", "hold": "bool", "at": "int"},
    "retry": {"step_id": "str", "from_gen": "int", "at": "int"},
    "verdict": {"signature_id": "str", "label": "str"},
    "refute": {"target": "str", "why": "str"},
}
STEP_FIELDS = {"step_id": "str", "work_key": "str", "cmd_key": "str", "cmd": "strs", "kind": "str",
               "tags": "strs", "requires": "dict", "env_names": "strs", "timeout": "int"}
NO_BODIES = "no body has reported for this goal yet"


class MindError(Exception):
    """A mind operation that cannot go ahead; the message is for the user."""


# ---------------------------------------------------------------- pure: records and ids
# Everything from here to PURE is the planner and its helpers: Python 3.9
# syntax, json and hashlib only, no clock, files, environment or randomness.


def canonical(obj: object) -> bytes:
    """The one byte form of a JSON value: sorted keys, no spaces, ASCII. Raises
    on anything that is not JSON (paths, sets, objects, NaN)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def record_id(rec: dict) -> str:
    return hashlib.sha256(canonical(rec)).hexdigest()


def _is_hex(value: object, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and all(c in "0123456789abcdef" for c in value))


def _typed(value: object, tag: str) -> bool:
    if tag.endswith("?"):
        return value is None or _typed(value, tag[:-1])
    if tag == "str":
        return isinstance(value, str)
    if tag == "int":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if tag == "bool":
        return isinstance(value, bool)
    if tag == "dict":
        return isinstance(value, dict)
    if tag == "strs":
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if tag == "caps":
        return isinstance(value, dict) and _typed(value.get("bench"), "int")
    if tag == "problems":
        return isinstance(value, dict) and all(_typed(v, "strs") for v in value.values())
    if tag == "steps":
        return isinstance(value, list) and all(
            isinstance(s, dict) and all(_typed(s.get(f), t) for f, t in STEP_FIELDS.items()) for s in value)
    return False


def well_formed(rec: object) -> bool:
    """Whether rec has the envelope and the body fields the planner reads, with
    their types. Anything else is dropped before planning."""
    if not isinstance(rec, dict) or set(rec) != {"v", "kind", "key", "body", "mac"}:
        return False
    # Exactly the int 1: True and 1.0 compare equal to it but are other bytes,
    # so they would make a second record with the same meaning.
    if type(rec["v"]) is not int or rec["v"] != VERSION:
        return False
    if not isinstance(rec["kind"], str) or rec["kind"] not in REQUIRED:
        return False
    if not _is_hex(rec["key"], 16) or not _is_hex(rec["mac"], 64):
        return False
    body = rec["body"]
    if not isinstance(body, dict):
        return False
    return all(_typed(body.get(f), t) for f, t in REQUIRED[rec["kind"]].items())


def step_id_of(tree: str, work: dict) -> str:
    """Step id: the tree plus receipts.work_fields without env_mac. env_mac is
    keyed per install, so dropping it gives every HOME the same id."""
    fields = {k: v for k, v in work.items() if k != "env_mac"}
    return hashlib.sha256(canonical({"v": VERSION, "tree": tree, **fields})).hexdigest()


def cmd_key_of(step: dict) -> str:
    """Cost-history key: the command, its kind and tags, but not the tree."""
    key = {"cmd": list(step["cmd"]), "kind": step["kind"], "tags": sorted(step.get("tags") or [])}
    return hashlib.sha256(canonical(key)).hexdigest()[:16]


def goal_id_of(body: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in body.items() if k != "goal_id"})).hexdigest()


def lower_median(values: list) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


# ---------------------------------------------------------------- pure: planner


def refuted(recs: dict) -> set:
    """Ids of records knocked out by a refute. X is out when a refute R that is
    not itself out targets id(X) under X's own key: only the key holder can
    refute. Iterative DFS with memo, so a long refute chain cannot overflow
    the stack. Real chains are acyclic (a refute hashes over its target's id);
    an impossible cycle edge counts as not out."""
    by_target = {}
    for rid in sorted(recs):
        r = recs[rid]
        if r["kind"] != "refute":
            continue
        target = r["body"]["target"]
        if target in recs and recs[target]["key"] == r["key"]:
            by_target.setdefault(target, []).append(rid)
    memo = {}
    for start in sorted(by_target):
        if start in memo:
            continue
        stack, on_stack = [start], {start}
        while stack:
            x = stack[-1]
            nxt = None
            for r in by_target.get(x, ()):
                if r not in memo and r not in on_stack:
                    nxt = r
                    break
            if nxt is not None:
                stack.append(nxt)
                on_stack.add(nxt)
                continue
            memo[x] = any(not memo.get(r, False) for r in by_target.get(x, ()))
            stack.pop()
            on_stack.discard(x)
    return {x for x, out in memo.items() if out}


def _step_state(step: dict, live: list, trust: set, fits: dict, problems: dict, noise: set, author: str) -> dict:
    """One step's state from the live records about it (sorted by id). Only
    records under a trusted key count. Returns the D fields plus duplicates,
    the ids consulted and the signatures of the results seen. Only the
    author's launches keep a step running: another trusted key's launch moves
    the next gen on, but its runner is not watched from here."""
    sid = step["step_id"]
    mine = [(rid, r) for rid, r in live
            if r["key"] in trust and r["kind"] in ("result", "launch", "lost", "retry")
            and r["body"]["step_id"] == sid]
    retries = [(rid, r) for rid, r in mine if r["kind"] == "retry"]
    base = max([r["body"]["from_gen"] for _, r in retries], default=0)
    ended = [(rid, r) for rid, r in mine if r["kind"] == "result" and r["body"]["gen"] >= base]
    # An env step passes or fails only under its own work key (env_mac is per
    # install), but any result ends the attempt at its gen.
    results = [(rid, r) for rid, r in ended
               if not step["env_names"] or r["body"]["work_key"] == step["work_key"]]
    used = [rid for rid, _ in retries] + [rid for rid, _ in ended]
    state = {"step": sid, "state": "pending", "gen": 0, "body": None, "result": None, "capable": [],
             "excluded": [], "why": [], "duplicates": [], "inputs": used, "cmd_key": step["cmd_key"],
             "signatures": sorted({r["body"]["signature_id"] for _, r in results
                                   if r["body"]["signature_id"] is not None})}

    passed = [(rid, r) for rid, r in results if r["body"]["state"] == "succeeded"]
    if passed:
        rid, r = passed[0]
        state.update(state="passed", gen=r["body"]["gen"], body=r["body"]["body_id"], result=rid)
        return state

    launches = {}
    losts = []
    for rid, r in mine:
        if r["kind"] == "launch" and r["body"]["gen"] >= base:
            launches.setdefault(r["body"]["gen"], []).append((rid, r))
            used.append(rid)
        elif r["kind"] == "lost" and r["body"]["gen"] >= base:
            losts.append((rid, r))
            used.append(rid)
    state["duplicates"] = sorted(g for g, ls in launches.items() if len(ls) > 1)
    lostgens = {r["body"]["gen"] for _, r in losts}
    closed = lostgens | {r["body"]["gen"] for _, r in ended}

    # A kill at a lost gen is mind's own terminate, not the step failing.
    fails = [(rid, r) for rid, r in results
             if r["body"]["state"] in FAIL_STATES and r["body"]["gen"] not in lostgens]
    if fails:
        f_id, f = sorted(fails, key=lambda x: (-x[1]["body"]["gen"], x[0]))[0]
        if not (len(fails) == 1 and f["body"]["signature_id"] in noise):
            state.update(state="failed", gen=f["body"]["gen"], body=f["body"]["body_id"], result=f_id)
            return state

    own = {g: [x for x in ls if x[1]["key"] == author] for g, ls in launches.items()}
    open_gens = [g for g in own if own[g] and g not in closed]
    if open_gens:
        g = max(open_gens)
        state.update(state="running", gen=g, body=own[g][0][1]["body"]["body_id"])
        return state

    gen = max(base, 1 + max(closed | set(launches), default=0))
    state["gen"] = gen
    if gen - max(base, 1) >= MAX_GEN:
        state.update(state="stranded", why=["gave up after %d attempts" % MAX_GEN])
        return state
    hard = {r["body"]["body_id"] for _, r in losts if not r["body"]["hold"]}
    held = {r["body"]["body_id"] for _, r in losts if r["body"]["hold"]}
    allowed = sorted(b for b in fits if sid in fits[b] and b not in hard)
    # A held body (the owner came back) returns when nothing else fits.
    capable = [b for b in allowed if b not in held] or allowed
    state["capable"] = capable
    state["excluded"] = sorted(b for b in fits if b not in capable)
    if not capable:
        why = {p for b in sorted(problems) for p in problems[b].get(sid, ())}
        why |= {r["body"]["why"] for _, r in losts}
        state.update(state="stranded", why=sorted(why) or ([] if fits else [NO_BODIES]))
    return state


def _assign(pending: list, running: list, fits: dict, bench: dict, history: dict) -> tuple:
    """LPT, integers only. work(s) is the lower median of elapsed_ms * bench over
    past passes of the same command, or DEFAULT_MS times the median body bench.
    pred(s, b) = work // bench[b] (bench is higher-is-faster). Longest steps
    are placed first, each on the body that would finish it soonest. Returns
    (queues, costs): queues lists a body's running steps first."""
    default = DEFAULT_MS * (lower_median([bench.get(b, 1) for b in sorted(fits)]) or 1)

    def work(st: dict) -> int:
        known = lower_median(history.get(st["cmd_key"], []))
        return default if known is None else known

    queues, load, costs, preds = {}, {}, {}, {}
    for st in sorted(running, key=lambda s: s["step"]):
        b, w = st["body"], work(st)
        p = w // max(1, bench.get(b, 1))
        load[b] = load.get(b, 0) + p
        queues.setdefault(b, []).append(st["step"])
        costs[st["step"]] = {"work": w, "pred": {b: p}}
    for st in pending:
        w = work(st)
        preds[st["step"]] = {b: w // max(1, bench.get(b, 1)) for b in st["capable"]}
        costs[st["step"]] = {"work": w, "pred": preds[st["step"]]}
    order = sorted(pending, key=lambda s: (-min(preds[s["step"]].values()), s["step"]))
    for st in order:
        p = preds[st["step"]]
        best = min(st["capable"], key=lambda c: (load.get(c, 0) + p[c], -bench.get(c, 1), c))
        queues.setdefault(best, []).append(st["step"])
        load[best] = load.get(best, 0) + p[best]
    return queues, costs


def _effective_refutes(recs: dict, out: set, touches) -> set:
    """Live refutes that knocked out a record the planner would have consulted.
    A refute of a refute counts when the chain's root record is consulted."""
    found = set()
    for rid in sorted(recs):
        r = recs[rid]
        if r["kind"] != "refute" or rid in out:
            continue
        target = r["body"]["target"]
        if target not in out or recs[target]["key"] != r["key"]:
            continue
        root, hops = target, 0
        while recs[root]["kind"] == "refute" and recs[root]["body"]["target"] in recs and hops < len(recs):
            root, hops = recs[root]["body"]["target"], hops + 1
        if touches(recs[root]):
            found.add(rid)
    return found


def _sha(obj: object) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def plan(records: Iterable, goal_id: str) -> dict:
    """The plan for one goal from a set of records: D (section 3 of the design)
    plus digest, inputs (record ids consulted), inputs_digest, and for display
    costs, author and trust. Input order and unrelated records never change
    digest or inputs_digest."""
    recs = {}
    for r in records:
        if well_formed(r):
            recs[record_id(r)] = r
    out = refuted(recs)
    live = [(rid, recs[rid]) for rid in sorted(recs) if rid not in out]

    # A goal names its author, so the same goal made on two installs has two
    # ids, and a record under one key claiming another's goal is ignored.
    goals = [(rid, r) for rid, r in live if r["kind"] == "goal" and r["body"]["goal_id"] == goal_id
             and goal_id_of(r["body"]) == goal_id and r["body"]["author"] == r["key"]]
    if not goals:
        d = {"v": VERSION, "goal": goal_id, "state": "unknown", "steps": [], "queues": {}, "duplicates": []}
        inputs = sorted(_effective_refutes(
            recs, out, lambda x: x["kind"] == "goal" and x["body"]["goal_id"] == goal_id))
        return dict(d, digest=_sha(d), inputs=inputs, inputs_digest=_sha(inputs), costs={},
                    author=None, trust=[])

    goal_rid, goal = goals[0]
    author = goal["key"]
    trust = set(goal["body"]["trust"]) | {author}
    steps = {}
    for s in sorted(goal["body"]["steps"], key=lambda s: s["step_id"]):
        steps.setdefault(s["step_id"], s)
    cmd_keys = {s["cmd_key"] for s in steps.values()}
    used = {goal_rid}

    fits, caps, problems = {}, {}, {}
    for rid, r in live:
        if r["kind"] != "fit" or r["key"] != author or r["body"]["goal_id"] != goal_id:
            continue
        b = r["body"]["body_id"]
        fits[b] = set(r["body"]["fits"]) if b not in fits else fits[b] & set(r["body"]["fits"])
        caps.setdefault(b, r["body"]["caps"])
        for sid, strs in r["body"]["problems"].items():
            problems.setdefault(b, {}).setdefault(sid, set()).update(strs)
        used.add(rid)
    bench = {b: max(1, caps[b]["bench"]) for b in caps}
    verdicts = [(rid, r) for rid, r in live
                if r["kind"] == "verdict" and r["key"] in trust and r["body"]["label"] in NOISE_LABELS]
    noise = {r["body"]["signature_id"] for _, r in verdicts}

    about = {}
    for rid, r in live:
        sid = r["body"].get("step_id")
        if sid in steps and r["kind"] in ("result", "launch", "lost", "retry"):
            about.setdefault(sid, []).append((rid, r))
    states = [_step_state(steps[sid], about.get(sid, []), trust, fits, problems, noise, author)
              for sid in sorted(steps)]

    history = {}
    for rid, r in live:
        if (r["kind"] == "result" and r["key"] in trust and r["body"]["state"] == "succeeded"
                and r["body"]["cmd_key"] in cmd_keys):
            history.setdefault(r["body"]["cmd_key"], []).append(r["body"]["elapsed_ms"] * max(1, r["body"]["bench"]))
            used.add(rid)
    pending = [st for st in states if st["state"] == "pending"]
    running = [st for st in states if st["state"] == "running"]
    queues, costs = _assign(pending, running, fits, bench, history)

    signatures = set()
    for st in states:
        used.update(st["inputs"])
        signatures.update(st["signatures"])
    used.update(rid for rid, r in verdicts if r["body"]["signature_id"] in signatures)

    def touches(x: dict) -> bool:
        kind, body = x["kind"], x["body"]
        if kind == "goal":
            return body["goal_id"] == goal_id
        if kind == "fit":
            return x["key"] == author and body["goal_id"] == goal_id
        if x["key"] not in trust:
            return False
        if kind in ("result", "launch", "lost", "retry") and body["step_id"] in steps:
            return True
        if kind == "result":
            return body["state"] == "succeeded" and body["cmd_key"] in cmd_keys
        return kind == "verdict" and body["label"] in NOISE_LABELS and body["signature_id"] in signatures

    used |= _effective_refutes(recs, out, touches)

    stepped = {st["state"] for st in states}
    if stepped & {"running", "pending"}:
        overall = "running"
    elif "stranded" in stepped:
        overall = "stranded"
    elif "failed" in stepped:
        overall = "failed"
    else:
        overall = "passed"
    keep = ("step", "state", "gen", "body", "result", "capable", "excluded", "why")
    d = {"v": VERSION, "goal": goal_id, "state": overall,
         "steps": [{k: st[k] for k in keep} for st in states],
         "queues": queues,
         "duplicates": sorted([st["step"], g] for st in states for g in st["duplicates"])}
    inputs = sorted(used)
    return dict(d, digest=_sha(d), inputs=inputs, inputs_digest=_sha(inputs), costs=costs,
                author=author, trust=sorted(trust))


# The planner: pure functions only (Python 3.9, json and hashlib). Tests check
# that none of them reaches for a clock, a file, the environment or orchard.
PURE = (canonical, record_id, _is_hex, _typed, well_formed, step_id_of, cmd_key_of, goal_id_of, lower_median,
        refuted, _step_state, _assign, _effective_refutes, _sha, plan)


# ---------------------------------------------------------------- record envelope


def key_id() -> str:
    """This install's public key id: names the receipt key without revealing it."""
    return hashlib.sha256(b"orchard-mind-key\0" + receipts._key()).hexdigest()[:16]


def _mac_of(kind: str, key: str, body: dict) -> str:
    # "mind": 1 keeps this domain apart from receipt seals under the same key.
    return receipts._mac(canonical({"mind": VERSION, "kind": kind, "key": key, "body": body}))


def make_record(kind: str, body: dict) -> dict:
    kid = key_id()
    try:
        rec = {"v": VERSION, "kind": kind, "key": kid, "body": body, "mac": _mac_of(kind, kid, body)}
    except (TypeError, ValueError) as e:
        raise MindError(f"mind: a {kind} record is not plain JSON ({e})") from e
    if not well_formed(rec):
        raise MindError(f"mind: a {kind} record is missing a field or has one of the wrong type")
    return rec


def _status(rec: dict, kid: str) -> str:
    if rec.get("key") != kid:
        return "unverifiable"
    try:
        want = _mac_of(rec["kind"], rec["key"], rec["body"])
    except (KeyError, TypeError, ValueError):
        return "forged"
    mac = rec.get("mac")
    return "verified" if isinstance(mac, str) and hmac.compare_digest(want, mac) else "forged"


def verify_record(rec: dict) -> str:
    """verified (this install's key, mac checks), forged (this install's key,
    mac fails) or unverifiable (another install's key)."""
    return _status(rec, key_id())


def project_id(cwd: Path, tree: str) -> str:
    """A project id with no path in it: the hash of the repo's root commit, or
    of the tree when there is no git history."""
    try:
        out = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, timeout=30)
        lines = out.stdout.split() if out.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        lines = []
    if lines:
        return "git:" + hashlib.sha256(lines[0].encode()).hexdigest()[:16]
    return "tree:" + tree[:16]


# ---------------------------------------------------------------- store


def _link_once(path: Path, data: bytes) -> bool:
    """Write data at path exactly once: temp file, then os.link, which fails if
    path exists. True when this call created it. Safe across sessions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        return True
    finally:
        os.unlink(tmp)


def _parse(data: bytes) -> list:
    """(id, record) for each well-formed line, first occurrence only."""
    seen = {}
    for raw in data.splitlines():
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        if well_formed(rec):
            seen.setdefault(record_id(rec), rec)
    return sorted(seen.items())


def _via_lines() -> list:
    try:
        raw = VIA_PATH.read_bytes()
    except OSError:
        return []
    lines = []
    for ln in raw.splitlines():
        try:
            obj = json.loads(ln)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("segment"), str):
            lines.append(obj)
    return lines


def _append_via(line: dict) -> bool:
    """Append one provenance line under flock, unless this arrival (segment,
    source, origin) is already recorded. True when it was appended."""
    VIA_PATH.parent.mkdir(parents=True, exist_ok=True)
    arrival = (line["segment"], line["source"], line["origin"])
    with open(VIA_PATH, "a+b") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            for raw in fh:
                try:
                    old = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(old, dict) and (old.get("segment"), old.get("source"), old.get("origin")) == arrival:
                    return False
            fh.write(canonical(line) + b"\n")
            fh.flush()
            return True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def store_segment(data: bytes, source: str, origin: str | None) -> dict:
    """Store one segment under the sha256 of its bytes and record where it came
    from. Each record is checked once, at ingest: one under this install's key
    whose mac fails is forged and gets a refute. Returns the via line plus
    new (whether the segment file was created by this call)."""
    if source not in SOURCES:
        raise MindError(f"mind: unknown segment source {source!r}")
    if len(data) > SEGMENT_MAX_BYTES:
        raise MindError("mind: refused a segment over 1 MiB")
    name = hashlib.sha256(data).hexdigest()
    new = _link_once(SEG_DIR / f"{name}.jsonl", data)
    recs = _parse(data)
    kid = key_id()
    verified, forged, other = 0, [], {}
    for rid, rec in recs:
        status = _status(rec, kid)
        if status == "verified":
            verified += 1
        elif status == "forged":
            forged.append(rid)
        else:
            other[rec["key"]] = other.get(rec["key"], 0) + 1
    line = {"segment": name, "source": source, "origin": origin, "at": int(time.time()),
            "records": len(recs), "verified": verified, "forged": sorted(forged), "other": other}
    if _append_via(line) and forged:
        write_records([make_record("refute", {"target": rid, "why": "mac does not match this install's key"})
                       for rid in sorted(forged)])
    return dict(line, new=new)


def write_records(records: list) -> str | None:
    """Store records as one segment (sorted by id, so the same set gives the
    same segment), split only past SEGMENT_MAX_BYTES. Returns the name of the
    last segment written, or None when there is nothing to write."""
    uniq = {}
    for r in records:
        if not well_formed(r):
            raise MindError("mind: refused to store a malformed record")
        uniq[record_id(r)] = r
    if not uniq:
        return None
    chunks, cur, size = [], [], 0
    for rid in sorted(uniq):
        ln = canonical(uniq[rid]) + b"\n"
        if len(ln) > SEGMENT_MAX_BYTES:
            raise MindError("mind: one record is over 1 MiB")
        if cur and size + len(ln) > SEGMENT_MAX_BYTES:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(ln)
        size += len(ln)
    chunks.append(cur)
    name = None
    for chunk in chunks:
        name = store_segment(b"".join(chunk), "self", None)["segment"]
    return name


def load_records() -> list:
    """Every well-formed record in memory, deduplicated, sorted by id. A segment
    with no via line (copied in by hand, or left by a crash) is ingested as
    found, so it is checked like any other arrival. Files whose name is not
    the sha256 of their bytes are not segments and are skipped."""
    if not SEG_DIR.is_dir():
        return []
    known = {v["segment"] for v in _via_lines()}
    seen, refuted_some = {}, False
    for path in sorted(SEG_DIR.glob("*.jsonl")):
        try:
            if path.stat().st_size > SEGMENT_MAX_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() != path.stem:
            continue
        if path.stem not in known:
            refuted_some |= bool(store_segment(data, "found", None)["forged"])
        for rid, rec in _parse(data):
            seen.setdefault(rid, rec)
    if refuted_some:
        return load_records()  # pick up the refute segments just written; every segment is known now
    return [seen[rid] for rid in sorted(seen)]


def _load_bodies() -> dict:
    """body id -> local target, for display only. Never replicated."""
    try:
        data = json.loads(BODIES_PATH.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}


def _save_bodies(mapping: dict) -> None:
    """Merge mapping into bodies.json under flock: temp file, then os.replace."""
    MIND_DIR.mkdir(parents=True, exist_ok=True)
    with open(MIND_DIR / ".bodies.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            merged = _load_bodies()
            merged.update(mapping)
            fd, tmp = tempfile.mkstemp(dir=MIND_DIR, prefix=".bodies-")
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(merged, fh, sort_keys=True, indent=1)
                os.replace(tmp, BODIES_PATH)
            except BaseException:
                os.unlink(tmp)
                raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ---------------------------------------------------------------- seam
# Stage 2 appends goal building, lifting, bodies and sensing here (step_specs,
# build_steps, goal_record, lift_*, local_body_id, caps_of, fit_record,
# survey, launch_check); stage 3 the sessions, claims and execution.


# ---------------------------------------------------------------- timing, exit codes, network ledger
# Every module mind calls (transport, engine, fleet, agent, dispatcher, env,
# placement, consensus, events) is reached through this module's globals, and
# processes start through _popen and stop through _killpg, so tests swap any
# of them for a fake without touching the real modules.

TICK_S = 1.0              # the session loop; a launcher exit is seen within one tick
SENSE_TIMEOUT_S = 20      # one sense call, covering every job this session holds on a body
KILL_GRACE_S = 5          # SIGTERM, then SIGKILL, for a dropped launcher group
QUIET_DEFAULT_S = 600     # replica.LOST_AFTER_S: a body silent this long is lost
QUIET_MIN_S = 60          # the least --quiet-after accepts
TERMINAL = ("succeeded", "failed", "timed_out", "killed")
NET_PURPOSES = ("sense", "caps", "sync", "collect", "terminate", "launch")  # never "coord"
NET_ERRORS = (OSError, subprocess.SubprocessError, transport.RemoteError)
UNREACHABLE = "unreachable at launch"
# What `mind run` returns. "usage" is no steps, or no body has reported at all.
EXIT_CODES = {"passed": 0, "failed": 1, "usage": 2, "stranded": 3, "tree_changed": 4, "detached": 130}

_popen = subprocess.Popen
_killpg = os.killpg


def exit_code(state: str) -> int:
    """The exit code `mind run` returns for a final session state."""
    return EXIT_CODES.get(state, EXIT_CODES["usage"])


def _net(ledger: Counter, purpose: str, fn, *args, **kwargs):
    """Every network call mind makes goes through here and is counted by
    purpose. There is no "coord" purpose: sessions never message each other,
    so ledger["coord"] (coord_msgs) is 0 by construction."""
    if purpose not in NET_PURPOSES:
        raise MindError(f"mind: {purpose!r} is not a network purpose")
    ledger[purpose] += 1
    return fn(*args, **kwargs)


# ---------------------------------------------------------------- bodies and sensing

# Python 3.9 stdlib only: it runs on a worker as `python3 - <b64 json job ids>`.
# One call per body per tick answers for every job this session holds there.
SENSE_SCRIPT = b"""\
import base64, json, os, re, secrets, subprocess, sys
H = os.path.expanduser("~/.orchard")
TERMINAL = ("succeeded", "failed", "timed_out", "killed")
def _read(p):
    try:
        with open(p) as f: return json.load(f)
    except (OSError, ValueError): return None
def body_id():
    p = os.path.join(H, "body.id")
    try:
        with open(p) as f: return f.read().strip()
    except FileNotFoundError: pass
    os.makedirs(H, exist_ok=True)
    try: fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with open(p) as f: return f.read().strip()
    v = secrets.token_hex(16)
    with os.fdopen(fd, "w") as f: f.write(v)
    return v
def boot():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/sys/kernel/random/boot_id") as f: return f.read().strip() or None
        return subprocess.run(["sysctl", "-n", "kern.bootsessionuuid"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception: return None
def group_alive(pgid):
    try: os.killpg(int(pgid), 0); return True
    except ProcessLookupError: return False
    except (PermissionError, TypeError, ValueError): return True
ids = json.loads(base64.b64decode(sys.argv[1])) if len(sys.argv) > 1 else []
b = boot()
out = {"body": body_id(), "boot": b, "load": os.getloadavg()[0], "cores": os.cpu_count(), "jobs": {}}
for j in ids:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", j): continue
    m = _read(os.path.join(H, "state", "running", j + ".json"))      # marker FIRST
    s = _read(os.path.join(H, "jobs", j, "status.json"))             # then status
    if m:
        st = "orphan" if (m.get("boot_id") and b and m["boot_id"] != b) or not group_alive(m.get("pgid")) else "running"
    elif s and s.get("state") in TERMINAL: st = "done"
    elif s: st = "orphan"
    else: st = "absent"
    out["jobs"][j] = {"state": st, "status": (s or {}).get("state")}
print(json.dumps(out))
"""


def local_body_id() -> str:
    """This machine's body id: ~/.orchard/body.id, random and minted once with
    O_EXCL. It is the same file SENSE_SCRIPT reads when this machine is also a
    configured worker, so both paths name one body."""
    try:
        return BODY_ID_PATH.read_text().strip()
    except FileNotFoundError:
        pass
    BODY_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(BODY_ID_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return BODY_ID_PATH.read_text().strip()
    value = secrets.token_hex(16)
    with os.fdopen(fd, "w") as fh:
        fh.write(value)
    return value


def sense_remote(worker: Worker, job_ids: list, ledger: Counter) -> dict | None:
    """One probe of a remote body: {body, boot, load, cores, jobs{id: {state,
    status}}} with state running, done, orphan or absent. None when the body
    does not answer, or answers with something that is not a sense report."""
    arg = base64.b64encode(json.dumps(sorted(set(job_ids))).encode()).decode()
    try:
        proc = _net(ledger, "sense", transport.run_script, worker, SENSE_SCRIPT, [arg], timeout=SENSE_TIMEOUT_S)
    except NET_ERRORS:
        return None
    if proc.returncode != 0:
        return None
    try:
        out = json.loads(proc.stdout)
    except (TypeError, ValueError):
        return None
    if not isinstance(out, dict) or not _is_hex(out.get("body"), 32) or not isinstance(out.get("jobs"), dict):
        return None
    return out


def caps_of(target: str, worker: Worker | None, ledger: Counter) -> dict | None:
    """A body's capabilities now: collected here for "local", asked of the
    worker's agent otherwise. None when the body cannot say."""
    try:
        if worker is None:
            return agent.collect_caps(env.ensure_bench_python())
        return _net(ledger, "caps", engine.remote_caps, worker)
    except NET_ERRORS:
        return None


# Hold and problem texts can name a job, a worker or a path. Records carry only
# these fixed codes; the full text stays in this session's own messages.
_HOLD_CODES = (("worker paused", "held: paused"), ("worker running ", "held: busy"),
               ("worker busy with other work", "held: busy"), ("worker host status", "held: host not ok"),
               ("worker idle time unknown", "held: idle time unknown"), ("worker in use", "held: owner active"),
               ("worker on battery", "held: on battery"), ("worker battery low", "held: battery low"),
               ("worker thermal throttle", "held: thermal"))
_PROBLEM_KEEP = ("missing tags: ", "worker OS is ", "needs ")


def _hold_code(hold: str) -> str:
    """The fixed code a record carries for a dispatcher hold."""
    for prefix, code in _HOLD_CODES:
        if hold.startswith(prefix):
            return code
    return "held"


def _problem_code(problem: str) -> str:
    """The text a record carries for a placement problem. Tags, OS families and
    sizes are kept; a resolver note (it can quote a path) is not."""
    if problem.startswith("platform resolve failed"):
        return "platform resolve failed"
    if problem.startswith(_PROBLEM_KEEP):
        return problem
    return "placement problem"


def launch_check(target: str, worker: Worker | None, job: JobSpec, ledger: Counter) -> tuple:
    """(why, hold, detail) just before a spawn. why is None when the body can
    take the step now. A hold (the owner is active, it is on battery, it runs
    another job) comes back with hold True; problems and silence with hold
    False. why is a fixed code fit for a record; detail is the full text, for
    this session's own messages only."""
    caps = caps_of(target, worker, ledger)
    if caps is None:
        return UNREACHABLE, False, UNREACHABLE
    caps = dict(caps)
    hold = None
    if worker is not None:
        if job.kind == "run":
            caps["platform_ok"], caps["platform_note"] = env.resolve_for_worker(job.requirements, job.python, caps)
        # Includes "worker running X": a foreground child skips dispatcher
        # holds, so the one-job-at-a-time rule is kept here.
        hold = dispatcher.idle_hold(caps, dispatcher.load_config())
    probs = [str(p) for p in placement.problems(job, caps)]
    if probs:
        return "; ".join(_problem_code(p) for p in probs), False, "; ".join(probs)
    if hold:
        return _hold_code(str(hold)), True, "held: " + str(hold)
    return None, False, None


# ---------------------------------------------------------------- attempts: spawn, outcome, drop, reap


@dataclass
class Attempt:
    """One launch this session observes (it spawned it, or adopted it). Local
    only: the job name, launcher pid and target never enter a record."""
    step_id: str
    gen: int
    body_id: str
    target: str                  # "local" or a worker name
    name: str                    # the job name the launcher runs under
    proc: object = None          # our own launcher child, when this session spawned it
    job_id: str | None = None    # the job folder, once the launcher has made it
    rjob: str | None = None      # the remote job folder, once known
    last_heard: float = 0.0      # observer clock, when the body last answered
    orphan_seen: int = 0         # consecutive "orphan" probes
    adopted: bool = False
    t_launch: float = 0.0
    launcher: dict | None = None  # an adopted launcher that is still alive: {pid, boot, ident}
    absent_seen: int = 0         # consecutive "absent" probes
    terminal_at: float | None = None  # when an adopted job was first seen finished
    dead_seen: int = 0           # consecutive looks at a local job whose process group is gone


def spawn(target: str, name: str, job: JobSpec, cwd: Path):
    """Start `orchard run -w target --name name -- cmd` as a detached child in
    its own session, so Ctrl-C here leaves it running and a drop can signal its
    whole group. Not run_on: that blocks, and watches a remote job forever."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    argv = fleet.child_argv(target, name, job.cmd, job.tags, job.env, job.timeout_s)
    with open(LOG_DIR / f"{name}.log", "ab") as log:
        return _popen(argv, env=fleet.child_env(), cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=log,
                      stderr=subprocess.STDOUT, start_new_session=True)


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def lift_receipt(receipt: dict, step: dict, gen: int, body_id: str | None, bench: int,
                 signature_id: str | None) -> dict | None:
    """A result record from a sealed receipt. None when the receipt does not
    verify under this install's key, is not finished, or is for other work
    than this step (the step id is recomputed from the receipt's own tree and
    work fields, never trusted from the launch)."""
    if not isinstance(receipt, dict) or not receipts.verify(receipt):
        return None
    try:
        if step_id_of(receipt["tree"], receipt["work"]) != step["step_id"]:
            return None
        state, worker = receipt["state"], receipt.get("worker") or {}
        elapsed = receipt.get("elapsed_s")
        body = {
            "step_id": step["step_id"], "gen": gen, "body_id": body_id, "state": state,
            "exit_code": _count(receipt.get("exit_code")),
            "elapsed_ms": int(round(elapsed * 1000)) if isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool) and elapsed >= 0 else 0,
            "bench": max(0, int(bench)), "work_key": receipt["work_key"], "tree": receipt["tree"],
            "cmd_key": step["cmd_key"], "log_sha256": receipt.get("log_sha256"),
            "out_sha256": receipt.get("out_sha256"),
            "receipt_sha": hashlib.sha256(canonical(receipt)).hexdigest(),
            "os_family": worker.get("os_family") if isinstance(worker, dict) else None,
            "signature_id": signature_id, "at": int(time.time()),
        }
    except (KeyError, TypeError, AttributeError, ValueError):
        return None
    if state not in TERMINAL:
        return None
    try:
        return make_record("result", body)
    except MindError:
        return None


def lost_record(step_id: str, gen: int, body_id: str, why: str, hold: bool = False) -> dict:
    return make_record("lost", {"step_id": step_id, "gen": gen, "body_id": body_id, "why": why, "hold": hold,
                                "at": int(time.time())})


def _lost(att: Attempt, why: str, hold: bool = False) -> dict:
    return lost_record(att.step_id, att.gen, att.body_id, why, hold)


def _last_line(path: Path) -> str:
    try:
        lines = [ln.strip() for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return ""
    return lines[-1][:200] if lines else ""


def finish(att: Attempt, step: dict, bench: int) -> list:
    """The record an attempt ends with once its launcher has exited: a result
    from a sealed, verified receipt for this very step, or a lost record that
    says why not. The launcher's own log stays local: a log line could name a
    machine, so the lost reason never quotes it."""
    sdir = fleet.find_job(att.name)
    if sdir is not None:
        att.job_id = sdir.name
    res = fleet.result_of(sdir, att.target) if sdir is not None else {}
    if res.get("state") not in TERMINAL:
        if att.target == "local":
            return [_lost(att, "launcher died")]
        code = att.proc.poll() if att.proc is not None else None
        return [_lost(att, "unavailable: launcher exited" + (f" {code}" if code is not None else "")
                      + " with no job status")]
    if not res.get("sealed"):
        return [_lost(att, "not sealed")]
    try:
        receipt = json.loads((sdir / "receipt.json").read_text())
    except (OSError, ValueError):
        receipt = None
    if not isinstance(receipt, dict) or not receipts.verify(receipt):
        return [_lost(att, "receipt failed verification")]
    sig_id = None
    if receipt.get("state") in FAIL_STATES:
        try:
            text = (sdir / "log.txt").read_text(errors="replace")
        except OSError:
            text = ""
        sig_id = consensus.signature_id(consensus.signature(text))
    rec = lift_receipt(receipt, step, att.gen, att.body_id, bench, sig_id)
    return [rec] if rec is not None else [_lost(att, "tree moved under the step")]


def drop_launcher(att: Attempt, sleep=time.sleep) -> None:
    """Stop an attempt's launcher group so it can never collect and seal later:
    SIGTERM, up to KILL_GRACE_S for the leader to go, then SIGKILL to whatever
    is left of the group. An adopted launcher is signalled only while its pid
    still names the same process."""
    if att.proc is not None:
        pid = att.proc.pid
    else:
        info = att.launcher or {}
        pid = info.get("pid")
        if not pid_alive(pid, info.get("boot"), info.get("ident")):
            return
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return
    try:
        _killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    waited = 0.0
    while waited < KILL_GRACE_S:
        if att.proc is not None and att.proc.poll() is not None:
            break
        sleep(0.25)
        waited += 0.25
    try:
        _killpg(pid, signal.SIGKILL)  # the leader's children (ssh, rsync) share its group
    except (ProcessLookupError, PermissionError):
        pass
    if att.proc is not None:
        try:
            att.proc.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            pass


def _zombie_lines() -> list:
    try:
        raw = ZOMBIES_PATH.read_text().splitlines()
    except OSError:
        return []
    out = []
    for ln in raw:
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if isinstance(e, dict) and isinstance(e.get("target"), str) and isinstance(e.get("job"), str):
            out.append(e)
    return out


def add_zombie(target: str, job_id: str, rjob: str | None) -> None:
    """Remember a remote job whose launcher was dropped, to terminate it once
    its body answers again. Local file, never replicated."""
    MIND_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"target": target, "job": job_id, "rjob": rjob, "at": int(time.time())}, sort_keys=True)
    with open(MIND_DIR / ".zombies.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(ZOMBIES_PATH, "a") as fh:
                fh.write(line + "\n")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reap(workers: dict, ledger: Counter, answered: set | None = None) -> int:
    """Terminate each zombie job whose body answers now, and forget it. With
    answered, only zombies on those targets are tried: a body that has not
    answered this session since the last reap is not probed again. The
    network calls run outside the lock; settled entries are then removed under
    it, so a zombie added meanwhile is kept. Returns how many were settled."""
    settled = []
    for e in _zombie_lines():
        w = workers.get(e["target"])
        if w is None or (answered is not None and e["target"] not in answered):
            continue
        try:
            rjob = e.get("rjob") or engine.remote_job_dir(_net(ledger, "terminate", engine.remote_home, w), e["job"])
            ok = _net(ledger, "terminate", engine.terminate_remote, w, rjob)
        except NET_ERRORS:
            ok = None
        if ok is not None:  # True: signalled; False: the agent says it is not running
            settled.append(e)
    if not settled:
        return 0
    with open(MIND_DIR / ".zombies.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            keep = [e for e in _zombie_lines() if e not in settled]
            fd, tmp = tempfile.mkstemp(dir=MIND_DIR, prefix=".zombies-")
            with os.fdopen(fd, "w") as fh:
                fh.writelines(json.dumps(e, sort_keys=True) + "\n" for e in keep)
            os.replace(tmp, ZOMBIES_PATH)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return len(settled)


def _local_group_gone(status: dict) -> bool | None:
    """Whether a local job's process group is known gone: another boot, its
    leader's number now names another process, or no member is left. None
    when the status names no group yet (the launcher died before it spawned
    the command, or has not written it)."""
    pgid = status.get("pgid")
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 0:
        return None
    here, boot = agent.boot_id(), status.get("boot_id")
    if here and boot and here != boot:
        return True
    start = status.get("pid_start")
    try:
        if start is not None and agent.process_start(pgid) not in (None, start):
            return True
        return not agent._group_alive(pgid)
    except (OSError, OverflowError, TypeError, ValueError):
        return False


def watch_adopted(att: Attempt, worker: Worker | None, cwd: Path, ledger: Counter, *, step: dict, bench: int,
                  clock=time.monotonic) -> list | None:
    """One look at an adopted attempt whose launcher is gone or unknown. Returns
    the records that end it, or None to keep watching. A finished job whose
    receipt has not appeared after SEAL_GRACE_S is collected and sealed with
    the same tail calls as cli.run_remote and cli.run_local."""
    sdir = fleet.find_job(att.name)
    if sdir is None:
        # A launcher nobody knows the pid of may still be making the folder.
        if clock() - att.t_launch < SEAL_GRACE_S:
            return None
        return [_lost(att, "launcher died before the job started")]
    att.job_id = sdir.name
    if (sdir / "receipt.json").exists():
        return finish(att, step, bench)
    if worker is None:
        status = engine.local_status(sdir)
        out_dir = cwd / "out"
    else:
        try:
            if att.rjob is None:
                att.rjob = engine.remote_job_dir(_net(ledger, "collect", engine.remote_home, worker), att.job_id)
            status = _net(ledger, "collect", engine.remote_status, worker, att.rjob)
            if status is None:
                missing = _net(ledger, "collect", engine.remote_status_missing, worker, att.rjob)
                if missing and att.absent_seen >= ORPHAN_CONFIRM:
                    return [_lost(att, "launcher died before the job started")]
                return None
        except NET_ERRORS:
            return None
        att.last_heard = clock()
        out_dir = cwd / ".orchard-out" / att.job_id / "out"
    if not isinstance(status, dict) or status.get("state") not in TERMINAL:
        if worker is not None:
            return None
        # Local: the launcher ran the job's agent itself, so with the launcher
        # gone nobody will write a final status. A job whose group is gone
        # ended without one.
        gone = _local_group_gone(status) if isinstance(status, dict) else None
        if gone is None:
            if clock() - att.t_launch < SEAL_GRACE_S:
                return None
            return [_lost(att, "launcher died before the job started")]
        att.dead_seen = att.dead_seen + 1 if gone else 0
        return [_lost(att, "launcher died")] if att.dead_seen >= ORPHAN_CONFIRM else None
    if att.terminal_at is None:
        att.terminal_at = clock()
    if clock() - att.terminal_at < SEAL_GRACE_S:
        return None  # a slow but live launcher may still seal
    if (sdir / "receipt.json").exists():
        return finish(att, step, bench)
    if worker is not None:
        try:
            final = _net(ledger, "collect", engine.collect_remote, worker, att.job_id, att.rjob, sdir, cwd)
        except NET_ERRORS:
            return None
        if isinstance(final, dict) and final.get("state") in TERMINAL:
            status = final
    receipts.seal(sdir, status, worker.name if worker is not None else "local", out_dir)
    return finish(att, step, bench)


# ---------------------------------------------------------------- sessions and claims
# claims/<step>.<gen> is the first holder (k = 0); <step>.<gen>.h<k> are later
# holders, each adopting from a dead one; <step>.<gen>.run names the launcher.
# All are write-once (_link_once), local, and never replicated.


def _proc_identity(pid: int) -> str | None:
    """Who pid is beyond its number: when it started. None when unknown."""
    try:
        start = agent.process_start(pid)
    except Exception:
        start = None
    if start is not None:
        return "t" + str(start)
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(int(pid))], capture_output=True, text=True,
                             timeout=5)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return None
    text = out.stdout.strip()
    return "l" + text if out.returncode == 0 and text else None


def pid_alive(pid: object, boot: str | None, ident: str | None) -> bool:
    """Whether the process a claim names may still be alive. It is dead only
    when known dead: another boot, no such pid, or the pid now names a process
    that started at another time (reused). Unknown counts as alive, so mind
    never adopts from a holder that might still be running. A claim with no
    usable pid is malformed and counts as dead."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    here = agent.boot_id()
    if boot and here and boot != here:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OverflowError:
        return False
    now = _proc_identity(pid)
    return not (ident and now and ident != now)


def new_session() -> dict:
    """This process as a claim holder: {session, pid, boot, ident}."""
    pid = os.getpid()
    return {"session": secrets.token_hex(8), "pid": pid, "boot": agent.boot_id(), "ident": _proc_identity(pid)}


def _claim_path(step_id: str, gen: int, k: int = 0) -> Path:
    base = f"{step_id}.{gen}"
    return CLAIM_DIR / (base if k == 0 else f"{base}.h{k}")


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def claim(step_id: str, gen: int, info: dict) -> bool:
    """Take (step, gen) for one session. Write-once, so exactly one of any
    number of racing sessions wins, and the content is there from the first
    instant. Keyed by step and gen, not goal: two goals that share a step
    launch it once on this install."""
    return _link_once(_claim_path(step_id, gen), canonical(info))


def current_holder(step_id: str, gen: int) -> tuple | None:
    """(k, info) for the newest holder of (step, gen), or None when unclaimed.
    An unreadable holder file reads as {} (no pid: dead)."""
    if not CLAIM_DIR.is_dir():
        return None
    base = f"{step_id}.{gen}"
    best = -1
    for path in CLAIM_DIR.glob(base + "*"):
        name = path.name
        if name == base:
            k = 0
        elif name.startswith(base + ".h") and name[len(base) + 2:].isdigit():
            k = int(name[len(base) + 2:])
        else:
            continue
        best = max(best, k)
    if best < 0:
        return None
    return best, _read_json(_claim_path(step_id, gen, best)) or {}


def adopt(step_id: str, gen: int, session: dict, info: dict | None = None) -> bool:
    """Become the holder of (step, gen) after a dead one (or hand over from this
    same session): write-once .h<k+1>, so of two adopters exactly one wins.
    The new holder file is info, or the old holder's with this session's
    identity. Writes no memory record."""
    held = current_holder(step_id, gen)
    if held is None:
        return False
    k, cur = held
    if cur.get("session") != session["session"] and pid_alive(cur.get("pid"), cur.get("boot"), cur.get("ident")):
        return False
    who = {f: session[f] for f in ("session", "pid", "boot", "ident")}
    body = dict(info if info is not None else cur, **who, adopted_from=k)
    return _link_once(_claim_path(step_id, gen, k + 1), canonical(body))


def mark_launched(step_id: str, gen: int, pid: int, job_name: str, target: str) -> bool:
    """claims/<step>.<gen>.run: the launcher an adopter can still watch."""
    info = {"pid": pid, "boot": agent.boot_id(), "ident": _proc_identity(pid), "job_name": job_name,
            "target": target}
    return _link_once(CLAIM_DIR / f"{step_id}.{gen}.run", canonical(info))


def launcher_of(step_id: str, gen: int) -> dict | None:
    return _read_json(CLAIM_DIR / f"{step_id}.{gen}.run")


# ---------------------------------------------------------------- the session loop


def own_goal(goal_id: str, records: list) -> dict:
    """The goal record a session may execute: one the planner reads, made and
    signed by this install. A goal from another key (synced or imported) is
    shown and planned but never run here: launching is its author's."""
    recs = {record_id(r): r for r in records if well_formed(r)}
    out = refuted(recs)
    me, seen = key_id(), False
    for rid in sorted(recs):
        r = recs[rid]
        if rid in out or r["kind"] != "goal":
            continue
        if (r["body"]["goal_id"] != goal_id or goal_id_of(r["body"]) != goal_id
                or r["body"]["author"] != r["key"]):
            continue
        seen = True
        if r["key"] == me and verify_record(r) == "verified":
            return r
    if seen:
        raise MindError(f"mind: goal {goal_id[:12]} was made by another install; only its author runs it")
    raise MindError(f"mind: no goal {goal_id[:12]} in memory here")


def job_for(step: dict, jobs: dict | None = None) -> JobSpec:
    """The JobSpec a step launches: the caller's when it has one, else rebuilt
    from the goal's own step fields."""
    if jobs and step["step_id"] in jobs:
        return jobs[step["step_id"]]
    return JobSpec(name=step["step_id"][:12], cmd=list(step["cmd"]), timeout_s=step["timeout"] or None,
                   requires=dict(step["requires"]), kind=step["kind"], tags=list(step["tags"]),
                   env=list(step["env_names"]))


def final_state(p: dict) -> str:
    """The session state for a plan that is no longer running. A goal stranded
    only because no body has reported is a usage error: survey first."""
    if p["state"] == "stranded":
        why = [st["why"] for st in p["steps"] if st["state"] == "stranded"]
        if why and all(w == [NO_BODIES] for w in why):
            return "usage"
    return p["state"]


class _Session:
    """One `mind run`: plans from memory each tick, watches what it launched or
    adopted, and launches each body's next step. Sessions never talk to each
    other: they share only memory and the write-once claim files."""

    def __init__(self, goal: dict, cwd: Path, jobs: dict | None, *, quiet_s: float, say, clock, sleep,
                 workers: dict, tree_now, session: dict, survey):
        self.goal_id = goal["body"]["goal_id"]
        self.author = goal["key"]
        self.goal_tree = goal["body"]["tree"]
        self.steps = {s["step_id"]: s for s in goal["body"]["steps"]}
        self.cwd = Path(cwd)
        self.jobs = jobs or {}
        self.quiet_s = quiet_s
        self.say = say or (lambda msg: None)
        self.clock, self.sleep = clock, sleep
        self.workers = workers
        self.tree_now = tree_now
        self.me = session
        self.survey = survey
        self.ledger = Counter()
        self.attempts = {}          # step id -> Attempt
        self.arbitration_writes = 0  # write-once claims this session lost to a live holder
        self.lost_at = {}           # step id -> (clock, detect_s) for a step this session lost
        self.recovered = []         # recover_s of each relaunch after such a loss
        self.next_sense = {}        # target -> clock
        self.next_check = {}        # (step, body) -> clock, while a sole body is held
        self.next_reap = 0.0
        self.heard = set()          # targets that answered since the last reap
        self.next_survey = 0.0
        self.unknown = set()        # bodies already reported as unknown here
        self.last = None

    # ---- one tick

    def tick(self) -> tuple:
        """(plan, stop): stop is "tree_changed" when the tree moved, else None."""
        now = self.clock()
        if self.survey is not None and now >= self.next_survey:
            self.next_survey = now + FIT_EVERY_S
            found = self.survey(self.ledger)
            if isinstance(found, dict):
                self.heard.update(t for t in found if t != "local")
        if now >= self.next_reap and self.heard:
            self.next_reap = now + SENSE_EVERY_S
            reap(self.workers, self.ledger, set(self.heard))
            self.heard.clear()
        records = load_records()
        bench = self._bench(records)
        p = plan(records, self.goal_id)
        out = self.advance(bench)
        out += self.adopt_running(p)
        if out:
            write_records(out)
            p = plan(load_records(), self.goal_id)  # recover in this same tick
        stop = None
        if p["state"] == "running":
            stop, wrote = self.launch_heads(p)
            if wrote:
                p = plan(load_records(), self.goal_id)
        self.last = p
        return p, stop

    def _bench(self, records: list) -> dict:
        out = {}
        for r in records:
            if (well_formed(r) and r["kind"] == "fit" and r["key"] == self.author
                    and r["body"]["goal_id"] == self.goal_id):
                out.setdefault(r["body"]["body_id"], r["body"]["caps"]["bench"])
        return out

    # ---- attempts already out

    def sense(self) -> set:
        """One sense call per remote body due one; returns the targets sensed."""
        now = self.clock()
        by_target = {}
        for att in self.attempts.values():
            if att.target != "local":
                by_target.setdefault(att.target, []).append(att)
        sensed = set()
        for target in sorted(by_target):
            if now < self.next_sense.get(target, 0.0):
                continue
            self.next_sense[target] = now + SENSE_EVERY_S
            w = self.workers.get(target)
            if w is None:
                continue
            atts = by_target[target]
            for att in atts:
                if att.job_id is None:
                    sdir = fleet.find_job(att.name)
                    att.job_id = sdir.name if sdir is not None else None
            sensed.add(target)
            rep = sense_remote(w, [a.job_id for a in atts if a.job_id], self.ledger)
            if rep is None:
                continue
            self.heard.add(target)
            heard = self.clock()
            for att in atts:
                att.last_heard = heard
                st = (rep["jobs"].get(att.job_id) or {}).get("state") if att.job_id else None
                if rep["body"] != att.body_id:
                    st = "orphan"  # another body answers under that name now: ours is gone
                att.orphan_seen = att.orphan_seen + 1 if st == "orphan" else 0
                att.absent_seen = att.absent_seen + 1 if st == "absent" else 0
        return sensed

    def advance(self, bench: dict) -> list:
        sensed = self.sense()
        out = []
        for sid in sorted(self.attempts):
            att = self.attempts[sid]
            recs = self._advance_one(att, self.steps[sid], max(1, bench.get(att.body_id, 1)), sensed)
            if recs is None:
                continue
            del self.attempts[sid]
            if any(r["kind"] == "lost" for r in recs):
                now = self.clock()
                self.lost_at[sid] = (now, round(max(0.0, now - att.last_heard), 3))
                self.say(f"lost {sid[:12]} gen {att.gen} on {att.target}: "
                         + next(r["body"]["why"] for r in recs if r["kind"] == "lost"))
            else:
                self.say(f"{recs[0]['body']['state']} {sid[:12]} gen {att.gen} on {att.target}")
            out += recs
        return out

    def _sealed(self, att: Attempt) -> bool:
        sdir = fleet.find_job(att.name)
        return sdir is not None and (sdir / "receipt.json").exists()

    def _advance_one(self, att: Attempt, step: dict, bench: int, sensed: set) -> list | None:
        now = self.clock()
        if att.proc is not None and att.proc.poll() is not None:
            return finish(att, step, bench)
        live = att.proc is not None or (att.launcher is not None and pid_alive(
            att.launcher.get("pid"), att.launcher.get("boot"), att.launcher.get("ident")))
        if att.target == "local" and live:
            att.last_heard = now
        if att.orphan_seen >= ORPHAN_CONFIRM:
            drop_launcher(att, self.sleep)
            return finish(att, step, bench) if self._sealed(att) else [_lost(att, "body restarted or job died")]
        if att.target != "local" and now - att.last_heard >= self.quiet_s:
            drop_launcher(att, self.sleep)
            if self._sealed(att):
                return finish(att, step, bench)
            if att.job_id:
                add_zombie(att.target, att.job_id, att.rjob)
            return [_lost(att, f"quiet for {int(self.quiet_s)}s")]
        if live:
            return None
        # An adopted attempt whose launcher is gone: finish its job ourselves.
        att.launcher = None
        if att.target != "local" and att.target not in sensed:
            return None  # remote looks go at the sense cadence
        w = self.workers.get(att.target) if att.target != "local" else None
        if att.target != "local" and w is None:
            return None
        return watch_adopted(att, w, self.cwd, self.ledger, step=step, bench=bench, clock=self.clock)

    def adopt_running(self, p: dict) -> list:
        """Take over running steps whose holder session is dead. Returns lost
        records for any whose launch cannot be traced."""
        out = []
        for st in p["steps"]:
            sid, gen = st["step"], st["gen"]
            if st["state"] != "running" or sid in self.attempts or sid not in self.steps:
                continue
            held = current_holder(sid, gen)
            if held is None:
                continue  # launched elsewhere: that install watches it
            _, cur = held
            if cur.get("session") == self.me["session"]:
                continue
            if pid_alive(cur.get("pid"), cur.get("boot"), cur.get("ident")):
                continue
            if not adopt(sid, gen, self.me):
                self.arbitration_writes += 1
                continue
            run = launcher_of(sid, gen) or {}
            name = run.get("job_name") or cur.get("job_name")
            target = run.get("target") or cur.get("target")
            if not isinstance(name, str) or not isinstance(target, str):
                out.append(lost_record(sid, gen, st["body"], "launch cannot be traced"))
                continue
            now = self.clock()
            launcher = {f: run.get(f) for f in ("pid", "boot", "ident")} if run else None
            self.attempts[sid] = Attempt(sid, gen, st["body"], target, name, adopted=True, t_launch=now,
                                         last_heard=now, launcher=launcher)
            self.say(f"adopted {sid[:12]} gen {gen} on {target}")
        return out

    # ---- new launches

    def _claim_for(self, sid: str, gen: int, info: dict) -> dict | None:
        """The claim info this session holds (sid, gen) under, or None when a
        live session holds it."""
        if claim(sid, gen, info):
            return info
        held = current_holder(sid, gen)
        cur = held[1] if held else {}
        if cur.get("session") == self.me["session"]:
            if cur.get("target") == info["target"] and isinstance(cur.get("job_name"), str):
                return cur  # still ours from a held tick: keep its name
            return info if adopt(sid, gen, self.me, info) else None
        if pid_alive(cur.get("pid"), cur.get("boot"), cur.get("ident")):
            self.arbitration_writes += 1
            return None
        if adopt(sid, gen, self.me, info):
            return info
        self.arbitration_writes += 1
        return None

    def launch_heads(self, p: dict) -> tuple:
        """Launch each idle body's next pending step. Returns (stop, wrote)."""
        by_step = {st["step"]: st for st in p["steps"]}
        bodies = _load_bodies()
        busy = {a.body_id for a in self.attempts.values()}
        wrote = False
        for b in sorted(p["queues"]):
            q = p["queues"][b]
            if not q or b in busy:
                continue
            st = by_step[q[0]]
            if st["state"] != "pending" or st["step"] in self.attempts:
                continue
            sid, gen = st["step"], st["gen"]
            target = bodies.get(b)
            if target is None or (target != "local" and target not in self.workers):
                if b not in self.unknown:
                    self.unknown.add(b)
                    self.say(f"body {b[:12]} is not a worker here; its steps wait")
                continue
            if self.clock() < self.next_check.get((sid, b), 0.0):
                continue
            if self.tree_now() != self.goal_tree:
                return "tree_changed", wrote
            token = fleet.new_token()
            info = dict(self.me, job_name=fleet.job_name("mind", token, target, gen), target=target, token=token,
                        goal_id=self.goal_id)
            held = self._claim_for(sid, gen, info)
            if held is None:
                continue
            name, token = held["job_name"], held.get("token") or token
            job = job_for(self.steps[sid], self.jobs)
            w = self.workers.get(target) if target != "local" else None
            why, hold, detail = launch_check(target, w, job, self.ledger)
            if w is not None and why != UNREACHABLE:
                self.heard.add(target)
            if why is not None:
                if hold and len(st["capable"]) < 2:
                    self.next_check[(sid, b)] = self.clock() + SENSE_EVERY_S
                    self.say(f"waiting: {target} {detail}")
                    continue
                write_records([lost_record(sid, gen, b, why, hold)])
                self.lost_at.setdefault(sid, (self.clock(), 0.0))
                self.say(f"not launching {sid[:12]} on {target}: {detail}")
                return None, True  # the plan moves; the next tick sees it
            write_records([make_record("launch", {"goal_id": self.goal_id, "step_id": sid, "gen": gen,
                                                  "body_id": b, "token": token, "at": int(time.time())})])
            wrote = True
            try:
                if w is None:
                    proc = spawn(target, name, job, self.cwd)
                else:
                    proc = _net(self.ledger, "launch", spawn, target, name, job, self.cwd)
            except OSError:
                write_records([lost_record(sid, gen, b, "launcher failed to start")])
                self.lost_at.setdefault(sid, (self.clock(), 0.0))
                return None, True
            mark_launched(sid, gen, proc.pid, name, target)
            now = self.clock()
            self.attempts[sid] = Attempt(sid, gen, b, target, name, proc=proc, last_heard=now, t_launch=now)
            busy.add(b)
            self.say(f"launched {sid[:12]} gen {gen} on {target}")
            if sid in self.lost_at:
                t_lost, detect_s = self.lost_at.pop(sid)
                recover_s = round(max(0.0, now - t_lost), 3)
                self.recovered.append(recover_s)
                events.append("mind.recovered", goal=self.goal_id, step=sid, gen=gen, detect_s=detect_s,
                              recover_s=recover_s)
        return None, wrote

    # ---- the end

    def summary(self, p: dict, state: str) -> dict:
        out = {
            "goal": self.goal_id, "state": state,
            "exit": exit_code(state) if state in EXIT_CODES else 0,
            "coord_msgs": self.ledger["coord"],  # no purpose is "coord": always 0
            "arbitration_writes": self.arbitration_writes,
            "sense_calls": self.ledger["sense"],
            "duplicate_launches": len(p.get("duplicates") or []),
            "recover_s": list(self.recovered),
            "ledger": dict(self.ledger),
            "plan": p,
        }
        events.append("mind.session", goal=self.goal_id, state=state, exit=out["exit"], coord_msgs=out["coord_msgs"],
                      arbitration_writes=self.arbitration_writes, sense_calls=out["sense_calls"],
                      duplicate_launches=out["duplicate_launches"], recover_s=out["recover_s"])
        return out


def execute(goal_id: str, cwd: Path, jobs: dict | None = None, *, quiet_s: float = QUIET_DEFAULT_S,
            plan_only: bool = False, say=None, clock=time.monotonic, sleep=time.sleep, workers: dict | None = None,
            tree_now=None, session: dict | None = None, survey=None, max_ticks: int | None = None) -> dict:
    """Run this install's goal to an end: passed, failed, stranded, usage (no
    steps, or no body has reported), tree_changed, or detached (Ctrl-C, or
    max_ticks). Returns the session summary; its "exit" is what `mind run`
    exits with. Ctrl-C writes no record: launchers run in their own sessions,
    and the next session adopts them through the claims."""
    records = load_records()
    goal = own_goal(goal_id, records)
    if workers is None:
        workers = {w.name: w for w in config.load_workers()}
    if tree_now is None:
        def tree_now() -> str:
            return receipts.tree_digest(Path(cwd))[0]
    s = _Session(goal, Path(cwd), jobs, quiet_s=quiet_s, say=say, clock=clock, sleep=sleep, workers=workers,
                 tree_now=tree_now, session=session or new_session(), survey=survey)
    p = plan(records, goal_id)
    if not s.steps:
        return s.summary(p, "usage")
    if plan_only:
        return s.summary(p, p["state"] if p["state"] != "stranded" else final_state(p))
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    spath = SESSION_DIR / f"{s.me['session']}.json"
    spath.write_text(json.dumps({"session": s.me["session"], "pid": s.me["pid"], "boot": s.me["boot"],
                                 "ident": s.me["ident"], "goal": goal_id,
                                 "started": int(time.time())}, sort_keys=True))
    ticks = 0
    try:
        while True:
            p, stop = s.tick()
            if stop:
                return s.summary(p, stop)
            if p["state"] != "running":
                return s.summary(p, final_state(p))
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                return s.summary(p, "detached")
            sleep(TICK_S)
    except KeyboardInterrupt:
        return s.summary(s.last or p, "detached")
    finally:
        try:
            spath.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------- memory: summary, export, import, sync
# Memory moves only as whole segments, named by the sha256 of their bytes, so
# a copy is checked by its name and every arrival goes through store_segment.


def segment_names() -> list:
    if not SEG_DIR.is_dir():
        return []
    return sorted(p.stem for p in SEG_DIR.glob("*.jsonl") if _is_hex(p.stem, 64))


def _segment_ok(name: str, data: bytes) -> bool:
    return len(data) <= SEGMENT_MAX_BYTES and hashlib.sha256(data).hexdigest() == name


def memory_summary() -> dict:
    """Counts for `mind memory`: segments, records by kind and by status under
    this install's key, arrivals by source, and how many keys wrote records."""
    recs = load_records()
    return {
        "segments": len(segment_names()),
        "records": len(recs),
        "kinds": dict(sorted(Counter(r["kind"] for r in recs).items())),
        "status": dict(sorted(Counter(verify_record(r) for r in recs).items())),
        "arrivals": dict(sorted(Counter(str(v.get("source")) for v in _via_lines()).items())),
        "keys": len({r["key"] for r in recs}),
        "key_id": key_id(),
    }


def export_memory(path: Path) -> int:
    """Copy every segment into the directory path (made if missing) as
    <sha256>.jsonl. Segments already there are left alone. Returns how many
    were written."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for name in segment_names():
        dest = path / f"{name}.jsonl"
        if dest.exists():
            continue
        try:
            data = (SEG_DIR / f"{name}.jsonl").read_bytes()
        except OSError:
            continue
        if not _segment_ok(name, data):
            continue
        tmp = path / f".{name}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        wrote += 1
    return wrote


def import_memory(path: Path) -> dict:
    """Store each <sha256>.jsonl in the directory path as an import. A file
    whose bytes do not hash to its name, or over 1 MiB, is skipped. Returns
    {segments, new, skipped, forged}."""
    path = Path(path)
    if not path.is_dir():
        raise MindError("mind: import needs a directory of segments")
    out = {"segments": 0, "new": 0, "skipped": 0, "forged": 0}
    for f in sorted(path.glob("*.jsonl")):
        try:
            data = f.read_bytes() if f.stat().st_size <= SEGMENT_MAX_BYTES else b""
        except OSError:
            data = b""
        if not _is_hex(f.stem, 64) or not _segment_ok(f.stem, data):
            out["skipped"] += 1
            continue
        line = store_segment(data, "import", None)
        out["segments"] += 1
        out["new"] += int(line["new"])
        out["forged"] += len(line["forged"])
    return out


def sync(workers: dict, ledger: Counter) -> dict:
    """Swap segments with each worker's ~/.orchard/mind/segments: pull the ones
    missing here (each checked by its name, then stored as source "sync"),
    push the ones missing there (temp name, then mv, so a reader never sees a
    partial file). Workers are relays: nothing on them is trusted. Returns
    {worker: {pulled, pushed, bad}} or {worker: {error}}."""
    out = {}
    for name in sorted(workers):
        w = workers[name]
        got = {"pulled": 0, "pushed": 0, "bad": 0}
        try:
            rhome = _net(ledger, "sync", engine.remote_home, w)
            rdir = f"{rhome}/.orchard/mind/segments"
            qdir = shlex.quote(rdir)
            proc = _net(ledger, "sync", transport.run, w, f"mkdir -p {qdir} && ls {qdir}", timeout=60)
            if proc.returncode != 0:
                raise transport.RemoteError("segment listing failed")
            text = proc.stdout.decode("utf-8", "replace") if isinstance(proc.stdout, bytes) else str(proc.stdout)
            there = {ln[:-6] for ln in text.split() if ln.endswith(".jsonl") and _is_hex(ln[:-6], 64)}
            here = set(segment_names())
            for seg in sorted(there - here):
                data = _net(ledger, "sync", transport.pull_bytes, w, f"{rdir}/{seg}.jsonl")
                if not _segment_ok(seg, data):
                    got["bad"] += 1
                    continue
                store_segment(data, "sync", w.name)
                got["pulled"] += 1
            for seg in sorted(here - there):
                try:
                    data = (SEG_DIR / f"{seg}.jsonl").read_bytes()
                except OSError:
                    continue
                tmp = f"{rdir}/.tmp-{seg}-{secrets.token_hex(4)}"
                _net(ledger, "sync", transport.push_bytes, w, tmp, data)
                _net(ledger, "sync", transport.run_checked, w,
                     f"mv {shlex.quote(tmp)} {shlex.quote(rdir + '/' + seg + '.jsonl')}", timeout=60)
                got["pushed"] += 1
        except NET_ERRORS:
            out[name] = {"error": "unreachable"}
            continue
        out[name] = got
    return out


# ---------------------------------------------------------------- goals: steps, the goal record, lifting
# A goal is built exactly as `orchard run` would build each step's JobSpec
# (fleet.job_like_run), so a step's work key is the one today's receipts use.


def step_specs(cmd: list | None, step_texts: list, tags: list, env_names: list, timeout: int | None,
               checks: list | None = None) -> list:
    """One {cmd, tags, env, timeout} per step, in the order given: the command
    after --, each --step (split like a shell would), then each speculate
    check with its own tags, env and timeout. Exact repeats are dropped."""
    specs = []

    def add(c, t, e, to) -> None:
        spec = {"cmd": list(c), "tags": list(t or []), "env": list(dict.fromkeys(e or [])), "timeout": to}
        if spec not in specs:
            specs.append(spec)

    if cmd:
        add(cmd, tags, env_names, timeout)
    for text in step_texts or []:
        try:
            parts = shlex.split(text)
        except ValueError as e:
            raise MindError(f"mind: cannot split --step {text!r} ({e})") from e
        if not parts:
            raise MindError("mind: --step needs a command")
        add(parts, tags, env_names, timeout)
    for c in checks or []:
        if isinstance(c, dict) and c.get("cmd"):
            add(c["cmd"], c.get("tags"), c.get("env"), c.get("timeout"))
    return specs


def build_steps(cwd: Path, tree: str, specs: list, keyless: bool = False) -> tuple:
    """(steps, jobs): the goal's step fields, in the order given (the goal
    record sorts them), and the JobSpec each step launches. Two specs that
    make the same step are one. keyless (a plan on an install with no receipt
    key yet) leaves env values out of the work key, since keying them would
    make the key."""
    steps, jobs = {}, {}
    for spec in specs:
        job = fleet.job_like_run(spec["cmd"], Path(cwd), spec["tags"], spec["env"], spec["timeout"])
        work = receipts.work_fields(replace(job, env_values=None) if keyless else job)
        s = {"step_id": step_id_of(tree, work), "work_key": receipts.work_key(tree, work), "cmd": list(job.cmd),
             "kind": job.kind, "tags": list(job.tags), "requires": dict(job.requires), "env_names": list(job.env),
             "timeout": job.timeout_s or 0}
        s["cmd_key"] = cmd_key_of(s)
        steps.setdefault(s["step_id"], s)
        jobs.setdefault(s["step_id"], job)
    return list(steps.values()), jobs


NO_KEY = "0" * 16  # the author of a --plan-only goal on an install with no receipt key yet


def goal_record(cwd: Path, tree: str, steps: list, trust: list, keyless: bool = False) -> dict:
    """The signed goal: its author (this install's key id), project id, tree,
    trusted keys (the author's own is implied) and steps. The same inputs on
    the same install give the same goal id; another install's is another goal.
    keyless gives an unsigned goal by NO_KEY, for a plan that must not make
    the receipt key."""
    me = NO_KEY if keyless else key_id()
    keys = sorted({k for k in trust or [] if k != me})
    bad = [k for k in keys if not _is_hex(k, 16)]
    if bad:
        raise MindError(f"mind: --trust takes a key id (16 hex, from `orchard mind key`), not {bad[0]!r}")
    body = {"author": me, "project": project_id(Path(cwd), tree), "tree": tree, "trust": keys,
            "steps": sorted(steps, key=lambda s: s["step_id"])}
    body["goal_id"] = goal_id_of(body)
    if not keyless:
        return make_record("goal", body)
    rec = {"v": VERSION, "kind": "goal", "key": me, "body": body, "mac": "0" * 64}
    if not well_formed(rec):
        raise MindError("mind: a goal record is missing a field or has one of the wrong type")
    return rec


def _bench_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        return 0
    return int(round(value))


def fit_record(goal: dict, body_id: str, caps: dict, jobs: dict | None, remote: bool) -> dict:
    """What one body can run of a goal: the steps placement allows on its caps,
    and placement.problems' own strings for the rest. Only the bench and the
    OS family of the caps go into the record: never a name or a path."""
    fits, problems = [], {}
    for s in goal["body"]["steps"]:
        job = job_for(s, jobs)
        c = dict(caps)
        if remote and job.kind == "run":
            c["platform_ok"], c["platform_note"] = env.resolve_for_worker(job.requirements, job.python, c)
        probs = [_problem_code(str(p)) for p in placement.problems(job, c)]
        if probs:
            problems[s["step_id"]] = probs
        else:
            fits.append(s["step_id"])
    family = caps.get("os_family")
    return make_record("fit", {"goal_id": goal["body"]["goal_id"], "body_id": body_id,
                               "caps": {"bench": _bench_int(caps.get("bench")),
                                        "os_family": family if isinstance(family, str) else None},
                               "fits": sorted(fits), "problems": problems})


def survey(goal: dict, jobs: dict | None, workers: dict, ledger: Counter) -> dict:
    """Write one fit record for each body that answers: this machine, then
    every worker (its body id comes from a sense probe, which mints one on a
    fresh body). The id -> target mapping is kept here in bodies.json, never
    in a record. Returns {target: body id}."""
    recs, mapping = [], {}
    caps = caps_of("local", None, ledger)
    if caps is not None:
        b = local_body_id()
        recs.append(fit_record(goal, b, caps, jobs, remote=False))
        mapping[b] = "local"
    for name in sorted(workers):
        w = workers[name]
        rep = sense_remote(w, [], ledger)
        if rep is None or rep["body"] in mapping:
            continue  # silent, or this very machine reached as a worker
        caps = caps_of(name, w, ledger)
        if caps is None:
            continue
        recs.append(fit_record(goal, rep["body"], caps, jobs, remote=True))
        mapping[rep["body"]] = name
    if mapping:
        _save_bodies(mapping)
    write_records(recs)
    return {t: b for b, t in mapping.items()}


def lift_reusable(goal: dict, p: dict) -> list:
    """Result records from sealed passing receipts made before mind (orchard
    run, replica), one for each pending step whose work key has one. Only with
    --reuse: reuse is opt-in everywhere in Orchard."""
    by_id = {s["step_id"]: s for s in goal["body"]["steps"]}
    out = []
    for st in p["steps"]:
        s = by_id.get(st["step"])
        if s is None or st["state"] != "pending":
            continue
        receipt = receipts.find_reusable(s["work_key"])
        if receipt is None:
            continue
        worker = receipt.get("worker") if isinstance(receipt.get("worker"), dict) else {}
        rec = lift_receipt(receipt, s, st["gen"], None, _bench_int(worker.get("bench")), None)
        if rec is not None and rec["body"]["state"] == "succeeded":
            out.append(rec)
    return out


def lift_verdicts() -> list:
    """Verdict records for the failure signatures `orchard consensus` already
    labelled flaky or load-sensitive here. The record holds only the
    signature's short hash, never the log line."""
    out = {}
    for v in consensus._memory():
        sig, label = v.get("signature"), v.get("label")
        if label in NOISE_LABELS and isinstance(sig, str) and sig:
            rec = make_record("verdict", {"signature_id": consensus.signature_id(sig), "label": label})
            out[record_id(rec)] = rec
    return [out[k] for k in sorted(out)]


def _write_new(records: list, have: set) -> list:
    """Store the records memory does not already hold; returns them."""
    new = {}
    for r in records:
        rid = record_id(r)
        if rid not in have:
            new[rid] = r
    if new:
        write_records(list(new.values()))
        have.update(new)
    return list(new.values())


# ---------------------------------------------------------------- read-only views: goals, steps, why, memory


def _holders() -> dict:
    """record id -> [(arrival index, source, origin)] for every via line whose
    segment holds it, oldest arrival first."""
    out = {}
    for i, v in enumerate(_via_lines()):
        try:
            data = (SEG_DIR / f"{v['segment']}.jsonl").read_bytes()
        except OSError:
            continue
        for rid, _ in _parse(data):
            out.setdefault(rid, []).append((i, v.get("source"), v.get("origin")))
    return out


def _live(records: list) -> list:
    recs = {record_id(r): r for r in records if well_formed(r)}
    out = refuted(recs)
    return [(rid, recs[rid]) for rid in sorted(recs) if rid not in out]


def _is_goal(r: dict) -> bool:
    """A goal record the planner would read: its id checks and its author signed it."""
    return (r["kind"] == "goal" and goal_id_of(r["body"]) == r["body"]["goal_id"]
            and r["body"]["author"] == r["key"])


def goal_of(records: list, goal_id: str) -> dict | None:
    """The goal record the planner reads for goal_id, or None."""
    for _, r in _live(records):
        if _is_goal(r) and r["body"]["goal_id"] == goal_id:
            return r
    return None


def goals_here(cwd: Path, records: list | None = None, tree: str | None = None) -> list:
    """This project's goals, newest first: goals on the tree as it is now lead,
    then by when their segment arrived here."""
    records = load_records() if records is None else records
    tree = tree or receipts.tree_digest(Path(cwd))[0]
    pid = project_id(Path(cwd), tree)
    first = {rid: hs[0][0] for rid, hs in _holders().items()}
    goals, seen = [], set()
    live = [(rid, r) for rid, r in _live(records) if _is_goal(r) and r["body"]["project"] == pid]
    live.sort(key=lambda x: (x[1]["body"]["tree"] == tree, first.get(x[0], -1), x[0]), reverse=True)
    for _, r in live:
        if r["body"]["goal_id"] not in seen:
            seen.add(r["body"]["goal_id"])
            goals.append(r)
    return goals


def resolve_goal(cwd: Path, ref: str | None, records: list | None = None) -> str:
    """A goal id from a prefix, or this project's newest goal."""
    records = load_records() if records is None else records
    if ref:
        ids = sorted({r["body"]["goal_id"] for _, r in _live(records)
                      if _is_goal(r) and r["body"]["goal_id"].startswith(ref)})
        if len(ids) != 1:
            raise MindError(f"mind: {'no goal' if not ids else 'more than one goal'} matches {ref!r}")
        return ids[0]
    goals = goals_here(cwd, records)
    if not goals:
        raise MindError("mind: no goal for this project in memory here; start one with `orchard mind run -- <cmd>`")
    return goals[0]["body"]["goal_id"]


def find_step(goal: dict, ref: str) -> dict:
    """A goal step by id prefix (4 hex or more) or by its command text."""
    steps = goal["body"]["steps"]
    hits = []
    if len(ref) >= 4 and all(c in "0123456789abcdef" for c in ref):
        hits = [s for s in steps if s["step_id"].startswith(ref)]
    if not hits:
        hits = [s for s in steps if ref in (shlex.join(s["cmd"]), " ".join(s["cmd"]))]
    if len(hits) != 1:
        raise MindError(f"mind: {'no step' if not hits else 'more than one step'} matches {ref!r}")
    return hits[0]


def _step_of(p: dict, step_id: str) -> dict:
    return next(st for st in p["steps"] if st["step"] == step_id)


def explain(goal_id: str, ref: str, records: list | None = None) -> dict:
    """Why one step is where it is: per body, fits or placement's exact
    problems; launch, lost (held or not) and result history by gen; and the
    results seen under keys the goal does not trust."""
    records = load_records() if records is None else records
    goal = goal_of(records, goal_id)
    if goal is None:
        raise MindError(f"mind: no goal {goal_id[:12]} in memory here")
    s = find_step(goal, ref)
    sid = s["step_id"]
    p = plan(records, goal_id)
    st = _step_of(p, sid)
    trust = set(p["trust"])
    bodies, history, untrusted = {}, [], []
    for rid, r in _live(records):
        kind, body = r["kind"], r["body"]
        if kind == "fit" and r["key"] == goal["key"] and body["goal_id"] == goal_id:
            b = bodies.setdefault(body["body_id"], {"body": body["body_id"], "fits": True, "problems": [],
                                                    "bench": body["caps"]["bench"]})
            b["fits"] = b["fits"] and sid in body["fits"]
            b["problems"] = sorted(set(b["problems"]) | set(body["problems"].get(sid, [])))
            continue
        if kind not in ("launch", "lost", "result", "retry") or body.get("step_id") != sid:
            continue
        if r["key"] not in trust:
            if kind == "result":
                untrusted.append({"gen": body["gen"], "state": body["state"], "key": r["key"],
                                  "status": verify_record(r)})
            continue
        entry = {"kind": kind, "gen": body["from_gen"] if kind == "retry" else body["gen"],
                 "body": body.get("body_id"), "record": rid}
        if kind == "lost":
            entry.update(why=body["why"], hold=body["hold"])
        elif kind == "result":
            entry.update(state=body["state"], elapsed_ms=body["elapsed_ms"])
        history.append(entry)
    order = {"retry": 0, "launch": 1, "lost": 2, "result": 3}
    history.sort(key=lambda e: (e["gen"], order[e["kind"]], e["record"]))
    return {"goal": goal_id, "step": sid, "cmd": shlex.join(s["cmd"]), "state": st["state"], "gen": st["gen"],
            "why": list(st["why"]), "capable": list(st["capable"]), "excluded": list(st["excluded"]),
            "bodies": [bodies[b] for b in sorted(bodies)], "history": history,
            "untrusted": sorted(untrusted, key=lambda u: (u["gen"], u["key"], u["state"]))}


def _retry_from(st: dict) -> int:
    """The gen a retry reopens a step at: past a pass or a failure; at the gen
    a stranded step is waiting at (it is one past the last attempt already)."""
    return st["gen"] + 1 if st["state"] in ("passed", "failed") else st["gen"]


def retry_step(goal_id: str, ref: str) -> dict:
    """Reopen a failed, stranded (or passed) step of this install's goal: a
    retry record makes every earlier gen history, exclusions included."""
    records = load_records()
    goal = own_goal(goal_id, records)
    s = find_step(goal, ref)
    st = _step_of(plan(records, goal_id), s["step_id"])
    if st["state"] == "running":
        raise MindError("mind: that step is running; let it finish, or let `orchard mind run` adopt it")
    if st["state"] == "pending":
        raise MindError("mind: that step is already waiting to run; nothing to retry")
    write_records([make_record("retry", {"step_id": s["step_id"], "from_gen": _retry_from(st),
                                         "at": int(time.time())})])
    after = _step_of(plan(load_records(), goal_id), s["step_id"])
    return {"goal": goal_id, "step": s["step_id"], "from_gen": _retry_from(st), "state": after["state"],
            "gen": after["gen"]}


def status_rows(cwd: Path) -> list:
    """This project's goals: state, step counts, whether a live session is
    working on it, and whether it is for the tree as it is now."""
    records = load_records()
    tree = receipts.tree_digest(Path(cwd))[0]
    watching = Counter()
    if SESSION_DIR.is_dir():
        for f in sorted(SESSION_DIR.glob("*.json")):
            info = _read_json(f) or {}
            if pid_alive(info.get("pid"), info.get("boot"), info.get("ident")):
                watching[info.get("goal")] += 1
    rows = []
    for g in goals_here(cwd, records, tree):
        gid = g["body"]["goal_id"]
        p = plan(records, gid)
        rows.append({"goal": gid, "state": p["state"], "total": len(p["steps"]),
                     "steps": dict(sorted(Counter(st["state"] for st in p["steps"]).items())),
                     "watching": watching[gid] > 0, "this_tree": g["body"]["tree"] == tree, "digest": p["digest"]})
    return rows


def memory_view(with_records: bool = False) -> dict:
    """memory_summary plus records by key (this install's marked) and, with
    with_records, every record with its status and the sources it came by."""
    out = memory_summary()
    recs = load_records()
    me = out["key_id"]
    out["by_key"] = [{"key": k, "mine": k == me, "records": n}
                     for k, n in sorted(Counter(r["key"] for r in recs).items())]
    if with_records:
        held = _holders()
        out["entries"] = [{"id": rid, "kind": r["kind"], "key": r["key"], "mine": r["key"] == me,
                           "status": verify_record(r), "sources": sorted({str(h[1]) for h in held.get(rid, [])})}
                          for rid, r in ((record_id(r), r) for r in recs)]
    return out


def body_ids(records: list) -> set:
    """Every body id the records and bodies.json mention."""
    ids = set(_load_bodies())
    for r in records:
        b = r["body"].get("body_id") if isinstance(r.get("body"), dict) else None
        if isinstance(b, str):
            ids.add(b)
    return ids


def public(obj, ids: set):
    """obj with every body id (as a value or a key) shortened to 8 hex: what
    --json prints. No name is ever in obj; this only makes ids readable."""
    if isinstance(obj, dict):
        return {(k[:8] if isinstance(k, str) and k in ids else k): public(v, ids) for k, v in obj.items()}
    if isinstance(obj, list):
        return [public(v, ids) for v in obj]
    if isinstance(obj, str) and obj in ids:
        return obj[:8]
    return obj


def bodies_view(ids: set) -> dict:
    """The local mapping --bodies adds: short body id -> target."""
    return {b[:8]: t for b, t in sorted(_load_bodies().items()) if b in ids}


def hide_names(msg: str, names: Iterable) -> str:
    """A session message with each worker name replaced by its body's short id
    (or "a worker" when no body id is known for it here)."""
    by_target = {t: b for b, t in _load_bodies().items()}
    for name in sorted(set(names), key=len, reverse=True):
        if not name or name == "local":
            continue
        label = f"body {by_target[name][:8]}" if name in by_target else "a worker"
        msg = re.sub(r"(?<![\w.-])" + re.escape(name) + r"(?![\w.-])", label, msg)
    return msg


# ---------------------------------------------------------------- export and import as one file
# `orchard mind export FILE` writes one JSON line per segment,
# {"segment": <sha256>, "data": <text>}; import checks each line's name
# against its bytes, the size cap and that every line is a well-formed record.


def export_file(path: Path) -> int:
    path = Path(path)
    lines = []
    for name in segment_names():
        try:
            data = (SEG_DIR / f"{name}.jsonl").read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if _segment_ok(name, data):
            lines.append(json.dumps({"segment": name, "data": text}, sort_keys=True, ensure_ascii=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{secrets.token_hex(4)}")
    tmp.write_text("".join(ln + "\n" for ln in lines))
    os.replace(tmp, path)
    return len(lines)


def _all_records(data: bytes) -> bool:
    return sum(1 for ln in data.splitlines() if ln.strip()) == len(_parse(data)) > 0


def import_file(path: Path) -> dict:
    """Store each segment in an export file as source "import", origin the
    sha256 of the file. Returns {segments, new, rejected, forged, origin}."""
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise MindError(f"mind: cannot read the import file ({e.strerror or e})") from e
    out = {"segments": 0, "new": 0, "rejected": 0, "forged": 0, "origin": hashlib.sha256(raw).hexdigest()}
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
            name, data = obj["segment"], obj["data"].encode("utf-8")
        except (ValueError, KeyError, TypeError, AttributeError):
            out["rejected"] += 1
            continue
        if not _is_hex(name, 64) or not _segment_ok(name, data) or not _all_records(data):
            out["rejected"] += 1
            continue
        line = store_segment(data, "import", out["origin"])
        out["segments"] += 1
        out["new"] += int(line["new"])
        out["forged"] += len(line["forged"])
    return out


# ---------------------------------------------------------------- mind run: one goal from a command line


def workers_here() -> dict:
    """This install's workers by name, as config has them."""
    return {w.name: w for w in config.load_workers()}


def _fresh_retries(p: dict) -> list:
    at = int(time.time())
    return [make_record("retry", {"step_id": st["step"], "from_gen": _retry_from(st), "at": at})
            for st in p["steps"] if st["state"] in ("passed", "failed", "stranded")]


def step_lines(steps: list, p: dict, records: list, before: set) -> list:
    """One entry per step, in the order given: its command, state, gen, the
    body whose result decided it, that result's elapsed time, and whether the
    result was already in memory when this run began."""
    by_rid = {record_id(r): r for r in records if well_formed(r)}
    planned = {st["step"]: st for st in p["steps"]}
    out = []
    for s in steps:
        st = planned.get(s["step_id"], {})
        res = by_rid.get(st.get("result") or "")
        out.append({"step": s["step_id"], "cmd": shlex.join(s["cmd"]), "state": st.get("state", "pending"),
                    "gen": st.get("gen", 0), "body": st.get("body"),
                    "elapsed_ms": res["body"]["elapsed_ms"] if res else None,
                    "from_memory": res is not None and st["result"] in before})
    return out


def sync_totals(res: dict) -> dict:
    out = Counter()
    for got in res.values():
        if "error" in got:
            out["unreachable"] += 1
        else:
            out.update({k: got[k] for k in ("pulled", "pushed", "bad")})
    return {k: out[k] for k in ("pulled", "pushed", "bad", "unreachable")}


def run_goal(cwd: Path, cmd: list | None, step_texts: list = (), *, use_speculate: bool = False,
             tags: list = (), env_names: list = (), timeout: int | None = DEFAULT_TIMEOUT_S,
             quiet_s: float = QUIET_DEFAULT_S, trust: list = (), reuse: bool = False, fresh: bool = False,
             plan_only: bool = False, sync_too: bool = False, say=None, workers: dict | None = None,
             clock=time.monotonic, sleep=time.sleep, max_ticks: int | None = None) -> dict:
    """`orchard mind run`: build the goal from the command line and store it,
    then answer from memory when it already passed or failed (no network at
    all), else run it to an end with execute. --reuse lifts sealed passing
    receipts first; --fresh reopens every finished step; --sync swaps memory
    with every worker before and after. --plan-only plans from memory plus
    the goal in hand: it writes nothing and reaches nothing, so no body has
    reported unless one already did for this goal."""
    if quiet_s < QUIET_MIN_S:
        raise MindError(f"mind: --quiet-after takes at least {QUIET_MIN_S} seconds")
    cwd = Path(cwd)
    checks = speculate.load(cwd) if use_speculate else []
    if use_speculate and not checks:
        raise MindError("mind: --speculate found no checks for this project; add one with `orchard speculate add`")
    specs = step_specs(cmd, list(step_texts), list(tags), list(env_names), timeout, checks)
    if not specs:
        raise MindError("mind: nothing to run; give a command after --, a --step, or --speculate")
    tree = receipts.tree_digest(cwd)[0]
    keyless = plan_only and not receipts.KEY_PATH.exists()  # a plan never makes the key
    steps, jobs = build_steps(cwd, tree, specs, keyless)
    goal = goal_record(cwd, tree, steps, list(trust), keyless)
    gid = goal["body"]["goal_id"]
    if plan_only:
        records = load_records() + [goal]
        p = plan(records, gid)
        state = final_state(p) if p["state"] == "stranded" else p["state"]
        return {"goal": gid, "state": state, "exit": 0, "plan_only": True, "from_memory": False,
                "coord_msgs": 0, "arbitration_writes": 0, "sense_calls": 0,
                "duplicate_launches": len(p["duplicates"]), "recover_s": [], "ledger": {}, "plan": p,
                "steps": step_lines(steps, p, records, {record_id(r) for r in records}),
                "sync": None, "sync_workers": None}
    workers = workers_here() if workers is None else workers
    pre = Counter()
    synced = {"before": sync(workers, pre) if sync_too else {}, "after": {}}
    records = load_records()
    have = {record_id(r) for r in records}
    _write_new([goal] + lift_verdicts(), have)
    if reuse:
        _write_new(lift_reusable(goal, plan(load_records(), gid)), have)
    p = plan(load_records(), gid)
    if fresh:
        _write_new(_fresh_retries(p), have)
        p = plan(load_records(), gid)
    before = set(have)
    if p["state"] in ("passed", "failed"):
        out = {"goal": gid, "state": p["state"], "exit": exit_code(p["state"]), "from_memory": True,
               "coord_msgs": 0, "arbitration_writes": 0, "sense_calls": 0,
               "duplicate_launches": len(p["duplicates"]), "recover_s": [], "ledger": dict(pre), "plan": p}
        events.append("mind.session", goal=gid, state=p["state"], exit=out["exit"], from_memory=True,
                      coord_msgs=0, arbitration_writes=0, sense_calls=0,
                      duplicate_launches=out["duplicate_launches"], recover_s=[])
    else:
        out = execute(gid, cwd, jobs, quiet_s=quiet_s, say=say, clock=clock, sleep=sleep, workers=workers,
                      survey=lambda ledger: survey(goal, jobs, workers, ledger), max_ticks=max_ticks)
        out["from_memory"] = False
    if sync_too:
        synced["after"] = sync(workers, pre)
    out["ledger"] = dict(Counter(out["ledger"]) + pre)
    out["plan_only"] = False
    out["steps"] = step_lines(steps, out["plan"], load_records(), before)
    out["sync"] = {k: sync_totals(v) for k, v in synced.items()} if sync_too else None
    out["sync_workers"] = synced if sync_too else None
    return out
