"""One mind: the record envelope, the write-once store and the pure planner.

Every test runs against an isolated home (its own receipt key and mind
folder) with sockets blocked: nothing here touches a real worker or the real
~/.orchard."""

import ast
import hashlib
import inspect
import json
import os
import random
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from orchard import agent, consensus, dispatcher, engine, env, events, fleet, mind, placement, receipts, transport
from orchard.config import Worker
from orchard.job import JobSpec, capture_env, new_job_id

K1 = "a" * 16  # the author install's key id in planner tests
K2 = "b" * 16  # another install
FAKE_MAC = "0" * 64  # the planner never checks macs: verification is not its input


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(receipts, "KEY_PATH", home / "receipt.key")
    monkeypatch.setattr(receipts, "INDEX_DIR", home / "state" / "receipts")
    mdir = home / "mind"
    monkeypatch.setattr(mind, "MIND_DIR", mdir)
    monkeypatch.setattr(mind, "SEG_DIR", mdir / "segments")
    monkeypatch.setattr(mind, "VIA_PATH", mdir / "via.jsonl")
    monkeypatch.setattr(mind, "BODIES_PATH", mdir / "bodies.json")
    monkeypatch.setattr(mind, "CLAIM_DIR", mdir / "claims")
    monkeypatch.setattr(mind, "SESSION_DIR", mdir / "sessions")
    monkeypatch.setattr(mind, "ZOMBIES_PATH", mdir / "zombies.jsonl")
    monkeypatch.setattr(mind, "LOG_DIR", mdir / "logs")
    monkeypatch.setattr(mind, "BODY_ID_PATH", home / "body.id")
    # A fixed local body id: a random one would race the fake worker's id
    # wherever a tie on bench falls back to body id order.
    home.mkdir(parents=True)
    (home / "body.id").write_text("f" * 32)
    return home


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("mind tests must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def other_install(tmp_path, monkeypatch, name="other"):
    """Switch this process to a second install: a different receipt key."""
    monkeypatch.setattr(receipts, "KEY_PATH", tmp_path / name / "receipt.key")


# ---------------------------------------------------------------- planner fixtures


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rec(kind, body, key=K1):
    return {"v": 1, "kind": kind, "key": key, "body": body, "mac": FAKE_MAC}


def step(name, env_names=(), tags=()):
    s = {"step_id": h("step " + name), "work_key": h("work " + name), "cmd": [name], "kind": "shell",
         "tags": list(tags), "requires": {}, "env_names": list(env_names), "timeout": 600}
    s["cmd_key"] = mind.cmd_key_of(s)
    return s


def goal(steps, trust=(), key=K1, author=None):
    body = {"author": author or key, "project": "tree:" + "c" * 16, "tree": "d" * 64, "trust": list(trust),
            "steps": sorted(steps, key=lambda s: s["step_id"])}
    body["goal_id"] = mind.goal_id_of(body)
    return rec("goal", body, key)


def gid(g):
    return g["body"]["goal_id"]


def fit(g, body_id, steps, bench=100, problems=None, key=K1):
    return rec("fit", {"goal_id": gid(g), "body_id": body_id,
                       "caps": {"os_family": "macos", "machine": "arm64", "cores": 8, "ram_gb": 16,
                                "bench": bench, "tags": []},
                       "fits": [s["step_id"] for s in steps], "problems": problems or {}}, key)


def launch(g, s, gen, body_id, key=K1, token="t1"):
    return rec("launch", {"goal_id": gid(g), "step_id": s["step_id"], "gen": gen, "body_id": body_id,
                          "token": token, "at": 100 + gen}, key)


def result(s, gen, body_id, state="succeeded", elapsed=1000, bench=100, sig=None, key=K1, work_key=None,
           cmd_key=None, step_id=None):
    return rec("result", {"step_id": step_id or s["step_id"], "gen": gen, "body_id": body_id, "state": state,
                          "exit_code": 0 if state == "succeeded" else 1, "elapsed_ms": elapsed, "bench": bench,
                          "work_key": work_key or s["work_key"], "tree": "d" * 64,
                          "cmd_key": cmd_key or s["cmd_key"], "log_sha256": None, "out_sha256": None,
                          "receipt_sha": h("receipt %s %d %s" % (s["step_id"], gen, state)), "os_family": "macos",
                          "signature_id": sig, "at": 200 + gen}, key)


def lost(s, gen, body_id, hold=False, why="went quiet", key=K1):
    return rec("lost", {"step_id": s["step_id"], "gen": gen, "body_id": body_id, "why": why, "hold": hold,
                        "at": 300 + gen}, key)


def retry(s, from_gen, key=K1):
    return rec("retry", {"step_id": s["step_id"], "from_gen": from_gen, "at": 400}, key)


def verdict(sig, label, key=K1):
    return rec("verdict", {"signature_id": sig, "label": label}, key)


def refute(target, key=K1, why="mac does not match"):
    return rec("refute", {"target": mind.record_id(target), "why": why}, key)


def st(p, s):
    return next(x for x in p["steps"] if x["step"] == s["step_id"])


BODY_A, BODY_B, BODY_C = "1" * 32, "2" * 32, "3" * 32


# ---------------------------------------------------------------- import, canonical, envelope


def test_import_has_no_side_effects(tmp_path):
    src = str(Path(mind.__file__).parents[1])
    env = {"HOME": str(tmp_path / "fresh"), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": src}
    (tmp_path / "fresh").mkdir()
    out = subprocess.run([sys.executable, "-c", "import orchard.mind, orchard.cli_agentic"], env=env,
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert list((tmp_path / "fresh").iterdir()) == []


def test_canonical_is_one_byte_form_and_refuses_non_json():
    assert mind.canonical({"b": 1, "a": [1, "é"]}) == b'{"a":[1,"\\u00e9"],"b":1}'
    assert mind.canonical({"a": 1, "b": 2}) == mind.canonical({"b": 2, "a": 1})
    for bad in (Path("/Users/x"), {1, 2}, float("nan"), object()):
        with pytest.raises((TypeError, ValueError)):
            mind.canonical({"x": bad})


def test_mac_round_trip_and_key_id(_isolated):
    kid = mind.key_id()
    assert len(kid) == 16 and all(c in "0123456789abcdef" for c in kid)
    assert kid == mind.key_id()
    assert receipts._key().hex()[:16] != kid  # the id does not reveal the key
    r = mind.make_record("verdict", {"signature_id": "s1", "label": "flaky"})
    assert r["key"] == kid and mind.well_formed(r)
    assert mind.verify_record(r) == "verified"
    assert mind.verify_record(json.loads(mind.canonical(r))) == "verified"


def test_tampered_record_is_forged():
    r = mind.make_record("verdict", {"signature_id": "s1", "label": "flaky"})
    body = dict(r, body={"signature_id": "s1", "label": "deterministic"})
    assert mind.verify_record(body) == "forged"
    assert mind.verify_record(dict(r, mac="f" * 64)) == "forged"
    assert mind.verify_record(dict(r, kind="refute")) == "forged"


def test_receipt_seal_is_not_a_mind_mac():
    # Same key, different domain: a receipt-style mac over the body does not verify.
    body = {"signature_id": "s1", "label": "flaky"}
    r = mind.make_record("verdict", body)
    assert mind.verify_record(dict(r, mac=receipts._mac(mind.canonical(body)))) == "forged"


def test_foreign_key_is_unverifiable(tmp_path, monkeypatch):
    mine = mind.make_record("verdict", {"signature_id": "s1", "label": "flaky"})
    other_install(tmp_path, monkeypatch)
    theirs = mind.make_record("verdict", {"signature_id": "s1", "label": "flaky"})
    assert theirs["key"] != mine["key"]
    assert mind.verify_record(theirs) == "verified"
    assert mind.verify_record(mine) == "unverifiable"
    assert mind.verify_record(dict(mine, mac="f" * 64)) == "unverifiable"


def test_make_record_refuses_bad_bodies():
    with pytest.raises(mind.MindError):
        mind.make_record("verdict", {"signature_id": "s1"})
    with pytest.raises(mind.MindError):
        mind.make_record("retry", {"step_id": "s", "from_gen": True, "at": 1})
    with pytest.raises(mind.MindError):
        mind.make_record("verdict", {"signature_id": "s1", "label": "x", "path": Path("/Users/x")})
    with pytest.raises(mind.MindError):
        mind.make_record("nonsense", {})


def test_well_formed_rejects_wrong_shapes():
    s = step("npm test")
    good = result(s, 1, BODY_A)
    assert mind.well_formed(good)
    assert not mind.well_formed(dict(good, extra=1))
    assert not mind.well_formed(dict(good, v=2))
    for v in (True, 1.0, "1"):  # equal to 1 in Python, but other bytes: another record id for one meaning
        assert not mind.well_formed(dict(good, v=v)), v
    assert not mind.well_formed(dict(good, kind=["result"]))
    assert not mind.well_formed(dict(good, key="NOTHEX0123456789"))
    assert not mind.well_formed(dict(good, body=dict(good["body"], gen=1.0)))
    assert not mind.well_formed(dict(good, body=dict(good["body"], gen=-1)))
    assert not mind.well_formed(dict(good, body={k: v for k, v in good["body"].items() if k != "state"}))
    assert not mind.well_formed(rec("lost", dict(lost(s, 1, BODY_A)["body"], hold=1)))
    assert not mind.well_formed(rec("fit", dict(fit(goal([s]), BODY_A, [s])["body"], caps={"bench": None})))
    assert not mind.well_formed("not a record")


# ---------------------------------------------------------------- identifiers


def test_ids_are_stable_and_free_of_names_and_paths(tmp_path, monkeypatch):
    work = {"cmd": ["npm", "test"], "kind": "shell", "python": None, "requirements": h(""), "tags": ["x86_64"],
            "requires": {}, "env": ["API_TOKEN"], "env_mac": "e" * 64}
    tree = "d" * 64
    sid = mind.step_id_of(tree, work)
    assert sid == mind.step_id_of(tree, dict(work))
    # env_mac is keyed per install; another install computes the same step id.
    assert sid == mind.step_id_of(tree, dict(work, env_mac="f" * 64))
    assert sid != mind.step_id_of("e" * 64, work)
    assert len(sid) == 64 and all(c in "0123456789abcdef" for c in sid)

    s = {"cmd": ["npm", "test"], "kind": "shell", "tags": ["b", "a"]}
    assert mind.cmd_key_of(s) == mind.cmd_key_of(dict(s, tags=["a", "b"], tree="anything"))
    assert len(mind.cmd_key_of(s)) == 16

    g = goal([step("npm test")])
    assert mind.goal_id_of(g["body"]) == gid(g)
    assert mind.goal_id_of(dict(g["body"], goal_id="whatever")) == gid(g)
    assert mind.goal_id_of(dict(g["body"], trust=[K2])) != gid(g)

    # project ids name no path: root commit hash, or the tree when there is no git.
    proj = tmp_path / "proj"
    proj.mkdir()
    assert mind.project_id(proj, tree) == "tree:" + tree[:16]
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], cwd=proj, check=True, timeout=60)
    (proj / "f").write_text("x")
    subprocess.run(git + ["add", "f"], cwd=proj, check=True, timeout=60)
    subprocess.run(git + ["commit", "-q", "-m", "init"], cwd=proj, check=True, timeout=60)
    pid = mind.project_id(proj, tree)
    assert pid.startswith("git:") and len(pid) == 20
    assert pid == mind.project_id(proj / ".", "0" * 64)
    blob = mind.canonical(mind.make_record("goal", dict(g["body"], project=pid)))
    for leak in (str(tmp_path).encode(), b"/Users/", os.environ.get("USER", "nobody-here").encode()):
        assert leak not in blob


# ---------------------------------------------------------------- store


def test_segments_are_write_once_and_named_by_hash(_isolated):
    recs = [mind.make_record("verdict", {"signature_id": "s%d" % i, "label": "flaky"}) for i in range(3)]
    name = mind.write_records(recs)
    path = mind.SEG_DIR / f"{name}.jsonl"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == name
    assert data.count(b"\n") == 3
    # Same set in any order: same segment, nothing rewritten, no second via line.
    assert mind.write_records(list(reversed(recs)) + recs[:1]) == name
    assert path.read_bytes() == data
    assert len(mind._via_lines()) == 1
    assert mind._link_once(path, b"something else") is False
    assert path.read_bytes() == data
    assert not [p for p in mind.SEG_DIR.iterdir() if p.name.startswith(".tmp-")]
    assert mind.write_records([]) is None

    via = mind._via_lines()[0]
    assert (via["source"], via["origin"], via["records"], via["verified"], via["forged"]) == ("self", None, 3, 3, [])
    assert {mind.record_id(r) for r in mind.load_records()} == {mind.record_id(r) for r in recs}

    # The same segment arriving from a worker is one more via line, not a new file.
    out = mind.store_segment(data, "sync", "w1")
    assert out["new"] is False and out["segment"] == name
    assert [(v["source"], v["origin"]) for v in mind._via_lines()] == [("self", None), ("sync", "w1")]
    mind.store_segment(data, "sync", "w1")
    assert len(mind._via_lines()) == 2


def test_store_refuses_oversize_and_unknown_source():
    with pytest.raises(mind.MindError):
        mind.store_segment(b"x" * (mind.SEGMENT_MAX_BYTES + 1), "sync", "w1")
    with pytest.raises(mind.MindError):
        mind.store_segment(b"", "carrier pigeon", None)
    with pytest.raises(mind.MindError):
        mind.write_records([{"kind": "goal"}])


def test_forged_record_is_refuted_at_ingest(_isolated):
    good = mind.make_record("verdict", {"signature_id": "s1", "label": "flaky"})
    forged = dict(good, body={"signature_id": "s2", "label": "flaky"})
    data = mind.canonical(good) + b"\n" + mind.canonical(forged) + b"\nnot json\n"
    out = mind.store_segment(data, "import", h("file"))
    assert out["new"] and out["verified"] == 1 and out["forged"] == [mind.record_id(forged)]
    records = mind.load_records()
    refutes = [r for r in records if r["kind"] == "refute"]
    assert [r["body"]["target"] for r in refutes] == [mind.record_id(forged)]
    assert mind.verify_record(refutes[0]) == "verified"
    assert mind.refuted({mind.record_id(r): r for r in records}) == {mind.record_id(forged)}


def test_foreign_records_count_as_other(tmp_path, monkeypatch, _isolated):
    other_install(tmp_path, monkeypatch)
    theirs = mind.make_record("verdict", {"signature_id": "s1", "label": "flaky"})
    monkeypatch.setattr(receipts, "KEY_PATH", _isolated / "receipt.key")
    out = mind.store_segment(mind.canonical(theirs) + b"\n", "sync", "w1")
    assert out["verified"] == 0 and out["forged"] == [] and out["other"] == {theirs["key"]: 1}
    assert not [r for r in mind.load_records() if r["kind"] == "refute"]


def test_found_segments_are_ingested_and_misnamed_files_skipped(_isolated):
    good = mind.make_record("verdict", {"signature_id": "s1", "label": "flaky"})
    forged = dict(good, body={"signature_id": "s9", "label": "flaky"})
    data = mind.canonical(good) + b"\n" + mind.canonical(forged) + b"\n"
    mind.SEG_DIR.mkdir(parents=True)
    (mind.SEG_DIR / f"{hashlib.sha256(data).hexdigest()}.jsonl").write_bytes(data)
    (mind.SEG_DIR / ("0" * 64 + ".jsonl")).write_bytes(mind.canonical(good) + b"\n")
    records = mind.load_records()
    kinds = sorted(r["kind"] for r in records)
    assert kinds == ["refute", "verdict", "verdict"]
    assert [v["source"] for v in mind._via_lines()] == ["found", "self"]
    assert len(mind.load_records()) == 3  # nothing new on a second read
    assert len(mind._via_lines()) == 2


def test_bodies_merge_under_lock(_isolated):
    assert mind._load_bodies() == {}
    mind._save_bodies({BODY_A: "local"})
    mind._save_bodies({BODY_B: "w1"})
    assert mind._load_bodies() == {BODY_A: "local", BODY_B: "w1"}
    mind.BODIES_PATH.write_text("[1, 2]")
    assert mind._load_bodies() == {}


# ---------------------------------------------------------------- planner rules


def test_pure_functions_use_no_clock_files_env_or_orchard():
    banned = {"time", "os", "open", "receipts", "config", "subprocess", "random", "secrets", "Path", "datetime",
              "fcntl", "tempfile", "sys", "events", "socket"}
    for fn in mind.PURE:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)), feature_version=(3, 9))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not names & banned, (fn.__name__, names & banned)
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom, ast.Global))]


def test_unknown_goal():
    p = mind.plan([], "f" * 64)
    assert p["state"] == "unknown" and p["steps"] == [] and len(p["digest"]) == 64
    s = step("npm test")
    g = goal([s])
    bad = rec("goal", dict(g["body"], tree="e" * 64))  # goal_id no longer matches content
    assert mind.plan([bad], gid(g))["state"] == "unknown"


def test_pending_then_running_then_passed():
    s = step("npm test")
    g = goal([s])
    base = [g, fit(g, BODY_A, [s])]
    p = mind.plan(base, gid(g))
    assert (p["state"], st(p, s)["state"], st(p, s)["gen"], st(p, s)["capable"]) == ("running", "pending", 1,
                                                                                    [BODY_A])
    assert p["queues"] == {BODY_A: [s["step_id"]]}
    p = mind.plan(base + [launch(g, s, 1, BODY_A)], gid(g))
    assert (st(p, s)["state"], st(p, s)["body"]) == ("running", BODY_A)
    ok = result(s, 1, BODY_A)
    p = mind.plan(base + [launch(g, s, 1, BODY_A), ok], gid(g))
    assert p["state"] == "passed"
    assert (st(p, s)["state"], st(p, s)["result"], st(p, s)["gen"]) == ("passed", mind.record_id(ok), 1)


def test_no_fit_records_strands_with_a_reason():
    s = step("npm test")
    g = goal([s])
    p = mind.plan([g], gid(g))
    assert (p["state"], st(p, s)["state"], st(p, s)["why"]) == ("stranded", "stranded", [mind.NO_BODIES])


def test_success_wins_over_any_failure():
    s = step("npm test")
    g = goal([s])
    base = [g, fit(g, BODY_A, [s])]
    later_pass = result(s, 2, BODY_A)
    p = mind.plan(base + [result(s, 1, BODY_A, "failed"), later_pass], gid(g))
    assert st(p, s)["state"] == "passed" and st(p, s)["result"] == mind.record_id(later_pass)
    early_pass = result(s, 1, BODY_A)
    p = mind.plan(base + [early_pass, result(s, 2, BODY_B, "failed")], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("passed", 1)


def test_failure_is_terminal_at_highest_gen():
    s = step("npm test")
    g = goal([s])
    f2 = result(s, 2, BODY_A, "timed_out")
    p = mind.plan([g, fit(g, BODY_A, [s]), result(s, 1, BODY_A, "failed"), f2], gid(g))
    assert p["state"] == "failed"
    assert (st(p, s)["state"], st(p, s)["gen"], st(p, s)["result"]) == ("failed", 2, mind.record_id(f2))


def test_duplicate_launches_at_one_gen():
    s = step("npm test")
    g = goal([s])
    recs = [g, fit(g, BODY_A, [s]), fit(g, BODY_B, [s]), launch(g, s, 1, BODY_A, token="x"),
            launch(g, s, 1, BODY_B, token="y")]
    p = mind.plan(recs, gid(g))
    assert p["duplicates"] == [[s["step_id"], 1]]
    assert st(p, s)["state"] == "running"


def test_lost_gen_closes_and_moves_the_step():
    s = step("npm test")
    g = goal([s])
    base = [g, fit(g, BODY_A, [s]), fit(g, BODY_B, [s]), launch(g, s, 1, BODY_A)]
    p = mind.plan(base + [lost(s, 1, BODY_A)], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"], st(p, s)["capable"], st(p, s)["excluded"]) == (
        "pending", 2, [BODY_B], [BODY_A])
    # A kill at the lost gen is mind's own terminate, not a failure.
    p = mind.plan(base + [lost(s, 1, BODY_A), result(s, 1, BODY_A, "killed")], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("pending", 2)


def test_hard_lost_only_body_strands_with_reasons():
    s = step("swift test", tags=["x86_64"])
    g = goal([s])
    recs = [g, fit(g, BODY_A, [s]), fit(g, BODY_B, [], problems={s["step_id"]: ["needs tag x86_64"]}),
            launch(g, s, 1, BODY_A), lost(s, 1, BODY_A, why="went quiet")]
    p = mind.plan(recs, gid(g))
    assert p["state"] == "stranded"
    assert st(p, s)["state"] == "stranded"
    assert st(p, s)["why"] == ["needs tag x86_64", "went quiet"]
    assert st(p, s)["excluded"] == [BODY_A, BODY_B]


def test_held_body_returns_when_nothing_else_fits():
    s = step("npm test")
    g = goal([s])
    only = [g, fit(g, BODY_A, [s]), launch(g, s, 1, BODY_A), lost(s, 1, BODY_A, hold=True, why="owner active")]
    p = mind.plan(only, gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"], st(p, s)["capable"]) == ("pending", 2, [BODY_A])
    p = mind.plan(only + [fit(g, BODY_B, [s])], gid(g))
    assert st(p, s)["capable"] == [BODY_B]


def test_single_fail_in_noise_gets_one_rerun():
    s = step("npm test")
    g = goal([s])
    base = [g, fit(g, BODY_A, [s]), launch(g, s, 1, BODY_A), result(s, 1, BODY_A, "failed", sig="sig1")]
    assert st(mind.plan(base, gid(g)), s)["state"] == "failed"
    p = mind.plan(base + [verdict("sig1", "flaky")], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("pending", 2)
    assert st(mind.plan(base + [verdict("sig1", "load-sensitive")], gid(g)), s)["state"] == "pending"
    assert st(mind.plan(base + [verdict("sig1", "deterministic")], gid(g)), s)["state"] == "failed"
    # An untrusted verdict does not buy a rerun.
    assert st(mind.plan(base + [verdict("sig1", "flaky", key=K2)], gid(g)), s)["state"] == "failed"
    # The second failure is terminal even in noise.
    p = mind.plan(base + [verdict("sig1", "flaky"), launch(g, s, 2, BODY_A),
                          result(s, 2, BODY_A, "failed", sig="sig1")], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("failed", 2)


def test_max_gen_strands():
    s = step("npm test")
    g = goal([s])
    recs = [g, fit(g, BODY_A, [s]), fit(g, BODY_B, [s])]
    for gen in range(1, mind.MAX_GEN):
        recs += [launch(g, s, gen, BODY_A), lost(s, gen, BODY_A, hold=True)]
    p = mind.plan(recs, gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("pending", mind.MAX_GEN)
    recs += [launch(g, s, mind.MAX_GEN, BODY_B), lost(s, mind.MAX_GEN, BODY_B, hold=True)]
    p = mind.plan(recs, gid(g))
    assert p["state"] == "stranded"
    assert (st(p, s)["state"], st(p, s)["why"]) == ("stranded", ["gave up after 6 attempts"])


def test_retry_sets_a_new_base():
    s = step("npm test")
    g = goal([s])
    base = [g, fit(g, BODY_A, [s]), launch(g, s, 1, BODY_A), result(s, 1, BODY_A)]
    assert st(mind.plan(base, gid(g)), s)["state"] == "passed"
    p = mind.plan(base + [retry(s, 2)], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("pending", 2)
    # Attempts are counted from the base, so a retried step gets MAX_GEN more.
    recs = base + [retry(s, 7)]
    for gen in range(7, 7 + mind.MAX_GEN - 1):
        recs += [launch(g, s, gen, BODY_A), lost(s, gen, BODY_A, hold=True)]
    assert st(mind.plan(recs, gid(g)), s)["state"] == "pending"
    assert st(mind.plan(recs + [lost(s, 12, BODY_A, hold=True)], gid(g)), s)["state"] == "stranded"
    # An untrusted retry moves nothing.
    assert st(mind.plan(base + [retry(s, 2, key=K2)], gid(g)), s)["state"] == "passed"


def test_env_steps_need_a_matching_work_key():
    s = step("deploy check", env_names=["API_TOKEN"])
    g = goal([s])
    other_env = result(s, 1, BODY_A, work_key=h("different env values"))
    p = mind.plan([g, fit(g, BODY_A, [s]), other_env], gid(g))
    # It does not pass the step, but it ends gen 1: the next launch is gen 2.
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("pending", 2)
    # The launch it answers is over too, so the step does not stay running.
    p = mind.plan([g, fit(g, BODY_A, [s]), launch(g, s, 1, BODY_A), other_env], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("pending", 2)
    assert mind.record_id(other_env) in p["inputs"]
    mine = result(s, 2, BODY_A)
    p = mind.plan([g, fit(g, BODY_A, [s]), launch(g, s, 1, BODY_A), other_env, launch(g, s, 2, BODY_A), mine],
                  gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"], st(p, s)["result"]) == ("passed", 2, mind.record_id(mine))


def test_a_goal_names_its_author_and_a_copy_under_another_key_is_ignored():
    s = step("npm test")
    g = goal([s])
    # The same steps on two installs are two goals, not one id two authors fight over.
    assert gid(goal([s], key=K1)) != gid(goal([s], key=K2))
    # A goal claimed by a key that did not make it is no goal at all.
    assert mind.plan([goal([s], key=K2, author=K1)], gid(g))["state"] == "unknown"
    # Copies of the author's goal under other keys, some with lower record ids
    # than the real one, cannot take it over: the author's records still count.
    forged = [rec("goal", g["body"], key=f"{i:016x}") for i in range(64)]
    forged = [f for f in forged if mind.record_id(f) < mind.record_id(g)]
    assert forged
    recs = [*forged, g, fit(g, BODY_A, [s]), launch(g, s, 1, BODY_A), result(s, 1, BODY_A)]
    p = mind.plan(recs, gid(g))
    assert (p["state"], st(p, s)["state"]) == ("passed", "passed")
    assert mind.plan(list(reversed(recs)), gid(g))["digest"] == p["digest"]


def test_a_trusted_launch_does_not_keep_a_step_running():
    s = step("npm test")
    g = goal([s], trust=[K2])
    base = [g, fit(g, BODY_A, [s])]
    # Another install's launch is not watched by this one: it never ends, so it
    # must not hold the step running. Its gen is spent, so the next is gen 2.
    p = mind.plan(base + [launch(g, s, 1, BODY_A, key=K2)], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("pending", 2)
    mine = [launch(g, s, 2, BODY_A), result(s, 2, BODY_A)]
    p = mind.plan(base + [launch(g, s, 1, BODY_A, key=K2)] + mine, gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("passed", 2)
    # The author's own launch does hold it running.
    p = mind.plan(base + [launch(g, s, 1, BODY_A)], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("running", 1)


def test_refute_only_by_the_same_key():
    s = step("npm test")
    g = goal([s], trust=[K2])
    ok = result(s, 1, BODY_A)
    base = [g, fit(g, BODY_A, [s]), ok]
    assert st(mind.plan(base, gid(g)), s)["state"] == "passed"
    r1 = refute(ok, key=K1)
    p = mind.plan(base + [r1], gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"]) == ("pending", 1)
    assert mind.record_id(r1) in p["inputs"] and mind.record_id(ok) not in p["inputs"]
    # A trusted but different key cannot refute this install's record.
    assert st(mind.plan(base + [refute(ok, key=K2)], gid(g)), s)["state"] == "passed"
    # A refute of the refute (same key) brings the result back.
    r2 = refute(r1, key=K1)
    p = mind.plan(base + [r1, r2], gid(g))
    assert st(p, s)["state"] == "passed"
    assert mind.record_id(r2) in p["inputs"]


def test_long_refute_chain_does_not_overflow():
    s = step("npm test")
    g = goal([s])
    ok = result(s, 1, BODY_A)
    chain = [ok]
    for _ in range(3000):
        chain.append(refute(chain[-1]))
    out = mind.refuted({mind.record_id(r): r for r in chain})
    # Odd links are out: ok is refuted by link 1, which is refuted by link 2, ...
    assert mind.record_id(chain[-2]) in out and mind.record_id(chain[-1]) not in out
    assert mind.record_id(ok) not in out  # 3000 links: an even count, so the result stands
    assert st(mind.plan([g, fit(g, BODY_A, [s])] + chain, gid(g)), s)["state"] == "passed"


def test_untrusted_records_only_add_work():
    s = step("npm test")
    g = goal([s])
    base = [g, fit(g, BODY_A, [s])]
    noise = [result(s, 1, BODY_A, key=K2), launch(g, s, 1, BODY_A, key=K2), lost(s, 1, BODY_A, key=K2),
             fit(g, BODY_B, [s], key=K2), retry(s, 3, key=K2)]
    p = mind.plan(base + noise, gid(g))
    assert (st(p, s)["state"], st(p, s)["gen"], st(p, s)["capable"]) == ("pending", 1, [BODY_A])
    assert p["digest"] == mind.plan(base, gid(g))["digest"]
    # Trust is goal data: the same pass counts once the goal trusts its key.
    g2 = goal([s], trust=[K2])
    p = mind.plan([g2, fit(g2, BODY_A, [s]), result(s, 1, BODY_A, key=K2)], gid(g2))
    assert st(p, s)["state"] == "passed"


def test_only_the_author_fits_count_and_fits_intersect():
    s1, s2 = step("npm test"), step("npm run lint")
    g = goal([s1, s2], trust=[K2])
    recs = [g, fit(g, BODY_A, [s1, s2]), fit(g, BODY_A, [s1], bench=50), fit(g, BODY_B, [s1, s2], key=K2)]
    p = mind.plan(recs, gid(g))
    assert st(p, s1)["capable"] == [BODY_A]
    assert st(p, s2)["state"] == "stranded"


# ---------------------------------------------------------------- assignment and digest


def test_lpt_assignment():
    s1, s2, s3 = step("long"), step("medium"), step("short")
    g = goal([s1, s2, s3])
    hist = [result(s, 1, BODY_C, elapsed=ms, bench=200, step_id=h("old tree " + s["cmd"][0]))
            for s, ms in ((s1, 3000), (s2, 2000), (s3, 1000))]
    recs = [g, fit(g, BODY_A, [s1, s2, s3], bench=200), fit(g, BODY_B, [s1, s2, s3], bench=100)] + hist
    p = mind.plan(recs, gid(g))
    # work = elapsed * bench: 600000, 400000, 200000. A (bench 200) takes long
    # (3000), B takes medium (4000 vs A's 5000), A takes short (4000 vs 6000).
    assert p["queues"] == {BODY_A: [s1["step_id"], s3["step_id"]], BODY_B: [s2["step_id"]]}
    assert p["costs"][s1["step_id"]] == {"work": 600000, "pred": {BODY_A: 3000, BODY_B: 6000}}
    assert set(p["inputs"]) >= {mind.record_id(r) for r in hist}


def test_running_step_leads_its_queue_and_adds_load():
    s1, s2 = step("long"), step("short")
    g = goal([s1, s2])
    recs = [g, fit(g, BODY_A, [s1, s2], bench=100), fit(g, BODY_B, [s1, s2], bench=100),
            launch(g, s2, 1, BODY_A)]
    p = mind.plan(recs, gid(g))
    # No history: work = DEFAULT_MS * median bench for both, so s1 goes to idle B.
    assert p["queues"] == {BODY_A: [s2["step_id"]], BODY_B: [s1["step_id"]]}
    assert p["costs"][s1["step_id"]]["work"] == mind.DEFAULT_MS * 100


def test_assign_ties_prefer_faster_then_lower_id():
    pending = [{"step": "s1", "cmd_key": "k", "capable": [BODY_A, BODY_B]}]
    q, _ = mind._assign(pending, [], {BODY_A: set(), BODY_B: set()}, {BODY_A: 100, BODY_B: 100}, {})
    assert q == {BODY_A: ["s1"]}
    q, _ = mind._assign(pending, [], {BODY_A: set(), BODY_B: set()}, {BODY_A: 100, BODY_B: 101}, {"k": [0]})
    assert q == {BODY_B: ["s1"]}
    assert mind.lower_median([4, 1, 3, 2]) == 2 and mind.lower_median([]) is None


def test_overall_state_order():
    a, b = step("a"), step("b")
    g = goal([a, b])
    fits = [g, fit(g, BODY_A, [a])]
    assert mind.plan(fits + [result(a, 1, BODY_A, "failed")], gid(g))["state"] == "stranded"
    fits = [g, fit(g, BODY_A, [a, b])]
    assert mind.plan(fits + [result(a, 1, BODY_A, "failed"), result(b, 1, BODY_A)], gid(g))["state"] == "failed"
    assert mind.plan(fits + [result(a, 1, BODY_A, "failed")], gid(g))["state"] == "running"


def test_digest_ignores_order_duplicates_and_unrelated_records():
    s1, s2 = step("npm test"), step("npm run lint")
    g = goal([s1, s2])
    recs = [g, fit(g, BODY_A, [s1, s2], bench=150), fit(g, BODY_B, [s1]), launch(g, s1, 1, BODY_A),
            lost(s1, 1, BODY_A), result(s2, 1, BODY_B, "failed", sig="z"), verdict("z", "flaky"),
            launch(g, s2, 2, BODY_B)]
    p = mind.plan(recs, gid(g))
    shuffled = recs + recs[:3]
    random.Random(7).shuffle(shuffled)
    other = goal([step("cargo test")])
    unrelated = [other, fit(other, BODY_C, []), verdict("q", "flaky"), result(step("x"), 1, BODY_C)]
    q = mind.plan(shuffled + unrelated, gid(g))
    for k in ("digest", "inputs", "inputs_digest", "steps", "queues"):
        assert p[k] == q[k], k
    d = {k: p[k] for k in ("v", "goal", "state", "steps", "queues", "duplicates")}
    assert p["digest"] == hashlib.sha256(mind.canonical(d)).hexdigest()
    assert p["inputs_digest"] == hashlib.sha256(mind.canonical(p["inputs"])).hexdigest()
    assert mind.record_id(unrelated[0]) not in p["inputs"]


def test_same_records_on_two_installs_give_the_same_digest(tmp_path, monkeypatch, _isolated):
    s = step("npm test")
    body = dict(goal([s])["body"])
    g = mind.make_record("goal", body)
    f = mind.make_record("fit", dict(fit(g, BODY_A, [s])["body"]))
    mind.write_records([g, f])
    here = mind.plan(mind.load_records(), body["goal_id"])
    seg = next(mind.SEG_DIR.glob("*.jsonl")).read_bytes()
    other_install(tmp_path, monkeypatch)
    monkeypatch.setattr(mind, "SEG_DIR", tmp_path / "other" / "mind" / "segments")
    monkeypatch.setattr(mind, "VIA_PATH", tmp_path / "other" / "mind" / "via.jsonl")
    out = mind.store_segment(seg, "sync", "w1")
    assert out["other"] == {g["key"]: 2}
    there = mind.plan(mind.load_records(), body["goal_id"])
    assert (there["digest"], there["inputs_digest"]) == (here["digest"], here["inputs_digest"])


# ================================================================ stage 2: sensing, sessions, claims, sync
# Remote bodies are folders (FakeFleet) where the REAL SENSE_SCRIPT runs under
# a local python with HOME pointed at the body. The launcher is FakeLauncher:
# it makes the job folder `orchard run` would and seals with the REAL
# receipts. Time is FakeClock. Nothing reaches a real worker or ~/.orchard.

W1 = "1" * 32  # the fake worker body's id
IN_USE = "worker in use (idle 3s < 180s)"  # dispatcher.idle_hold's text for an active owner
WORKER_HOST = "10.0.0.9"
DEAD_PID = 4000999  # no such process on any test machine (pids stop far below this)


@pytest.fixture(autouse=True)
def _isolated_runtime(_isolated, monkeypatch):
    """The rest of what mind's runtime writes, under the isolated home too."""
    monkeypatch.setattr(events, "EVENTS_PATH", _isolated / "state" / "events.jsonl")
    monkeypatch.setattr(fleet, "JOBS_DIR", _isolated / "jobs")
    monkeypatch.setattr(consensus, "MEMORY_PATH", _isolated / "state" / "consensus.jsonl")


class _Over:
    """A module stand-in: the attributes given here, the real module's for the rest."""

    def __init__(self, real, **attrs):
        self._real = real
        self.__dict__.update(attrs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeProc:
    """A launcher child that is not a process: a pid nothing owns, and a
    returncode the fake launcher or a kill sets."""

    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("launcher", timeout)
        return self.returncode


class FakeFleet:
    """Workers as folders: worker w's HOME is <root>/<w>. A body in `down`
    raises RemoteError like an unreachable ssh."""

    def __init__(self, root, names=("w1",)):
        self.root = root
        self.down = set()
        self.holds = {}      # target -> idle-hold reason
        self.problems = {}   # target -> placement problems
        self.terminated = []
        self.workers = {}
        for i, name in enumerate(names):
            self.workers[name] = Worker(name, host=WORKER_HOST)
            (self.home(name) / ".orchard").mkdir(parents=True)
            (self.home(name) / ".orchard" / "body.id").write_text(str(i + 1) * 32)

    def home(self, name):
        return self.root / name

    def _up(self, worker):
        if worker.name in self.down:
            raise transport.RemoteError(f"{worker.name} does not answer")
        return self.home(worker.name)

    def caps(self, name):
        return {"name": name, "bench": 100, "tags": ["os:macos"], "ram_gb": 16, "disk_free_gb": 100,
                "platform_ok": True}

    # what a launcher leaves on a body
    def write_status(self, name, job_id, status):
        d = self.home(name) / ".orchard" / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "status.json").write_text(json.dumps(status))

    def write_marker(self, name, job_id, pgid, boot):
        d = self.home(name) / ".orchard" / "state" / "running"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{job_id}.json").write_text(json.dumps({"pgid": pgid, "boot_id": boot}))

    # transport
    def run_script(self, worker, script, script_args, timeout=30):
        home = self._up(worker)
        return subprocess.run([sys.executable, "-", *script_args], input=script, capture_output=True,
                              env=dict(os.environ, HOME=str(home)), timeout=timeout)

    def run(self, worker, remote_cmd, stdin=None, timeout=30, **_):
        home = self._up(worker)
        return subprocess.run(["sh", "-c", remote_cmd], input=stdin, capture_output=True, cwd=home,
                              env=dict(os.environ, HOME=str(home)), timeout=timeout)

    def run_checked(self, worker, remote_cmd, stdin=None, timeout=30, **_):
        proc = self.run(worker, remote_cmd, stdin, timeout)
        if proc.returncode != 0:
            raise transport.RemoteError(proc.stderr.decode("utf-8", "replace"))
        return proc.stdout.decode("utf-8", "replace")

    def pull_bytes(self, worker, path):
        self._up(worker)
        try:
            return Path(path).read_bytes()
        except OSError as e:
            raise transport.RemoteError(str(e)) from e

    def push_bytes(self, worker, path, data):
        self._up(worker)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)

    # engine
    def remote_caps(self, worker):
        self._up(worker)
        return self.caps(worker.name)

    def remote_home(self, worker):
        return str(self._up(worker))

    def remote_status(self, worker, rjob):
        self._up(worker)
        try:
            return json.loads((Path(rjob) / "status.json").read_text())
        except (OSError, ValueError):
            return None

    def remote_status_missing(self, worker, rjob):
        self._up(worker)
        return not (Path(rjob) / "status.json").exists()

    def collect_remote(self, worker, job_id, rjob, sdir, cwd, artifacts=None):
        self._up(worker)
        for f in ("log.txt", "status.json"):
            if (Path(rjob) / f).exists():
                (Path(sdir) / f).write_bytes((Path(rjob) / f).read_bytes())
        return agent.read_status(str(sdir))

    def terminate_remote(self, worker, rjob):
        self._up(worker)
        self.terminated.append((worker.name, rjob))
        return True


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Swap every module mind reaches the network, the dispatcher config, the
    benchmark or placement through."""
    ff = FakeFleet(tmp_path / "bodies")
    monkeypatch.setattr(mind, "transport", _Over(
        transport, run_script=ff.run_script, run=ff.run, run_checked=ff.run_checked, pull_bytes=ff.pull_bytes,
        push_bytes=ff.push_bytes))
    monkeypatch.setattr(mind, "engine", _Over(
        engine, remote_caps=ff.remote_caps, remote_home=ff.remote_home, remote_status=ff.remote_status,
        remote_status_missing=ff.remote_status_missing, collect_remote=ff.collect_remote,
        terminate_remote=ff.terminate_remote))
    monkeypatch.setattr(mind, "dispatcher", _Over(
        dispatcher, load_config=lambda: {}, idle_hold=lambda caps, cfg: ff.holds.get(caps.get("name"))))
    monkeypatch.setattr(mind, "env", _Over(
        env, ensure_bench_python=lambda *a, **k: None, resolve_for_worker=lambda *a, **k: (True, "")))
    monkeypatch.setattr(mind, "agent", _Over(agent, collect_caps=lambda *a, **k: ff.caps("local")))
    monkeypatch.setattr(mind, "placement", _Over(
        placement, problems=lambda job, caps: list(ff.problems.get(caps.get("name"), []))))
    return ff


class FakeLauncher:
    """Stands in for `orchard run` as mind._popen, and records kills as
    mind._killpg. Modes, per target in launch order (default "pass"):
    pass/fail seal at once; die makes no job; hang leaves the job running on
    the body; hang-down then takes the body down; orphan leaves a marker whose
    process group is gone; slow is still starting (inputs only, no status
    anywhere); moved seals a pass for other work than the step's."""

    def __init__(self, ff):
        self.fleet = ff
        self.modes = {}
        self.launched = []  # (target, name, mode)
        self.procs = []
        self.killed = []    # (pid, signal)

    def mode(self, target, *modes):
        self.modes.setdefault(target, []).extend(modes)

    def popen(self, argv, *, cwd, start_new_session=False, **_):
        assert start_new_session, "a launcher must run in its own session"
        cut = argv.index("--")
        opts, cmd = argv[argv.index("run") + 1:cut], argv[cut + 1:]
        target = name = timeout = None
        tags, env_names = [], []
        for flag, value in zip(opts[::2], opts[1::2]):
            if flag == "-w":
                target = value
            elif flag == "--name":
                name = value
            elif flag == "--timeout":
                timeout = int(value)
            elif flag == "-t":
                tags.append(value)
            elif flag == "--env":
                env_names.append(value)
        job = JobSpec(name=name, cmd=cmd, timeout_s=timeout, tags=tags, env=env_names)
        job.env_values, _ = capture_env(env_names)  # as `orchard run` does
        queue = self.modes.get(target) or []
        mode = queue.pop(0) if queue else "pass"
        self.launched.append((target, name, mode))
        proc = FakeProc(4000000 + len(self.launched))
        self.procs.append(proc)
        if mode == "die":
            proc.returncode = 1
            return proc
        sdir = fleet.JOBS_DIR / new_job_id(name)
        sdir.mkdir(parents=True)
        hashed = job if mode != "moved" else JobSpec(name=name, cmd=[*cmd, "moved"], timeout_s=timeout, tags=tags,
                                                     env=env_names)
        assert receipts.record_inputs(sdir, Path(cwd), hashed, target) is not None
        if mode == "slow":
            return proc
        remote = target != "local"
        if mode in ("pass", "fail", "moved"):
            ok = mode != "fail"
            (sdir / "log.txt").write_text("all good\n" if ok else "collecting\nError: boom in step\n")
            status = {"state": "succeeded" if ok else "failed", "exit_code": 0 if ok else 1, "elapsed_s": 1.5,
                      "machine": "x86_64" if remote else "arm64", "os_family": "macos"}
            (sdir / "status.json").write_text(json.dumps(status))
            if remote:
                self.fleet.write_status(target, sdir.name, status)
            assert receipts.seal(sdir, status, target, None) is not None
            proc.returncode = 0 if ok else 1
            return proc
        status = {"state": "running"}
        (sdir / "status.json").write_text(json.dumps(status))
        if remote:
            self.fleet.write_status(target, sdir.name, status)
            pgid = DEAD_PID if mode == "orphan" else os.getpgid(0)
            self.fleet.write_marker(target, sdir.name, pgid, agent.boot_id())
        if mode == "hang-down":
            self.fleet.down.add(target)
        return proc

    def killpg(self, pid, sig):
        procs = [p for p in self.procs if p.pid == pid]
        if not procs:
            raise ProcessLookupError(pid)
        self.killed.append((pid, sig))
        if procs[0].returncode is None:
            procs[0].returncode = -int(sig)


@pytest.fixture
def launcher(fake, monkeypatch):
    fl = FakeLauncher(fake)
    monkeypatch.setattr(mind, "_popen", fl.popen)
    monkeypatch.setattr(mind, "_killpg", fl.killpg)
    return fl


def as_home(tmp_path, monkeypatch, name):
    """Switch this process to install `name`: its own receipt key, mind folder
    and job folders."""
    home = tmp_path / name
    monkeypatch.setattr(receipts, "KEY_PATH", home / "receipt.key")
    monkeypatch.setattr(receipts, "INDEX_DIR", home / "state" / "receipts")
    mdir = home / "mind"
    for attr, path in (("MIND_DIR", mdir), ("SEG_DIR", mdir / "segments"), ("VIA_PATH", mdir / "via.jsonl"),
                       ("BODIES_PATH", mdir / "bodies.json"), ("CLAIM_DIR", mdir / "claims"),
                       ("SESSION_DIR", mdir / "sessions"), ("ZOMBIES_PATH", mdir / "zombies.jsonl"),
                       ("LOG_DIR", mdir / "logs"), ("BODY_ID_PATH", home / "body.id")):
        monkeypatch.setattr(mind, attr, path)
    monkeypatch.setattr(fleet, "JOBS_DIR", home / "jobs")
    return home


def project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "main.py").write_text("print('hi')\n")
    return proj


def make_goal(cwd, cmds):
    """A goal for cmds (one step each) on cwd's tree, signed and stored as
    `mind run` would. Returns (goal record, {step id: JobSpec})."""
    tree = receipts.tree_digest(cwd)[0]
    steps, jobs = [], {}
    for i, cmd in enumerate(cmds):
        job = JobSpec(name=f"s{i}", cmd=list(cmd))
        work = receipts.work_fields(job)
        s = {"step_id": mind.step_id_of(tree, work), "work_key": receipts.work_key(tree, work), "cmd": list(cmd),
             "kind": job.kind, "tags": list(job.tags), "requires": dict(job.requires), "env_names": list(job.env),
             "timeout": job.timeout_s or 0}
        s["cmd_key"] = mind.cmd_key_of(s)
        steps.append(s)
        jobs[s["step_id"]] = job
    body = {"author": mind.key_id(), "project": mind.project_id(cwd, tree), "tree": tree, "trust": [],
            "steps": sorted(steps, key=lambda s: s["step_id"])}
    body["goal_id"] = mind.goal_id_of(body)
    g = mind.make_record("goal", body)
    mind.write_records([g])
    return g, jobs


def add_bodies(g, benches, ff=None):
    """Fit records and local targets for {target: bench}; "local" is this machine."""
    mapping, recs = {}, []
    for target, bench in benches.items():
        b = mind.local_body_id() if target == "local" else (ff.home(target) / ".orchard" / "body.id").read_text()
        mapping[b] = target
        recs.append(mind.make_record("fit", {
            "goal_id": g["body"]["goal_id"], "body_id": b, "caps": {"bench": bench, "os_family": "macos"},
            "fits": [s["step_id"] for s in g["body"]["steps"]], "problems": {}}))
    mind._save_bodies(mapping)
    mind.write_records(recs)
    return {t: b for b, t in mapping.items()}


def run_goal(g, cwd, ff, clock=None, max_ticks=500, **kw):
    clock = clock or FakeClock()
    said = []
    out = mind.execute(g["body"]["goal_id"], cwd, workers=ff.workers, clock=clock, sleep=clock.sleep,
                       say=said.append, max_ticks=max_ticks, **kw)
    return out, said


def kinds(kind):
    return [r for r in mind.load_records() if r["kind"] == kind]


def logged(event):
    try:
        lines = events.EVENTS_PATH.read_text().splitlines()
    except OSError:
        return []
    return [e for e in map(json.loads, lines) if e.get("event") == event]


# ---------------------------------------------------------------- sensing


def test_sense_reports_running_done_orphan_and_absent(fake):
    boot = agent.boot_id()
    alive = os.getpgid(0)
    fake.write_status("w1", "j-run", {"state": "running"})
    fake.write_marker("w1", "j-run", alive, boot)
    fake.write_status("w1", "j-done", {"state": "succeeded"})
    fake.write_status("w1", "j-gone", {"state": "running"})
    fake.write_marker("w1", "j-gone", DEAD_PID, boot)
    fake.write_status("w1", "j-boot", {"state": "running"})
    fake.write_marker("w1", "j-boot", alive, "another-boot")
    fake.write_status("w1", "j-nomark", {"state": "running"})
    ledger = Counter()
    rep = mind.sense_remote(fake.workers["w1"], ["j-run", "j-done", "j-gone", "j-boot", "j-nomark", "j-none",
                                                 "../escape"], ledger)
    assert rep["body"] == W1 and isinstance(rep["cores"], int)
    assert {j: v["state"] for j, v in rep["jobs"].items()} == {
        "j-run": "running", "j-done": "done", "j-gone": "orphan", "j-boot": "orphan" if boot else "running",
        "j-nomark": "orphan", "j-none": "absent"}
    assert rep["jobs"]["j-done"]["status"] == "succeeded"
    assert ledger == Counter(sense=1)
    fake.down.add("w1")
    assert mind.sense_remote(fake.workers["w1"], ["j-run"], ledger) is None
    assert ledger == Counter(sense=2)


def test_sense_mints_a_body_id_once_on_a_fresh_body(fake):
    (fake.home("w1") / ".orchard" / "body.id").unlink()
    first = mind.sense_remote(fake.workers["w1"], [], Counter())["body"]
    assert mind._is_hex(first, 32) and first != W1
    assert mind.sense_remote(fake.workers["w1"], [], Counter())["body"] == first
    assert mind.local_body_id() == mind.local_body_id() != first


def test_launch_check_gives_why_and_hold(fake):
    job = JobSpec(name="t", cmd=["true"])
    w, ledger = fake.workers["w1"], Counter()
    assert mind.launch_check("w1", w, job, ledger) == (None, False, None)
    assert mind.launch_check("local", None, job, ledger) == (None, False, None)
    fake.holds["w1"] = IN_USE
    assert mind.launch_check("w1", w, job, ledger) == ("held: owner active", True, "held: " + IN_USE)
    fake.problems["w1"] = ["needs 64 GB RAM", "platform resolve failed (no wheel in /Users/x/cache)", "odd"]
    assert mind.launch_check("w1", w, job, ledger) == (
        "needs 64 GB RAM; platform resolve failed; placement problem", False,
        "needs 64 GB RAM; platform resolve failed (no wheel in /Users/x/cache); odd")
    fake.down.add("w1")
    assert mind.launch_check("w1", w, job, ledger) == (mind.UNREACHABLE, False, mind.UNREACHABLE)
    assert set(ledger) == {"caps"}


# ---------------------------------------------------------------- receipts become results


def test_lift_receipt_recomputes_the_step_from_the_receipt(tmp_path):
    cwd = project(tmp_path)
    g, jobs = make_goal(cwd, [["make", "test"], ["make", "lint"]])
    s_test, s_lint = sorted(g["body"]["steps"], key=lambda s: s["cmd"] != ["make", "test"])
    sdir = fleet.JOBS_DIR / new_job_id("lift")
    sdir.mkdir(parents=True)
    receipts.record_inputs(sdir, cwd, jobs[s_test["step_id"]], "local")
    (sdir / "log.txt").write_text("ok\n")
    receipt = receipts.seal(sdir, {"state": "succeeded", "exit_code": 0, "elapsed_s": 2.0, "os_family": "macos"},
                            "local", None)
    r = mind.lift_receipt(receipt, s_test, 3, "f" * 32, 150, None)
    assert mind.verify_record(r) == "verified"
    assert (r["body"]["step_id"], r["body"]["gen"], r["body"]["elapsed_ms"], r["body"]["bench"]) == (
        s_test["step_id"], 3, 2000, 150)
    assert mind.lift_receipt(receipt, s_lint, 3, "f" * 32, 150, None) is None   # other work
    assert mind.lift_receipt(dict(receipt, state="failed"), s_test, 3, None, 150, None) is None  # seal broken
    assert mind.lift_receipt(dict(receipt, seal="0" * 64), s_test, 3, None, 150, None) is None


# ---------------------------------------------------------------- sessions


def test_launch_to_result_on_this_machine(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"local": 100})
    out, said = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("passed", 0)
    assert [(t, m) for t, _, m in launcher.launched] == [("local", "pass")]
    st = out["plan"]["steps"][0]
    assert (st["state"], st["gen"], st["body"]) == ("passed", 1, ids["local"])
    (res,) = kinds("result")
    assert mind.verify_record(res) == "verified" and res["body"]["body_id"] == ids["local"]
    assert len(kinds("launch")) == 1 and not kinds("lost")
    assert out["coord_msgs"] == 0 and "coord" not in out["ledger"] and "launch" not in out["ledger"]
    assert not list(mind.SESSION_DIR.glob("*.json"))  # the session file goes with the session
    assert any(s.startswith("launched") for s in said) and any(s.startswith("succeeded") for s in said)
    (ev,) = logged("mind.session")
    assert (ev["state"], ev["exit"], ev["coord_msgs"]) == ("passed", 0, 0)


def test_launch_to_result_on_a_worker(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"w1": 100}, fake)
    out, _ = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("passed", 0)
    assert [t for t, _, _ in launcher.launched] == ["w1"]
    assert out["ledger"]["launch"] == 1 and out["ledger"]["caps"] == 1 and out["coord_msgs"] == 0
    (res,) = kinds("result")
    assert res["body"]["body_id"] == W1 and res["body"]["os_family"] == "macos"


def test_a_failing_step_fails_the_goal_with_a_signature(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100})
    launcher.mode("local", "fail")
    out, _ = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("failed", 1)
    (res,) = kinds("result")
    assert res["body"]["state"] == "failed" and mind._is_hex(res["body"]["signature_id"], 12)


def test_lost_launch_is_retried_at_the_next_gen_on_another_body(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"local": 200, "w1": 100}, fake)  # this machine is faster: it goes first
    launcher.mode("local", "die")
    out, said = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("passed", 0)
    assert [(t, m) for t, _, m in launcher.launched] == [("local", "die"), ("w1", "pass")]
    (lost_rec,) = kinds("lost")
    assert (lost_rec["body"]["gen"], lost_rec["body"]["body_id"], lost_rec["body"]["why"],
            lost_rec["body"]["hold"]) == (1, ids["local"], "launcher died", False)
    (res,) = kinds("result")
    assert (res["body"]["gen"], res["body"]["body_id"]) == (2, W1)
    assert len(out["recover_s"]) == 1 and out["recover_s"][0] >= 0
    (ev,) = logged("mind.recovered")
    assert (ev["step"], ev["gen"], ev["goal"]) == (res["body"]["step_id"], 2, g["body"]["goal_id"])
    assert any(s.startswith("lost") and "launcher died" in s for s in said)


def test_held_body_hands_the_step_to_another(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"local": 100, "w1": 200}, fake)  # the worker is preferred, but its owner is active
    fake.holds["w1"] = IN_USE
    out, _ = run_goal(g, cwd, fake)
    assert out["state"] == "passed"
    assert [t for t, _, _ in launcher.launched] == ["local"]
    (lost_rec,) = kinds("lost")
    assert (lost_rec["body"]["body_id"], lost_rec["body"]["why"], lost_rec["body"]["hold"]) == (
        W1, "held: owner active", True)
    assert kinds("result")[0]["body"]["body_id"] == ids["local"]


def test_a_sole_held_body_waits_without_a_record(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"w1": 100}, fake)
    fake.holds["w1"] = IN_USE
    out, said = run_goal(g, cwd, fake, max_ticks=40)
    assert (out["state"], out["exit"]) == ("detached", 130)
    assert launcher.launched == [] and not kinds("lost") and not kinds("launch")
    assert 2 <= out["ledger"]["caps"] <= 4  # rechecked at the sense cadence, not every tick
    assert any(s.startswith("waiting") for s in said)


def test_unreachable_body_at_launch_is_lost_and_strands_a_sole_body(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"w1": 100}, fake)
    fake.down.add("w1")
    out, _ = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("stranded", 3)
    assert out["plan"]["steps"][0]["why"] == [mind.UNREACHABLE]
    assert launcher.launched == []


def test_orphan_is_confirmed_twice_before_the_launcher_is_dropped(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"w1": 100}, fake)
    launcher.mode("w1", "orphan")
    out, said = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("stranded", 3)
    assert out["sense_calls"] == mind.ORPHAN_CONFIRM == 2
    (lost_rec,) = kinds("lost")
    assert lost_rec["body"]["why"] == "body restarted or job died" and lost_rec["body"]["hold"] is False
    pid = launcher.procs[0].pid
    assert launcher.killed == [(pid, mind.signal.SIGTERM), (pid, mind.signal.SIGKILL)]
    assert mind._zombie_lines() == []  # the body answered: nothing is left to reap
    assert out["plan"]["steps"][0]["why"] == ["body restarted or job died"]


def test_quiet_body_is_dropped_zombied_and_reaped_later(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"w1": 100}, fake)
    launcher.mode("w1", "hang-down")
    clock = FakeClock()
    out, _ = run_goal(g, cwd, fake, clock=clock, quiet_s=60)
    assert (out["state"], out["exit"]) == ("stranded", 3)
    assert 60 <= clock.t - 1000 <= 62
    (lost_rec,) = kinds("lost")
    assert lost_rec["body"]["why"] == "quiet for 60s"
    pid = launcher.procs[0].pid
    assert launcher.killed == [(pid, mind.signal.SIGTERM), (pid, mind.signal.SIGKILL)]
    (z,) = mind._zombie_lines()
    job_id = fleet.find_job(launcher.launched[0][1]).name
    assert (z["target"], z["job"], z["rjob"]) == ("w1", job_id, None)
    assert mind.reap(fake.workers, Counter()) == 0  # still down: kept
    assert len(mind._zombie_lines()) == 1
    fake.down.clear()
    ledger = Counter()
    assert mind.reap(fake.workers, ledger) == 1
    assert fake.terminated == [("w1", f"{fake.home('w1')}/.orchard/jobs/{job_id}")]
    assert mind._zombie_lines() == [] and ledger == Counter(terminate=2)


def test_tree_change_stops_the_session(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100})
    out, _ = run_goal(g, cwd, fake, tree_now=lambda: "e" * 64)
    assert (out["state"], out["exit"]) == ("tree_changed", 4)
    assert launcher.launched == [] and not kinds("launch")


def test_no_reports_and_no_steps_are_usage(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    out, _ = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("usage", 2)
    empty, _ = make_goal(cwd, [])
    assert run_goal(empty, cwd, fake)[0]["exit"] == 2
    assert launcher.launched == []


def test_plan_only_launches_nothing(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100})
    out, _ = run_goal(g, cwd, fake, plan_only=True)
    assert out["state"] == "running" and launcher.launched == [] and not kinds("launch")


def test_ctrl_c_detaches_without_writing(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"w1": 100}, fake)
    launcher.mode("w1", "hang")
    clock = FakeClock()

    def interrupt(s):
        clock.sleep(s)
        if clock.t > 1003:
            raise KeyboardInterrupt

    before = len(mind.load_records())
    out = mind.execute(g["body"]["goal_id"], cwd, workers=fake.workers, clock=clock, sleep=interrupt)
    assert (out["state"], out["exit"]) == ("detached", 130)
    assert len(mind.load_records()) == before + 1  # the launch record only
    assert launcher.killed == []  # the launcher keeps running in its own session


def test_sessions_never_send_coordination_messages():
    with pytest.raises(mind.MindError):
        mind._net(Counter(), "coord", lambda: None)
    assert "coord" not in mind.NET_PURPOSES
    purposes = set()
    for node in ast.walk(ast.parse(inspect.getsource(mind))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_net":
            assert isinstance(node.args[1], ast.Constant), ast.unparse(node)
            purposes.add(node.args[1].value)
    assert purposes and purposes <= set(mind.NET_PURPOSES)


# ---------------------------------------------------------------- claims and adoption


def test_pid_alive_is_false_only_when_known_dead(monkeypatch):
    me = os.getpid()
    ident = mind._proc_identity(me)
    assert ident
    monkeypatch.setattr(mind, "agent", _Over(agent, boot_id=lambda: "boot-a"))
    assert mind.pid_alive(me, "boot-a", ident) and mind.pid_alive(me, None, None)
    assert not mind.pid_alive(me, "boot-b", ident)          # rebooted since
    assert not mind.pid_alive(me, "boot-a", ident + "x")    # same number, another process: reused
    for bad in (None, 0, -1, True, "123", 2 ** 80, DEAD_PID):
        assert not mind.pid_alive(bad, None, None)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not mind.pid_alive(child.pid, None, None)        # exited and reaped
    monkeypatch.setattr(mind, "_proc_identity", lambda pid: None)
    assert mind.pid_alive(me, "boot-a", "lanother start")   # cannot tell: counts as alive


def test_racing_claims_have_exactly_one_winner():
    sid, n = "e" * 64, 12
    gate, got = threading.Barrier(n), []

    def race(i):
        gate.wait()
        got.append((i, mind.claim(sid, 1, {"session": f"s{i}", "pid": os.getpid()})))

    threads = [threading.Thread(target=race, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [i for i, ok in got if ok]
    assert len(got) == n and len(winners) == 1
    assert mind.current_holder(sid, 1) == (0, {"session": f"s{winners[0]}", "pid": os.getpid()})


def test_racing_adopters_of_a_dead_holder_have_one_winner():
    sid, n = "e" * 64, 8
    assert mind.claim(sid, 1, {"session": "dead", "pid": DEAD_PID, "boot": None, "ident": None, "job_name": "j"})
    me = mind.new_session()
    gate, got = threading.Barrier(n), []

    def race(i):
        gate.wait()
        got.append((i, mind.adopt(sid, 1, dict(me, session=f"s{i}"))))

    threads = [threading.Thread(target=race, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [i for i, ok in got if ok]
    assert len(winners) == 1
    k, holder = mind.current_holder(sid, 1)
    assert (k, holder["session"], holder["adopted_from"], holder["job_name"]) == (1, f"s{winners[0]}", 0, "j")
    assert not mind.adopt(sid, 1, mind.new_session())  # the new holder is this live process


def test_a_live_holder_keeps_its_claim(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100})
    sid = g["body"]["steps"][0]["step_id"]
    other = mind.new_session()  # another session in this very (live) process
    assert mind.claim(sid, 1, dict(other, job_name="mind-aaaaaa-1-local", target="local"))
    out, _ = run_goal(g, cwd, fake, max_ticks=1)
    assert (out["state"], out["arbitration_writes"]) == ("detached", 1)
    assert launcher.launched == [] and not kinds("launch")
    assert mind.current_holder(sid, 1)[0] == 0


def test_a_dead_launcher_is_adopted_and_its_job_sealed(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, jobs = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"local": 100})
    sid, gid_ = g["body"]["steps"][0]["step_id"], g["body"]["goal_id"]
    name = fleet.job_name("mind", "abc123", "local", 1)
    # A session that died after launching: its claim, launch record and launcher pid...
    dead = {"session": "0" * 16, "pid": DEAD_PID, "boot": agent.boot_id(), "ident": None}
    assert mind.claim(sid, 1, dict(dead, job_name=name, target="local", token="abc123", goal_id=gid_))
    mind.write_records([mind.make_record("launch", {"goal_id": gid_, "step_id": sid, "gen": 1,
                                                    "body_id": ids["local"], "token": "abc123", "at": 1})])
    assert mind.mark_launched(sid, 1, DEAD_PID, name, "local")
    # ...and a job that finished after its launcher died, so nothing sealed it.
    sdir = fleet.JOBS_DIR / new_job_id(name)
    sdir.mkdir(parents=True)
    receipts.record_inputs(sdir, cwd, jobs[sid], "local")
    (sdir / "log.txt").write_text("ok\n")
    (sdir / "status.json").write_text(json.dumps({"state": "succeeded", "exit_code": 0, "elapsed_s": 3.0}))
    me, clock = mind.new_session(), FakeClock()
    out, said = run_goal(g, cwd, fake, clock=clock, session=me)
    assert (out["state"], out["exit"]) == ("passed", 0)
    assert launcher.launched == []  # watched to the end, never relaunched
    assert clock.t - 1000 >= mind.SEAL_GRACE_S  # a live launcher had its chance to seal first
    h1 = json.loads((mind.CLAIM_DIR / f"{sid}.1.h1").read_text())
    assert (h1["session"], h1["adopted_from"], h1["job_name"]) == (me["session"], 0, name)
    assert receipts.verify(json.loads((sdir / "receipt.json").read_text()))
    (res,) = kinds("result")
    assert (res["body"]["gen"], res["body"]["body_id"], res["body"]["elapsed_ms"]) == (1, ids["local"], 3000)
    assert any(s.startswith("adopted") for s in said)


def test_a_running_step_without_a_trace_is_lost(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"local": 200, "w1": 100}, fake)
    sid, gid_ = g["body"]["steps"][0]["step_id"], g["body"]["goal_id"]
    assert mind.claim(sid, 1, {"session": "0" * 16, "pid": DEAD_PID})  # no job name, no .run
    mind.write_records([mind.make_record("launch", {"goal_id": gid_, "step_id": sid, "gen": 1,
                                                    "body_id": ids["local"], "token": "t", "at": 1})])
    out, _ = run_goal(g, cwd, fake)
    assert out["state"] == "passed"
    (lost_rec,) = kinds("lost")
    assert (lost_rec["body"]["gen"], lost_rec["body"]["body_id"]) == (1, ids["local"])
    assert [t for t, _, _ in launcher.launched] == ["w1"]  # the lost body is excluded at gen 2
    assert (kinds("result")[0]["body"]["gen"], kinds("result")[0]["body"]["body_id"]) == (2, W1)
    assert lost_rec["body"]["why"] == "launch cannot be traced"


# ---------------------------------------------------------------- installs: foreign goals, export, import, sync


def test_a_goal_from_another_install_cannot_execute(fake, launcher, tmp_path, monkeypatch):
    as_home(tmp_path, monkeypatch, "A")
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100})
    carry = tmp_path / "carry"
    assert mind.export_memory(carry) == len(mind.segment_names())
    as_home(tmp_path, monkeypatch, "B")
    mind.import_memory(carry)
    assert verify_kinds() == {"unverifiable"}
    with pytest.raises(mind.MindError, match="another install"):
        run_goal(g, cwd, fake)
    with pytest.raises(mind.MindError, match="no goal"):
        mind.execute("f" * 64, cwd, workers=fake.workers, max_ticks=1)
    assert launcher.launched == [] and not list(mind.SESSION_DIR.glob("*"))


def verify_kinds():
    return {mind.verify_record(r) for r in mind.load_records()}


def test_export_and_import_carry_segments_between_installs(tmp_path, monkeypatch):
    as_home(tmp_path, monkeypatch, "A")
    mind.write_records([mind.make_record("verdict", {"signature_id": "a" * 12, "label": "flaky"})])
    mind.write_records([mind.make_record("retry", {"step_id": "e" * 64, "from_gen": 2, "at": 1})])
    names = mind.segment_names()
    carry = tmp_path / "carry"
    assert mind.export_memory(carry) == 2 and mind.export_memory(carry) == 0
    (carry / ("0" * 64 + ".jsonl")).write_bytes(b"not the bytes this name promises\n")
    as_home(tmp_path, monkeypatch, "B")
    assert mind.import_memory(carry) == {"segments": 2, "new": 2, "skipped": 1, "forged": 0}
    assert mind.import_memory(carry)["new"] == 0
    s = mind.memory_summary()
    assert (s["segments"], s["records"], s["status"], s["keys"]) == (2, 2, {"unverifiable": 2}, 1)
    assert s["arrivals"] == {"import": 2} and mind.segment_names() == names
    with pytest.raises(mind.MindError):
        mind.import_memory(tmp_path / "missing")


def test_sync_swaps_segments_through_a_worker(fake, tmp_path, monkeypatch):
    as_home(tmp_path, monkeypatch, "A")
    mind.write_records([mind.make_record("verdict", {"signature_id": "a" * 12, "label": "flaky"})])
    ledger = Counter()
    assert mind.sync(fake.workers, ledger) == {"w1": {"pulled": 0, "pushed": 1, "bad": 0}}
    rdir = fake.home("w1") / ".orchard" / "mind" / "segments"
    assert sorted(p.name for p in rdir.iterdir()) == [mind.segment_names()[0] + ".jsonl"]
    assert mind.sync(fake.workers, ledger) == {"w1": {"pulled": 0, "pushed": 0, "bad": 0}}  # nothing new either way
    as_home(tmp_path, monkeypatch, "B")
    mind.write_records([mind.make_record("verdict", {"signature_id": "b" * 12, "label": "flaky"})])
    (rdir / ("0" * 64 + ".jsonl")).write_bytes(b"tampered on the relay\n")
    assert mind.sync(fake.workers, ledger) == {"w1": {"pulled": 1, "pushed": 1, "bad": 1}}
    assert mind.memory_summary()["arrivals"] == {"self": 1, "sync": 1}
    assert {v["origin"] for v in mind._via_lines() if v["source"] == "sync"} == {"w1"}
    assert not [p for p in rdir.iterdir() if p.name.startswith(".tmp-")]
    (rdir / ("0" * 64 + ".jsonl")).unlink()
    big = b"x" * (mind.SEGMENT_MAX_BYTES + 1)  # named by its own hash, but over the size limit
    (rdir / (hashlib.sha256(big).hexdigest() + ".jsonl")).write_bytes(big)
    assert mind.sync(fake.workers, ledger) == {"w1": {"pulled": 0, "pushed": 0, "bad": 1}}
    assert hashlib.sha256(big).hexdigest() not in mind.segment_names()
    fake.down.add("w1")
    assert mind.sync(fake.workers, ledger) == {"w1": {"error": "unreachable"}}
    assert set(ledger) == {"sync"}


# ---------------------------------------------------------------- mind run, the views and the CLI


def run_cmd(cwd, fake, cmd=("make", "test"), clock=None, **kw):
    """mind.run_goal as `orchard mind run -- cmd` calls it, on the fake workers."""
    clock = clock or FakeClock()
    said = []
    out = mind.run_goal(cwd, list(cmd), workers=fake.workers, clock=clock, sleep=clock.sleep, say=said.append,
                        max_ticks=500, **kw)
    return out, said


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


@pytest.fixture
def cli(fake, launcher, tmp_path, monkeypatch):
    """`orchard mind ...` through Typer in a project, with the fake workers as config's."""
    from typer.testing import CliRunner

    from orchard import cli as cli_mod
    from orchard import config

    monkeypatch.setattr(mind, "config", _Over(config, load_workers=lambda: list(fake.workers.values())))
    monkeypatch.setattr(mind, "TICK_S", 0.0)
    monkeypatch.chdir(project(tmp_path))
    runner = CliRunner()

    def invoke(*args):
        return runner.invoke(cli_mod.app, ["mind", *args])

    return invoke


def test_existing_commands_and_x86_64_placement_unchanged():
    from orchard import cli as cli_mod
    from orchard import cli_agentic

    cmds = {c.name or c.callback.__name__: c.callback for c in cli_mod.app.registered_commands}
    names = set(cmds)
    assert {"run", "build", "receipt", "consensus", "fork", "live", "workers"} <= names and "mind" not in names
    assert cmds["run"] is cli_mod.run and cmds["build"] is cli_mod.build
    assert {cmds["run"].__module__, cmds["build"].__module__, cli_mod.submit.__module__} == {"orchard.cli"}
    groups = {g.name: g for g in cli_mod.app.registered_groups}
    assert groups["mind"].typer_instance is cli_agentic.mind_app
    assert {"replica", "speculate"} <= set(groups)
    job = JobSpec(name="j", cmd=["swift", "test"], tags=["x86_64"])
    assert placement.problems(job, {"tags": ["arm64", "os:macos"], "ram_gb": 16, "disk_free_gb": 100}) == [
        "missing tags: x86_64"]
    assert placement.problems(job, {"tags": ["x86_64", "os:macos"], "ram_gb": 16, "disk_free_gb": 100}) == []
    # placement picks the x86_64 worker the same way before and after mind and its CLI are imported
    code = textwrap.dedent("""
        from orchard import placement
        from orchard.job import JobSpec

        def pick(background):
            base = {"ram_gb": 16, "disk_free_gb": 100, "platform_ok": True}
            here = placement.Candidate("local", "local", dict(base, tags=["arm64", "os:macos"]))
            w1 = placement.Candidate("w1", "worker", dict(base, tags=["x86_64", "os:macos"]))
            job = JobSpec(name="j", cmd=["swift", "test"], tags=["x86_64"])
            return placement.choose(job, [here, w1], background=background).candidate.name

        before = [pick(False), pick(True)]
        import orchard.mind, orchard.cli_agentic  # noqa: E401
        print(*before, pick(False), pick(True))
    """)
    src = str(Path(mind.__file__).parents[1])
    home = Path(os.environ["HOME"]) / "placement-home"
    home.mkdir()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60,
                         env={"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": src})
    assert out.returncode == 0, out.stderr
    assert out.stdout.decode().split() == ["w1"] * 4
    from typer.testing import CliRunner

    helped = CliRunner().invoke(cli_mod.app, ["mind", "--help"])
    assert helped.exit_code == 0 and "run" in helped.output


def test_x86_64_step_fits_only_tagged_body(tmp_path, monkeypatch):
    monkeypatch.setattr(mind, "env", _Over(env, resolve_for_worker=lambda *a, **k: (True, "")))
    cwd = project(tmp_path)
    tree = receipts.tree_digest(cwd)[0]
    specs = mind.step_specs(["swift", "test"], ["make lint"], [], [], 60)
    specs[0]["tags"] = ["x86_64"]
    steps, jobs = mind.build_steps(cwd, tree, specs)
    assert [s["cmd"] for s in steps] == [["swift", "test"], ["make", "lint"]]  # the order given
    g = mind.goal_record(cwd, tree, steps, [])
    swift, lint = steps
    base = {"ram_gb": 16, "disk_free_gb": 100, "bench": 250.4, "name": "w1", "host": WORKER_HOST}
    arm = mind.fit_record(g, "a" * 32, dict(base, tags=["arm64", "os:macos"], os_family="macos"), jobs, True)
    x86 = mind.fit_record(g, "b" * 32, dict(base, tags=["x86_64", "os:macos"], os_family="macos"), jobs, True)
    assert arm["body"]["fits"] == [lint["step_id"]]
    assert arm["body"]["problems"] == {swift["step_id"]: ["missing tags: x86_64"]}
    assert x86["body"]["fits"] == sorted([swift["step_id"], lint["step_id"]]) and x86["body"]["problems"] == {}
    assert arm["body"]["caps"] == {"bench": 250, "os_family": "macos"}  # never a name or a host
    mind.write_records([g, arm, x86])
    p = mind.plan(mind.load_records(), g["body"]["goal_id"])
    assert st(p, swift)["capable"] == ["b" * 32]


def test_step_specs_split_steps_and_refuse_bad_ones(tmp_path):
    specs = mind.step_specs(["pytest", "-q"], ["make 'a b'", "pytest -q"], ["t"], ["E", "E"], 9,
                            [{"cmd": ["npm", "test"], "tags": ["node"], "env": [], "timeout": 30}])
    assert [s["cmd"] for s in specs] == [["pytest", "-q"], ["make", "a b"], ["npm", "test"]]
    assert specs[0] == {"cmd": ["pytest", "-q"], "tags": ["t"], "env": ["E"], "timeout": 9}
    assert specs[2]["tags"] == ["node"] and specs[2]["timeout"] == 30
    for bad in (["make 'open"], ["   "]):
        with pytest.raises(mind.MindError):
            mind.step_specs(None, bad, [], [], None)


def test_step_id_ignores_env_mac_but_work_key_does_not(tmp_path, monkeypatch):
    monkeypatch.setenv("MIND_TEST_TOKEN", "same value on both installs")
    cwd = project(tmp_path)
    tree = receipts.tree_digest(cwd)[0]
    spec = mind.step_specs(["make", "test"], [], [], ["MIND_TEST_TOKEN"], None)
    seen = []
    for name in ("A", "B"):
        as_home(tmp_path, monkeypatch, name)
        (s,), _ = mind.build_steps(cwd, tree, spec)
        seen.append(s)
    a, b = seen
    assert a["step_id"] == b["step_id"] and a["cmd_key"] == b["cmd_key"]
    assert a["work_key"] != b["work_key"]  # env_mac is keyed per install
    assert a["env_names"] == ["MIND_TEST_TOKEN"] and "same value" not in json.dumps(a)


def test_trust_changes_the_goal_id_and_takes_only_key_ids(tmp_path):
    cwd = project(tmp_path)
    tree = receipts.tree_digest(cwd)[0]
    steps, _ = mind.build_steps(cwd, tree, mind.step_specs(["make", "test"], [], [], [], None))
    plain = mind.goal_record(cwd, tree, steps, [])
    trusted = mind.goal_record(cwd, tree, steps, ["c" * 16])
    assert plain["body"]["goal_id"] != trusted["body"]["goal_id"] and trusted["body"]["trust"] == ["c" * 16]
    assert mind.goal_record(cwd, tree, steps, [mind.key_id()])["body"]["goal_id"] == plain["body"]["goal_id"]
    with pytest.raises(mind.MindError, match="key id"):
        mind.goal_record(cwd, tree, steps, ["w1"])


def test_max_gen_strands_and_retry_reopens(tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"local": 100})
    (s,) = g["body"]["steps"]
    gid_ = g["body"]["goal_id"]
    with pytest.raises(mind.MindError, match="already waiting"):
        mind.retry_step(gid_, s["step_id"][:8])
    mind.write_records([mind.lost_record(s["step_id"], gen, ids["local"], "went quiet", hold=True)
                        for gen in range(1, mind.MAX_GEN + 1)])
    p = mind.plan(mind.load_records(), gid_)
    assert st(p, s)["state"] == "stranded"
    ex = mind.explain(gid_, "make test")
    assert ex["state"] == "stranded" and [h["gen"] for h in ex["history"]] == list(range(1, mind.MAX_GEN + 1))
    assert all(h["kind"] == "lost" and h["hold"] for h in ex["history"])
    assert ex["bodies"] == [{"body": ids["local"], "fits": True, "problems": [], "bench": 100}]
    out = mind.retry_step(gid_, s["step_id"][:8])
    assert (out["state"], out["gen"]) == ("pending", mind.MAX_GEN + 1)
    assert st(mind.plan(mind.load_records(), gid_), s)["state"] == "pending"
    with pytest.raises(mind.MindError, match="no step"):
        mind.retry_step(gid_, "nothing like it")


def test_run_goal_passes_then_answers_from_memory_with_no_network(fake, launcher, tmp_path, monkeypatch):
    cwd = project(tmp_path)
    out, said = run_cmd(cwd, fake)
    assert (out["state"], out["exit"], out["from_memory"]) == ("passed", 0, False)
    (line,) = out["steps"]
    assert (line["cmd"], line["state"], line["from_memory"]) == ("make test", "passed", False)
    assert line["elapsed_ms"] == 1500 and len(launcher.launched) == 1
    fake.down.add("w1")  # nothing may be reached now

    def no_caps(*a, **k):
        raise AssertionError("answering from memory must not sense this machine")

    monkeypatch.setattr(mind, "agent", _Over(mind.agent, collect_caps=no_caps))
    again, said2 = run_cmd(cwd, fake)
    assert (again["state"], again["exit"], again["from_memory"]) == ("passed", 0, True)
    assert again["ledger"] == {} and again["steps"][0]["from_memory"] and said2 == []
    assert len(launcher.launched) == 1 and again["goal"] == out["goal"]
    assert logged("mind.session")[-1]["from_memory"] is True


def test_fresh_reruns_and_reuse_lifts_a_sealed_receipt(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    first, _ = run_cmd(cwd, fake)
    again, _ = run_cmd(cwd, fake, fresh=True)
    assert (again["state"], again["from_memory"]) == ("passed", False) and len(launcher.launched) == 2
    assert again["steps"][0]["gen"] == first["steps"][0]["gen"] + 1
    # A new goal for the same check (another trust set) can lift the sealed receipt with --reuse.
    lifted, _ = run_cmd(cwd, fake, trust=["c" * 16], reuse=True)
    assert lifted["goal"] != first["goal"] and lifted["from_memory"] and len(launcher.launched) == 2


def test_run_goal_failure_then_why_and_retry(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    launcher.mode("local", "fail")
    launcher.mode("w1", "fail")
    out, _ = run_cmd(cwd, fake)
    assert (out["state"], out["exit"]) == ("failed", 1)
    again, _ = run_cmd(cwd, fake)
    assert (again["state"], again["exit"], again["from_memory"]) == ("failed", 1, True)
    gid_ = mind.resolve_goal(cwd, None)
    assert gid_ == out["goal"] and mind.resolve_goal(cwd, gid_[:10]) == gid_
    ex = mind.explain(gid_, "make test")
    assert ex["state"] == "failed" and ex["history"][-1]["kind"] == "result"
    assert ex["history"][-1]["state"] == "failed"
    (row,) = mind.status_rows(cwd)
    assert (row["goal"], row["state"], row["watching"], row["this_tree"]) == (gid_, "failed", False, True)
    r = mind.retry_step(gid_, "make test")
    assert r["state"] == "pending" and r["gen"] == out["steps"][0]["gen"] + 1
    fixed, _ = run_cmd(cwd, fake)
    assert (fixed["state"], fixed["exit"], fixed["from_memory"]) == ("passed", 0, False)


def test_run_goal_refuses_bad_input(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    for kw in ({"quiet_s": 10}, {"trust": ["not-a-key"]}, {"use_speculate": True}):
        with pytest.raises(mind.MindError):
            run_cmd(cwd, fake, **kw)
    with pytest.raises(mind.MindError, match="nothing to run"):
        mind.run_goal(cwd, None, workers=fake.workers)
    with pytest.raises(mind.MindError, match="no goal"):
        mind.resolve_goal(cwd, None)
    assert launcher.launched == [] and mind.segment_names() == []


def test_no_names_in_replicated_records(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    launcher.mode("w1", "die")  # the first launch is lost, so every kind of record is written
    out, said = run_cmd(cwd, fake, sync_too=True)
    assert out["state"] == "passed" and launcher.launched[0][::2] == ("w1", "die") and len(launcher.launched) == 2
    assert {r["kind"] for r in mind.load_records()} >= {"goal", "fit", "launch", "lost", "result"}
    assert out["sync"]["after"]["pushed"] >= 1 and out["sync_workers"]["after"]["w1"]["pushed"] >= 1
    mine = [p.read_bytes() for p in mind.SEG_DIR.glob("*.jsonl")]
    theirs = [p.read_bytes() for p in (fake.home("w1") / ".orchard" / "mind" / "segments").glob("*.jsonl")]
    blob = b"".join(mine + theirs)
    for leak in ("w1", WORKER_HOST, str(tmp_path), str(cwd)):
        assert leak.encode() not in blob, leak
    assert any(" on w1" in m for m in said)  # names stay in this session's own messages


def test_hide_names_and_public_view(tmp_path):
    mind._save_bodies({W1: "w1"})
    names = ["w1", "w10", "local"]
    assert mind.hide_names("launched abc gen 1 on w1", names) == f"launched abc gen 1 on body {W1[:8]}"
    assert mind.hide_names("waiting: w10 busy", names) == "waiting: a worker busy"
    assert mind.hide_names("launched abc gen 1 on local", names) == "launched abc gen 1 on local"
    obj = {"queues": {W1: ["s1"]}, "steps": [{"body": W1, "capable": [W1]}], "w1": 1}
    assert mind.public(obj, {W1}) == {"queues": {W1[:8]: ["s1"]}, "steps": [{"body": W1[:8], "capable": [W1[:8]]}],
                                      "w1": 1}
    assert mind.bodies_view({W1}) == {W1[:8]: "w1"}


def test_plan_only_writes_nothing(fake, launcher, tmp_path):
    cwd = project(tmp_path)  # a fresh install: no receipt key yet, and a plan must not make one
    before = tree_bytes(tmp_path)
    out = mind.run_goal(cwd, ["make", "test"], ["make lint"], plan_only=True, workers=fake.workers)
    assert out["plan_only"] and out["exit"] == 0 and out["state"] == "usage"  # no body has reported here
    assert [s["cmd"] for s in out["steps"]] == ["make test", "make lint"]
    assert tree_bytes(tmp_path) == before and launcher.launched == []


def test_mind_source_has_no_listeners():
    tree = ast.parse(Path(mind.__file__).read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names} | {(n.module or "").split(".")[0] for n in ast.walk(tree)
                                     if isinstance(n, ast.ImportFrom)}
    assert not imported & {"socket", "socketserver", "http", "asyncio", "selectors", "ssl", "urllib"}
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"listen", "bind", "accept", "serve_forever", "start_server"}


def test_sense_script_parses_at_python_3_9():
    ast.parse(mind.SENSE_SCRIPT.decode("utf-8"), feature_version=(3, 9))


def test_export_file_round_trip_with_provenance(tmp_path, monkeypatch):
    as_home(tmp_path, monkeypatch, "A")
    mind.write_records([mind.make_record("verdict", {"signature_id": "a" * 12, "label": "flaky"})])
    mind.write_records([mind.make_record("retry", {"step_id": "e" * 64, "from_gen": 2, "at": 1})])
    f = tmp_path / "carry" / "memory.jsonl"
    assert mind.export_file(f) == 2
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    as_home(tmp_path, monkeypatch, "B")
    res = mind.import_file(f)
    assert res == {"segments": 2, "new": 2, "rejected": 0, "forged": 0, "origin": sha}
    assert {(v["source"], v["origin"]) for v in mind._via_lines()} == {("import", sha)}
    assert mind.import_file(f)["new"] == 0
    s = mind.memory_view(True)
    assert s["records"] == 2 and [k["mine"] for k in s["by_key"]] == [False]
    assert {e["status"] for e in s["entries"]} == {"unverifiable"} and {tuple(e["sources"]) for e in s["entries"]} == {
        ("import",)}
    with pytest.raises(mind.MindError, match="cannot read"):
        mind.import_file(tmp_path / "missing.jsonl")


def test_tampered_import_is_rejected_or_forged(tmp_path, monkeypatch):
    as_home(tmp_path, monkeypatch, "A")
    mind.write_records([mind.make_record("verdict", {"signature_id": "a" * 12, "label": "flaky"})])
    f = tmp_path / "memory.jsonl"
    mind.export_file(f)
    (line,) = [json.loads(ln) for ln in f.read_text().splitlines()]
    changed = line["data"].replace("flaky", "load-sensitive")
    renamed = hashlib.sha256(changed.encode()).hexdigest()
    lines = [
        {"segment": line["segment"], "data": changed},              # bytes no longer match the name
        {"segment": hashlib.sha256(b"junk\n").hexdigest(), "data": "junk\n"},  # named right, not records
        {"segment": "../" + line["segment"][3:], "data": line["data"]},  # not a segment name
        "not json at all",
    ]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("".join((ln if isinstance(ln, str) else json.dumps(ln)) + "\n" for ln in lines))
    key = receipts.KEY_PATH.read_bytes()
    as_home(tmp_path, monkeypatch, "A2")  # the same key, an empty memory
    receipts.KEY_PATH.parent.mkdir(parents=True)
    receipts.KEY_PATH.write_bytes(key)
    assert mind.import_file(bad)["rejected"] == 4 and mind.segment_names() == []
    forged = tmp_path / "forged.jsonl"
    forged.write_text(json.dumps({"segment": renamed, "data": changed}) + "\n")
    res = mind.import_file(forged)
    assert (res["segments"], res["forged"]) == (1, 1)
    # The forged record stays in memory as evidence, refuted by a record this install signs.
    (via,) = [v for v in mind._via_lines() if v["source"] == "import"]
    (refute,) = [r for r in mind.load_records() if r["kind"] == "refute"]
    assert refute["body"]["target"] == via["forged"][0] and mind.verify_record(refute) == "verified"
    assert mind.memory_summary()["status"] == {"forged": 1, "verified": 1}
    assert mind.lift_verdicts() == []


def test_cli_exit_codes(cli, launcher):
    ok = cli("run", "--", "make", "test")
    assert ok.exit_code == 0, ok.output
    assert "mind: all passed. 0 coordination messages, 0 recoveries" in ok.output
    launcher.mode("local", "fail")
    launcher.mode("w1", "fail")
    failed = cli("run", "--step", "make lint")
    assert failed.exit_code == 1 and "failed" in failed.output
    for args in (("run",), ("run", "--quiet-after", "10", "--", "make", "test"),
                 ("run", "--trust", "w1", "--", "make", "test"), ("why", "no such step"), ("retry", "zzzz"),
                 ("plan", "ffffffffffff"), ("sync", "nobody")):
        res = cli(*args)
        assert res.exit_code == 2, (args, res.output)
    assert cli("why", "make lint").exit_code == 0
    retried = cli("retry", "make lint")
    assert retried.exit_code == 0 and "reopened at gen 2" in retried.output
    assert cli("status").exit_code == 0 and cli("memory", "--records").exit_code == 0


def test_cli_json_hides_names_and_bodies_adds_them(cli, launcher):
    res = cli("run", "--json", "--", "make", "test")
    assert res.exit_code == 0 and launcher.launched[0][0] == "w1"  # a named worker ran it: there was a name to hide
    out = json.loads(res.stdout)
    assert out["state"] == "passed" and "sync_workers" not in out and "bodies" not in out
    assert "w1" not in res.stdout and WORKER_HOST not in res.stdout
    assert "w1" not in res.output  # the session's own messages name bodies by id too
    plan = cli("plan", "--json")
    assert plan.exit_code == 0 and "w1" not in plan.stdout
    p = json.loads(plan.stdout)
    assert all(len(b) == 8 for b in p["queues"]) and all(len(b) == 8 for st_ in p["steps"] for b in st_["capable"])
    shown = json.loads(cli("plan", "--json", "--bodies").stdout)
    assert shown["bodies"][W1[:8]] == "w1" and shown["digest"] == p["digest"]
    why = cli("why", "--json", "make test")
    assert why.exit_code == 0 and "w1" not in why.stdout
    status = json.loads(cli("status", "--json").stdout)
    assert [r["state"] for r in status] == ["passed"]


def test_cli_plan_only_writes_nothing(cli, tmp_path, launcher):
    assert not receipts.KEY_PATH.exists()  # a fresh install: a plan must not make the key
    before = tree_bytes(tmp_path)
    res = cli("run", "--plan-only", "--json", "--", "make", "test")
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["plan_only"] is True
    assert tree_bytes(tmp_path) == before and launcher.launched == []


def test_cli_export_import_round_trip_and_tamper(cli, tmp_path, monkeypatch):
    assert cli("run", "--", "make", "test").exit_code == 0
    f = tmp_path / "memory.jsonl"
    res = cli("export", str(f))
    n = len(mind.segment_names())
    assert res.exit_code == 0 and f"exported {n} segment(s)" in res.output
    as_home(tmp_path, monkeypatch, "B")
    imp = cli("import", "--json", str(f))
    assert imp.exit_code == 0 and json.loads(imp.stdout)["new"] == n
    rows = [json.loads(ln) for ln in f.read_text().splitlines()]
    rows[0]["data"] = rows[0]["data"].replace('"v":1', '"v":2', 1)
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    as_home(tmp_path, monkeypatch, "C")
    bad = cli("import", str(f))
    assert bad.exit_code == 1 and "1 rejected" in bad.output
    assert len(mind.segment_names()) == n - 1


def test_cli_key_prints_this_install_key_id(cli):
    res = cli("key")
    assert res.exit_code == 0 and res.stdout.strip() == mind.key_id()
    assert len(mind.key_id()) == 16 and all(c in "0123456789abcdef" for c in mind.key_id())


# ---------------------------------------------------------------- authors, codes in records, reaping


def test_the_same_goal_on_two_installs_is_two_goals_each_run_by_its_author(tmp_path, monkeypatch):
    cwd = project(tmp_path)
    tree = receipts.tree_digest(cwd)[0]
    specs = mind.step_specs(["make", "test"], [], [], [], 60)
    made = {}
    for name in ("A", "B"):
        as_home(tmp_path, monkeypatch, name)
        steps, _ = mind.build_steps(cwd, tree, specs)
        made[name] = mind.goal_record(cwd, tree, steps, [])
        assert made[name]["body"]["author"] == made[name]["key"] == mind.key_id()
    ga, gb = made["A"], made["B"]
    assert ga["body"]["steps"] == gb["body"]["steps"] and ga["body"]["goal_id"] != gb["body"]["goal_id"]
    both = [ga, gb]
    assert mind.own_goal(gb["body"]["goal_id"], both) == gb  # this process is install B now
    with pytest.raises(mind.MindError, match="another install"):
        mind.own_goal(ga["body"]["goal_id"], both)
    as_home(tmp_path, monkeypatch, "A")
    assert mind.own_goal(ga["body"]["goal_id"], both) == ga
    # Copies of A's goal under other keys, some with lower record ids, change nothing.
    forged = [dict(ga, key=f"{i:016x}") for i in range(64)]
    forged = [f for f in forged if mind.record_id(f) < mind.record_id(ga)]
    assert forged
    assert mind.own_goal(ga["body"]["goal_id"], forged + both) == ga
    assert mind.goal_of(forged + both, ga["body"]["goal_id"]) == ga


def test_fit_records_carry_problem_codes_not_resolver_notes(tmp_path, monkeypatch):
    note = f"no wheel in {tmp_path.joinpath('Users', 'someone', 'cache')}"
    monkeypatch.setattr(mind, "env", _Over(env, resolve_for_worker=lambda *a, **k: (False, note)))
    cwd = project(tmp_path)
    tree = receipts.tree_digest(cwd)[0]
    steps, jobs = mind.build_steps(cwd, tree, mind.step_specs(["pytest", "-q"], [], [], [], 60))
    g = mind.goal_record(cwd, tree, steps, [])
    caps = {"tags": ["os:macos"], "ram_gb": 16, "disk_free_gb": 100, "bench": 100, "os_family": "macos"}
    r = mind.fit_record(g, "a" * 32, caps, jobs, True)
    assert r["body"] == dict(r["body"], fits=[], problems={steps[0]["step_id"]: ["platform resolve failed"]})
    assert "someone" not in json.dumps(r) and str(tmp_path) not in json.dumps(r)
    assert mind._problem_code("needs 64 GB RAM") == "needs 64 GB RAM"
    assert mind._problem_code("something new that names w1") == "placement problem"


def test_every_dispatcher_hold_maps_to_a_fixed_code():
    cfg = {"idle_threshold_s": 180}
    idle = {"tags": ["os:macos"], "idle_seconds": 100000}
    cases = {
        "held: paused": {"policy_paused": True},
        "held: busy": {"busy_jobs": ["mind-0a1b2c3d-1-w1"]},
        "held: host not ok": {"host_status": "asleep"},
        "held: idle time unknown": {"tags": ["os:linux"], "idle_seconds": None},
        "held: owner active": {"idle_seconds": 3},
        "held: on battery": {"power": "battery"},
        "held: battery low": {"pause_on_low_battery_pct": 20, "battery_pct": 10},
        "held: thermal": {"thermal_limit": -1},
    }
    for code, extra in cases.items():
        hold = dispatcher.idle_hold(dict(idle, **extra), cfg)
        assert hold and mind._hold_code(hold) == code, (hold, code)
    other = dispatcher.idle_hold(dict(idle, max_other_cpu_pct=50, other_cpu_pct=90.0), cfg)
    assert mind._hold_code(other) == "held: busy"
    assert mind._hold_code("a hold nobody has written yet on w1") == "held"


def test_a_hold_reaches_records_as_a_code_and_messages_in_full(fake, launcher, tmp_path, monkeypatch):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100, "w1": 200}, fake)
    # The real dispatcher reads the hold off w1's caps: it runs another job.
    monkeypatch.setattr(mind, "dispatcher", _Over(dispatcher, load_config=lambda: {"idle_threshold_s": 180}))
    caps = fake.caps
    fake.caps = lambda name: dict(caps(name), busy_jobs=["mind-0a1b2c3d-1-w1"]) if name == "w1" else caps(name)
    out, said = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("passed", 0)
    (lost_rec,) = kinds("lost")
    assert (lost_rec["body"]["why"], lost_rec["body"]["hold"]) == ("held: busy", True)
    assert [t for t, _, _ in launcher.launched] == ["local"]
    blob = b"".join(p.read_bytes() for p in mind.SEG_DIR.glob("*.jsonl"))
    assert b"mind-0a1b2c3d" not in blob and b"w1" not in blob
    assert any("worker running mind-0a1b2c3d-1-w1" in m for m in said)  # the full text stays here


def test_reap_probes_only_bodies_that_answered(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    fake.workers["w2"] = Worker("w2", host=WORKER_HOST)
    (fake.home("w2") / ".orchard").mkdir(parents=True)
    (fake.home("w2") / ".orchard" / "body.id").write_text("2" * 32)
    add_bodies(g, {"w2": 100}, fake)
    mind.add_zombie("w1", "job-x", None)
    fake.down.add("w1")
    launcher.mode("w2", "hang")  # w2 answers every SENSE_EVERY_S, so reaps run
    out, _ = run_goal(g, cwd, fake, max_ticks=60)
    assert out["state"] == "detached" and out["sense_calls"] >= 2
    assert out["ledger"].get("terminate", 0) == 0  # but w1 is silent: it is not probed every time
    assert len(mind._zombie_lines()) == 1
    assert mind.reap(fake.workers, Counter(), {"w2"}) == 0 and len(mind._zombie_lines()) == 1
    fake.down.clear()

    def survey(ledger):
        return mind.survey(g, None, fake.workers, ledger)

    again, _ = run_goal(g, cwd, fake, max_ticks=1, survey=survey)
    assert fake.terminated == [("w1", f"{fake.home('w1')}/.orchard/jobs/job-x")]
    assert mind._zombie_lines() == [] and again["ledger"]["terminate"] == 2


# ---------------------------------------------------------------- adopted launchers


def dead_launch(g, jobs, ids, cwd, target="local", pid=DEAD_PID, status=None, folder=True):
    """What a session that died after launching leaves: its claim, the launch
    record, the launcher's pid in .run and, with folder, the job folder the
    launcher made (with status, when given). Returns (step id, folder)."""
    sid, gid_ = g["body"]["steps"][0]["step_id"], g["body"]["goal_id"]
    name = fleet.job_name("mind", "abc123", target, 1)
    dead = {"session": "0" * 16, "pid": DEAD_PID, "boot": agent.boot_id(), "ident": None}
    assert mind.claim(sid, 1, dict(dead, job_name=name, target=target, token="abc123", goal_id=gid_))
    mind.write_records([mind.make_record("launch", {"goal_id": gid_, "step_id": sid, "gen": 1,
                                                    "body_id": ids[target], "token": "abc123", "at": 1})])
    assert mind.mark_launched(sid, 1, pid, name, target)
    if not folder:
        return sid, None
    sdir = fleet.JOBS_DIR / new_job_id(name)
    sdir.mkdir(parents=True)
    receipts.record_inputs(sdir, cwd, jobs[sid], target)
    if status is not None:
        (sdir / "status.json").write_text(json.dumps(status))
    return sid, sdir


@pytest.mark.parametrize("group, why", [("gone", "launcher died"), (None, "launcher died before the job started"),
                                        ("alive", None)])
def test_an_adopted_local_job_ends_with_its_process_group(fake, launcher, tmp_path, group, why):
    cwd = project(tmp_path)
    g, jobs = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"local": 200, "w1": 100}, fake)
    status = {"state": "running"}
    if group == "gone":
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        status.update(pgid=child.pid, boot_id=agent.boot_id())
    elif group == "alive":
        status.update(pgid=os.getpgid(0), boot_id=agent.boot_id())
    dead_launch(g, jobs, ids, cwd, status=status)
    clock = FakeClock()
    out, _ = run_goal(g, cwd, fake, clock=clock, max_ticks=300)
    if why is None:  # the job still runs: keep watching
        assert out["state"] == "detached" and not kinds("lost") and launcher.launched == []
        return
    assert (out["state"], out["exit"]) == ("passed", 0)
    (lost_rec,) = kinds("lost")
    assert (lost_rec["body"]["gen"], lost_rec["body"]["body_id"], lost_rec["body"]["why"]) == (1, ids["local"], why)
    assert [t for t, _, _ in launcher.launched] == ["w1"]
    waited = clock.t - 1000
    assert waited >= mind.SEAL_GRACE_S if group is None else waited < mind.SEAL_GRACE_S


def test_an_adopted_launcher_that_is_alive_is_left_to_seal(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, jobs = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"local": 100})
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        sid, sdir = dead_launch(g, jobs, ids, cwd, pid=child.pid, status={"state": "running"})
        clock = FakeClock()

        def sleep(s):
            clock.t += s
            if clock.t - 1000 >= 2 * mind.SEAL_GRACE_S and child.poll() is None:
                # The launcher outlived its session: it seals its job, then exits.
                status = {"state": "succeeded", "exit_code": 0, "elapsed_s": 2.0}
                (sdir / "log.txt").write_text("ok\n")
                (sdir / "status.json").write_text(json.dumps(status))
                assert receipts.seal(sdir, status, "local", None) is not None
                child.kill()
                child.wait()

        clock.sleep = sleep
        out, said = run_goal(g, cwd, fake, clock=clock, max_ticks=1000)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert (out["state"], out["exit"]) == ("passed", 0)
    assert launcher.launched == [] and not kinds("lost")  # never relaunched, never given up while it lived
    assert (mind.CLAIM_DIR / f"{sid}.1.h1").exists()
    (res,) = kinds("result")
    assert (res["body"]["gen"], res["body"]["elapsed_ms"]) == (1, 2000)
    assert any(s.startswith("adopted") for s in said)


def test_an_adopted_remote_job_that_finished_is_collected_once(fake, launcher, tmp_path, monkeypatch):
    cwd = project(tmp_path)
    g, jobs = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"w1": 100}, fake)
    sid, sdir = dead_launch(g, jobs, ids, cwd, target="w1")
    status = {"state": "succeeded", "exit_code": 0, "elapsed_s": 4.0, "machine": "x86_64", "os_family": "macos"}
    fake.write_status("w1", sdir.name, status)
    (fake.home("w1") / ".orchard" / "jobs" / sdir.name / "log.txt").write_text("ok\n")
    calls, real = [], mind.engine.collect_remote
    monkeypatch.setattr(mind.engine, "collect_remote", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    clock = FakeClock()
    out, _ = run_goal(g, cwd, fake, clock=clock)
    assert (out["state"], out["exit"]) == ("passed", 0)
    assert calls == [sdir.name] and clock.t - 1000 >= mind.SEAL_GRACE_S
    assert launcher.launched == [] and receipts.verify(json.loads((sdir / "receipt.json").read_text()))
    (res,) = kinds("result")
    assert (res["body"]["gen"], res["body"]["body_id"], res["body"]["elapsed_ms"]) == (1, ids["w1"], 4000)


@pytest.mark.parametrize("folder", [True, False])
def test_an_adopted_remote_launch_that_never_started_is_lost(fake, launcher, tmp_path, folder):
    cwd = project(tmp_path)
    g, jobs = make_goal(cwd, [["make", "test"]])
    ids = add_bodies(g, {"w1": 100}, fake)
    dead_launch(g, jobs, ids, cwd, target="w1", folder=folder)
    clock = FakeClock()
    out, _ = run_goal(g, cwd, fake, clock=clock)
    assert (out["state"], out["exit"]) == ("stranded", 3)  # w1 was its only body
    (lost_rec,) = kinds("lost")
    assert lost_rec["body"]["why"] == "launcher died before the job started"
    if folder:  # the body said twice that it has no such job
        assert out["sense_calls"] >= mind.ORPHAN_CONFIRM and clock.t - 1000 < mind.SEAL_GRACE_S
    else:  # no job folder here to ask the body about: wait out the grace
        assert clock.t - 1000 >= mind.SEAL_GRACE_S
    assert launcher.launched == []


# ---------------------------------------------------------------- sessions: races, recovery, slow and odd launches


def test_two_sessions_on_one_goal_launch_it_once(fake, launcher, tmp_path, monkeypatch):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100})
    barrier, lock, waited = threading.Barrier(2, timeout=10), threading.Lock(), set()
    real_claim = mind.claim

    def claim(*a, **k):
        with lock:
            first = threading.get_ident() not in waited
            waited.add(threading.get_ident())
        if first:
            barrier.wait()  # both sessions try the same claim at once
        return real_claim(*a, **k)

    monkeypatch.setattr(mind, "claim", claim)
    outs, errors = {}, []

    def session(i):
        clock = FakeClock()

        def sleep(s):
            clock.t += s
            time.sleep(0.001)  # let the other session move

        clock.sleep = sleep
        try:
            outs[i], _ = run_goal(g, cwd, fake, clock=clock, max_ticks=2000)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=session, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors and sorted(outs) == [0, 1]
    assert len(launcher.launched) == 1 and len(kinds("launch")) == 1 and len(kinds("result")) == 1
    assert [o["state"] for o in outs.values()] == ["passed", "passed"]
    assert sum(o["arbitration_writes"] for o in outs.values()) >= 1
    assert outs[0]["plan"]["digest"] == outs[1]["plan"]["digest"]
    assert all(o["duplicate_launches"] == 0 and o["coord_msgs"] == 0 for o in outs.values())


def test_a_quiet_worker_hands_its_step_to_this_machine_in_the_same_tick(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100, "w1": 200}, fake)
    launcher.mode("w1", "hang-down")
    out, _ = run_goal(g, cwd, fake, quiet_s=60)
    assert (out["state"], out["exit"]) == ("passed", 0)
    assert [(t, m) for t, _, m in launcher.launched] == [("w1", "hang-down"), ("local", "pass")]
    assert (out["duplicate_launches"], out["coord_msgs"], out["recover_s"]) == (0, 0, [0.0])
    (lost_rec,) = kinds("lost")
    assert lost_rec["body"]["why"] == "quiet for 60s"
    assert logged("mind.recovered")[-1]["recover_s"] == 0.0


@pytest.mark.parametrize("target", ["local", "w1"])
def test_a_launch_still_starting_is_never_lost(fake, launcher, tmp_path, target):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {target: 100}, fake)
    launcher.mode(target, "slow")
    out, _ = run_goal(g, cwd, fake, quiet_s=60, max_ticks=200)
    assert out["state"] == "detached" and not kinds("lost") and launcher.killed == []
    assert len(launcher.launched) == 1


def test_an_orphan_probe_followed_by_a_running_one_loses_nothing(fake, launcher, tmp_path, monkeypatch):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"w1": 100}, fake)
    launcher.mode("w1", "hang")
    real, probes = mind.sense_remote, []

    def flicker(w, job_ids, ledger):
        if job_ids:  # every other probe finds the job's group gone, the next finds it again
            probes.append(job_ids[0])
            pgid = DEAD_PID if len(probes) % 2 else os.getpgid(0)
            fake.write_marker("w1", job_ids[0], pgid, agent.boot_id())
        return real(w, job_ids, ledger)

    monkeypatch.setattr(mind, "sense_remote", flicker)
    out, _ = run_goal(g, cwd, fake, max_ticks=100)
    assert out["state"] == "detached" and len(probes) >= 4
    assert not kinds("lost") and launcher.killed == []


def test_a_receipt_for_other_work_is_lost_as_a_moved_tree(fake, launcher, tmp_path):
    cwd = project(tmp_path)
    g, _ = make_goal(cwd, [["make", "test"]])
    add_bodies(g, {"local": 100})
    launcher.mode("local", "moved")
    out, _ = run_goal(g, cwd, fake)
    assert (out["state"], out["exit"]) == ("stranded", 3)
    (lost_rec,) = kinds("lost")
    assert lost_rec["body"]["why"] == "tree moved under the step" and not kinds("result")


def test_no_env_values_or_home_paths_in_replicated_records(fake, launcher, tmp_path, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "S3CR3T")
    cwd = tmp_path.joinpath("Users", "someone", "proj")
    cwd.mkdir(parents=True)
    (cwd / "main.py").write_text("print('hi')\n")
    out, _ = run_cmd(cwd, fake, env_names=["API_TOKEN"], sync_too=True)
    assert out["state"] == "passed" and [t for t, _, _ in launcher.launched] == ["w1"]
    assert {r["kind"] for r in mind.load_records()} >= {"goal", "fit", "launch", "result"}
    mine = [p.read_bytes() for p in mind.SEG_DIR.glob("*.jsonl")]
    theirs = [p.read_bytes() for p in (fake.home("w1") / ".orchard" / "mind" / "segments").glob("*.jsonl")]
    assert theirs
    blob = b"".join(mine + theirs)
    for leak in (b"S3CR3T", b"someone", str(cwd).encode()):
        assert leak not in blob, leak
