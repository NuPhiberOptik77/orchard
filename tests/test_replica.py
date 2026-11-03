"""Living replicas, end to end against fake bodies.

Each fake body is a folder standing in for a remote home: the real agent runs
there under /usr/bin/python3 with HOME pointed at it, reached through a fake
transport instead of ssh."""

import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from orchard import agent, cli, dispatcher, engine, env, events, fleet, receipts, replica, speculate, transport
from orchard.config import Worker

AGENT_SRC = Path(agent.__file__)
PREP = ("mkdir -p \"$ORCHARD_REPLICA_TOOLS/bin\" && echo ran >> \"$ORCHARD_REPLICA_TOOLS/prep.log\" && "
        "t=\"$ORCHARD_REPLICA_TOOLS/bin/mytool\" && echo '#!/bin/sh' > \"$t\" && "
        "echo 'echo augmented' >> \"$t\" && chmod +x \"$t\"")


def sh(script):
    return ["/bin/sh", "-c", script]


@pytest.fixture
def bodies(tmp_path, monkeypatch):
    workers = [Worker("pro", "10.0.0.9"), Worker("laptop", "10.0.0.9")]
    homes = {}
    for w in workers:
        h = tmp_path / "remote" / w.name
        (h / ".orchard" / "bin").mkdir(parents=True)
        shutil.copy(AGENT_SRC, h / ".orchard" / "bin" / "orchard-agent.py")
        homes[w.name] = h
    quiet = set()

    def fake_run(worker, remote_cmd, stdin=None, timeout=30, stdin_file=None, stdout_file=None):
        if worker.name in quiet:
            return subprocess.CompletedProcess([], 255, b"", b"ssh: connect to host: timed out")
        argv = ["/bin/sh", "-c", remote_cmd]
        e = {"HOME": str(homes[worker.name]), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
        if stdin_file is not None or stdout_file is not None:
            fin = open(stdin_file, "rb") if stdin_file is not None else None
            fout = open(stdout_file, "wb") if stdout_file is not None else None
            try:
                return subprocess.run(argv, stdin=fin or subprocess.DEVNULL, stdout=fout or subprocess.PIPE,
                                      stderr=subprocess.PIPE, timeout=timeout, env=e)
            finally:
                for f in (fin, fout):
                    if f is not None:
                        f.close()
        return subprocess.run(argv, input=stdin if stdin is not None else b"", capture_output=True,
                              timeout=timeout, env=e)

    def fast_watch(worker, rjob, interval=4.0, lost_after=None):
        deadline = time.time() + 60
        while time.time() < deadline:
            if worker.name in quiet and lost_after is not None:
                return None
            st = engine.remote_status(worker, rjob)
            if st and st.get("state") in agent.TERMINAL_STATES:
                return st
            time.sleep(0.05)
        raise AssertionError(f"{rjob} never finished")

    real_payload = replica.agent_job_payload
    home = tmp_path / "home"
    monkeypatch.setattr(transport, "run", fake_run)
    monkeypatch.setattr(engine, "watch_remote", fast_watch)
    # The owner's activity on this Mac must not pause test jobs.
    monkeypatch.setattr(replica, "agent_job_payload", lambda *a, **k: {**real_payload(*a, **k), "preempt": False})
    monkeypatch.setattr(replica, "load_workers", lambda: workers)
    monkeypatch.setattr(replica, "REPLICAS_DIR", home / "replicas")
    monkeypatch.setattr(replica, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(replica, "_AGENT_READY", set())
    monkeypatch.setattr(fleet, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(fleet, "LOG_DIR", tmp_path / "fleet")
    monkeypatch.setattr(receipts, "KEY_PATH", home / "receipt.key")
    monkeypatch.setattr(receipts, "INDEX_DIR", home / "receipts")
    monkeypatch.setattr(events, "EVENTS_PATH", home / "events.jsonl")
    monkeypatch.setattr(speculate, "SPEC_DIR", home / "speculate")
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(dispatcher, "load_config", lambda: {})
    monkeypatch.setattr(dispatcher, "idle_hold", lambda caps, cfg: None)
    monkeypatch.setattr(env, "ensure_agent_uploaded", lambda w: None)
    monkeypatch.setattr(env, "ensure_uv_remote", lambda w: None)
    monkeypatch.setattr(env, "sync_env_remote", lambda *a, **k: None)
    return SimpleNamespace(workers=workers, by={w.name: w for w in workers}, homes=homes, quiet=quiet,
                           jobs=tmp_path / "jobs")


@pytest.fixture
def proj(tmp_path):
    p = tmp_path / "proj"
    (p / "sub").mkdir(parents=True)
    (p / "a.txt").write_text("alpha\n")
    (p / "sub" / "b.txt").write_text("beta\n")
    (p / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (p / "run.sh").chmod(0o755)
    (p / "link").symlink_to("a.txt")
    return p


def rep_dir(b, name, key):
    return b.homes[name] / ".orchard" / "replicas" / key


def in_process_children(monkeypatch):
    """check/compare/blame start `orchard replica run` children; run them here
    instead, so they see the fake bodies."""
    def child(cwd, label, argv, result_file):
        opts, cmd = argv[argv.index("run") + 1:argv.index("--")], argv[argv.index("--") + 1:]
        kw = {"result_file": Path(result_file)}
        it = iter(opts)
        for a in it:
            if a == "--result-file":
                next(it)
            elif a == "--as-is":
                kw["as_is"] = True
            elif a == "--no-local":
                kw["no_local"] = True
            elif a == "--pooled":
                kw["pooled"] = True
            elif a == "--target-file":
                kw["target_file"] = Path(next(it))
            elif a == "-w":
                kw["worker_name"] = next(it)
            elif a == "-t":
                kw.setdefault("tags", []).append(next(it))
            elif a == "--env":
                kw.setdefault("env_names", []).append(next(it))
            elif a == "--timeout":
                kw["timeout"] = int(next(it))
        try:
            replica.run(cmd, cwd, **kw)
        except replica.ReplicaError as exc:
            return {"state": "unavailable", "why": str(exc)}
        return json.loads(Path(result_file).read_text())
    monkeypatch.setattr(replica, "_run_child", child)


# --- the tree ---------------------------------------------------------------------

def test_the_three_digests_agree(bodies, proj, capsys):
    key = replica.project_key(proj)
    local, _sources = replica.local_manifest(proj, key)
    assert local["run.sh"][1] == "x"
    assert local["link"] == ["L", "a.txt", None]
    d = receipts.digest_manifest(local)
    assert d == receipts.tree_digest(proj)[0] == agent.manifest_digest(local)
    assert replica.up(proj, "pro") == 0
    st = replica.replica_state(bodies.by["pro"], key)
    assert st["digest"] == d and st["files"] == 4


def test_up_then_run_sends_only_what_changed(bodies, proj, tmp_path, capsys):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    out = capsys.readouterr().out
    assert "replica: pro holds tree" in out and "4 file(s) sent" in out and "is up as a day replica" in out
    work = rep_dir(bodies, "pro", key) / "work"
    assert (work / "a.txt").read_text() == "alpha\n"
    assert os.access(work / "run.sh", os.X_OK)
    assert os.readlink(work / "link") == "a.txt"

    rf = tmp_path / "r1.json"
    cmd = sh("./run.sh && mkdir -p out && cat a.txt > out/copy.txt")
    assert replica.run(cmd, proj, worker_name="pro", result_file=rf) == 0
    assert "day: pro already holds this tree" in capsys.readouterr().out
    r = json.loads(rf.read_text())
    assert r["state"] == "succeeded" and r["replica"] == "pro" and r["role"] == "day" and r["sealed"]
    assert not (rep_dir(bodies, "pro", key) / "current").exists()
    assert (proj / ".orchard-out" / r["job"] / "out" / "copy.txt").read_text() == "alpha\n"

    (proj / "a.txt").write_text("alpha two\n")
    (proj / "sub" / "b.txt").unlink()
    (proj / "sub").rmdir()
    assert replica.run(cmd, proj, worker_name="pro") == 0
    out = capsys.readouterr().out
    assert "day: sent 1 file(s)" in out and "removed 1 on pro" in out
    assert not (work / "sub").exists()
    assert (work / "a.txt").read_text() == "alpha two\n"

    assert replica.run(cmd, proj, worker_name="pro", reuse=True) == 0
    assert "reuse:" in capsys.readouterr().out


def test_a_check_that_rewrites_its_inputs_is_reported_and_undone(bodies, proj, capsys):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    assert replica.run(sh("echo more >> a.txt"), proj, worker_name="pro") == 0
    assert "hygiene: the check changed 1 tracked file(s) in the copy: a.txt" in capsys.readouterr().out
    drift = [json.loads(p.read_text()) for p in bodies.jobs.glob("*/drift.json")]
    assert [d["drifted"] for d in drift] == [["a.txt"]]
    assert replica.run(["true"], proj, worker_name="pro") == 0
    assert "day: sent 1 file(s)" in capsys.readouterr().out
    assert (rep_dir(bodies, "pro", key) / "work" / "a.txt").read_text() == "alpha\n"


def test_sync_recovers_when_the_copy_moved_under_it(bodies, proj, capsys):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    w = bodies.by["pro"]
    stale = replica.replica_state(w, key)
    (rep_dir(bodies, "pro", key) / "work" / "a.txt").write_text("someone was here\n")
    (proj / "sub" / "b.txt").write_text("beta two\n")
    local, sources = replica.local_manifest(proj, key)
    res = replica.sync(w, key, local, sources, stale)
    work = rep_dir(bodies, "pro", key) / "work"
    assert (work / "a.txt").read_text() == "alpha\n"
    assert (work / "sub" / "b.txt").read_text() == "beta two\n"
    assert res["digest"] == receipts.digest_manifest(local) == replica.replica_state(w, key)["digest"]


# --- succession ----------------------------------------------------------------------

def test_a_busy_body_is_passed_over_and_cannot_be_synced_or_dropped(bodies, proj, tmp_path, capsys):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    replica.up(proj, "laptop")
    stale = replica.replica_state(bodies.by["pro"], key)
    (rep_dir(bodies, "pro", key) / "current").write_text("someone-elses-job")
    rf = tmp_path / "r.json"
    assert replica.run(["true"], proj, result_file=rf) == 0
    assert "succession: pro cannot take it (busy with someone-elses-job)" in capsys.readouterr().out
    assert json.loads(rf.read_text())["replica"] == "laptop"

    (proj / "a.txt").write_text("changed\n")
    local, sources = replica.local_manifest(proj, key)
    with pytest.raises(replica.ReplicaBusy):
        replica.sync(bodies.by["pro"], key, local, sources, stale)
    with pytest.raises(replica.ReplicaBusy):
        replica.down(proj, "pro")
    assert (rep_dir(bodies, "pro", key) / "work").is_dir()


def test_a_quiet_body_is_succeeded_by_the_next(bodies, proj, tmp_path, capsys):
    replica.up(proj, "pro")
    replica.up(proj, "laptop")
    bodies.quiet.add("pro")
    rf = tmp_path / "r.json"
    assert replica.run(["true"], proj, result_file=rf) == 0
    assert "succession: pro cannot take it (quiet: no answer)" in capsys.readouterr().out
    assert json.loads(rf.read_text())["replica"] == "laptop"
    bodies.quiet.add("laptop")
    with pytest.raises(replica.ReplicaError, match="no replica is free"):
        replica.run(["true"], proj, no_local=True)


def test_the_nearest_mind_goes_first(bodies, proj):
    key = replica.project_key(proj)
    replica.up(proj, "laptop")
    (proj / "big.bin").write_bytes(b"x" * 50_000)
    replica.up(proj, "pro")
    local, sources = replica.local_manifest(proj, key)
    job = fleet.job_like_run(["true"], proj)
    ranked = replica.choose(key, replica.load_registry(key), job, None, local, sources)
    assert [(w.name, est) for w, est in ranked] == [("pro", 0), ("laptop", 50_000)]


# --- Dawn and augment ----------------------------------------------------------------

def test_dawn_runs_prep_only_when_its_keys_change_and_augments_the_path(bodies, proj, capsys):
    key = replica.project_key(proj)
    (proj / "package.json").write_text("{}\n")
    replica.up(proj, "pro", prep=PREP, augment=["node"])
    out = capsys.readouterr().out
    assert "dawn: pro is learning the project" in out and "augmented with node" in out
    tools = rep_dir(bodies, "pro", key) / "tools"
    assert (tools / "prep.log").read_text() == "ran\n"

    rf = proj.parent / "r.json"
    assert replica.run(sh("mkdir -p out && mytool > out/tool.txt"), proj, worker_name="pro", result_file=rf) == 0
    assert "dawn:" not in capsys.readouterr().out
    job = json.loads(rf.read_text())["job"]
    assert (proj / ".orchard-out" / job / "out" / "tool.txt").read_text() == "augmented\n"

    (proj / "a.txt").write_text("not a prep key\n")
    assert replica.run(["true"], proj, worker_name="pro") == 0
    assert "dawn:" not in capsys.readouterr().out
    (proj / "package.json").write_text('{"name": "x"}\n')
    assert replica.run(["true"], proj, worker_name="pro") == 0
    assert "dawn: pro is learning the project" in capsys.readouterr().out
    assert (tools / "prep.log").read_text() == "ran\nran\n"


def test_a_failed_prep_stops_the_run(bodies, proj):
    (proj / "package.json").write_text("{}\n")
    with pytest.raises(replica.ReplicaError, match="prep failed"):
        replica.up(proj, "pro", prep="exit 3")


# --- Dusk ------------------------------------------------------------------------------

def test_bless_moves_the_dusk_and_up_never_moves_it(bodies, proj, capsys):
    key = replica.project_key(proj)
    replica.up(proj, "laptop", role="dusk")
    replica.up(proj, "pro")
    first = receipts.digest_manifest(replica.local_manifest(proj, key)[0])
    (proj / "a.txt").write_text("alpha two\n")
    assert replica.run(["true"], proj, worker_name="pro", bless=True) == 0
    second = receipts.digest_manifest(replica.local_manifest(proj, key)[0])
    assert "dusk: laptop now remembers tree" in capsys.readouterr().out
    assert replica.load_registry(key)["dusk_tree"] == second != first
    assert (rep_dir(bodies, "laptop", key) / "work" / "a.txt").read_text() == "alpha two\n"

    (proj / "a.txt").write_text("alpha three\n")
    replica.up(proj, "laptop")
    assert "dusk: laptop keeps tree" in capsys.readouterr().out
    assert (rep_dir(bodies, "laptop", key) / "work" / "a.txt").read_text() == "alpha two\n"
    assert replica.run(["false"], proj, worker_name="pro", bless=True) == 1
    assert replica.load_registry(key)["dusk_tree"] == second


def test_compare_says_your_change_broke_it(bodies, proj, monkeypatch, capsys):
    in_process_children(monkeypatch)
    replica.up(proj, "laptop", role="dusk")
    replica.up(proj, "pro")
    (proj / "a.txt").write_text("broken\n")
    assert replica.compare(proj, sh("grep -q alpha a.txt")) == 1
    out = capsys.readouterr().out
    assert "your change broke it" in out and "orchard replica blame" in out
    assert replica.compare(proj, ["true"]) == 0
    assert "compare: green now" in capsys.readouterr().out


def test_blame_finds_the_file_that_breaks_it(bodies, proj, monkeypatch, capsys):
    in_process_children(monkeypatch)
    key = replica.project_key(proj)
    replica.up(proj, "laptop", role="dusk")
    replica.up(proj, "pro")
    (proj / "a.txt").write_text("broken\n")
    (proj / "sub" / "b.txt").write_text("beta two\n")
    (proj / "c.txt").write_text("new\n")
    assert replica.blame(proj, sh("grep -q alpha a.txt && test -f run.sh")) == 1
    out = capsys.readouterr().out
    assert "3 file(s) differ" in out
    assert "the failure needs these 1 file(s) of yours together:\n  a.txt\n" in out
    assert "on 1 body" in out
    # Blame never touches the dusk's copy.
    assert (rep_dir(bodies, "laptop", key) / "work" / "a.txt").read_text() == "alpha\n"

    (proj / "a.txt").write_text("alpha\n")
    assert replica.blame(proj, sh("grep -q alpha a.txt")) == 0
    assert "passes on your tree" in capsys.readouterr().out


def test_ddmin_finds_minimal_failing_sets():
    files = list("abcdefgh")
    calls = []

    def test(subsets):
        calls.append(len(subsets))
        return [{"b", "e"} <= s for s in subsets]

    found = replica.ddmin(files, test)
    assert sorted(found) == ["b", "e"]
    # Minimal: the set fails, and it passes with any one file taken out.
    assert test([set(found)]) == [True]
    assert test([set(found) - {f} for f in found]) == [False] * len(found)
    assert replica.ddmin(files, lambda subs: ["f" in s for s in subs]) == ["f"]
    assert all(n >= 1 for n in calls)


def test_run_pool_moves_work_off_a_body_that_cannot_answer():
    tasks = [{"id": i, "bodies": {"pro", "laptop"}} for i in range(4)] + [{"id": 9, "bodies": set()}]

    def launch(task, body):
        if body == "pro":
            return {"state": "unavailable", "why": "went quiet"}
        return {"state": "succeeded", "replica": body}

    results, stranded = replica._run_pool(tasks, ["pro", "laptop"], launch)
    assert sorted(results) == [0, 1, 2, 3] and all(r["replica"] == "laptop" for r in results.values())
    assert [t["id"] for t in stranded] == [9]

    only_pro = [{"id": 0, "bodies": {"pro"}}]
    results, stranded = replica._run_pool(only_pro, ["pro", "laptop"], launch)
    assert results == {} and stranded[0]["why"] == ["pro: went quiet"]


def test_run_pool_waits_for_work_that_can_come_back():
    # laptop finishes first and finds nothing left while pro still holds a
    # task; that task must still reach it when pro gives it up.
    tasks = [{"id": i, "bodies": {"pro", "laptop"}} for i in range(2)]

    def launch(task, body):
        if body == "pro":
            time.sleep(0.3)
            return {"state": "unavailable", "why": "went quiet"}
        return {"state": "succeeded", "replica": body}

    results, stranded = replica._run_pool(tasks, ["pro", "laptop"], launch)
    assert sorted(results) == [0, 1] and stranded == []

    def crash(task, body):
        raise RuntimeError("boom")

    results, stranded = replica._run_pool([{"id": 0, "bodies": {"pro"}}], ["pro"], crash)
    assert results == {} and stranded[0]["why"] == ["pro: RuntimeError: boom"]


def test_check_borrows_the_body_that_can_run_it_and_advances_the_dusk(bodies, proj, monkeypatch, capsys):
    in_process_children(monkeypatch)
    key = replica.project_key(proj)
    replica.up(proj, "pro", augment=["node"])
    replica.up(proj, "laptop")
    speculate.add(proj, speculate.make_check(["true"], ["node"]))
    speculate.add(proj, speculate.make_check(sh("test -x run.sh")))
    assert replica.check(proj) == 0
    out = capsys.readouterr().out
    assert "borrow: true needs node: only pro can run it" in out
    assert "checks:" in out and "succeeded" in out

    replica.set_role(proj, "laptop", "dusk")
    assert replica.check(proj, split="test -f sub/b.txt -a {i} -le {n}", shards=2) == 0
    assert "dusk: laptop now remembers tree" in capsys.readouterr().out
    assert replica.load_registry(key)["dusk_tree"] == receipts.digest_manifest(replica.local_manifest(proj, key)[0])
    with pytest.raises(replica.ReplicaError, match="--shards"):
        replica.check(proj, split="true")


def test_down_removes_the_copy_and_the_entry(bodies, proj, capsys):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    replica.down(proj, "pro")
    assert "replica: pro is down" in capsys.readouterr().out
    assert not rep_dir(bodies, "pro", key).exists()
    assert replica.load_registry(key)["replicas"] == {}


# --- the agent's side ------------------------------------------------------------------

KEY = "0123456789ab"


def _tgz(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _apply(home, incoming, members, token="abc123"):
    rep = home / "replicas" / KEY
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"incoming-{token}.json").write_text(json.dumps({"base": agent.manifest_digest({}), **incoming}))
    (rep / f"incoming-{token}.tgz").write_bytes(_tgz(members))
    return agent.cmd_replica_apply([KEY, token])


@pytest.mark.parametrize("name", ["../evil", "/tmp/evil", "sub/../../evil"])
def test_apply_refuses_paths_that_leave_the_copy(tmp_path, monkeypatch, capsys, name):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "h"))
    assert _apply(tmp_path / "h", {"updates": [name]}, {name: b"x"}) == 6
    assert not (tmp_path / "evil").exists()
    assert not list((tmp_path / "h" / "replicas" / KEY / "work").iterdir())


def test_apply_refuses_members_it_did_not_ask_for(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "h"))
    assert _apply(tmp_path / "h", {"updates": ["a.txt"]}, {"a.txt": b"a", "b.txt": b"b"}) == 6


def test_apply_replaces_a_link_parent_with_a_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(tmp_path / "h"))
    outside = tmp_path / "outside"
    outside.mkdir()
    work = tmp_path / "h" / "replicas" / KEY / "work"
    work.mkdir(parents=True)
    (work / "d").symlink_to(outside)
    rep = work.parent
    manifest, meta = agent._replica_files(str(rep))
    agent._replica_refresh(str(rep), manifest)
    agent._replica_write(str(rep), manifest, meta)
    base = agent.manifest_digest(manifest)
    assert _apply(tmp_path / "h", {"base": base, "deleted": ["d"], "updates": ["d/f.txt"]}, {"d/f.txt": b"f"}) == 0
    assert (work / "d").is_dir() and not (work / "d").is_symlink()
    assert (work / "d" / "f.txt").read_bytes() == b"f"
    assert not list(outside.iterdir())


def test_clear_out_keeps_tracked_outputs(tmp_path):
    work = tmp_path / "work"
    (work / "out" / "old").mkdir(parents=True)
    (work / "out" / "keep.txt").write_text("k")
    (work / "out" / "old" / "stale.txt").write_text("s")
    agent._replica_clear_out(str(work), {"out/keep.txt": ["F", "-", "x", 1, 0]})
    assert (work / "out" / "keep.txt").exists()
    assert not (work / "out" / "old" / "stale.txt").exists()


def test_gc_keeps_a_busy_replica(tmp_path, monkeypatch, capsys):
    home = tmp_path / "h"
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(home))
    old = time.time() - 40 * 86400
    reps = {}
    for name, key in (("idle", "aaaaaaaaaaaa"), ("busy", "bbbbbbbbbbbb")):
        rep = home / "replicas" / key
        (rep / "work").mkdir(parents=True)
        (rep / "meta.json").write_text("{}")
        reps[name] = rep
    (reps["busy"] / "current").write_text("job-in-progress")
    for rep in reps.values():
        os.utime(rep / "meta.json", (old, old))
        os.utime(rep, (old, old))
    entries = {Path(e["path"]).name: e for e in agent._gc_entries(time.time(), 2) if e.get("replica")}
    assert entries["bbbbbbbbbbbb"]["protected"] and not entries["aaaaaaaaaaaa"]["protected"]
    assert entries["aaaaaaaaaaaa"]["max_age_days"] == agent.REPLICA_MIN_AGE_DAYS
    monkeypatch.setattr(agent, "_gc_free_gb", lambda: 100.0)
    assert agent.cmd_gc(["2", "0"]) == 0
    assert not reps["idle"].exists() and reps["busy"].exists()


# --- CLI ------------------------------------------------------------------------------------

def test_cli_lists_replica_commands_and_refuses_without_replicas(bodies, proj, monkeypatch):
    out = CliRunner().invoke(cli.app, ["replica", "--help"]).output
    for verb in ("up", "run", "check", "compare", "blame", "list", "role", "down"):
        assert verb in out
    monkeypatch.chdir(proj)
    res = CliRunner().invoke(cli.app, ["replica", "run", "--", "true"])
    assert res.exit_code == 2 and "error: no replicas yet" in res.output


# --- review fixes: the Air's side ------------------------------------------------------------

def test_begin_starts_only_on_the_tree_it_names(bodies, proj):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    w = bodies.by["pro"]
    st = replica.replica_state(w, key)
    job = replica._project_job(proj, ["true"])
    (rep_dir(bodies, "pro", key) / "work" / "a.txt").write_text("someone was here\n")
    with pytest.raises(replica._Moved) as moved:
        replica._start(w, key, st["home"], job, replica.new_job_id("t"), None, st["digest"])
    assert moved.value.drifted == ["a.txt"]
    assert not (rep_dir(bodies, "pro", key) / "current").exists()
    replica.down(proj, "pro")
    with pytest.raises(replica.ReplicaError, match="has no copy"):
        replica._start(w, key, st["home"], job, replica.new_job_id("t"), None, st["digest"])


def test_prep_that_rewrites_a_tracked_file_is_sent_back_before_the_run(bodies, proj, capsys):
    (proj / "package.json").write_text("{}\n")
    replica.up(proj, "pro", prep="echo x >> a.txt")
    (proj / "package.json").write_text('{"name": "x"}\n')
    capsys.readouterr()
    assert replica.run(sh("grep -qx alpha a.txt"), proj, worker_name="pro") == 0
    out = capsys.readouterr().out
    assert "dawn: prep changed 1 tracked file(s) (a.txt); sending them back" in out
    assert "hygiene:" not in out


def test_a_body_runs_one_job_at_a_time(bodies, proj, monkeypatch):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    real = replica.replica_state

    def state(worker, k, *a, **kw):
        st = real(worker, k, *a, **kw)
        if st:
            st["caps"]["busy_jobs"] = ["other-job"]
        return st

    monkeypatch.setattr(replica, "replica_state", state)
    local, sources = replica.local_manifest(proj, key)
    job = fleet.job_like_run(["true"], proj)
    with pytest.raises(replica.ReplicaBusy, match="other-job"):
        replica._launch(bodies.by["pro"], key, proj, job, replica.load_registry(key), local, sources, {}, False)


def test_check_compare_and_blame_honour_idle_holds(bodies, proj, monkeypatch, capsys):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    replica.up(proj, "laptop")
    real = replica.replica_state

    def state(worker, k, *a, **kw):
        st = real(worker, k, *a, **kw)
        if st:
            st["caps"]["who"] = worker.name
        return st

    monkeypatch.setattr(replica, "replica_state", state)
    monkeypatch.setattr(dispatcher, "idle_hold", lambda caps, cfg: "owner active" if caps.get("who") == "pro" else None)
    capsys.readouterr()
    with pytest.raises(replica.ReplicaError, match="no replica is free"):
        replica.run(["true"], proj, worker_name="pro", pooled=True, no_local=True)
    assert "succession: pro cannot take it (owner active)" in capsys.readouterr().out
    assert replica.run(["true"], proj, worker_name="pro") == 0
    assert "note: pro: owner active (running anyway: you named it)" in capsys.readouterr().out
    days, _probes = replica._usable_days(key, replica.load_registry(key))
    assert days == ["laptop"] and "succession: pro is held (owner active)" in capsys.readouterr().out


def test_a_body_that_goes_quiet_mid_run_is_given_up_on(bodies, proj, tmp_path, monkeypatch, capsys):
    replica.up(proj, "pro")
    monkeypatch.setattr(engine, "watch_remote", lambda *a, **kw: None)
    rf = tmp_path / "r.json"
    assert replica.run(["true"], proj, worker_name="pro", result_file=rf) == 1
    assert "succession: pro went quiet mid-run" in capsys.readouterr().out
    r = json.loads(rf.read_text())
    assert r["state"] == "unavailable" and r["replica"] == "pro" and r["job"]


def test_watch_remote_gives_up_on_a_silent_worker(monkeypatch):
    monkeypatch.setattr(engine, "remote_status", lambda worker, rjob: None)
    assert engine.watch_remote(Worker("pro", "10.0.0.9"), "/nowhere", lost_after=0) is None


def test_a_case_only_rename_reaches_the_copy(bodies, proj):
    key = replica.project_key(proj)
    (proj / "Readme.txt").write_text("r\n")
    replica.up(proj, "pro")
    (proj / "Readme.txt").rename(proj / "tmp-name")
    (proj / "tmp-name").rename(proj / "README.txt")
    assert replica.run(["true"], proj, worker_name="pro") == 0
    names = os.listdir(rep_dir(bodies, "pro", key) / "work")
    assert "README.txt" in names and "Readme.txt" not in names


def test_an_entry_that_is_not_a_file_or_link_is_named_and_refused(bodies, proj):
    key = replica.project_key(proj)
    os.mkfifo(proj / "pipe")
    local, _sources = replica.local_manifest(proj, key)
    assert local["pipe"][0] == "?"
    assert receipts.digest_manifest(local) == receipts.tree_digest(proj)[0] == agent.manifest_digest(local)
    with pytest.raises(replica.ReplicaError, match="cannot send pipe"):
        replica.up(proj, "pro")


def test_digest_entries_names_a_path_once(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("a\n")
    assert receipts.digest_entries([(f, "a.txt"), (f, "a.txt")]) == receipts.digest_entries([(f, "a.txt")])


def test_the_dusk_moves_only_on_green_and_compare_and_blame_refuse_a_drifted_one(bodies, proj, monkeypatch, capsys):
    in_process_children(monkeypatch)
    key = replica.project_key(proj)
    replica.up(proj, "laptop", role="dusk")
    replica.up(proj, "pro")
    with pytest.raises(replica.ReplicaError, match="moves only on green"):
        replica.run(["true"], proj, worker_name="laptop")
    assert replica.run(["true"], proj, worker_name="laptop", as_is=True) == 0
    assert replica.run(["true"], proj, worker_name="pro", bless=True) == 0
    dusk_a = rep_dir(bodies, "laptop", key) / "work" / "a.txt"

    dusk_a.write_text("someone was here\n")
    capsys.readouterr()
    assert replica.compare(proj, ["true"]) == 2
    assert "no verdict while the dusk is away from its blessed tree" in capsys.readouterr().out
    (proj / "a.txt").write_text("broken\n")
    with pytest.raises(replica.ReplicaError, match="not the tree it was blessed at"):
        replica.blame(proj, sh("grep -q alpha a.txt"))

    # A green run on the blessed tree puts the dusk's copy back.
    (proj / "a.txt").write_text("alpha\n")
    assert replica.run(["true"], proj, worker_name="pro", bless=True) == 0
    assert "drifted from tree" in capsys.readouterr().out
    assert dusk_a.read_text() == "alpha\n"


def test_compare_says_why_the_run_did_not_finish(bodies, proj, monkeypatch, capsys):
    in_process_children(monkeypatch)
    replica.up(proj, "laptop", role="dusk")
    replica.up(proj, "pro")
    bodies.quiet.add("pro")
    monkeypatch.setattr(fleet, "run_on", lambda *a, **kw: {"state": "unavailable", "why": "no room here"})
    assert replica.compare(proj, ["true"]) == 1
    assert "compare: the run did not finish (unavailable: no room here)" in capsys.readouterr().out


def test_blame_gives_no_verdict_when_a_tree_does_not_finish(bodies, proj, monkeypatch):
    in_process_children(monkeypatch)
    replica.up(proj, "laptop", role="dusk")
    replica.up(proj, "pro")
    (proj / "a.txt").write_text("broken\n")
    with pytest.raises(replica.ReplicaError, match="no verdict"):
        replica.blame(proj, sh("sleep 30"), timeout=1)


def test_a_green_run_here_still_blesses_the_dusk(bodies, proj, tmp_path, monkeypatch):
    key = replica.project_key(proj)
    replica.up(proj, "laptop", role="dusk")
    replica.up(proj, "pro")
    bodies.quiet.add("pro")
    blessed = []
    monkeypatch.setattr(fleet, "run_on", lambda *a, **kw: {"state": "succeeded", "job": "local-1"})
    monkeypatch.setattr(replica, "advance_dusk", lambda cwd, k, tree, source, env=None: blessed.append(tree) or True)
    rf = tmp_path / "r.json"
    assert replica.run(["true"], proj, bless=True, result_file=rf) == 0
    tree = receipts.digest_manifest(replica.local_manifest(proj, key)[0])
    assert blessed == [tree] and json.loads(rf.read_text())["tree"] == tree


def test_up_and_the_cli_report_a_quiet_body_without_a_traceback(bodies, proj, monkeypatch):
    def quiet(*a, **kw):
        raise transport.RemoteError("ssh: timed out")

    monkeypatch.setattr(replica, "_up", quiet)
    with pytest.raises(replica.ReplicaError, match="pro went quiet"):
        replica.up(proj, "pro")
    monkeypatch.setattr(replica, "run", quiet)
    monkeypatch.chdir(proj)
    res = CliRunner().invoke(cli.app, ["replica", "run", "--", "true"])
    assert res.exit_code == 2 and "error: a body went quiet" in res.output


def test_down_keeps_the_entry_when_the_copy_is_not_all_gone(bodies, proj, monkeypatch):
    key = replica.project_key(proj)
    replica.up(proj, "pro")
    monkeypatch.setattr(replica, "remote_call", lambda *a, **kw: subprocess.CompletedProcess(
        [], 7, json.dumps({"error": "some files could not be removed", "left": "/x"}).encode(), b""))
    with pytest.raises(replica.ReplicaError, match="some files could not be removed; it stays listed"):
        replica.down(proj, "pro")
    assert "pro" in replica.load_registry(key)["replicas"]


# --- review fixes: the agent's side ----------------------------------------------------------

def _state(capsys):
    capsys.readouterr()
    assert agent.cmd_replica_state([KEY]) == 0
    return json.loads(capsys.readouterr().out)


def test_a_failed_apply_leaves_copy_and_manifest_agreeing(tmp_path, monkeypatch, capsys):
    home = tmp_path / "h"
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(home))
    assert _apply(home, {"updates": ["a.txt", "b.txt"]}, {"a.txt": b"a"}) == 6
    rep = home / "replicas" / KEY
    assert _state(capsys)["digest"] == receipts.tree_digest(rep / "work")[0]
    assert not (rep / "applying.json").exists() and not list(rep.glob("incoming*"))


def test_an_interrupted_apply_is_rehashed_from_its_journal(tmp_path, monkeypatch, capsys):
    home = tmp_path / "h"
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(home))
    assert _apply(home, {"updates": ["a.txt"]}, {"a.txt": b"a"}) == 0
    rep = home / "replicas" / KEY
    a = rep / "work" / "a.txt"
    before = a.stat()
    # Same size and mtime: only the journal says this file may have changed.
    a.write_bytes(b"b")
    os.utime(a, ns=(before.st_atime_ns, before.st_mtime_ns))
    (rep / "work" / ".orchard-incoming-123").write_bytes(b"half")
    (rep / "applying.json").write_text(json.dumps(["a.txt"]))
    assert _state(capsys)["digest"] == receipts.tree_digest(rep / "work")[0]
    assert not (rep / "applying.json").exists() and not (rep / "work" / ".orchard-incoming-123").exists()


def test_each_update_has_its_own_names(tmp_path, monkeypatch):
    home = tmp_path / "h"
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(home))
    rep = home / "replicas" / KEY
    rep.mkdir(parents=True)
    (rep / "incoming-def456.json").write_text("{}")
    (rep / "incoming-def456.tgz").write_bytes(b"")
    assert _apply(home, {"updates": ["a.txt"]}, {"a.txt": b"a"}, token="abc123") == 0
    assert (rep / "incoming-def456.json").exists() and (rep / "incoming-def456.tgz").exists()
    with pytest.raises(ValueError, match="bad update token"):
        agent.cmd_replica_apply([KEY, "../x"])


def _begin_job(home, job_id):
    rep = home / "replicas" / KEY
    job_dir = home / "jobs" / job_id
    job_dir.mkdir(parents=True)
    job_json = job_dir / "job.json"
    job_json.write_text(json.dumps({"id": job_id, "workdir": str(rep / "work"), "job_dir": str(job_dir),
                                    "cmd": ["true"]}))
    return str(job_json)


def test_begin_refuses_while_another_job_runs_on_the_machine(tmp_path, monkeypatch, capsys):
    home = tmp_path / "h"
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(home))
    assert _apply(home, {"updates": ["a.txt"]}, {"a.txt": b"a"}) == 0
    digest = _state(capsys)["digest"]
    monkeypatch.setattr(agent, "running_jobs", lambda: ["other-1"])
    monkeypatch.setattr(agent, "read_status", lambda job_dir: {"state": "running"})
    assert agent.cmd_replica_begin([KEY, _begin_job(home, "mine-1"), digest]) == 3
    assert json.loads(capsys.readouterr().out) == {"error": "busy", "busy": "other-1"}
    assert not (home / "replicas" / KEY / "current").exists()


def test_a_finished_job_holds_its_copy_until_released(tmp_path, monkeypatch, capsys):
    home = tmp_path / "h"
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(home))
    assert _apply(home, {"updates": ["a.txt"]}, {"a.txt": b"a"}) == 0
    rep = home / "replicas" / KEY
    (rep / "current").write_text("job-a")
    (home / "jobs" / "job-a").mkdir(parents=True)
    (home / "jobs" / "job-a" / "status.json").write_text(json.dumps({"state": "succeeded"}))
    assert agent._replica_job(str(rep)) == ("job-a", "collecting")
    capsys.readouterr()
    assert agent.cmd_replica_release([KEY, "job-b"]) == 0
    assert json.loads(capsys.readouterr().out) == {"released": False} and (rep / "current").exists()
    assert agent.cmd_replica_release([KEY, "job-a"]) == 0
    assert json.loads(capsys.readouterr().out) == {"released": True} and not (rep / "current").exists()


def test_down_says_when_files_are_left(tmp_path, monkeypatch, capsys):
    home = tmp_path / "h"
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(home))
    assert _apply(home, {"updates": ["a.txt"]}, {"a.txt": b"a"}) == 0
    monkeypatch.setattr(agent, "_rmtree_force", lambda path: False)
    capsys.readouterr()
    assert agent.cmd_replica_down([KEY]) == 7
    assert json.loads(capsys.readouterr().out)["error"] == "some files could not be removed"


def test_gc_spares_a_replica_used_today_even_on_low_disk(tmp_path, monkeypatch):
    home = tmp_path / "h"
    monkeypatch.setattr(agent, "ORCHARD_HOME", str(home))
    old, few_days = time.time() - 40 * 86400, time.time() - 3 * 86400
    reps = {}
    for name, key in (("stale", "aaaaaaaaaaaa"), ("used", "bbbbbbbbbbbb"), ("cooling", "cccccccccccc")):
        rep = home / "replicas" / key
        (rep / "work").mkdir(parents=True)
        (rep / "meta.json").write_text("{}")
        reps[name] = rep
    (reps["used"] / "used").write_text("")
    for name, rep in reps.items():
        when = few_days if name == "cooling" else old
        os.utime(rep / "meta.json", (when, when))
        os.utime(rep, (when, when))
    monkeypatch.setattr(agent, "_gc_free_gb", lambda: 1.0)
    assert agent.cmd_gc(["2", "50"]) == 0
    assert not reps["stale"].exists() and not reps["cooling"].exists() and reps["used"].exists()
