import fcntl
import json
import subprocess
import time

import pytest
from typer.testing import CliRunner

from orchard import cli, dispatcher, events, fleet, receipts, speculate
from orchard.job import slugify
from orchard.speculate import make_check


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(speculate, "SPEC_DIR", tmp_path / "home" / "speculate")
    monkeypatch.setattr(receipts, "KEY_PATH", tmp_path / "home" / "receipt.key")
    monkeypatch.setattr(receipts, "INDEX_DIR", tmp_path / "home" / "receipts")
    monkeypatch.setattr(events, "EVENTS_PATH", tmp_path / "home" / "events.jsonl")
    monkeypatch.setattr(fleet, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(fleet, "LOG_DIR", tmp_path / "fleet")
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    return root


def put(project, name, work, key="k", state="succeeded", elapsed=3.0):
    receipt = receipts.sign({"v": receipts.VERSION, "job": name, "project": str(project), "work": work,
                             "work_key": key, "worker": {"name": "local"}, "state": state,
                             "elapsed_s": elapsed, "sealed_at": time.time()})
    folder = receipts.INDEX_DIR / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(receipt))
    return receipt


def work(project, cmd, tags=(), env=(), **extra):
    """Receipt work for a check, with the kind and requires a check runs with."""
    base = receipts.work_fields(fleet.job_like_run(["x"], project))
    return {"cmd": cmd, "tags": list(tags), "env": list(env), "kind": base["kind"],
            "requires": base["requires"], **extra}


def key_now(project, cmd):
    return receipts.describe(project, fleet.job_like_run(cmd, project))["work_key"]


def entry(tmp_path, job_id, cmd, project, state, launched_key=None):
    sdir = tmp_path / "jobs" / job_id
    sdir.mkdir(parents=True, exist_ok=True)
    if launched_key:
        (sdir / "inputs.json").write_text(json.dumps({"work_key": launched_key}))
    spec = dict(fleet.job_like_run(cmd, project).__dict__)
    return {"id": job_id, "spec": spec, "cwd": str(project), "sdir": str(sdir), "state": state}


def test_checks_live_outside_the_project_and_update_in_place(proj):
    check = make_check(["pytest"], ["x86_64", "x86_64"], ["TOKEN"], 60)
    assert check == {"cmd": ["pytest"], "tags": ["x86_64"], "env": ["TOKEN"], "timeout": 60}
    assert speculate.add(proj, check) == "added"
    assert speculate.add(proj, {**check, "timeout": 90}) == "updated"
    speculate.add(proj, make_check(["pytest"], ["os:linux"]))
    speculate.add(proj, make_check(["npm", "test"]))
    assert [c["timeout"] for c in speculate.load(proj)] == [90, None, None]
    assert speculate.load(proj.parent / "other") == []
    assert sorted(p.name for p in proj.iterdir()) == ["a.py"]
    assert speculate.remove(proj, ["pytest"]) == 2
    assert [c["cmd"] for c in speculate.load(proj)] == [["npm", "test"]]


def test_learn_counts_this_projects_runs_but_not_its_own(proj, tmp_path):
    for i in range(3):
        put(proj, f"pytest-{i}", work(proj, ["pytest"]))
    put(proj, "spec-user-0", work(proj, ["pytest"]))  # a user's job, whatever its name
    for i in range(2):
        put(proj, f"npm-{i}", work(proj, ["npm", "test"], ["os:any"], ["CI"]))
    for i in range(5):
        put(proj, f"spec-x-{i}", work(proj, ["make"]))
        speculate._record(proj, f"spec-x-{i}")
        put(tmp_path / "other", f"other-{i}", work(proj, ["cargo", "test"]))
        put(proj, f"build-{i}", work(proj, ["swift", "build"], kind="build"))
        put(proj, f"big-{i}", work(proj, ["train"], requires={"mem_gb": 6}))
    forged = put(proj, "forged", work(proj, ["rm"]))
    forged["work"]["cmd"] = ["curl"]
    for i in range(4):
        (receipts.INDEX_DIR / "k" / f"forged-{i}.json").write_text(json.dumps(forged))
    assert speculate.learn(proj, top=5) == [
        {**make_check(["pytest"]), "runs": 4},
        {**make_check(["npm", "test"], ["os:any"], ["CI"]), "runs": 2},
        {**make_check(["rm"]), "runs": 1},
    ]
    assert len(speculate.learn(proj, top=1)) == 1


def test_run_queues_only_what_is_unanswered_and_cancels_only_its_own_stale_jobs(proj, tmp_path, monkeypatch):
    checks = [make_check([c]) for c in ("check-a", "check-b", "check-c", "check-d", "check-f")]
    put(proj, "check-a-20261103-120000-aaaa", {"cmd": ["check-a"]}, key=key_now(proj, ["check-a"]))
    entries = [
        entry(tmp_path, "spec-1-check-b", ["check-b"], proj, "queued"),
        entry(tmp_path, "spec-2-check-d", ["check-d"], proj, "running", launched_key="old"),
        entry(tmp_path, "spec-3-check-f", ["check-f"], proj, "running", launched_key=key_now(proj, ["check-f"])),
        entry(tmp_path, "check-d-mine", ["check-d"], proj, "running", launched_key="old"),
        entry(tmp_path, "spec-4-check-d", ["check-d"], tmp_path / "other", "running", launched_key="old"),
        entry(tmp_path, "spec-5-check-d", ["check-d"], proj, "done", launched_key="old"),
        entry(tmp_path, "spec-6-check-b", ["check-b"], proj, "queued"),
        entry(tmp_path, "spec-9-check-d", ["check-d"], proj, "running", launched_key="old"),
    ]
    for job in ("spec-1-check-b", "spec-2-check-d", "spec-3-check-f", "spec-5-check-d", "spec-6-check-b"):
        speculate._record(proj, job)  # spec-9 is a user's job that only looks like one
    monkeypatch.setattr(dispatcher, "load_entries", lambda: entries)
    submitted, cancelled = [], []
    monkeypatch.setattr(speculate, "submit",
                        lambda project, check: submitted.append(check["cmd"]) or (f"spec-new-{check['cmd'][0]}", None))
    monkeypatch.setattr(speculate, "cancel", lambda project, job: cancelled.append(job) or True)

    report = speculate.run(proj, checks, limit=4)
    states = {r["cmd"][0]: (r["state"], r["job"]) for r in report["checks"]}
    assert states == {
        "check-a": ("answered", "check-a-20261103-120000-aaaa"),
        "check-b": ("in flight", "spec-1-check-b"),
        "check-c": ("queued", "spec-new-check-c"),
        "check-d": ("queued", "spec-new-check-d"),
        "check-f": ("in flight", "spec-3-check-f"),
    }
    assert submitted == [["check-c"], ["check-d"]]
    # The stale run and the duplicate of check-b go; the user's spec-9 stays.
    assert cancelled == report["cancelled"] == ["spec-2-check-d", "spec-6-check-b"]
    assert [e["event"] for e in events.load()] == ["speculated"]

    # The cap counts this project's unfinished speculative jobs. Here nothing
    # will cancel, so four stay in flight.
    monkeypatch.setattr(speculate, "cancel", lambda project, job: False)
    submitted.clear()
    capped = speculate.run(proj, checks, limit=4)
    assert [r["state"] for r in capped["checks"]] == ["answered", "in flight", "held", "held", "in flight"]
    assert submitted == [] and "4 speculative jobs already in flight" in capped["checks"][2]["why"]
    # A submit that fails takes no slot, so the next check still goes.
    monkeypatch.setattr(speculate, "submit", lambda project, check: submitted.append(check["cmd"]) or (
        (None, "error: no worker") if check["cmd"] == ["check-c"] else ("spec-new", None)))
    capped = speculate.run(proj, checks, limit=5)
    assert [r["state"] for r in capped["checks"]][2:4] == ["not queued", "queued"]
    assert submitted == [["check-c"], ["check-d"]]


def test_a_failed_run_is_not_repeated_on_the_same_tree(proj, tmp_path, monkeypatch):
    put(proj, "check-g-1", {"cmd": ["check-g"]}, key=key_now(proj, ["check-g"]), state="failed")
    entries = [entry(tmp_path, "spec-7-check-g", ["check-g"], proj, "queued")]
    speculate._record(proj, "spec-7-check-g")
    monkeypatch.setattr(dispatcher, "load_entries", lambda: entries)
    monkeypatch.setattr(speculate, "submit", lambda project, check: pytest.fail("must not queue"))
    monkeypatch.setattr(speculate, "cancel", lambda project, job: True)
    report = speculate.run(proj, [make_check(["check-g"])])
    assert report["checks"][0]["state"] == "failed" and report["checks"][0]["job"] == "check-g-1"
    assert report["cancelled"] == ["spec-7-check-g"]


def test_one_run_at_a_time_per_project(proj, monkeypatch):
    def survey(project, checks):
        with open(speculate._path(project).with_suffix(".lock"), "a") as fh:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return [], []

    monkeypatch.setattr(speculate, "survey", survey)
    monkeypatch.setattr(dispatcher, "load_entries", lambda: [])
    assert speculate.run(proj, []) == {"checks": [], "cancelled": []}


def test_submit_queues_a_named_background_job(proj, monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        name = argv[argv.index("--name") + 1]
        job_id = f"{slugify(name)}-20261103-120000-abcd"
        (fleet.JOBS_DIR / job_id).mkdir(parents=True)
        dispatcher.QUEUE_DIR.mkdir(exist_ok=True)
        dispatcher.entry_path(job_id).write_text("{}")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(speculate.subprocess, "run", fake_run)
    job, why = speculate.submit(proj, make_check(["npm", "test"], ["x86_64"], ["CI"], 120))
    argv, kw = calls[0]
    assert job.startswith("spec-") and job.endswith("-20261103-120000-abcd") and why is None
    opts = argv[argv.index("run") + 1:argv.index("--")]
    assert opts[:2] == ["-w", "auto"] and "-b" in opts and "--no-local" in opts
    assert {"--timeout", "120", "-t", "x86_64", "--env", "CI"} <= set(opts)
    assert argv[-3:] == ["--", "npm", "test"] and kw["cwd"] == proj and "PYTHONPATH" in kw["env"]
    assert speculate.own_ids(proj) == {job}

    # The child makes the job folder before placement, then fails: no entry,
    # so nothing was queued, and the folder it left is cleared away.
    def fails_after_mkdir(argv, **kw):
        name = argv[argv.index("--name") + 1]
        (fleet.JOBS_DIR / f"{slugify(name)}-20261103-120001-abcd").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 2, "", "error: no worker with required tags: x86_64\n")

    monkeypatch.setattr(speculate.subprocess, "run", fails_after_mkdir)
    assert speculate.submit(proj, make_check(["x"], ["x86_64"])) == (None, "error: no worker with required tags: x86_64")
    assert [d.name for d in fleet.JOBS_DIR.iterdir()] == [job] and speculate.own_ids(proj) == {job}

    monkeypatch.setattr(speculate.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    assert speculate.cancel(proj, "spec-abcdef-pytest-20261103-120000-abcd") is False
    assert speculate.cancel(proj.parent / "other", job) is False
    monkeypatch.setattr(speculate.subprocess, "run",
                        lambda argv, **kw: calls.append((argv, kw)) or subprocess.CompletedProcess(argv, 0, "", ""))
    assert speculate.cancel(proj, job) is True and calls[-1][0][-2:] == ["cancel", job]


def test_fingerprint_sees_edits_but_not_collected_results(proj):
    before = speculate.fingerprint(proj)
    out = proj / ".orchard-out" / "job" / "out"
    out.mkdir(parents=True)
    (out / "result.txt").write_text("done\n")
    assert speculate.fingerprint(proj) == before
    (proj / "a.py").chmod(0o755)
    moved = speculate.fingerprint(proj)
    assert moved != before
    (proj / "a.py").write_text("x = 2  # changed\n")
    assert speculate.fingerprint(proj) != moved


def test_watch_runs_once_the_tree_holds_still(proj, monkeypatch):
    seen = iter(["A", "A", "B", "C", "C", "C", "C", "D", "D", "D"])
    monkeypatch.setattr(speculate, "fingerprint", lambda project: next(seen))
    monkeypatch.setattr(speculate, "run", lambda project, limit: {"checks": [], "cancelled": []})
    said, slept = [], []
    speculate.watch(proj, settle=2, interval=1, rounds=9, sleep=slept.append, say=said.append)
    # At start, once C held still for two looks, once D did; never mid-change.
    assert len(said) == 3 and slept == [1] * 9


def test_watch_sees_an_edit_made_during_its_first_run(proj, monkeypatch):
    seen = iter(["A", "B", "B", "B"])
    monkeypatch.setattr(speculate, "fingerprint", lambda project: next(seen))
    monkeypatch.setattr(speculate, "run", lambda project, limit: {"checks": [], "cancelled": []})
    said = []
    speculate.watch(proj, settle=2, interval=1, rounds=3, sleep=lambda s: None, say=said.append)
    assert len(said) == 2


def test_watch_retries_held_checks_and_errors_without_a_change(proj, monkeypatch):
    monkeypatch.setattr(speculate, "fingerprint", lambda project: "A")
    outcomes = iter([RuntimeError("queue unreadable"), {"checks": [{"state": "held"}], "cancelled": []},
                     {"checks": [{"state": "queued"}], "cancelled": []}])

    def run(project, limit):
        out = next(outcomes)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(speculate, "run", run)
    said = []
    speculate.watch(proj, settle=2, interval=1, rounds=8, sleep=lambda s: None, say=said.append)
    assert said[0]["error"] == "RuntimeError: queue unreadable"
    assert [r["checks"][0]["state"] for r in said[1:]] == ["held", "queued"]


def test_stats_counts_hits_saved_wait_and_waste(proj, tmp_path):
    put(proj, "spec-a", {"cmd": ["a"]}, elapsed=10.0)
    put(proj, "spec-b", {"cmd": ["b"]}, elapsed=4.0)
    put(proj, "pytest-1", {"cmd": ["pytest"]}, elapsed=99.0)
    put(tmp_path / "other", "spec-c", {"cmd": ["c"]}, elapsed=7.0)
    put(proj, "spec-user", {"cmd": ["d"]}, elapsed=50.0)  # named like one, never queued by speculation
    speculate._record(proj, "spec-a")
    speculate._record(proj, "spec-b")
    speculate._record(tmp_path / "other", "spec-c")
    for job in ("spec-a", "spec-a", "pytest-1", "spec-c"):
        events.append("reused", job=job)
    assert speculate.stats(proj) == {"speculated": 2, "asked_for": 1, "hit_rate": 0.5,
                                     "saved_s": 20.0, "wasted_s": 4.0}


def test_cli_add_list_run_remove(proj, monkeypatch):
    monkeypatch.chdir(proj)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["speculate", "run"]).exit_code == 2
    added = runner.invoke(cli.app, ["speculate", "add", "-t", "x86_64", "--", "pytest", "-q"])
    assert added.exit_code == 0 and "added pytest -q  (-t x86_64)" in added.output
    monkeypatch.setattr(dispatcher, "load_entries", lambda: [])
    listed = json.loads(runner.invoke(cli.app, ["speculate", "list", "--json"]).output)
    assert listed["checks"][0]["state"] == "to queue" and listed["checks"][0]["cmd"] == ["pytest", "-q"]
    monkeypatch.setattr(speculate, "submit", lambda project, check: (None, "error: no worker matches"))
    ran = runner.invoke(cli.app, ["speculate", "run"])
    assert ran.exit_code == 1 and "not queued pytest -q  error: no worker matches" in ran.output
    assert "no speculative runs" in runner.invoke(cli.app, ["speculate", "stats"]).output
    assert runner.invoke(cli.app, ["speculate", "watch", "--interval", "0"]).exit_code == 2
    # learn adds what is new and keeps a timeout already set on a known check.
    runner.invoke(cli.app, ["speculate", "add", "--timeout", "60", "--", "pytest", "-q"])
    for i in range(2):
        put(proj, f"pytest-{i}", work(proj, ["pytest", "-q"]))
    put(proj, "lint-0", work(proj, ["lint"]))
    assert runner.invoke(cli.app, ["speculate", "learn"]).exit_code == 0
    assert [(c["cmd"], c["timeout"]) for c in speculate.load(proj)] == [
        (["pytest", "-q"], None), (["pytest", "-q"], 60), (["lint"], None)]
    assert runner.invoke(cli.app, ["speculate", "remove", "--", "pytest", "-q"]).exit_code == 0
    assert runner.invoke(cli.app, ["speculate", "remove", "--", "pytest", "-q"]).exit_code == 1
