import json

import pytest
from typer.testing import CliRunner

from orchard import cli, consensus, fleet
from orchard.job import JobSpec


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(consensus, "MEMORY_PATH", tmp_path / "consensus.jsonl")
    monkeypatch.setattr(cli, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(fleet, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(fleet, "LOG_DIR", tmp_path / "fleet")


def run(target, state, machine="x86_64", os_family="macos", load=0.1, log="", job=None):
    return {"target": target, "state": state, "machine": machine, "os_family": os_family,
            "load_per_core": load, "log_tail": log, "job": job or f"j-{target}"}


FAIL_LOG = "collected 3 items\nFAILED tests/test_a.py::test_x - AssertionError: assert 3 == 4\n1 failed"


def test_signature_normalises_numbers_paths_hex_and_colour():
    a = consensus.signature("ok\nFAILED tests/test_a.py::test_x - assert 3 == 4 at 0x7f3a2b\n")
    b = consensus.signature("\x1b[31mFAILED  src/x/test_a.py::test_x - assert 15 == 16 at 0x99\x1b[0m\n")
    assert a == b == "FAILED <path>::test_x - assert <n> == <n> at <hex>"
    assert consensus.signature("all good\n3 passed\n") is None
    assert consensus.signature_id(a) == consensus.signature_id(b)
    assert consensus.signature_id(None) is None


def test_signature_takes_the_last_error_line():
    assert consensus.signature(FAIL_LOG) == "<n> failed"
    assert consensus.signature("Traceback (most recent call last):\nValueError: bad\nok") == "ValueError: bad"


def test_classify_every_label():
    first = run("pro", "failed")
    assert consensus.classify(run("pro", "succeeded"), [])[0] == "pass"
    assert consensus.classify(first, [run("pro", "succeeded"), run("local", "failed")])[0] == "flaky"
    label, reason = consensus.classify(first, [run("pro", "failed"), run("local", "failed", "arm64")])
    assert label == "real" and "local, pro" in reason
    busy = run("pro", "failed", load=0.9)
    assert consensus.classify(busy, [run("pro", "failed", load=1.2), run("mini", "succeeded", load=0.2)])[0] == "load-sensitive"
    assert consensus.classify(first, [run("pro", "failed"), run("local", "succeeded", "arm64")])[0] == "platform-only"
    assert consensus.classify(first, [run("pro", "failed"), run("mini", "succeeded")])[0] == "machine-only"


def test_classify_ignores_runs_that_never_finished():
    first = run("pro", "failed")
    label, _ = consensus.classify(first, [{"target": "pro", "state": "unavailable"}, run("local", "failed", "arm64")])
    assert label == "real"
    assert not consensus.busy({"load_per_core": None})


def test_exit_codes():
    assert [consensus.exit_code(x) for x in consensus.LABELS] == [0, 2, 1, 2, 2, 2]


def test_memory_recall_noise_and_offenders(tmp_path):
    sig = "FAILED <path>::test_x"
    for label in ("flaky", "flaky", "real"):
        consensus.remember({"label": label, "signature": sig})
    consensus.remember({"label": "platform-only", "signature": "other"})
    with open(consensus.MEMORY_PATH, "a") as fh:
        fh.write("not json\n")
    seen = consensus.recall(sig)
    assert seen == {"flaky": 2, "real": 1}
    assert consensus.known_noise(sig)
    assert not consensus.known_noise("other")
    assert not consensus.known_noise(None)
    assert consensus.describe_memory(seen) == "seen 3 times before: flaky 2, real 1"
    assert consensus.describe_memory(consensus.recall("nope")) is None
    assert [s for s, _ in consensus.offenders()] == [sig, "other"]


def test_memory_write_failure_is_quiet(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    consensus.remember({"label": "flaky"}, path=blocker / "sub" / "m.jsonl")


def test_job_names_are_unique_and_found_exactly(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    one = fleet.job_name("consensus", "abc123", "a-very-long-worker-name-indeed", 1)
    ten = fleet.job_name("consensus", "abc123", "a-very-long-worker-name-indeed", 10)
    assert one.startswith("consensus-abc123-1-") and ten.startswith("consensus-abc123-10-")
    (jobs / "consensus-abc123-10-pro-20261103-101010-abcd").mkdir()
    (jobs / "consensus-abc123-1-pro-20261103-101011-beef").mkdir()
    assert fleet.find_job("consensus-abc123-1-pro").name.endswith("beef")
    assert fleet.find_job("consensus-abc123-2-pro") is None


def test_child_argv_passes_options_through():
    argv = fleet.child_argv("pro", "n", ["pytest", "-x"], tags=["x86_64"], env_names=["TOKEN"], timeout=60)
    assert argv[1:] == ["-m", "orchard", "run", "-w", "pro", "--name", "n", "--timeout", "60",
                        "-t", "x86_64", "--env", "TOKEN", "--", "pytest", "-x"]


def test_run_spread_keeps_order_per_target_and_stops(monkeypatch):
    calls = []

    def fake_run_on(target, cmd, cwd, name, **kw):
        calls.append(name)
        return run(target, "succeeded" if name == "a1" else "failed", job=name)

    monkeypatch.setattr(fleet, "run_on", fake_run_on)
    plans = [{"target": "a", "cmd": ["x"], "cwd": ".", "name": n} for n in ("a1", "a2")]
    plans.append({"target": "b", "cmd": ["x"], "cwd": ".", "name": "b1"})
    results = fleet.run_spread(plans, stop=consensus.passed)
    assert [r["state"] for r in results][:2] == ["succeeded", "skipped"]
    assert "a2" not in calls
    assert results[2]["target"] == "b"


def test_result_of_reads_status_log_and_placement(tmp_path):
    sdir = tmp_path / "jobs" / "t-20261103-101010-abcd"
    sdir.mkdir(parents=True)
    (sdir / "status.json").write_text(json.dumps({"state": "failed", "exit_code": 1, "machine": "x86_64",
                                                  "os_family": "linux", "load_per_core": 0.3}))
    (sdir / "placement.json").write_text(json.dumps({"worker": "laptop"}))
    (sdir / "log.txt").write_text(FAIL_LOG)
    got = fleet.result_of(sdir)
    assert got["target"] == "laptop" and got["state"] == "failed" and got["os_family"] == "linux"
    assert got["log_tail"].endswith("1 failed") and got["sealed"] is False


def fake_fleet(monkeypatch, outcomes, targets=("local", "pro")):
    """outcomes: target -> list of states, consumed in order."""
    seen = []

    def fake_run_on(target, cmd, cwd, name, **kw):
        seen.append((target, name, kw))
        actual = "pro" if target == "auto" else target
        state = outcomes[actual].pop(0)
        return run(actual, state, "arm64" if actual == "local" else "x86_64",
                   log=FAIL_LOG if state == "failed" else "ok", job=name)

    monkeypatch.setattr(fleet, "run_on", fake_run_on)
    monkeypatch.setattr(fleet, "capable_targets", lambda job: list(targets))
    monkeypatch.setattr(fleet, "job_like_run", lambda cmd, cwd, *a: JobSpec(name="t", cmd=list(cmd)))
    return seen


def test_consensus_cli_flaky_then_remembered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    fake_fleet(monkeypatch, {"pro": ["failed", "succeeded"], "local": ["succeeded"]})
    first = runner.invoke(cli.app, ["consensus", "-t", "x86_64", "--", "pytest"])
    assert first.exit_code == 2, first.output
    assert "consensus: flaky" in first.output and "memory:" not in first.output
    seen = fake_fleet(monkeypatch, {"pro": ["failed", "succeeded"], "local": ["succeeded"]})
    again = runner.invoke(cli.app, ["consensus", "-t", "x86_64", "--", "pytest"])
    assert "memory: seen 1 time before: flaky 1" in again.output
    assert all(kw["tags"] == ["x86_64"] for _, _, kw in seen)
    assert len({name for _, name, _ in seen}) == 3


def test_consensus_cli_pass_does_not_rerun(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_fleet(monkeypatch, {"pro": ["succeeded"]})
    monkeypatch.setattr(fleet, "capable_targets", lambda job: pytest.fail("a pass must not rerun"))
    result = CliRunner().invoke(cli.app, ["consensus", "--json", "--", "pytest"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["label"] == "pass"
    assert not consensus.MEMORY_PATH.exists()


def test_consensus_cli_from_job_real(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sdir = tmp_path / "jobs" / "pytest-20261103-101010-abcd"
    sdir.mkdir(parents=True)
    (sdir / "spec.json").write_text(json.dumps({"cmd": ["pytest", "-q"], "tags": [], "env": [], "timeout_s": 30}))
    (sdir / "status.json").write_text(json.dumps({"state": "failed", "machine": "x86_64", "os_family": "macos"}))
    (sdir / "placement.json").write_text(json.dumps({"worker": "pro"}))
    (sdir / "log.txt").write_text(FAIL_LOG)
    seen = fake_fleet(monkeypatch, {"pro": ["failed"], "local": ["failed"]})
    result = CliRunner().invoke(cli.app, ["consensus", "--job", "pytest-2026"])
    assert result.exit_code == 1, result.output
    assert "consensus: real" in result.output and "error:   " not in result.output
    assert {t for t, _, _ in seen} == {"local", "pro"}
    assert all(kw["timeout"] == 30 for _, _, kw in seen)
    remembered = json.loads(consensus.MEMORY_PATH.read_text().splitlines()[0])
    assert remembered["cmd"] == ["pytest", "-q"] and remembered["signature"] == "<n> failed"
    offenders = CliRunner().invoke(cli.app, ["consensus", "--offenders"])
    assert "no failure has been labelled noise yet" in offenders.output


def test_consensus_cli_unfinished_first_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fleet, "run_on", lambda *a, **k: {"target": "pro", "state": "unavailable", "why": "offline"})
    result = CliRunner().invoke(cli.app, ["consensus", "--", "pytest"])
    assert result.exit_code == 2
    assert "did not finish" in result.output
