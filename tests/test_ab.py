import json
import os
import sys
import time

import pytest
from typer.testing import CliRunner

from orchard import ab, cli, dispatcher, receipts

runner = CliRunner()

# Prints a number from the arm's V, plus the run number so values differ inside an arm.
PRINTS = [sys.executable, "-c",
          "import os; print('took', float(os.environ['V']) + int(os.environ['ORCHARD_AB_RUN']) / 100); print('ANSWER: same')"]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "AB_DIR", tmp_path / "ab")
    monkeypatch.setattr(ab, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(receipts, "_key", lambda: b"test key")


def invoke(*args):
    return runner.invoke(cli.app, ["ab", *args])


def sealed_record():
    return json.loads(sorted(ab.AB_DIR.glob("ab-*.json"))[-1].read_text())


def test_parse_arm_keeps_commas_inside_values():
    assert ab.parse_arm("plain") == {"name": "plain", "env": {}}
    assert ab.parse_arm("fast:T=4,LIST=a,b,MODE=x") == {"name": "fast", "env": {"T": "4", "LIST": "a,b", "MODE": "x"}}
    with pytest.raises(ValueError):
        ab.parse_arm("no spaces:T=4")
    with pytest.raises(ValueError):
        ab.parse_arm("fast:novalue")


def test_parse_arms_and_hooks_refuse_mistakes():
    with pytest.raises(ValueError):
        ab.parse_arms(["one"])
    with pytest.raises(ValueError):
        ab.parse_arms(["a", "a"])
    arms = ab.parse_arms(["a", "b"])
    assert ab.parse_hooks(["b=sh x=1.sh"], arms) == {"b": "sh x=1.sh"}
    with pytest.raises(ValueError):
        ab.parse_hooks(["c=sh x.sh"], arms)
    with pytest.raises(ValueError):
        ab.parse_hooks(["b"], arms)


def test_run_order_is_mirrored():
    assert ab.run_order(["a", "b"], 4) == list("abbaabba")
    assert ab.run_order(["a", "b", "c"], 2) == list("abccba")


def test_number_and_same_take_the_right_match():
    assert ab.last_number(r"took ([\d.]+)", "took 1.5\ntook 2.5\n") == 2.5
    assert ab.last_number(r"\d+", "9 then 12") == 12
    assert ab.last_number(r"took (\w+)", "took many") is None
    assert ab.last_number(r"took (\w+)", "took nan") is None
    assert ab.last_number(r"took", "nothing") is None
    assert ab.same_digest("A: (.*)", "A: x\nB") == ab.same_digest("A: (.*)", "A: x\nC")
    assert ab.same_digest("A: (.*)", "A: x") != ab.same_digest("A: (.*)", "A: y")
    assert ab.same_digest("A: (.*)", "none") is None


def test_compare_every_verdict():
    assert ab.compare([4, 5, 6, 7], [1, 2, 3, 3.5], lower_better=True)["verdict"] == "better"
    assert ab.compare([4, 5, 6, 7], [1, 2, 3, 3.5], lower_better=False)["verdict"] == "worse"
    assert ab.compare([1, 2, 3, 3.5], [4, 5, 6, 7], lower_better=True)["verdict"] == "worse"
    assert ab.compare([1, 3, 5, 7], [2, 4, 6, 8], lower_better=True)["verdict"] == "not separated"
    # Two runs each separate by luck one time in three: not enough to call it.
    two = ab.compare([4, 5], [1, 2], lower_better=True)
    assert two["verdict"] == "too few runs" and two["chance"] == pytest.approx(1 / 3)
    assert ab.compare([4, 5, 6], [1, 2, 3], lower_better=True)["verdict"] == "better"


def test_judge_needs_enough_counted_runs():
    arms = ab.parse_arms(["a", "b"])
    runs = [{"arm": "a", "value": v, "counted": True} for v in (5, 6, 7)]
    runs += [{"arm": "b", "value": v, "counted": c} for v, c in ((1, True), (2, False), (3, False))]
    state, reason, comparisons = ab.judge(arms, runs, True)
    assert state == "void" and "b" in reason and comparisons == []


def test_proven_run_is_sealed_and_checks_out():
    result = invoke("--arm", "slow:V=9", "--arm", "fast:V=1", "--metric", r"took ([\d.]+)", "--", *PRINTS)
    assert result.exit_code == 0, result.output
    assert "proven" in result.output and "better" in result.output and "luck 1 in 35" in result.output
    record = sealed_record()
    assert [r["arm"] for r in record["runs"]] == ["slow", "fast", "fast", "slow", "slow", "fast", "fast", "slow"]
    assert receipts.verify(record) and ab.check(record) == []
    assert record["comparisons"][0]["verdict"] == "better"
    shown = invoke("--show", record["id"][:12], "--json")
    assert shown.exit_code == 0 and json.loads(shown.output)["verified"] is True


def test_check_catches_an_edited_record_and_an_edited_log():
    invoke("--arm", "a:V=9", "--arm", "b:V=1", "--metric", r"took ([\d.]+)", "--", *PRINTS)
    record = sealed_record()
    (ab.AB_DIR / record["id"] / record["runs"][0]["log"]).write_text("changed")
    assert any("log changed" in p for p in ab.check(record))
    assert invoke("--show", record["id"]).exit_code == 1
    record["state"] = "proven by hand"
    assert any("seal" in p for p in ab.check(record))


def test_higher_better_flips_the_verdict():
    result = invoke("--arm", "a:V=9", "--arm", "b:V=1", "--metric", r"took ([\d.]+)", "--higher-better", "--", *PRINTS)
    assert result.exit_code == 0 and "worse" in result.output


def test_overlapping_values_are_not_separated():
    result = invoke("--arm", "a:V=1", "--arm", "b:V=1", "--metric", r"took ([\d.]+)", "--", *PRINTS)
    assert result.exit_code == 1, result.output
    assert sealed_record()["state"] == "not separated"


def test_wall_seconds_is_the_default_metric():
    result = invoke("--arm", "a:V=1", "--arm", "b:V=1", "--rounds", "1", "--", *PRINTS)
    record = sealed_record()
    assert result.exit_code == 2 and record["metric"] == "wall seconds"
    assert all(r["value"] == r["seconds"] for r in record["runs"])


def test_failed_command_stops_and_after_hook_still_runs(tmp_path):
    mark = tmp_path / "after.txt"
    result = invoke("--arm", "a", "--arm", "b", "--after", f"a=echo back >> {mark}", "--", "/bin/sh", "-c", "exit 3")
    assert result.exit_code == 2
    record = sealed_record()
    assert record["state"] == "void" and "failed" in record["reason"] and len(record["runs"]) == 1
    assert mark.read_text().count("back") == 1


def test_failed_before_hook_stops_before_the_command(tmp_path):
    mark = tmp_path / "ran.txt"
    result = invoke("--arm", "a", "--arm", "b", "--before", "a=exit 1", "--", "/bin/sh", "-c", f"echo x > {mark}")
    assert result.exit_code == 2 and "before hook" in sealed_record()["reason"] and not mark.exists()


def test_after_hook_that_keeps_failing_is_said_loudly(monkeypatch):
    monkeypatch.setattr(ab, "AFTER_TRIES", 2)
    result = invoke("--arm", "a:V=1", "--arm", "b:V=1", "--after", "a=exit 1", "--metric", r"took ([\d.]+)", "--", *PRINTS)
    assert result.exit_code == 2 and "may NOT be back" in sealed_record()["reason"]


def test_missing_metric_stops():
    result = invoke("--arm", "a", "--arm", "b", "--metric", "nothing like this", "--", "/bin/sh", "-c", "echo hi")
    assert result.exit_code == 2 and "found no number" in sealed_record()["reason"]


def test_changed_output_stops():
    cmd = [sys.executable, "-c", "import os; print('ANSWER:', os.environ['ORCHARD_AB_ARM'])"]
    result = invoke("--arm", "a", "--arm", "b", "--same", "ANSWER: (.*)", "--", *cmd)
    assert result.exit_code == 2 and "output changed" in sealed_record()["reason"]
    assert len(sealed_record()["runs"]) == 2


def test_same_output_passes_when_identical():
    result = invoke("--arm", "a:V=9", "--arm", "b:V=1", "--metric", r"took ([\d.]+)", "--same", "ANSWER: (.*)", "--", *PRINTS)
    assert result.exit_code == 0, result.output


def test_timeout_kills_the_run_and_stop_if_too(tmp_path):
    began = time.monotonic()
    result = invoke("--arm", "a", "--arm", "b", "--timeout", "1", "--", "/bin/sh", "-c", "sleep 30")
    assert result.exit_code == 2 and "timeout" in sealed_record()["reason"]
    result = invoke("--arm", "a", "--arm", "b", "--stop-if", "exit 0", "--poll", "0", "--", "/bin/sh", "-c", "sleep 30")
    assert result.exit_code == 2 and "stopped" in sealed_record()["reason"]
    assert time.monotonic() - began < 20


def test_stop_if_that_says_no_leaves_the_run_alone():
    result = invoke("--arm", "a:V=9", "--arm", "b:V=1", "--rounds", "1", "--stop-if", "exit 1", "--poll", "0",
                    "--metric", r"took ([\d.]+)", "--", *PRINTS)
    assert sealed_record()["runs"][0]["state"] == "ok" and result.exit_code == 2  # one run each is too few


def test_signal_stops_the_test_and_after_hook_still_runs(tmp_path):
    mark = tmp_path / "after.txt"
    arms = ab.parse_arms(["a", "b"])
    cmd = ["/bin/sh", "-c", f"kill -TERM {os.getpid()}; sleep 30"]
    began = time.monotonic()
    record = ab.run_test(cmd, arms, after={"a": f"echo back >> {mark}"}, say=lambda _: None)
    assert record["state"] == "void" and record["reason"] == "interrupted"
    assert mark.read_text().count("back") == 1
    assert time.monotonic() - began < 20


def test_interrupt_still_reports_an_after_hook_that_failed(monkeypatch):
    monkeypatch.setattr(ab, "AFTER_TRIES", 1)
    arms = ab.parse_arms(["a", "b"])
    cmd = ["/bin/sh", "-c", f"kill -TERM {os.getpid()}; sleep 30"]
    record = ab.run_test(cmd, arms, after={"a": "exit 1"}, say=lambda _: None)
    assert record["state"] == "void"
    assert "may NOT be back" in record["reason"] and "interrupted" in record["reason"]


def test_overlapped_runs_do_not_count(monkeypatch):
    calls = []

    def one_overlap(workers, start, end):
        calls.append(workers)
        return ["other-job"] if len(calls) == 1 else []

    monkeypatch.setattr(ab, "overlapping_jobs", one_overlap)
    result = invoke("--arm", "a:V=9", "--arm", "b:V=1", "--watch", "box", "--metric", r"took ([\d.]+)", "--", *PRINTS)
    assert result.exit_code == 0, result.output
    assert "DOES NOT COUNT: other-job" in result.output and "3 of 4 runs count" in result.output
    record = sealed_record()
    assert record["runs"][0]["overlapped"] == ["other-job"] and not record["runs"][0]["counted"]
    assert calls[0] == ["box"]


def test_overlapping_jobs_reads_the_queue_and_finished_jobs(tmp_path):
    def job(name, worker, began, ended):
        folder = ab.JOBS_DIR / name
        folder.mkdir(parents=True)
        (folder / "placement.json").write_text(json.dumps({"worker": worker}))
        (folder / "status.json").write_text(json.dumps({"started_at": began, "ended_at": ended}))

    job("inside", "box", 110, 120)
    job("spans", "box", 50, 500)
    job("before", "box", 10, 90)
    job("after", "box", 300, 400)
    job("elsewhere", "other", 110, 120)
    dispatcher.QUEUE_DIR.mkdir(parents=True)
    (dispatcher.QUEUE_DIR / "live.json").write_text(json.dumps({"id": "live", "worker": "box", "state": "running"}))
    (dispatcher.QUEUE_DIR / "waiting.json").write_text(json.dumps({"id": "waiting", "worker": "box", "state": "queued"}))
    assert ab.overlapping_jobs(["box"], 100, 200) == ["inside", "live", "spans"]


def test_cli_refuses_bad_input():
    assert invoke("--arm", "a", "--", "true").exit_code == 2
    assert invoke("--arm", "a", "--arm", "b").exit_code == 2
    assert invoke("--arm", "a", "--arm", "b", "--metric", "(", "--", "true").exit_code == 2
    assert invoke("--arm", "a", "--arm", "b", "--before", "c=x", "--", "true").exit_code == 2
    assert invoke("--show", "nope").exit_code == 1


def test_list_shows_recent_tests():
    invoke("--arm", "a:V=9", "--arm", "b:V=1", "--metric", r"took ([\d.]+)", "--", *PRINTS)
    listed = invoke("--list")
    assert listed.exit_code == 0 and "proven" in listed.output and sealed_record()["id"] in listed.output
