import json
import os
import subprocess

import pytest
from typer.testing import CliRunner

from orchard import cli, receipts
from orchard.job import JobSpec


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "KEY_PATH", tmp_path / "home" / "receipt.key")
    monkeypatch.setattr(receipts, "INDEX_DIR", tmp_path / "home" / "receipts")
    monkeypatch.setattr(cli, "JOBS_DIR", tmp_path / "home" / "jobs")


def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("print('a')\n")
    (root / "run.sh").write_text("#!/bin/sh\necho hi\n")
    os.chmod(root / "run.sh", 0o755)
    return root


def sealed(tmp_path, root, job, state="succeeded", worker="pro", name="j-1"):
    sdir = tmp_path / "home" / "jobs" / name
    sdir.mkdir(parents=True)
    (sdir / "log.txt").write_text("ok\n")
    receipts.record_inputs(sdir, root, job, worker)
    status = {"state": state, "exit_code": 0 if state == "succeeded" else 1, "elapsed_s": 3,
              "machine": "x86_64", "os_family": "macos", "sealed_at": 1}
    return sdir, receipts.seal(sdir, status, worker, None)


def test_tree_digest_is_stable_and_sees_content_mode_and_links(tmp_path):
    root = project(tmp_path)
    first, count = receipts.tree_digest(root)
    assert count == 2
    assert receipts.tree_digest(root)[0] == first
    os.chmod(root / "run.sh", 0o644)
    no_exec = receipts.tree_digest(root)[0]
    assert no_exec != first
    (root / "a.py").write_text("print('b')\n")
    assert receipts.tree_digest(root)[0] != no_exec
    os.symlink("a.py", root / "link")
    with_link = receipts.tree_digest(root)[0]
    os.unlink(root / "link")
    os.symlink("run.sh", root / "link")
    assert receipts.tree_digest(root)[0] != with_link


def test_tree_digest_follows_gitignore(tmp_path):
    root = project(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("cache/\n")
    before = receipts.tree_digest(root)[0]
    (root / "cache").mkdir()
    (root / "cache" / "big.bin").write_text("x" * 100)
    assert receipts.tree_digest(root)[0] == before


def test_work_key_covers_env_values_without_storing_them(tmp_path):
    root = project(tmp_path)
    a = JobSpec(name="t", cmd=["pytest"], env=["TOKEN"], env_values={"TOKEN": "one"})
    b = JobSpec(name="t", cmd=["pytest"], env=["TOKEN"], env_values={"TOKEN": "two"})
    da, db = receipts.describe(root, a), receipts.describe(root, b)
    assert da["work_key"] != db["work_key"]
    assert "one" not in json.dumps(da)
    renamed = JobSpec(name="other-name", cmd=["pytest"], env=["TOKEN"], env_values={"TOKEN": "one"})
    assert receipts.describe(root, renamed)["work_key"] == da["work_key"]
    assert receipts.describe(root, JobSpec(name="t", cmd=["pytest", "-x"]))["work_key"] != da["work_key"]


def test_key_file_is_private_and_reused(tmp_path):
    first = receipts._key()
    assert receipts._key() == first
    assert oct(receipts.KEY_PATH.stat().st_mode & 0o777) == "0o600"


def test_seal_verify_and_tamper(tmp_path):
    root = project(tmp_path)
    sdir, rec = sealed(tmp_path, root, JobSpec(name="t", cmd=["pytest"]))
    assert rec["worker"] == {"name": "pro", "machine": "x86_64", "os_family": "macos"}
    assert receipts.verify(rec)
    on_disk = json.loads((sdir / "receipt.json").read_text())
    assert receipts.verify(on_disk)
    on_disk["state"] = "failed"
    assert not receipts.verify(on_disk)
    assert not receipts.verify({"job": "x"})


def test_seal_without_inputs_is_a_quiet_none(tmp_path):
    sdir = tmp_path / "bare"
    sdir.mkdir()
    assert receipts.seal(sdir, {"state": "succeeded"}, "local", None) is None


def test_find_reusable_skips_failed_and_forged(tmp_path):
    root = project(tmp_path)
    job = JobSpec(name="t", cmd=["pytest"])
    key = receipts.describe(root, job)["work_key"]
    assert receipts.find_reusable(key) is None
    sealed(tmp_path, root, job, state="failed", name="j-fail")
    assert receipts.find_reusable(key) is None
    _sdir, good = sealed(tmp_path, root, job, name="j-good")
    forged = dict(good, job="j-forged", sealed_at=good["sealed_at"] + 10, seal="0" * 64)
    (receipts.INDEX_DIR / key / "j-forged.json").write_text(json.dumps(forged))
    assert receipts.find_reusable(key)["job"] == "j-good"
    (root / "a.py").write_text("changed\n")
    assert receipts.find_reusable(receipts.describe(root, job)["work_key"]) is None


def test_deep_check_flags_changed_log_and_out(tmp_path):
    root = project(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "r.txt").write_text("1")
    job = JobSpec(name="t", cmd=["pytest"])
    sdir = tmp_path / "home" / "jobs" / "j-deep"
    sdir.mkdir(parents=True)
    (sdir / "log.txt").write_text("ok\n")
    receipts.record_inputs(sdir, root, job, "local")
    rec = receipts.seal(sdir, {"state": "succeeded"}, "local", out)
    assert receipts.deep_check(rec, sdir, out) == []
    (out / "r.txt").write_text("2")
    (sdir / "log.txt").write_text("edited\n")
    problems = receipts.deep_check(rec, sdir, out)
    assert any("log.txt" in p for p in problems) and any("changed" in p for p in problems)


def test_run_reuse_answers_without_running(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr(cli.env, "capture_python", lambda cwd: "3.12")
    monkeypatch.setattr(cli.env, "capture_requirements", lambda cwd: (None, []))
    sealed(tmp_path, root, JobSpec(name="echo", cmd=["echo", "hi"], python="3.12"))
    monkeypatch.setattr(cli, "load_workers", lambda: pytest.fail("reuse must not place the job"))
    result = CliRunner().invoke(cli.app, ["run", "--reuse", "--", "echo", "hi"])
    assert result.exit_code == 0, result.output
    assert "reused: j-1 passed on pro" in result.output
    assert "nothing ran" in result.output


def test_answer_from_receipt_miss(tmp_path, capsys):
    root = project(tmp_path)
    assert cli.answer_from_receipt(JobSpec(name="t", cmd=["pytest"]), root) is None
    assert "no receipt for this exact work" in capsys.readouterr().out


def test_receipt_command(tmp_path):
    root = project(tmp_path)
    sdir, _rec = sealed(tmp_path, root, JobSpec(name="t", cmd=["pytest"]), name="t-20260101-000000-abcd")
    runner = CliRunner()
    shown = runner.invoke(cli.app, ["receipt", "t-2026"])
    assert shown.exit_code == 0, shown.output
    assert "seal:     verified" in shown.output and "pytest" in shown.output
    as_json = json.loads(runner.invoke(cli.app, ["receipt", "t-2026", "--json"]).output)
    assert as_json["verified"] is True
    data = json.loads((sdir / "receipt.json").read_text())
    data["exit_code"] = 0
    data["state"] = "failed"
    (sdir / "receipt.json").write_text(json.dumps(data))
    bad = runner.invoke(cli.app, ["receipt", "t-2026"])
    assert bad.exit_code == 1
    assert "DOES NOT VERIFY" in bad.output
    assert runner.invoke(cli.app, ["receipt", "nope"]).exit_code == 1
