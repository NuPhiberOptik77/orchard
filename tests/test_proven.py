import json

import pytest
from typer.testing import CliRunner

from orchard import cli, consensus, fleet, receipts
from orchard.job import JobSpec
from orchard.placement import allowed_oses, problems


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(receipts, "KEY_PATH", tmp_path / "home" / "receipt.key")
    monkeypatch.setattr(receipts, "INDEX_DIR", tmp_path / "home" / "receipts")
    monkeypatch.setattr(consensus, "MEMORY_PATH", tmp_path / "consensus.jsonl")
    monkeypatch.setattr(fleet, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(fleet, "LOG_DIR", tmp_path / "fleet")


def caps(tags):
    return {"ram_gb": 16, "disk_free_gb": 50, "bench": 1000, "platform_ok": True, "tags": tags}


def put(project, cmd, family, state, at, name):
    receipt = receipts.sign({"v": receipts.VERSION, "job": name, "project": str(project), "work": {"cmd": cmd},
                             "worker": {"name": "w", "os_family": family}, "state": state, "sealed_at": at})
    folder = receipts.INDEX_DIR / "k"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(receipt))
    return receipt


def test_without_the_tag_nothing_changes():
    assert allowed_oses(JobSpec(name="j", cmd=["t"])) == {"macos"}
    # A Mac job asking for x86_64 still places on an Intel Mac with no os tag.
    assert problems(JobSpec(name="j", cmd=["t"], tags=["x86_64"]), caps(["x86_64", "os:macos"])) == []
    assert any("-t os:linux" in p for p in problems(JobSpec(name="j", cmd=["t"]), caps(["os:linux"])))


def test_proven_allows_macos_plus_every_os_with_evidence():
    job = JobSpec(name="j", cmd=["t"], tags=["os:proven"], requires={"proven_os": ["linux"]})
    assert allowed_oses(job) == {"macos", "linux"}
    assert problems(job, caps(["os:linux"])) == []
    bare = JobSpec(name="j", cmd=["t"], tags=["os:proven"])
    assert problems(bare, caps(["os:macos"])) == []
    probs = problems(bare, caps(["os:linux"]))
    assert len(probs) == 1 and "not proven on linux yet" in probs[0] and "orchard prove" in probs[0]


def test_evidence_never_changes_the_work_key():
    plain = JobSpec(name="j", cmd=["t"], tags=["os:proven"])
    proven = JobSpec(name="j", cmd=["t"], tags=["os:proven"], requires={"proven_os": ["linux"]})
    assert receipts.work_fields(plain) == receipts.work_fields(proven)
    assert receipts.work_fields(JobSpec(name="j", cmd=["t"], requires={"mem_gb": 4}))["requires"] == {"mem_gb": 4}


def test_proven_oses_reads_the_latest_verified_run_per_os(tmp_path):
    proj, cmd = tmp_path / "p", ["pytest"]
    put(proj, cmd, "macos", "succeeded", 1, "a")
    put(proj, cmd, "linux", "succeeded", 1, "b")
    assert receipts.proven_oses(cmd, proj) == ["linux", "macos"]
    put(proj, cmd, "linux", "failed", 2, "c")
    put(proj, cmd, "macos", "cancelled", 3, "d")
    assert receipts.proven_oses(cmd, proj) == ["macos"]
    put(proj, cmd, "linux", "succeeded", 4, "e")
    assert receipts.proven_oses(cmd, proj) == ["linux", "macos"]
    put(tmp_path / "other", cmd, "windows", "succeeded", 5, "f")
    put(proj, ["pytest", "-x"], "windows", "succeeded", 5, "g")
    forged = put(proj, cmd, "windows", "failed", 5, "h")
    forged["state"] = "succeeded"
    (receipts.INDEX_DIR / "k" / "h.json").write_text(json.dumps(forged))
    assert receipts.proven_oses(cmd, proj) == ["linux", "macos"]


def test_attach_proof_only_touches_tagged_jobs(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    put(proj, ["pytest"], "linux", "succeeded", 1, "a")
    plain = JobSpec(name="j", cmd=["pytest"], requires={"mem_gb": 2})
    receipts.attach_proof(plain, proj)
    assert plain.requires == {"mem_gb": 2}
    tagged = JobSpec(name="j", cmd=["pytest"], tags=["os:proven"], requires={"mem_gb": 2})
    receipts.attach_proof(tagged, proj)
    assert tagged.requires == {"mem_gb": 2, "proven_os": ["linux"]}
    assert fleet.job_like_run(["pytest"], proj, ["os:proven"]).requires == {"proven_os": ["linux"]}


def fake_fleet(monkeypatch, outcomes):
    """outcomes: target -> (os_family, state)."""
    seen, jobs = [], []

    def fake_run_on(target, cmd, cwd, name, **kw):
        seen.append((target, kw))
        family, state = outcomes[target]
        log = "FAILED tests/test_a.py::test_x - AssertionError: assert 3 == 4" if state == "failed" else "ok"
        return {"target": target, "state": state, "machine": "x86_64", "os_family": family,
                "load_per_core": 0.1, "log_tail": log, "job": name}

    monkeypatch.setattr(fleet, "run_on", fake_run_on)
    monkeypatch.setattr(fleet, "capable_targets", lambda job: jobs.append(job) or list(outcomes))
    monkeypatch.setattr(fleet, "job_like_run",
                        lambda cmd, cwd, tags, *a: JobSpec(name="t", cmd=list(cmd), tags=list(tags)))
    return seen, jobs


def test_prove_split_by_os_is_remembered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen, jobs = fake_fleet(monkeypatch, {"local": ("macos", "succeeded"), "pro": ("macos", "succeeded"),
                                          "laptop": ("linux", "failed")})
    result = CliRunner().invoke(cli.app, ["prove", "-t", "x86_64", "--json", "--", "pytest"])
    assert result.exit_code == 1, result.output
    out = json.loads(result.output)
    assert out["passed_on"] == ["macos"] and out["failed_on"] == ["linux"] and out["label"] == "platform-only"
    assert out["proven_allows"] == ["macos"]
    assert jobs[0].tags == ["x86_64", "os:any"] and all(kw["tags"] == ["x86_64", "os:any"] for _, kw in seen)
    sig = consensus.signature("FAILED tests/test_a.py::test_x - AssertionError: assert 3 == 4")
    assert consensus.recall(sig) == {"platform-only": 1}


def test_prove_passing_everywhere_opens_those_oses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_fleet(monkeypatch, {"local": ("macos", "succeeded"), "laptop": ("linux", "succeeded")})
    monkeypatch.setattr(receipts, "proven_oses", lambda cmd, cwd: ["linux", "macos"])
    result = CliRunner().invoke(cli.app, ["prove", "--", "pytest"])
    assert result.exit_code == 0, result.output
    assert "prove: passes on linux, macos" in result.output
    assert "-t os:proven may now run it on: linux, macos" in result.output
    assert not consensus.MEMORY_PATH.exists()


def test_prove_errors_and_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["prove"]).exit_code == 2
    seen, _ = fake_fleet(monkeypatch, {"laptop": ("linux", "unavailable")})
    monkeypatch.setattr(fleet, "capable_targets", lambda job: pytest.fail("--on must not probe machines"))
    result = runner.invoke(cli.app, ["prove", "--on", "laptop", "--", "pytest"])
    assert result.exit_code == 2 and "no run finished" in result.output
    assert [t for t, _ in seen] == ["laptop"]
    monkeypatch.setattr(fleet, "capable_targets", lambda job: [])
    assert runner.invoke(cli.app, ["prove", "--", "pytest"]).exit_code == 2
