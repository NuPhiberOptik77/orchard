import shutil
import subprocess
import tarfile
from io import BytesIO

import pytest

from orchard import engine
from orchard.config import Worker


def git_available():
    if not shutil.which("git"):
        return False
    return subprocess.run(["git", "--version"], capture_output=True).returncode == 0


def test_build_tar_excludes_junk(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    (proj / ".git" / "config").write_text("x")
    (proj / "__pycache__").mkdir()
    (proj / "__pycache__" / "a.pyc").write_text("x")
    (proj / "a.py").write_text("print(1)")
    data = engine.build_tar_bytes(proj)
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    assert "a.py" in names
    assert not any(".git" in n for n in names)
    assert not any("__pycache__" in n for n in names)


def test_build_tar_excludes_build_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".build").mkdir()
    (proj / ".build" / "cached.o").write_text("x")
    (proj / "main.swift").write_text("print(1)")
    data = engine.build_tar_bytes(proj)
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    assert "main.swift" in names
    assert not any(".build" in n for n in names)


@pytest.mark.skipif(not git_available(), reason="git not available")
def test_build_tar_tracked_only(tmp_path):
    proj = tmp_path / "repo"
    proj.mkdir()
    subprocess.run(["git", "-C", str(proj), "init", "-q"], check=True)
    (proj / "Package.swift").write_text("// swift-tools-version:5.9")
    (proj / "main.swift").write_text("print(1)")
    (proj / ".build").mkdir()
    (proj / ".build" / "cached.o").write_text("x")
    (proj / "untracked.txt").write_text("nope")
    subprocess.run(
        ["git", "-C", str(proj), "add", "Package.swift", "main.swift", ".build/cached.o"],
        check=True,
        capture_output=True,
    )
    data = engine.build_tar_bytes(proj, tracked_only=True)
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    assert "Package.swift" in names
    assert "main.swift" in names
    assert "untracked.txt" not in names
    assert not any(".build" in n for n in names)


@pytest.mark.skipif(not git_available(), reason="git not available")
def test_run_tar_respects_gitignore(tmp_path):
    proj = tmp_path / "repo"
    proj.mkdir()
    subprocess.run(["git", "-C", str(proj), "init", "-q"], check=True)
    (proj / ".gitignore").write_text("data/\nsecret.txt\n")
    (proj / "main.py").write_text("print(1)")
    (proj / "scratch.txt").write_text("untracked but not ignored")
    (proj / "secret.txt").write_text("nope")
    (proj / "data").mkdir()
    (proj / "data" / "big.bin").write_text("huge")
    subprocess.run(["git", "-C", str(proj), "add", "main.py"], check=True, capture_output=True)
    data = engine.build_tar_bytes(proj)
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    assert "main.py" in names
    assert "scratch.txt" in names
    assert "secret.txt" not in names
    assert not any(n.startswith("data/") for n in names)


def test_check_shipping_size_cap(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.bin").write_text("x" * 100)
    monkeypatch.setenv("ORCHARD_MAX_TAR_MB", "0")
    with pytest.raises(ValueError, match="ORCHARD_MAX_TAR_MB"):
        engine.check_shipping_size(plain, "run")
    monkeypatch.setenv("ORCHARD_MAX_TAR_MB", "10")
    engine.check_shipping_size(plain, "run")


def test_tar_is_deterministic(tmp_path, monkeypatch):
    # Same tree, different clock and different temp-file names: the bytes (and
    # so the inbox sha) must match, or every run re-uploads the tarball.
    proj = tmp_path / "proj"
    (proj / "pkg").mkdir(parents=True)
    (proj / "a.py").write_text("print(1)")
    (proj / "pkg" / "b.py").write_text("print(2)")
    blobs = []
    for i, clock in enumerate((1_000_000_000.0, 2_000_000_000.0)):
        monkeypatch.setattr(engine.gzip.time, "time", lambda c=clock: c)
        path = tmp_path / f"run-{i}.tgz"
        with open(path, "wb") as fh:
            engine._tar_into(fh, proj)
        blobs.append(path.read_bytes())
    assert blobs[0] == blobs[1]
    assert engine.build_tar_bytes(proj) == blobs[0]
    with tarfile.open(fileobj=BytesIO(blobs[0]), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == ["a.py", "pkg/b.py"]


def test_tracked_files_outside_repo(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.txt").write_text("x")
    assert engine.tracked_files(plain) is None


def test_artifact_pull_verdict():
    gb = 10**9
    warn = 2 * gb
    assert engine.artifact_pull_verdict(1 * gb, 50 * gb, warn)[0] == "ok"
    assert engine.artifact_pull_verdict(3 * gb, 100 * gb, warn)[0] == "warn"
    verdict, msg = engine.artifact_pull_verdict(10 * gb, 12 * gb, warn)
    assert verdict == "refuse"
    assert "ORCHARD_ARTIFACT_FORCE" in msg


def test_remote_job_running(monkeypatch):
    class Proc:
        returncode = 0

        def __init__(self, out):
            self.stdout = out

    worker = Worker(name="pro", host="10.0.0.9")
    monkeypatch.setattr(engine.transport, "run", lambda *a, **k: Proc(b"yes\n"))
    assert engine.remote_job_running(worker, "j1") is True
    monkeypatch.setattr(engine.transport, "run", lambda *a, **k: Proc(b"no\n"))
    assert engine.remote_job_running(worker, "j1") is False
    monkeypatch.setattr(engine.transport, "run", lambda *a, **k: Proc(b""))
    assert engine.remote_job_running(worker, "j1") is False


def test_remote_status_missing(monkeypatch):
    class Proc:
        def __init__(self, code, out):
            self.returncode, self.stdout = code, out

    worker = Worker(name="pro", host="10.0.0.9")
    for code, out, want in ((0, b"no\n", True), (0, b"yes\n", False), (255, b"", False)):
        monkeypatch.setattr(engine.transport, "run", lambda *a, c=code, o=out, **k: Proc(c, o))
        assert engine.remote_status_missing(worker, "/Users/x/.orchard/jobs/j1") is want


def test_remote_job_dir():
    assert engine.remote_job_dir("/Users/x", "abc-1") == "/Users/x/.orchard/jobs/abc-1"
