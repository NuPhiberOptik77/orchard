import hashlib
import os
import subprocess

from orchard import transport
from orchard.config import Worker


def _run_in_local_sh(path_dirs):
    """Stand-in for transport.run: the remote command runs in a local sh with a fixed PATH."""
    def fake_run(worker, remote_cmd, **kwargs):
        env = {"PATH": os.pathsep.join(str(d) for d in path_dirs)}
        return subprocess.run(["/bin/sh", "-c", remote_cmd], capture_output=True, env=env)
    return fake_run


def test_file_sha256_remote_uses_shasum_on_macos(tmp_path, monkeypatch):
    target = tmp_path / "agent.py"
    target.write_bytes(b"print('hi')\n")
    monkeypatch.setattr(transport, "run", _run_in_local_sh(["/usr/bin", "/bin"]))
    sha = transport.file_sha256_remote(Worker(name="pro", host="10.0.0.9"), str(target))
    assert sha == hashlib.sha256(target.read_bytes()).hexdigest()


def test_file_sha256_remote_falls_back_to_sha256sum(tmp_path, monkeypatch):
    target = tmp_path / "agent.py"
    target.write_bytes(b"print('hi')\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # A Linux image with sha256sum and no shasum.
    fake = bin_dir / "sha256sum"
    fake.write_text('#!/bin/sh\necho "feedface  $1"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(transport, "run", _run_in_local_sh([bin_dir]))
    assert transport.file_sha256_remote(Worker(name="box", host="10.0.0.9"), str(target)) == "feedface"


def test_file_sha256_remote_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "run", _run_in_local_sh(["/usr/bin", "/bin"]))
    assert transport.file_sha256_remote(Worker(name="pro", host="10.0.0.9"), str(tmp_path / "nope")) is None
