import hashlib
import io
import json
import subprocess
import tarfile

import pytest
from typer.testing import CliRunner

from orchard import ai_install as inst
from orchard import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _manifests(monkeypatch, tmp_path):
    monkeypatch.setattr(inst, "INSTALL_DIR", tmp_path / "install")


def _tarball(path, files):
    with tarfile.open(path, "w:gz") as tar:
        for name, (body, mode) in files.items():
            info = tarfile.TarInfo(f"llama-b1/{name}")
            info.size, info.mode = len(body), mode
            tar.addfile(info, io.BytesIO(body))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path, binary_body=b"#!/bin/sh\nexit 0\n", sha=None):
    ball = tmp_path / "helper.tar.gz"
    digest = _tarball(ball, {"ggml-rpc-server": (binary_body, 0o755), "lib/libggml.dylib": (b"lib", 0o644)})
    return {"role": "helper", "host": "", "folder": str(tmp_path / "llm" / "helper"), "platform": "test",
            "release": "b1", "folder_state": "absent", "binary": "ggml-rpc-server", "installed": False,
            "files": [{"name": "helper.tar.gz", "url": ball.as_uri(), "bytes": ball.stat().st_size,
                       "sha256": sha or digest}]}


def test_each_platform_gets_its_pinned_release():
    assert inst.bundle("Darwin", "arm64", False)[0] == ["llama-b12091-bin-macos-arm64.tar.gz"]
    assert inst.bundle("Darwin", "x86_64", False)[0] == ["llama-b12091-bin-macos-x64.tar.gz"]
    assert inst.bundle("Linux", "x86_64", True)[0] == ["llama-b12091-bin-ubuntu-cuda-12.8-x64.tar.gz",
                                                       "cudart-llama-b12091-bin-ubuntu-cuda-12.8-x64.tar.gz"]
    assert inst.bundle("Linux", "x86_64", False)[0] == ["llama-b12091-bin-ubuntu-x64.tar.gz"]
    assert inst.bundle("Windows_NT", "x86_64", False) is None
    for files, _ in filter(None, [inst.bundle(s, m, g) for s in ("Darwin", "Linux")
                                  for m in ("arm64", "x86_64", "aarch64") for g in (True, False)]):
        assert all(name in inst.ASSETS for name in files)


def test_the_plan_reads_the_machine_and_says_file_source_size_and_folder():
    seen = []
    p = inst.plan("engine", "big-box", run=lambda host, script, timeout: seen.append((host, script)) or
                  subprocess.CompletedProcess([], 0, "Linux\nx86_64\nnvidia\nabsent\n", ""))
    assert seen[0][0] == "big-box" and 'folder="$HOME"/llm/engine' in seen[0][1]
    text = "\n".join(inst.describe(p))
    assert "Install the engine on big-box (Linux x86_64 with an NVIDIA card), llama.cpp b12091:" in text
    assert "file: llama-b12091-bin-ubuntu-cuda-12.8-x64.tar.gz, 187.2 MB" in text
    assert "file: cudart-llama-b12091-bin-ubuntu-cuda-12.8-x64.tar.gz, 594.4 MB" in text
    assert "from: https://github.com/ggml-org/llama.cpp/releases/download/b12091/" in text
    assert "into: ~/llm/engine (a new folder)" in text and "build-engine.sh" in text


def test_bad_names_never_reach_a_shell():
    for folder in ("~/../etc", "/tmp/$(id)", "~", "/", "relative/path", "~/a b"):
        with pytest.raises(inst.InstallError):
            inst.plan("helper", folder=folder, run=lambda *a: pytest.fail("ran"))
    with pytest.raises(inst.InstallError):
        inst.plan("helper", "-oProxyCommand=x", run=lambda *a: pytest.fail("ran"))
    with pytest.raises(inst.InstallError):
        inst.plan("server", run=lambda *a: pytest.fail("ran"))


def test_install_checks_the_digest_starts_the_binary_and_undo_removes_only_its_files(tmp_path):
    p = _plan(tmp_path)
    record = inst.install(p)
    folder = tmp_path / "llm" / "helper"
    assert sorted(f["path"] for f in record["files"]) == ["ggml-rpc-server", "lib/libggml.dylib"]
    assert record["created_folder"] and record["dirs"] == ["lib"]
    assert all(len(f["sha256"]) == 64 for f in record["files"])
    assert json.loads(inst.manifest_path("", "helper").read_text())["files"] == record["files"]
    with pytest.raises(inst.InstallError, match="already installed"):
        inst.install(p | {"installed": True})
    (folder / "mine.txt").write_text("the owner's")
    (folder / "models").mkdir()
    (folder / "lib" / "libggml.dylib").write_text("patched since")
    result = inst.undo("helper")
    assert result["removed"] == 1 and result["listed"] == 2 and not result["folder_gone"]
    assert sorted(x.name for x in folder.iterdir()) == ["lib", "mine.txt", "models"]
    assert (folder / "lib" / "libggml.dylib").read_text() == "patched since"
    assert not inst.manifest_path("", "helper").exists()
    with pytest.raises(inst.InstallError, match="nothing to undo"):
        inst.undo("helper")


def test_a_file_put_in_the_folder_during_the_download_stops_the_install(tmp_path):
    p = _plan(tmp_path)
    folder = tmp_path / "llm" / "helper"
    url = p["files"][0]["url"]

    def run(host, script, timeout):
        # The owner makes the folder and drops a file in while the download runs.
        script = script.replace(f"curl -fsSL --retry 3 -o", f"mkdir -p {folder} && echo m > {folder}/model.gguf && "
                                "curl -fsSL --retry 3 -o", 1)
        return inst._sh(host, script, timeout)

    with pytest.raises(inst.InstallError, match="already has files"):
        inst.install(p, run=run)
    assert [x.name for x in folder.iterdir()] == ["model.gguf"] and url
    assert not inst.manifest_path("", "helper").exists() and not list((tmp_path / "llm").glob("*.orchard.*"))


def test_this_machine_and_a_host_never_share_a_record():
    assert inst.manifest_path("", "helper") != inst.manifest_path("this-machine", "helper")
    for bad in ("a@-oX", "-x", "a@b@c", "a b"):
        with pytest.raises(inst.BadInput):
            inst.plan("helper", bad, run=lambda *a: pytest.fail("ran"))


def test_a_broken_record_or_folder_keeps_the_record(tmp_path):
    inst.manifest_path("", "helper").parent.mkdir(parents=True)
    inst.manifest_path("", "helper").write_text("{not json")
    with pytest.raises(inst.InstallError, match="can't be read"):
        inst.undo("helper")
    inst.manifest_path("", "helper").write_text(json.dumps({"folder": "~", "files": []}))
    with pytest.raises(inst.InstallError, match="can't be read") as caught:
        inst.undo("helper")
    assert not isinstance(caught.value, inst.BadInput)
    inst.manifest_path("", "helper").write_text(json.dumps({"folder": str(tmp_path / "f"), "files": []}))
    (tmp_path / "f").write_text("a file, not a folder")
    with pytest.raises(inst.InstallError, match="record of the install is kept"):
        inst.undo("helper")
    assert inst.manifest_path("", "helper").exists()


def test_a_timeout_here_lets_the_install_clean_up(tmp_path):
    folder = tmp_path / "llm" / "helper"
    script = inst._install_script(_plan(tmp_path)).replace("rc=0\n", "sleep 30\nrc=0\n", 1)
    done = inst._sh("", script, 2)
    assert done.returncode == 255 and not folder.exists()
    assert not list((tmp_path / "llm").glob("*.orchard.*"))


def test_undo_removes_the_folder_it_made(tmp_path):
    inst.install(_plan(tmp_path))
    assert inst.undo("helper")["folder_gone"] and not (tmp_path / "llm" / "helper").exists()


def test_a_wrong_digest_or_a_binary_that_wont_start_leaves_the_folder_untouched(tmp_path):
    with pytest.raises(inst.InstallError, match="digest"):
        inst.install(_plan(tmp_path, sha="0" * 64))
    assert not (tmp_path / "llm" / "helper").exists()
    with pytest.raises(inst.InstallError, match="didn't start"):
        inst.install(_plan(tmp_path, binary_body=b"#!/no/such/shell\n"))
    assert not (tmp_path / "llm" / "helper").exists() and not inst.manifest_path("", "helper").exists()
    assert not (tmp_path / "llm").exists()  # the parent it made is gone again


def test_a_folder_with_files_in_it_is_refused(tmp_path):
    folder = tmp_path / "llm" / "helper"
    folder.mkdir(parents=True)
    (folder / "old").write_text("x")
    with pytest.raises(inst.InstallError, match="already has files"):
        inst.install(_plan(tmp_path) | {"folder_state": "full"})
    with pytest.raises(inst.InstallError, match="already has files"):
        inst.install(_plan(tmp_path))  # the machine is checked again at install time
    assert [x.name for x in folder.iterdir()] == ["old"]


def test_orchard_ai_install_plans_unless_told_yes(monkeypatch):
    calls = []
    monkeypatch.setattr(inst, "_sh", lambda host, script, timeout: calls.append(script) or
                        subprocess.CompletedProcess([], 0, "Darwin\nx86_64\nnone\nabsent\n", ""))
    result = runner.invoke(cli.app, ["ai", "install", "--role", "helper", "--host", "old-mac"])
    assert result.exit_code == 0 and "file: llama-b12091-bin-macos-x64.tar.gz, 12.3 MB" in result.stdout
    assert "Nothing was downloaded. To go ahead: orchard ai install --role helper --host old-mac --yes" in result.stdout
    assert len(calls) == 1 and "curl" not in calls[0]
    assert runner.invoke(cli.app, ["ai", "install", "--role", "nope"]).exit_code == 2
    assert runner.invoke(cli.app, ["ai", "install", "--role", "helper", "--undo"]).exit_code == 1
