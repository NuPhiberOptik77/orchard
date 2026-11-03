import json
import os
import subprocess

import pytest
from typer.testing import CliRunner

from orchard import cli, fleet, worlds
from orchard.job import JobSpec

GOOD = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-broken\n+fixed\n"
OTHER = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-broken\n+still broken\n"
STALE = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-nothing like this\n+fixed\n"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(worlds, "WORLDS_DIR", tmp_path / "worlds")
    monkeypatch.setattr(fleet, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(fleet, "LOG_DIR", tmp_path / "fleet")


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "x.txt").write_text("broken\n")
    (root / "src" / "a.py").write_text("print(1)\n")
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / "node_modules" / "dep" / "index.js").write_text("")
    (root / ".orchard-out").mkdir()
    os.symlink("src/a.py", root / "link.py")
    return root


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_copy_tree_copies_shipped_files_and_links_shared_dirs(proj, tmp_path):
    dest = tmp_path / "w"
    count = worlds.copy_tree(proj, dest)
    assert (dest / "x.txt").read_text() == "broken\n" and (dest / "src" / "a.py").exists()
    assert (dest / "link.py").is_symlink() and os.readlink(dest / "link.py") == "src/a.py"
    assert (dest / "node_modules").is_symlink() and (dest / "node_modules").resolve() == (proj / "node_modules").resolve()
    assert not (dest / ".orchard-out").exists()
    assert count == 3
    (dest / "x.txt").write_text("changed\n")
    assert (proj / "x.txt").read_text() == "broken\n"


def test_copy_tree_without_sharing_and_without_clonefile(proj, tmp_path, monkeypatch):
    monkeypatch.setattr(worlds, "_CLONEFILE", None)
    dest = tmp_path / "w"
    worlds.copy_tree(proj, dest, shared=())
    assert (dest / "x.txt").read_text() == "broken\n"
    assert not (dest / "node_modules").exists()


def test_apply_patch_is_all_or_nothing(proj, tmp_path):
    dest = tmp_path / "w"
    worlds.copy_tree(proj, dest)
    ok, err = worlds.apply_patch(dest, write(tmp_path, "stale.diff", STALE))
    assert not ok and err and (dest / "x.txt").read_text() == "broken\n"
    ok, _ = worlds.apply_patch(dest, write(tmp_path, "good.diff", GOOD))
    assert ok and (dest / "x.txt").read_text() == "fixed\n"
    assert "x.txt" in worlds.diff_stat(tmp_path / "good.diff", proj)


def test_apply_patch_ignores_a_repo_above_the_world(proj, tmp_path):
    # Inside an enclosing repo, git apply would take paths from that repo's
    # root and silently skip the file; the ceiling keeps it to the world.
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    dest = outer / "worlds" / "w"
    worlds.copy_tree(proj, dest)
    ok, _ = worlds.apply_patch(dest, write(tmp_path, "good.diff", GOOD))
    assert ok and (dest / "x.txt").read_text() == "fixed\n"


def test_assign_round_robin():
    assert worlds.assign(5, ["local", "pro"]) == ["local", "pro", "local", "pro", "local"]


def test_verdict_every_summary():
    def w(label, state, patch="p"):
        return {"label": label, "patch": None if label == "baseline" else patch, "state": state, "target": "pro"}

    assert worlds.verdict([w("a", "failed")])["winner"] is None
    assert worlds.verdict([w("a", "succeeded")])["summary"] == "a passed on pro"
    fixed = worlds.verdict([w("baseline", "failed"), w("a", "failed"), w("b", "succeeded"), w("c", "succeeded")])
    assert fixed["winner"] == "b" and fixed["also_passed"] == ["c"] and "fixes it" in fixed["summary"]
    assert "so does the unpatched tree" in worlds.verdict([w("baseline", "succeeded"), w("a", "succeeded")])["summary"]
    assert "did not finish" in worlds.verdict([w("baseline", "unavailable"), w("a", "succeeded")])["summary"]


def fake_fleet(monkeypatch, targets=("local", "pro")):
    """A world passes when its x.txt reads "fixed"."""
    seen = []

    def fake_run_on(target, cmd, cwd, name, **kw):
        seen.append((target, name, cwd))
        state = "succeeded" if (cwd / "x.txt").read_text() == "fixed\n" else "failed"
        return {"target": target, "state": state, "job": f"{name}-20261103-101010-abcd"}

    monkeypatch.setattr(fleet, "run_on", fake_run_on)
    monkeypatch.setattr(fleet, "capable_targets", lambda job: list(targets))
    monkeypatch.setattr(fleet, "job_like_run", lambda cmd, cwd, *a: JobSpec(name="t", cmd=list(cmd)))
    return seen


def test_fork_names_the_winner_and_keeps_only_its_world(proj, tmp_path, monkeypatch):
    monkeypatch.chdir(proj)
    seen = fake_fleet(monkeypatch)
    args = ["fork", "--baseline", "-p", str(write(tmp_path, "other.diff", OTHER)),
            "-p", str(write(tmp_path, "good.diff", GOOD)), "-p", str(write(tmp_path, "stale.diff", STALE)),
            "--json", "--", "check"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["winner"].endswith("good.diff") and "fixes it" in out["summary"] and "x.txt" in out["stat"]
    states = {r["label"].rsplit("/", 1)[-1]: r["state"] for r in out["worlds"]}
    assert states == {"baseline": "failed", "other.diff": "failed", "good.diff": "succeeded",
                      "stale.diff": "patch does not apply"}
    assert {t for t, _, _ in seen} == {"local", "pro"} and len(seen) == 3
    kept = [r["world"] for r in out["worlds"] if r["world"]]
    assert len(kept) == 1 and kept[0].endswith("good-diff")
    assert (proj / "x.txt").read_text() == "broken\n"


def test_fork_first_stops_and_baseline_pass_does_not(proj, tmp_path, monkeypatch):
    monkeypatch.chdir(proj)
    (proj / "x.txt").write_text("fixed\n")
    seen = fake_fleet(monkeypatch, targets=("local",))
    already = "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-fixed\n+fixed \n"
    args = ["fork", "--first", "--baseline", "-p", str(write(tmp_path, "a.diff", already)),
            "-p", str(write(tmp_path, "b.diff", already)), "--", "check"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 1, result.output
    assert len(seen) == 3 and "no patch passed" in result.output

    (proj / "x.txt").write_text("broken\n")
    seen = fake_fleet(monkeypatch, targets=("local",))
    args = ["fork", "--first", "-p", str(write(tmp_path, "g.diff", GOOD)),
            "-p", str(write(tmp_path, "o.diff", OTHER)), "--", "check"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert len(seen) == 1 and "skipped" in result.output and "world:" in result.output


def test_fork_on_skips_discovery_and_keep_keeps_all(proj, tmp_path, monkeypatch):
    monkeypatch.chdir(proj)
    seen = fake_fleet(monkeypatch)
    monkeypatch.setattr(fleet, "capable_targets", lambda job: pytest.fail("--on must not probe machines"))
    args = ["fork", "--on", "pro", "--keep", "-p", str(write(tmp_path, "o.diff", OTHER)), "--", "check"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 1
    assert [t for t, _, _ in seen] == ["pro"] and seen[0][2].exists()


def test_fork_errors(proj, tmp_path, monkeypatch):
    monkeypatch.chdir(proj)
    fake_fleet(monkeypatch)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["fork", "--", "check"]).exit_code == 2
    assert runner.invoke(cli.app, ["fork", "-p", "nope.diff", "--", "check"]).exit_code == 2
    stale = runner.invoke(cli.app, ["fork", "-p", str(write(tmp_path, "s.diff", STALE)), "--", "check"])
    assert stale.exit_code == 2 and "patch does not apply" in stale.output
    assert not (worlds.WORLDS_DIR.exists() and any(worlds.WORLDS_DIR.iterdir()))
