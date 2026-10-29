"""Remote execution engine.

One implementation of the remote job lifecycle shared by the CLI and the
dispatcher: candidate probing, input staging, environment sync, launch,
status polling, log/artifact collection, and kill.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from io import BytesIO
from pathlib import Path

from . import agent, env, events, store, transport
from .config import Worker
from .job import JobSpec, agent_job_payload

EXCLUDES = {
    ".git", ".venv", ".orchard", ".orchard-out", "node_modules", ".build",
    "DerivedData", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store",
}

IDLE_THRESHOLD_S = 180
RESUME_MARGIN_S = 30
THERMAL_PAUSE_BELOW = 50
THERMAL_RESUME_AT = 60

AIR_BUSY_LOAD = 0.75  # normalized 1-min load average at which the Air counts as busy


def air_load() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0


def air_busy(threshold: float | None = None) -> bool:
    cores = os.cpu_count() or 4
    return air_load() >= (threshold if threshold is not None else AIR_BUSY_LOAD) * cores


def remote_bench_python(worker: Worker) -> str | None:
    cmd = 'ls -d "$HOME"/.orchard/python/cpython-3.12*/bin/python3 2>/dev/null | tail -1'
    try:
        proc = transport.run(worker, cmd, timeout=20)
    except subprocess.TimeoutExpired:
        return None
    out = proc.stdout.decode("utf-8", "replace").strip()
    return out or None


def remote_caps(worker: Worker) -> dict | None:
    script = Path(agent.__file__).read_bytes()
    args = ["status"]
    bench_python = remote_bench_python(worker)
    if bench_python:
        args.append(bench_python)
    try:
        proc = transport.run_script(worker, script, args, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        caps = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    caps["tags"] = sorted(set(caps.get("tags") or []) | set(worker.tags))
    return caps


def remote_home(worker: Worker) -> str:
    return transport.run_checked(worker, "echo $HOME", timeout=20).strip()


def remote_job_running(worker: Worker, job_id: str) -> bool:
    """True when the worker's agent reports this job's process group is still alive.

    Used to adopt a job whose "running" state was lost (e.g. the Air died right
    after the remote agent started) instead of relaunching it twice.
    """
    script = (
        f'M="$HOME/.orchard/state/running/{job_id}.json"; '
        f'[ -f "$M" ] && /usr/bin/python3 "$HOME/.orchard/bin/orchard-agent.py" running "{job_id}" '
        f'|| echo no'
    )
    try:
        proc = transport.run(worker, script, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    return proc.stdout.decode("utf-8", "replace").strip() == "yes"


def remote_job_dir(rhome: str, job_id: str) -> str:
    return f"{rhome}/.orchard/jobs/{job_id}"


def _git_paths(root: Path, include_untracked: bool) -> list[str] | None:
    args = ["git", "-C", str(root), "ls-files", "-z"]
    if include_untracked:
        args += ["--cached", "--others", "--exclude-standard"]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return [name for name in proc.stdout.decode("utf-8", "replace").split("\0") if name]


def tracked_files(root: Path) -> list[str] | None:
    """Paths (relative to root) tracked by git, or None when root is not a repo."""
    return _git_paths(root, include_untracked=False)


def _excluded_rel(rel: Path) -> bool:
    return any(part in EXCLUDES for part in rel.parts) or rel.name.endswith((".pyc", ".pyo"))


def iter_shipped(root: Path, tracked_only: bool = False):
    """Yield (full_path, arcname) for files that ship to a worker.

    In a git repo, ignored files never ship: build jobs take tracked files
    only; run jobs take tracked plus untracked-but-not-ignored files
    (`.gitignore` is respected). Outside a repo, everything except EXCLUDES
    ships, as before.
    """
    names = _git_paths(root, include_untracked=not tracked_only)
    if names is not None:
        for name in names:
            rel = Path(name)
            if _excluded_rel(rel):
                continue
            full = root / rel
            if full.exists() or full.is_symlink():
                yield full, name
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDES and not d.endswith(".egg-info")]
        for name in filenames:
            if name in EXCLUDES or name.endswith((".pyc", ".pyo")):
                continue
            full = Path(dirpath) / name
            yield full, str(full.relative_to(root))


def _tar_into(fobj, root: Path, tracked_only: bool = False) -> int:
    # mode="w:gz" stamps the gzip header with the current time and the temp
    # file's random name, so an unchanged tree never hashed the same and the
    # worker's inbox check (`test -f <sha>.tgz`) never skipped an upload.
    count = 0
    with gzip.GzipFile(filename="", mode="wb", fileobj=fobj, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for full, arcname in iter_shipped(root, tracked_only):
                tar.add(full, arcname=arcname, recursive=False)
                count += 1
    return count


def build_tar_bytes(root: Path, tracked_only: bool = False) -> bytes:
    buf = BytesIO()
    _tar_into(buf, root, tracked_only)
    return buf.getvalue()


MAX_TAR_MB_DEFAULT = 2048


def max_tar_bytes() -> int:
    try:
        return int(float(os.environ.get("ORCHARD_MAX_TAR_MB", MAX_TAR_MB_DEFAULT)) * 1e6)
    except ValueError:
        return MAX_TAR_MB_DEFAULT * 1_000_000


def check_shipping_size(cwd: Path, kind: str) -> None:
    """Refuse to ship absurdly large trees (keeps the Air from OOMing on tar)."""
    total = 0
    count = 0
    for full, _arcname in iter_shipped(cwd, tracked_only=(kind == "build")):
        try:
            total += full.lstat().st_size
        except OSError:
            continue
        count += 1
    cap = max_tar_bytes()
    if total > cap:
        raise ValueError(
            f"project ships {total / 1e9:.1f} GB across {count} files (cap {cap / 1e9:.1f} GB); "
            "add large folders to .gitignore (or move them out), or raise ORCHARD_MAX_TAR_MB"
        )


def launch_remote(
    worker: Worker, job: JobSpec, job_id: str, sdir: Path, cwd: Path, rhome: str | None = None
) -> tuple[str, str]:
    rhome = rhome or remote_home(worker)
    rroot = f"{rhome}/.orchard"
    rjob = remote_job_dir(rhome, job_id)

    is_build = job.kind == "build"
    check_shipping_size(cwd, job.kind)
    tmp = tempfile.NamedTemporaryFile(suffix=".tgz", delete=False)
    tmp.close()
    tar_path = Path(tmp.name)
    try:
        with open(tar_path, "wb") as fh:
            count = _tar_into(fh, cwd, tracked_only=is_build)
        sha = store.add_blob_file(tar_path)
        size = tar_path.stat().st_size
        label = "tracked inputs" if is_build else "inputs"
        print(f"{label}: {size / 1e6:.1f} MB across {count} files (blob {sha[:12]})")

        env.ensure_agent_uploaded(worker)
        marker = f"{rroot}/inbox/{sha}.tgz"
        transport.run_checked(worker, f'mkdir -p "{rroot}/inbox"', timeout=30)
        if transport.run(worker, f'test -f "{marker}"', timeout=20).returncode != 0:
            transport.run_checked(
                worker,
                f'cat > "{marker}.tmp" && mv "{marker}.tmp" "{marker}"',
                stdin_file=tar_path,
                timeout=1800,
            )
    finally:
        tar_path.unlink(missing_ok=True)
    if is_build:
        venv = None
        key = hashlib.sha256(str(cwd.resolve()).encode()).hexdigest()[:12]
        rwork = f"{rroot}/builds/{key}/work"
        transport.run_checked(
            worker,
            f'rm -rf "{rwork}.tmp" && mkdir -p "{rwork}.tmp" && tar -xzf "{marker}" -C "{rwork}.tmp"'
            f' && {{ [ -d "{rwork}/.build" ] && mv "{rwork}/.build" "{rwork}.tmp/.build" || true; }}'
            f' && rm -rf "{rwork}" && mv "{rwork}.tmp" "{rwork}"',
            timeout=900,
        )
        print(f"staged build tree on {worker.name} (cache: {rroot}/builds/{key})")
    else:
        rwork = f"{rjob}/work"
        transport.run_checked(
            worker,
            f'rm -rf "{rwork}" && mkdir -p "{rwork}" && tar -xzf "{marker}" -C "{rwork}"',
            timeout=900,
        )
        print(f"staged inputs on {worker.name}")
        env.ensure_uv_remote(worker)
        venv = env.sync_env_remote(
            worker, env.env_key(job.python, job.requirements), job.python, job.requirements, rhome
        )
        print(f"environment ready ({job.python})")

    payload = agent_job_payload(job, job_id, rjob, rwork, venv, preempt=True)
    transport.push_bytes(worker, f"{rjob}/job.json", json.dumps(payload, indent=2).encode())

    start = (
        f'cd "{rroot}" && nohup /usr/bin/python3 "{rroot}/bin/orchard-agent.py" exec "{rjob}/job.json" '
        f'> "{rjob}/agent.out" 2>&1 & echo started'
    )
    transport.run_checked(worker, start, timeout=30)
    print(f"job started on {worker.name} (detached)")
    events.append("launched", job=job_id, worker=worker.name, kind=job.kind)
    return rhome, rjob


def remote_status(worker: Worker, rjob: str) -> dict | None:
    try:
        proc = transport.run(worker, f'cat "{rjob}/status.json" 2>/dev/null || true', timeout=20)
    except subprocess.TimeoutExpired:
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def watch_remote(worker: Worker, rjob: str, interval: float = 4.0) -> dict | None:
    last_state = None
    while True:
        status = remote_status(worker, rjob)
        if status is None:
            time.sleep(3)
            continue
        state = status.get("state")
        if state != last_state:
            detail = f" [{status.get('pause_reason')}]" if state == "paused" else ""
            print(
                f"  [{time.strftime('%H:%M:%S')}] {state}{detail} "
                f"(elapsed {status.get('elapsed_s', 0)}s, active {status.get('active_s', 0)}s)"
            )
            last_state = state
        if state in agent.TERMINAL_STATES:
            return status
        time.sleep(interval)


ARTIFACT_WARN_MB_DEFAULT = 2048
ARTIFACT_FREE_MARGIN_BYTES = 3_000_000_000


def artifact_warn_bytes() -> int:
    try:
        return int(float(os.environ.get("ORCHARD_ARTIFACT_WARN_MB", ARTIFACT_WARN_MB_DEFAULT)) * 1e6)
    except ValueError:
        return ARTIFACT_WARN_MB_DEFAULT * 1_000_000


def artifact_pull_verdict(size_bytes: int, free_bytes: int, warn_bytes: int) -> tuple[str, str]:
    """Decide whether pulling an artifact set is safe for the Air's disk."""
    if size_bytes > free_bytes - ARTIFACT_FREE_MARGIN_BYTES:
        return (
            "refuse",
            f"artifacts are {size_bytes / 1e9:.1f} GB but the Air has only "
            f"{free_bytes / 1e9:.1f} GB free — not pulling (ORCHARD_ARTIFACT_FORCE=1 to override)",
        )
    if size_bytes > warn_bytes:
        return "warn", f"artifacts are {size_bytes / 1e9:.1f} GB (large; pulling anyway)"
    return "ok", ""


def _remote_dir_bytes(worker: Worker, rpath: str) -> int:
    try:
        proc = transport.run(worker, f'du -sk "{rpath}" 2>/dev/null || true', timeout=120)
    except subprocess.TimeoutExpired:
        return 0
    try:
        return int(proc.stdout.decode("utf-8", "replace").split()[0]) * 1024
    except (ValueError, IndexError):
        return 0


def collect_remote(worker: Worker, job_id: str, rjob: str, sdir: Path, cwd: Path) -> dict:
    status = None
    try:
        (sdir / "log.txt").write_bytes(transport.pull_bytes(worker, f"{rjob}/log.txt"))
        (sdir / "status.json").write_bytes(transport.pull_bytes(worker, f"{rjob}/status.json"))
        status = agent.read_status(str(sdir))
    except transport.RemoteError as exc:
        print(f"warning: could not collect logs: {exc}")

    workdir = (status or {}).get("workdir") or f"{rjob}/work"
    artifacts = cwd / ".orchard-out" / job_id
    size_bytes = _remote_dir_bytes(worker, f"{workdir}/out")
    if size_bytes:
        verdict, msg = artifact_pull_verdict(size_bytes, shutil.disk_usage(cwd).free, artifact_warn_bytes())
        if verdict == "refuse" and os.environ.get("ORCHARD_ARTIFACT_FORCE") != "1":
            print(f"warning: {msg}")
            print(f"artifacts kept on {worker.name} at {workdir}/out (nothing pulled)")
            return status or {}
        if verdict != "ok":
            print(f"note: {msg}")
    tmp = tempfile.NamedTemporaryFile(suffix=".tgz", delete=False)
    tmp.close()
    tar_path = Path(tmp.name)
    try:
        transport.run(
            worker,
            f'test -d "{workdir}/out" && tar -czf - -C "{workdir}" out || true',
            timeout=600,
            stdout_file=tar_path,
        )
        if tar_path.stat().st_size:
            artifacts.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, mode="r:gz") as tar:
                tar.extractall(artifacts, filter="data")
            print(f"artifacts: {artifacts / 'out'}")
        else:
            kind = None
            try:
                kind = json.loads((sdir / "spec.json").read_text()).get("kind")
            except (OSError, json.JSONDecodeError):
                pass
            if kind == "build":
                print("no artifacts produced (build output stays on the worker)")
            else:
                print("no artifacts produced (out/ was empty or missing)")
    except (transport.RemoteError, subprocess.TimeoutExpired, tarfile.TarError) as exc:
        print(f"warning: could not collect artifacts: {exc}")
    finally:
        tar_path.unlink(missing_ok=True)

    return status or {}


def kill_remote(worker: Worker, rjob: str) -> bool:
    rroot = rjob.rsplit("/jobs/", 1)[0]
    try:
        proc = transport.run(
            worker, f'/usr/bin/python3 "{rroot}/bin/orchard-agent.py" kill "{rjob}"', timeout=30
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def prepare_local_job(job: JobSpec, job_id: str, sdir: Path, cwd: Path) -> tuple[Path | None, Path]:
    job_json = sdir / "job.json"
    if job.kind == "build":
        payload = agent_job_payload(job, job_id, str(sdir), str(cwd), None, preempt=False)
        job_json.write_text(json.dumps(payload, indent=2) + "\n")
        return None, job_json
    key = env.env_key(job.python, job.requirements)
    venv = env.ensure_env_local(key, job.python, job.requirements)
    payload = agent_job_payload(job, job_id, str(sdir), str(cwd), str(venv), preempt=False)
    job_json.write_text(json.dumps(payload, indent=2) + "\n")
    return venv, job_json


def launch_local(job: JobSpec, job_id: str, sdir: Path, cwd: Path) -> tuple[Path | None, int]:
    venv, job_json = prepare_local_job(job, job_id, sdir, cwd)
    proc = subprocess.Popen(
        [sys.executable, str(Path(agent.__file__)), "exec", str(job_json)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    events.append("launched", job=job_id, worker="local", kind=job.kind)
    return venv, proc.pid


def local_status(sdir: Path) -> dict | None:
    return agent.read_status(str(sdir))
