"""Living replicas: a warm copy of the project kept on each body.

Think of the Cleons in Asimov's Foundation: one genome at three ages.

- Dawn learns. `replica up` copies the project to a body, builds its Python
  env and runs the project's prep command (npm ci, uv sync, ...). Prep runs
  again only when one of its key files (lockfiles by default) changes.
- Day acts. `replica run` sends only the files that changed since the body's
  last copy and runs the check there, warm.
- Dusk advises. One replica can be the dusk. It moves only to a tree every
  check passed on (`replica check`) or one you bless (`run --bless`), so it
  remembers the last green tree. `compare` and `blame` ask it.

Demerzel, who outlives every Cleon, is the receipts ledger: every replica run
is sealed like any other run. A copy that drifted (a check rewrote tracked
files) is caught by digest, reported, and sent back on the next run.
"""
from __future__ import annotations

import contextlib
import fcntl
import fnmatch
import hashlib
import json
import os
import secrets
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import consensus, dispatcher, engine, env, events, fleet, placement, receipts, speculate, transport
from .config import JOBS_DIR, STATE_DIR, Worker, load_workers
from .job import JobSpec, agent_job_payload, capture_env, new_job_id, slugify

REPLICAS_DIR = STATE_DIR / "replicas"
AGENT = f'/usr/bin/python3 "{env.AGENT_REMOTE_PATH}"'
PREP_KEY_NAMES = (
    "package.json", "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    "requirements*.txt", "pyproject.toml", "uv.lock", "poetry.lock", "Pipfile.lock",
    "Package.swift", "Package.resolved", "Podfile.lock", "Cargo.lock", "go.sum", "Gemfile.lock", "composer.lock",
)
HASH_SETTLE_S = 2.0  # a file touched this recently is hashed again next time, whatever its stat says
ROLES = ("day", "dusk")
LOST_AFTER_S = 600  # a body silent this long while it runs a job is given up on


class ReplicaError(Exception):
    pass


class ReplicaBusy(ReplicaError):
    pass


class _Quiet(ReplicaError):
    pass


class _Moved(ReplicaError):
    """The copy changed between its sync and the job's start."""

    def __init__(self, message: str, drifted: list[str] | None = None):
        super().__init__(message)
        self.drifted = list(drifted or [])


# --- local state -------------------------------------------------------------

def project_key(cwd: Path) -> str:
    return hashlib.sha256(str(cwd.resolve()).encode()).hexdigest()[:12]


def local_dir(key: str) -> Path:
    d = REPLICAS_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextlib.contextmanager
def _flock(path: Path, block: bool = True):
    """Yields True with the lock held, or False when block is off and it is taken."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX if block else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load_json(path: Path, default):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default
    return value if default is None or isinstance(value, type(default)) else default


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_registry(key: str) -> dict:
    reg = _load_json(local_dir(key) / "registry.json", {})
    reg.setdefault("project", None)
    reg.setdefault("prep", None)
    reg.setdefault("prep_keys", None)
    reg.setdefault("replicas", {})
    reg.setdefault("dusk_tree", None)
    return reg


@contextlib.contextmanager
def edit_registry(key: str):
    with _flock(local_dir(key) / "registry.lock"):
        reg = load_registry(key)
        yield reg
        _save_json(local_dir(key) / "registry.json", reg)


def worker_lock(key: str, name: str, block: bool = True):
    """Held while this Mac changes a replica: from its sync to its job's start."""
    return _flock(local_dir(key) / f"{slugify(name)}.lock", block)


def worker_named(name: str) -> Worker:
    for w in load_workers():
        if w.name == name:
            return w
    raise ReplicaError(f"no worker named {name} (orchard workers)")


def dusk_of(reg: dict) -> str | None:
    return next((n for n, r in reg["replicas"].items() if r.get("role") == "dusk"), None)


def days_of(reg: dict) -> list[str]:
    return [n for n, r in reg["replicas"].items() if r.get("role") != "dusk"]


# --- talking to a body's agent -------------------------------------------------

_AGENT_READY: set[str] = set()
_AGENT_LOCK = threading.Lock()


def _ensure_agent(worker: Worker) -> None:
    with _AGENT_LOCK:
        if worker.name in _AGENT_READY:
            return
    env.ensure_agent_uploaded(worker)
    with _AGENT_LOCK:
        _AGENT_READY.add(worker.name)


def remote_call(worker: Worker, verb: str, *args: str, stdin_file: Path | None = None,
                stdout_file: Path | None = None, timeout: float = 60) -> subprocess.CompletedProcess:
    _ensure_agent(worker)
    cmd = " ".join([AGENT, verb, *(shlex.quote(a) for a in args)])
    return transport.run(worker, cmd, timeout=timeout, stdin_file=stdin_file, stdout_file=stdout_file)


def _parse(proc: subprocess.CompletedProcess) -> dict:
    out = proc.stdout.decode("utf-8", "replace") if isinstance(proc.stdout, bytes) else (proc.stdout or "")
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        value = json.loads(lines[-1])
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _stderr(proc: subprocess.CompletedProcess) -> str:
    err = proc.stderr.decode("utf-8", "replace") if isinstance(proc.stderr, bytes) else (proc.stderr or "")
    return err.strip()


def replica_state(worker: Worker, key: str, manifest: bool = False, timeout: float = 60) -> dict | None:
    """What the body says about its copy, or None when it does not answer."""
    try:
        proc = remote_call(worker, "replica-state", key, *(["manifest"] if manifest else []), timeout=timeout)
    except (subprocess.TimeoutExpired, transport.RemoteError, OSError):
        return None
    if proc.returncode != 0:
        return None
    state = _parse(proc)
    if "digest" not in state or "home" not in state:
        return None
    caps = state.get("caps") if isinstance(state.get("caps"), dict) else {}
    caps["tags"] = sorted(set(caps.get("tags") or []) | set(worker.tags))
    state["caps"] = caps
    state.setdefault("meta", {})
    return state


def _probe_all(names: list[str], key: str) -> tuple[dict[str, Worker], dict[str, dict | None]]:
    workers = {w.name: w for w in load_workers()}
    targets = [n for n in names if n in workers]
    if not targets:
        return workers, {}
    with ThreadPoolExecutor(max_workers=len(targets)) as ex:
        futures = {n: ex.submit(replica_state, workers[n], key) for n in targets}
    return workers, {n: f.result() for n, f in futures.items()}


# --- the tree on this Mac ------------------------------------------------------

def local_manifest(cwd: Path, key: str) -> tuple[dict, dict]:
    """The shipped tree as {arc: ["F", exe, sha] | ["L", target, None] | ["?", None, None]},
    and where each entry's bytes come from: {arc: (path | None, size, forced_exe | None)}.
    "?" is an entry that ships but is neither a readable file nor a link (a
    submodule or nested repo, say): it counts in the digest, as tree_digest
    counts it, and a sync refuses it. Hashes are cached by (size, mtime,
    inode) once a file has settled."""
    cache_path = local_dir(key) / "hashcache.json"
    cache = _load_json(cache_path, {})
    fresh, manifest, sources = {}, {}, {}
    now = time.time()
    for full, arc in engine.iter_shipped(cwd):
        try:
            st = full.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            st = None
        if st is not None and stat.S_ISLNK(st.st_mode):
            manifest[arc] = ["L", os.readlink(full), None]
            sources[arc] = (None, 0, None)
            continue
        if st is None or not stat.S_ISREG(st.st_mode):
            manifest[arc], sources[arc] = ["?", None, None], (None, 0, None)
            continue
        sig = [st.st_size, st.st_mtime_ns, st.st_ino]
        hit = cache.get(arc)
        if isinstance(hit, list) and hit[:3] == sig:
            sha = hit[3]
        else:
            try:
                sha = receipts._sha_file(full)
            except FileNotFoundError:
                continue
            except OSError:
                manifest[arc], sources[arc] = ["?", None, None], (None, 0, None)
                continue
        if now - st.st_mtime > HASH_SETTLE_S:
            fresh[arc] = sig + [sha]
        manifest[arc] = ["F", "x" if st.st_mode & 0o111 else "-", sha]
        sources[arc] = (full, st.st_size, None)
    with contextlib.suppress(OSError):
        _save_json(cache_path, fresh)
    return manifest, sources


def apply_overrides(manifest: dict, sources: dict, overrides: dict) -> None:
    """Swap entries for other versions: {arc: None | ["L", target] | ["F", exe, sha, path]}."""
    for arc, o in overrides.items():
        if o is None:
            manifest.pop(arc, None)
            sources.pop(arc, None)
        elif o[0] == "L":
            manifest[arc] = ["L", o[1], None]
            sources[arc] = (None, 0, None)
        else:
            path = Path(o[3])
            manifest[arc] = ["F", o[1], o[2]]
            sources[arc] = (path, path.stat().st_size, o[1])


def _same(a, b) -> bool:
    return (a or [None])[:3] == (b or [None])[:3]


def plan_update(local: dict, remote: dict) -> tuple[list[str], list[str]]:
    deleted = sorted(a for a in remote if a not in local)
    updates = sorted(a for a, e in local.items() if not _same(e, remote.get(a)))
    return deleted, updates


def changed_arcs(a: dict, b: dict) -> list[str]:
    return sorted(arc for arc in set(a) | set(b) if not _same(a.get(arc), b.get(arc)))


# --- sync (Day) ------------------------------------------------------------------

def _cache_path(key: str, name: str) -> Path:
    return local_dir(key) / f"{slugify(name)}.manifest.json"


def _remote_manifest(worker: Worker, key: str, state: dict, force: bool = False) -> tuple[str, dict]:
    """The body's manifest and the digest it belongs to, from cache when it still matches."""
    cached = _load_json(_cache_path(key, worker.name), {})
    if not force and cached.get("digest") == state["digest"] and isinstance(cached.get("manifest"), dict):
        return state["digest"], cached["manifest"]
    full = replica_state(worker, key, manifest=True, timeout=300)
    if full is None or not isinstance(full.get("manifest"), dict):
        raise ReplicaError(f"{worker.name} did not send its manifest")
    _save_json(_cache_path(key, worker.name), {"digest": full["digest"], "manifest": full["manifest"]})
    return full["digest"], full["manifest"]


class _HashingReader:
    def __init__(self, fh):
        self.fh = fh
        self.h = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        data = self.fh.read(n)
        self.h.update(data)
        return data


def _pack(local: dict, sources: dict, updates: list[str], tgz: Path) -> list[str]:
    """Write updates into tgz. Entries are corrected to the bytes actually
    sent; files that vanished are returned so the caller can drop them."""
    vanished = []
    with tarfile.open(tgz, "w:gz") as tar:
        for arc in updates:
            entry = local[arc]
            info = tarfile.TarInfo(arc)
            info.mtime = int(time.time())
            if entry[0] == "L":
                info.type, info.linkname, info.mode = tarfile.SYMTYPE, entry[1], 0o777
                tar.addfile(info)
                continue
            if entry[0] != "F":
                raise ReplicaError(f"cannot send {arc}: it is a folder (a submodule or nested repo) "
                                   "or cannot be read; replicas carry plain files and links only")
            path, _size, forced = sources[arc]
            try:
                fh = open(path, "rb")
            except FileNotFoundError:
                vanished.append(arc)
                continue
            except OSError as exc:
                raise ReplicaError(f"cannot send {arc}: {exc.strerror or exc}") from exc
            with fh:
                st = os.fstat(fh.fileno())
                exe = forced or ("x" if st.st_mode & 0o111 else "-")
                info.size, info.mode = st.st_size, 0o755 if exe == "x" else 0o644
                reader = _HashingReader(fh)
                try:
                    tar.addfile(info, reader)
                except OSError as exc:
                    raise ReplicaError(f"{arc} changed while sending; run again") from exc
                entry[1], entry[2] = exe, reader.h.hexdigest()
    return vanished


def _fmt_bytes(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 100_000 else f"{n / 1e3:.1f} kB"


def sync(worker: Worker, key: str, local: dict, sources: dict, state: dict) -> dict:
    """Make the body's copy equal local, sending only what differs."""
    if state.get("busy"):
        raise ReplicaBusy(f"{worker.name} is running {state['busy']}")
    rrep = f"{state['home']}/replicas/{key}"
    if state.get("exists") and state["digest"] == receipts.digest_manifest(local):
        return {"digest": state["digest"], "sent": 0, "deleted": 0, "bytes": 0, "manifest": local}
    if state.get("exists"):
        base, remote = _remote_manifest(worker, key, state)
    else:
        base, remote = state["digest"], {}
    for attempt in (1, 2):
        deleted, updates = plan_update(local, remote)
        # Names of its own, so two Macs syncing one copy never apply each other's files.
        token = secrets.token_hex(8)
        fd, tmp = tempfile.mkstemp(suffix=".tgz")
        os.close(fd)
        tgz = Path(tmp)
        try:
            for arc in _pack(local, sources, updates, tgz):
                local.pop(arc, None)
                updates.remove(arc)
                if arc in remote:
                    deleted.append(arc)
            size = tgz.stat().st_size
            digest = receipts.digest_manifest(local)
            transport.run_checked(worker, f'mkdir -p "{rrep}"', timeout=30)
            if updates:
                transport.run_checked(worker, f'cat > "{rrep}/incoming-{token}.tgz"', stdin_file=tgz, timeout=1800)
        finally:
            tgz.unlink(missing_ok=True)
        incoming = {"base": base, "deleted": sorted(deleted), "updates": updates, "digest": digest}
        transport.push_bytes(worker, f"{rrep}/incoming-{token}.json", json.dumps(incoming).encode())
        proc = remote_call(worker, "replica-apply", key, token, timeout=900)
        answer = _parse(proc)
        if proc.returncode == 0 and answer.get("digest") == digest:
            _save_json(_cache_path(key, worker.name), {"digest": digest, "manifest": local})
            return {"digest": digest, "sent": len(updates), "deleted": len(deleted), "bytes": size, "manifest": local}
        if proc.returncode == 0:
            _cache_path(key, worker.name).unlink(missing_ok=True)
            raise ReplicaError(f"{worker.name}: its copy came out as {str(answer.get('digest'))[:12]}, "
                               f"not {digest[:12]}; run again")
        _cache_path(key, worker.name).unlink(missing_ok=True)
        if proc.returncode == 3:
            raise ReplicaBusy(f"{worker.name} is running {answer.get('busy')}")
        if proc.returncode == 4 and attempt == 1:
            # The copy moved since we last looked (a check rewrote it, or another
            # Mac synced it): diff against what it really holds and try again.
            base, remote = _remote_manifest(worker, key, {"digest": answer.get("digest")}, force=True)
            continue
        raise ReplicaError(f"{worker.name}: replica-apply failed ({proc.returncode}): "
                           f"{answer.get('error') or _stderr(proc) or 'no answer'}")
    raise ReplicaError(f"{worker.name}: its copy keeps moving; run again")


def ship_estimate(worker: Worker, key: str, local: dict, sources: dict, local_digest: str, state: dict) -> int:
    """Bytes a sync would send: what makes one body nearer than another."""
    if state.get("exists") and state["digest"] == local_digest:
        return 0
    cached = _load_json(_cache_path(key, worker.name), {})
    remote = cached.get("manifest") if state.get("exists") and isinstance(cached.get("manifest"), dict) else {}
    _deleted, updates = plan_update(local, remote)
    return sum(sources[a][1] for a in updates if a in sources)


# --- Dawn: env and prep ------------------------------------------------------------

def prep_keys(reg: dict, manifest: dict) -> list[str]:
    patterns = reg.get("prep_keys") or PREP_KEY_NAMES
    return sorted(a for a in manifest
                  if any(fnmatch.fnmatch(a, p) or fnmatch.fnmatch(a.rsplit("/", 1)[-1], p) for p in patterns))


def prep_digest(reg: dict, manifest: dict) -> str | None:
    if not reg.get("prep"):
        return None
    h = hashlib.sha256(reg["prep"].encode())
    for arc in prep_keys(reg, manifest):
        h.update(f"\0{arc}\0{json.dumps(manifest[arc][:3])}".encode())
    return h.hexdigest()


def _project_job(cwd: Path, cmd: list[str], env_names: list[str] | None = None) -> JobSpec:
    job = JobSpec(name="replica", cmd=list(cmd), kind="replica")
    job.python = env.capture_python(cwd)
    job.requirements, _notes = env.capture_requirements(cwd)
    if env_names:
        job.env = list(dict.fromkeys(env_names))
        job.env_values, _missing = capture_env(job.env)
    return job


def ensure_venv(worker: Worker, key: str, state: dict, job: JobSpec) -> str:
    """The Python env a run there would get; reused while the requirements hold."""
    ekey = env.env_key(job.python, job.requirements)
    meta = state.setdefault("meta", {})
    venv = meta.get("venv")
    if meta.get("env_key") == ekey and venv:
        if transport.run(worker, f'test -x "{venv}/bin/python"', timeout=20).returncode == 0:
            return venv
    env.ensure_uv_remote(worker)
    rhome = state["home"].rsplit("/.orchard", 1)[0]
    venv = env.sync_env_remote(worker, ekey, job.python, job.requirements, rhome)
    _meta(worker, key, {"env_key": ekey, "venv": venv})
    meta.update(env_key=ekey, venv=venv)
    return venv


def _meta(worker: Worker, key: str, fields: dict) -> None:
    try:
        proc = remote_call(worker, "replica-meta", key, json.dumps(fields), timeout=60)
    except (transport.RemoteError, subprocess.TimeoutExpired) as exc:
        raise ReplicaError(f"{worker.name} went quiet ({exc})") from exc
    if proc.returncode != 0:
        raise ReplicaError(f"{worker.name}: replica-meta failed ({proc.returncode}): {_stderr(proc)}")


def ensure_prep(worker: Worker, key: str, state: dict, reg: dict, manifest: dict, venv: str,
                env_values: dict | None = None) -> str | None:
    """Run the prep command when its key files changed since the body last ran it."""
    want = prep_digest(reg, manifest)
    meta = state.setdefault("meta", {})
    if want is None or meta.get("prep_digest") == want:
        return None
    print(f"dawn: {worker.name} is learning the project ({reg['prep']})")
    job = JobSpec(name="replica-prep", cmd=["/bin/sh", "-c", reg["prep"]], kind="replica",
                  env=sorted(env_values or {}), env_values=dict(env_values or {}))
    job_id = new_job_id(job.name)
    sdir = JOBS_DIR / job_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "spec.json").write_text(json.dumps({**job.__dict__, "env_values": {}}, indent=2))
    (sdir / "placement.json").write_text(json.dumps({"worker": worker.name, "kind": "replica-prep"}, indent=2) + "\n")
    rjob = _start(worker, key, state["home"], job, job_id, venv, receipts.digest_manifest(manifest))
    events.append("launched", job=job_id, worker=worker.name, kind=job.kind)
    status = engine.watch_remote(worker, rjob, lost_after=LOST_AFTER_S)
    if status is None:
        raise _Quiet(f"{worker.name} went quiet during prep; {job_id} may still finish there")
    for name in ("log.txt", "status.json"):
        with contextlib.suppress(transport.RemoteError, subprocess.TimeoutExpired):
            (sdir / name).write_bytes(transport.pull_bytes(worker, f"{rjob}/{name}"))
    _release(worker, key, job_id)
    events.append("finished", job=job_id, worker=worker.name, state=status.get("state"),
                  exit_code=status.get("exit_code"))
    if status.get("state") != "succeeded":
        raise ReplicaError(f"dawn: prep failed on {worker.name} ({status.get('state')}); log: {sdir / 'log.txt'}")
    _meta(worker, key, {"prep_digest": want, "prep_job": job_id})
    meta.update(prep_digest=want, prep_job=job_id)
    return job_id


def _start(worker: Worker, key: str, home: str, job: JobSpec, job_id: str, venv: str | None, digest: str) -> str:
    """Start job in the copy, if the copy still holds tree digest."""
    rrep = f"{home}/replicas/{key}"
    rjob = f"{home}/jobs/{job_id}"
    payload = agent_job_payload(job, job_id, rjob, f"{rrep}/work", venv, preempt=True)
    # Augment: whatever the project installs into tools/bin is on the job's PATH.
    payload["path_prepend"] = [f"{rrep}/tools/bin"]
    payload["env"]["ORCHARD_REPLICA_TOOLS"] = f"{rrep}/tools"
    transport.push_bytes(worker, f"{rjob}/job.json", json.dumps(payload, indent=2).encode())
    proc = remote_call(worker, "replica-begin", key, f"{rjob}/job.json", digest, timeout=60)
    answer = _parse(proc)
    if proc.returncode == 3:
        raise ReplicaBusy(f"{worker.name} is running {answer.get('busy')}")
    if proc.returncode == 4 and answer.get("error") == "moved":
        raise _Moved(f"{worker.name}'s copy changed before the job started", answer.get("drifted"))
    if proc.returncode == 4:
        raise ReplicaError(f"{worker.name} has no copy: orchard replica up -w {worker.name}")
    if proc.returncode != 0:
        raise ReplicaError(f"{worker.name}: replica-begin failed ({proc.returncode}): {_stderr(proc)}")
    return rjob


def _release(worker: Worker, key: str, job_id: str) -> None:
    """Free the copy once this Mac has what the job left in it. Best effort:
    a copy nobody releases frees itself a while after the job ends."""
    with contextlib.suppress(transport.RemoteError, subprocess.TimeoutExpired, OSError):
        remote_call(worker, "replica-release", key, job_id, timeout=60)


# --- succession --------------------------------------------------------------------

def _caps_for(state: dict, entry: dict, job: JobSpec) -> dict:
    """The body's caps plus the tags its replica was augmented with."""
    caps = dict(state.get("caps") or {})
    caps["tags"] = sorted(set(caps.get("tags") or []) | set(entry.get("augment") or []))
    caps["platform_ok"], caps["platform_note"] = env.resolve_for_worker(job.requirements, job.python, caps)
    return caps


def choose(key: str, reg: dict, job: JobSpec, worker_name: str | None, local: dict | None,
           sources: dict | None, pooled: bool = False) -> list[tuple[Worker, int]]:
    """Free bodies that can take job, nearest mind first: the one whose copy
    needs the fewest bytes to become this tree, then the most cores. A body
    you name runs it through an idle hold, unless check, compare or blame
    named it (pooled): those honour holds like any other job."""
    if worker_name and worker_name not in reg["replicas"]:
        raise ReplicaError(f"no replica on {worker_name}: orchard replica up -w {worker_name}")
    names = [worker_name] if worker_name else days_of(reg)
    workers, probes = _probe_all(names, key)
    cfg = dispatcher.load_config()
    local_digest = receipts.digest_manifest(local) if local is not None else None
    ranked = []
    for name in names:
        st = probes.get(name)
        if name not in workers:
            why = "not configured"
        elif st is None:
            why = "quiet: no answer"
        elif not st.get("exists"):
            why = f"no copy: orchard replica up -w {name}"
        elif st.get("busy") or st["caps"].get("busy_jobs"):
            why = f"busy with {st.get('busy') or st['caps']['busy_jobs'][0]}"
        else:
            caps = _caps_for(st, reg["replicas"][name], job)
            problems = placement.problems(job, caps)
            hold = dispatcher.idle_hold(caps, cfg)
            if problems:
                why = "; ".join(problems)
            elif hold and (pooled or not worker_name):
                why = hold
            else:
                if hold:
                    print(f"note: {name}: {hold} (running anyway: you named it)")
                est = 0 if local is None else ship_estimate(workers[name], key, local, sources, local_digest, st)
                ranked.append((est, -(caps.get("cores") or 0), name))
                continue
        print(f"succession: {name} cannot take it ({why})")
    return [(workers[name], est) for est, _cores, name in sorted(ranked)]


# --- running ---------------------------------------------------------------------------

def _launch(w: Worker, key: str, cwd: Path, job: JobSpec, reg: dict, local: dict | None, sources: dict | None,
            fields: dict, tree_override: bool) -> tuple[str, str, Path, str, dict]:
    """Under the body's lock: bring its copy to the tree, ready it, start the job.
    The start names the tree, so the job runs on exactly the bytes recorded."""
    st = replica_state(w, key)
    if st is None:
        raise _Quiet("quiet: no answer")
    busy = st.get("busy") or (st.get("caps") or {}).get("busy_jobs")
    if busy:
        raise ReplicaBusy(f"{w.name} is running {busy if isinstance(busy, str) else busy[0]}")
    for attempt in (1, 2):
        if local is None:
            tree, sent = _remote_manifest(w, key, st, force=attempt > 1)
            print(f"day: running on {w.name}'s copy as it is (tree {tree[:12]})")
        else:
            res = sync(w, key, dict((a, list(e)) for a, e in local.items()), sources, st)
            tree, sent = res["digest"], res["manifest"]
            if res["sent"] or res["deleted"]:
                print(f"day: sent {res['sent']} file(s) ({_fmt_bytes(res['bytes'])}), removed {res['deleted']} on {w.name}")
            else:
                print(f"day: {w.name} already holds this tree")
        venv = ensure_venv(w, key, st, job)
        prepped = None
        try:
            prepped = ensure_prep(w, key, st, reg, sent, venv, job.env_values)
            job_id = new_job_id(job.name)
            rjob = _start(w, key, st["home"], job, job_id, venv, tree)
            break
        except _Moved as exc:
            if attempt == 2:
                raise ReplicaError(f"{w.name}'s copy keeps changing before the job starts; run again") from exc
            shown = ", ".join(exc.drifted[:5]) or "files it did not list"
            if prepped:
                print(f"dawn: prep changed {len(exc.drifted)} tracked file(s) ({shown}); "
                      + ("reading the copy again" if local is None else "sending them back"))
            else:
                print(f"day: {w.name}'s copy changed before the job started ({shown}); "
                      + ("reading it again" if local is None else "bringing it back"))
            st = replica_state(w, key)
            if st is None:
                raise _Quiet("quiet: no answer") from exc
    sdir = JOBS_DIR / job_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "spec.json").write_text(json.dumps({**job.__dict__, "env_values": {}}, indent=2))
    role = reg["replicas"][w.name].get("role", "day")
    (sdir / "placement.json").write_text(json.dumps(
        {"worker": w.name, "kind": "replica", "reasons": [f"replica ({role})"]}, indent=2) + "\n")
    inputs = {"v": receipts.VERSION, "project": str(cwd), "worker": w.name, "recorded_at": time.time(),
              "tree": tree, "files": len(sent), "work": fields, "work_key": receipts.work_key(tree, fields)}
    if tree_override:
        inputs["overrides"] = True
    (sdir / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    events.append("submitted", job=job_id, worker=w.name, kind=job.kind)
    events.append("launched", job=job_id, worker=w.name, kind=job.kind)
    return job_id, rjob, sdir, tree, sent


def _hygiene(w: Worker, key: str, tree: str, sent: dict, sdir: Path, job_id: str) -> list[str]:
    """Tracked files the check itself changed in the copy. Only while the copy
    is still this job's: once another job holds it, its changes are not ours."""
    post = replica_state(w, key)
    if post is None or post["digest"] == tree or post.get("busy") != job_id:
        return []
    try:
        _digest, after = _remote_manifest(w, key, post)
    except ReplicaError:
        return []
    drifted = changed_arcs(sent, after)
    if drifted:
        shown = ", ".join(drifted[:5]) + (f" and {len(drifted) - 5} more" if len(drifted) > 5 else "")
        print(f"hygiene: the check changed {len(drifted)} tracked file(s) in the copy: {shown}")
        print("hygiene: the next run sends them back; a check that rewrites its own inputs may pass only once")
        (sdir / "drift.json").write_text(json.dumps({"tree": tree, "drifted": drifted}, indent=2) + "\n")
    return drifted


def _write_result(result_file: Path | None, result: dict) -> None:
    if result_file is not None:
        _save_json(Path(result_file), result)


def run(cmd: list[str], cwd: Path, worker_name: str | None = None, tags: list[str] | None = None,
        env_names: list[str] | None = None, timeout: int | None = None, reuse: bool = False,
        as_is: bool = False, bless: bool = False, no_local: bool = False,
        target_file: Path | None = None, result_file: Path | None = None, pooled: bool = False,
        sight: bool = False) -> int:
    key = project_key(cwd)
    reg = load_registry(key)
    if not reg["replicas"]:
        raise ReplicaError("no replicas yet: orchard replica up -w <worker>")
    if sight and (as_is or target_file is not None):
        raise ReplicaError("--sight looks at your tree on other bodies, so it cannot use --as-is")
    if as_is and not worker_name:
        raise ReplicaError("--as-is needs -w: it runs on one body's copy as that body holds it")
    if worker_name and worker_name == dusk_of(reg) and not as_is:
        raise ReplicaError("the dusk moves only on green: use --as-is, or run on a day replica")
    job = fleet.job_like_run(cmd, cwd, tags, env_names, timeout)
    job.kind = "replica"
    fields = {**receipts.work_fields(job), "prep": reg.get("prep")}
    local = sources = None
    if not as_is:
        local, sources = local_manifest(cwd, key)
        if target_file is not None:
            overrides = _load_json(Path(target_file), {}).get("overrides")
            if not isinstance(overrides, dict):
                raise ReplicaError(f"{target_file}: no overrides")
            apply_overrides(local, sources, overrides)
        tree = receipts.digest_manifest(local)
        if reuse:
            hit = receipts.find_reusable(receipts.work_key(tree, fields))
            if hit:
                print(f"reuse: {hit['job']} already passed this check on this tree (sealed receipt)")
                _write_result(result_file, {"target": hit["worker"]["name"], "job": hit["job"], "state": "succeeded",
                                            "exit_code": hit.get("exit_code"), "reused": True, "tree": tree})
                return 0

    bodies = choose(key, reg, job, worker_name, local, sources, pooled)
    locked_out: list[Worker] = []
    for blocking, queue in ((False, [w for w, _est in bodies]), (True, locked_out)):
        for w in list(queue):
            try:
                with worker_lock(key, w.name, blocking) as got:
                    if not got:
                        locked_out.append(w)
                        continue
                    launched = _launch(w, key, cwd, job, reg, local, sources, fields, target_file is not None)
            except (ReplicaBusy, _Quiet) as exc:
                print(f"succession: {w.name} cannot take it ({exc})")
                continue
            except (transport.RemoteError, subprocess.TimeoutExpired) as exc:
                print(f"succession: {w.name} went quiet ({exc})")
                continue
            rc, result = _finish(w, key, cwd, job, reg, launched, bless and not as_is and target_file is None,
                                 result_file)
            if sight and result and result.get("state") == "failed":
                return _sight_run(cwd, key, reg, cmd, result, tags, env_names, timeout, no_local,
                                  bool(worker_name) and not pooled, result_file)
            return rc

    if target_file is not None or no_local or as_is:
        raise ReplicaError("no replica is free to take it")
    print("succession: no replica is free; running here")
    result = fleet.run_on("local", cmd, cwd, fleet.job_name("replica", fleet.new_token(), "local"),
                          tags, env_names, timeout)
    print(f"result: {result['state']} on this Mac ({result.get('job') or result.get('why')})")
    first = {**result, "replica": None, "role": None, "drifted": [], "tree": tree}
    _write_result(result_file, first)
    ok = result["state"] == "succeeded"
    if bless and ok:
        advance_dusk(cwd, key, tree, f"{result.get('job')} on this Mac", job.env)
    if sight and result["state"] == "failed":
        return _sight_run(cwd, key, reg, cmd, first, tags, env_names, timeout, no_local, False, result_file)
    return 0 if ok else 1


def _finish(w: Worker, key: str, cwd: Path, job: JobSpec, reg: dict, launched: tuple, bless: bool,
            result_file: Path | None) -> tuple[int, dict | None]:
    """Watch the launched job to its end: (exit code, the result written)."""
    job_id, rjob, sdir, tree, sent = launched
    print(f"job: {job_id} on {w.name} (replica)")
    try:
        status = engine.watch_remote(w, rjob, lost_after=LOST_AFTER_S)
    except KeyboardInterrupt:
        print(f"detached: {job_id} keeps running on {w.name}")
        return 130, None
    if status is None:
        # Succession: a pool hands the work to another body; a lone run reports it.
        why = f"{w.name} went quiet mid-run; {job_id} may still finish there"
        print(f"succession: {why}")
        events.append("finished", job=job_id, worker=w.name, state="unavailable", exit_code=None)
        result = {"state": "unavailable", "why": why, "job": job_id, "target": w.name, "replica": w.name,
                  "tree": tree}
        _write_result(result_file, result)
        return 1, result
    status = engine.collect_remote(w, job_id, rjob, sdir, cwd) or status
    drifted = _hygiene(w, key, tree, sent, sdir, job_id)
    _release(w, key, job_id)
    out_dir = cwd / ".orchard-out" / job_id / "out"
    receipts.seal(sdir, status, w.name, out_dir if out_dir.is_dir() else None)
    print(f"result: {status.get('state')} in {status.get('elapsed_s')}s (exit {status.get('exit_code')})")
    events.append("finished", job=job_id, worker=w.name, state=status.get("state"), exit_code=status.get("exit_code"))
    role = reg["replicas"].get(w.name, {}).get("role", "day")
    result = {**fleet.result_of(sdir, w.name), "replica": w.name, "role": role, "drifted": drifted, "tree": tree}
    _write_result(result_file, result)
    ok = status.get("state") == "succeeded"
    if bless and ok:
        advance_dusk(cwd, key, tree, f"{job_id} on {w.name}", job.env)
    return (0 if ok else 1), result


# --- Dusk ----------------------------------------------------------------------------------

def advance_dusk(cwd: Path, key: str, tree: str, source: str, env_names: list[str] | None = None) -> bool:
    """Move the dusk to tree, which just passed: it now remembers the last green tree."""
    reg = load_registry(key)
    dusk = dusk_of(reg)
    if dusk is None:
        return False
    local, sources = local_manifest(cwd, key)
    if receipts.digest_manifest(local) != tree:
        print("dusk: not advanced (the tree changed while the checks ran)")
        return False
    try:
        w = worker_named(dusk)
        with worker_lock(key, dusk) as _held:
            st = replica_state(w, key)
            if st is None:
                raise ReplicaError("quiet: no answer")
            # Trust the copy, not the registry: a run could have rewritten it.
            if reg.get("dusk_tree") == tree:
                if st.get("exists") and st.get("digest") == tree:
                    print(f"dusk: {dusk} already remembers tree {tree[:12]}")
                    return True
                print(f"dusk: {dusk}'s copy drifted from tree {tree[:12]}; restoring it")
            if sync(w, key, local, sources, st)["digest"] != tree:
                print("dusk: not advanced (the tree changed while syncing)")
                return False
            job = _project_job(cwd, ["true"], env_names)
            venv = ensure_venv(w, key, st, job)
            ensure_prep(w, key, st, reg, local, venv, job.env_values)
    except (ReplicaError, transport.RemoteError, subprocess.TimeoutExpired) as exc:
        print(f"dusk: not advanced ({exc})")
        return False
    with edit_registry(key) as fresh:
        fresh["dusk_tree"] = tree
    print(f"dusk: {dusk} now remembers tree {tree[:12]} (last green: {source})")
    return True


# --- many checks across bodies ------------------------------------------------------------------

def _child_argv(worker: str | None, cmd: list[str], tags: list[str], env_names: list[str], timeout: int | None,
                result_file: Path, extra: list[str] | None = None) -> list[str]:
    argv = [sys.executable, "-m", "orchard", "replica", "run", "--result-file", str(result_file), *(extra or [])]
    if worker:
        argv += ["-w", worker]
    if timeout:
        argv += ["--timeout", str(timeout)]
    for tag in tags:
        argv += ["-t", tag]
    for name in env_names:
        argv += ["--env", name]
    return argv + ["--", *cmd]


def _run_child(cwd: Path, label: str, argv: list[str], result_file: Path) -> dict:
    fleet.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = fleet.LOG_DIR / f"{slugify(label)}.log"
    with open(log, "wb") as out:
        proc = subprocess.run(argv, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=fleet.child_env())
    result = _load_json(result_file, None)
    if isinstance(result, dict) and result.get("state"):
        result["log"] = str(log)
        return result
    lines = log.read_text(errors="replace").strip().splitlines()
    return {"state": "unavailable", "why": lines[-1] if lines else f"exited {proc.returncode}", "log": str(log)}


def _run_pool(tasks: list[dict], bodies: list[str], launch) -> tuple[dict, list[dict]]:
    """One thread per body, each taking the next task it can run. A body that
    cannot answer stops taking work and its task goes back for the others. A
    body with nothing it can run waits while any task is still out, since a
    task can come back. Returns ({task id: result}, tasks nobody could run)."""
    pending = [t for t in tasks if t["bodies"]]
    stranded = [t for t in tasks if not t["bodies"]]
    results: dict = {}
    cond = threading.Condition()
    in_flight = 0

    def body_loop(body: str) -> None:
        nonlocal in_flight
        while True:
            with cond:
                while True:
                    task = next((t for t in pending if body in t["bodies"]), None)
                    if task is not None or in_flight == 0:
                        break
                    cond.wait()
                if task is None:
                    return
                pending.remove(task)
                in_flight += 1
            try:
                result = launch(task, body)
            except Exception as exc:  # a crash in one launch must not hang the others
                result = {"state": "unavailable", "why": f"{type(exc).__name__}: {exc}"}
            with cond:
                in_flight -= 1
                cond.notify_all()
                if result.get("state") == "unavailable":
                    task["bodies"].discard(body)
                    task.setdefault("why", []).append(f"{body}: {result.get('why')}")
                    pending.append(task)
                    return
                results[task["id"]] = result

    threads = [threading.Thread(target=body_loop, args=(b,), daemon=True) for b in bodies]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, stranded + pending


def _usable_days(key: str, reg: dict) -> tuple[list[str], dict]:
    """Day bodies with a copy and no idle hold (their owner is not using them)."""
    days = days_of(reg)
    _workers, probes = _probe_all(days, key)
    cfg = dispatcher.load_config()
    usable = []
    for d in days:
        st = probes.get(d)
        if not st or not st.get("exists"):
            continue
        hold = dispatcher.idle_hold(st.get("caps") or {}, cfg)
        if hold:
            print(f"succession: {d} is held ({hold})")
            continue
        usable.append(d)
    return usable, probes


def _tree_of(result: dict) -> str | None:
    """The tree a result ran on; a run on this Mac records it in its job inputs."""
    tree = result.get("tree")
    if tree is None and result.get("job"):
        tree = _load_json(JOBS_DIR / result["job"] / "inputs.json", {}).get("tree")
    return tree


def check(cwd: Path, split: str | None = None, shards: int = 0, no_local: bool = False,
          env_names: list[str] | None = None, sight: bool = False) -> int:
    """Every check at once, spread across the day bodies (each check borrows a
    body that can run it). When all pass on one tree, the dusk moves to it.
    With sight, each failure is looked at again (shared_sight), one at a time."""
    key = project_key(cwd)
    reg = load_registry(key)
    if split:
        if shards < 1:
            raise ReplicaError("--split needs --shards N")
        checks = [speculate.make_check(shlex.split(split.replace("{i}", str(i)).replace("{n}", str(shards))),
                                       [], env_names or [], None) for i in range(1, shards + 1)]
    else:
        checks = speculate.load(cwd)
    if not checks:
        raise ReplicaError("no checks: orchard speculate add -- <cmd>, or --split TEMPLATE --shards N")
    local, _sources = local_manifest(cwd, key)
    start = receipts.digest_manifest(local)
    days, probes = _usable_days(key, reg)
    tasks = []
    for i, c in enumerate(checks):
        job = fleet.job_like_run(c["cmd"], cwd, c["tags"], c["env"], c.get("timeout"))
        job.kind = "replica"
        can = {d for d in days if not placement.problems(job, _caps_for(probes[d], reg["replicas"][d], job))}
        if c["tags"] and can and len(can) < len(days):
            print(f"borrow: {' '.join(c['cmd'])} needs {', '.join(c['tags'])}: only {', '.join(sorted(can))} can run it")
        tasks.append({"id": i, "check": c, "bodies": can, "can": set(can)})  # the pool empties bodies
    token = fleet.new_token()
    scratch = local_dir(key) / "check" / token
    scratch.mkdir(parents=True, exist_ok=True)

    def launch(task: dict, body: str) -> dict:
        c = task["check"]
        rf = scratch / f"{task['id']}-{slugify(body)}.json"
        rf.unlink(missing_ok=True)
        result = _run_child(cwd, f"replica-check-{token}-{task['id']}-{body}",
                            _child_argv(body, c["cmd"], c["tags"], c["env"], c.get("timeout"), rf,
                                        ["--no-local", "--pooled"]), rf)
        print(f"check: {result.get('state'):10} {' '.join(c['cmd'])} on {body} ({result.get('job') or result.get('why')})")
        return result

    results, leftovers = _run_pool(tasks, days, launch)
    for task in leftovers:
        c = task["check"]
        if no_local:
            results[task["id"]] = {"state": "unplaceable", "target": None,
                                   "why": "; ".join(task.get("why") or ["no replica can run it"])}
            continue
        result = fleet.run_on("local", c["cmd"], cwd, fleet.job_name("replica", token, "local", task["id"]),
                              c["tags"], c["env"], c.get("timeout"))
        print(f"check: {result['state']:10} {' '.join(c['cmd'])} on this Mac ({result.get('job') or result.get('why')})")
        results[task["id"]] = result

    if sight:
        for task in tasks:
            r, c = results[task["id"]], task["check"]
            if r.get("state") == "failed":
                first = {**r, "tree": _tree_of(r)}
                r["sight"] = shared_sight(cwd, key, reg, c["cmd"], first, c["tags"], c["env"], c.get("timeout"),
                                          no_local, named=False, can=task["can"])
    print("checks:")
    for task in tasks:
        r = results[task["id"]]
        where = r.get("replica") or r.get("target") or "-"
        seen = f"  sight: {r['sight']['word']}" if r.get("sight") else ""
        print(f"  {r['state']:10} {' '.join(task['check']['cmd'])}  ({where}){seen}")
    bad = [results[t["id"]] for t in tasks if results[t["id"]]["state"] != "succeeded"]
    if not bad:
        if all(_tree_of(results[t["id"]]) == start for t in tasks):
            advance_dusk(cwd, key, start, f"{len(tasks)} check(s) green", env_names)
        else:
            print("dusk: not advanced (the checks did not all run on the same tree)")
        return 0
    if sight and all((r.get("sight") or {}).get("word") in ("body", "chance") for r in bad):
        return 2
    return 1


# --- shared sight -------------------------------------------------------------------------------

SIGHT_EXIT = {"code": 1, "body": 2, "chance": 2, "unsure": 1}
_RUN_FIELDS = ("target", "job", "state", "machine", "os_family", "load_per_core")


def sight_verdict(first: dict, same: dict | None, other: dict | None) -> dict:
    """What a failure on body a was, from a rerun on a (same) and a run on a
    second body (other), both on the same tree: code (fix it), body (a is the
    problem), chance (flaky) or unsure."""
    a = first.get("target") or "?"
    b = (other or {}).get("target") or "?"
    same_ok = bool(same) and consensus.usable(same)
    if same_ok and consensus.passed(same):
        return {"word": "chance", "label": "flaky", "why": f"failed then passed on {a}: flaky"}
    if not (other and consensus.usable(other)):
        if other is None:
            why = f"failed twice on {a}; no second body could look" if same_ok else "no second body could look"
        else:
            why = f"failed twice on {a}; the look on {b} did not finish" if same_ok else "neither look finished"
        return {"word": "unsure", "label": None, "why": why}
    if not consensus.passed(other):
        why = (f"fails twice on {a} and on {b}: fix the code" if same_ok
               else f"fails on {a} and on {b}: fix the code (the rerun on {a} did not finish)")
        return {"word": "code", "label": "real", "why": why}
    if not same_ok:
        return {"word": "unsure", "label": None,
                "why": f"passed on {b}, but the rerun on {a} did not finish: chance is not ruled out"}
    label, reason = consensus.classify(first, [same, other])
    return {"word": "body", "label": label, "why": f"{reason}: {a} is the problem, not your change"}


def _second_bodies(key: str, reg: dict, a: str, job: JobSpec, no_local: bool, can: set[str] | None) -> list[str]:
    """Who can take the second look, best first: free day bodies that can run
    the job (in check, only those the check could borrow), then this Mac."""
    days, probes = _usable_days(key, reg)
    out = []
    for d in days:
        st = probes[d]
        if d == a or (can is not None and d not in can):
            continue
        if st.get("busy") or (st.get("caps") or {}).get("busy_jobs"):
            print(f"sight: {d} is busy")
        elif not placement.problems(job, _caps_for(st, reg["replicas"][d], job)):
            out.append(d)
    if not no_local and a != "local":
        out.append("local")
    return out


def shared_sight(cwd: Path, key: str, reg: dict, cmd: list[str], first: dict, tags: list[str] | None = None,
                 env_names: list[str] | None = None, timeout: int | None = None, no_local: bool = False,
                 named: bool = False, can: set[str] | None = None) -> dict:
    """A failure on one body, looked at again by two: the same body (does it
    pass now? chance) and a second body (does it pass there? the body). Looks
    on two replicas run at once; a look on this Mac runs last, because a run
    here works in place and its writes would change the tree a remote look
    ships. A body that cannot start its look passes it to the next. A look
    that ran on another tree does not count. Remembers the verdict with the
    failure's signature, as consensus does."""
    a = first.get("target") or "local"
    sig = consensus.signature(first.get("log_tail") or "")
    if consensus.known_noise(sig):
        print("sight: this failure was called noise before; looking anyway")
    tags, env_names = list(tags or []), list(env_names or [])
    job = fleet.job_like_run(cmd, cwd, tags, env_names, timeout)
    job.kind = "replica"
    seconds = _second_bodies(key, reg, a, job, no_local, can)
    b = seconds[0] if seconds else None
    if b is None:
        plan = f"on {a} (no second body)"
    elif "local" in (a, b):
        plan = f"on {b}, then on {a}" if a == "local" else f"on {a}, then on {b}"
    else:
        plan = f"on {a} and on {b} at once"
    print(f"sight: failed on {a}; looking again {plan}")
    token = fleet.new_token()
    scratch = local_dir(key) / "sight" / token
    scratch.mkdir(parents=True, exist_ok=True)

    def look(body: str, i: int, extra: list[str]) -> dict:
        try:
            if body == "local":
                result = fleet.run_on("local", cmd, cwd, fleet.job_name("sight", token, "local", i),
                                      tags, env_names, timeout)
                result["tree"] = _tree_of(result)  # recorded at launch, before the run wrote anything
            else:
                rf = scratch / f"{i}-{slugify(body)}.json"
                result = _run_child(cwd, f"replica-sight-{token}-{i}-{body}",
                                    _child_argv(body, cmd, tags, env_names, timeout, rf, extra), rf)
        except Exception as exc:  # one look crashing must not lose the other
            return {"target": body, "state": "unavailable", "why": f"{type(exc).__name__}: {exc}"}
        result.setdefault("target", body)
        if consensus.usable(result) and result.get("tree") != first.get("tree"):
            result = {**result, "state": "unavailable", "why": "the tree changed since the first run"}
        return result

    def second(bodies: list[str], r: dict | None = None) -> dict | None:
        for body in bodies:
            if r is not None:
                if r["state"] != "unavailable" or r.get("job"):
                    break
                print(f"sight: {r['target']} could not look ({r.get('why')}); trying {body}")
            r = look(body, seconds.index(body) + 1, ["--no-local", "--pooled"])
        return r

    # A body you named looks again through its idle hold, as the first run did.
    same_extra = ["--no-local"] + ([] if named else ["--pooled"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        same_f = pool.submit(look, a, 0, same_extra) if a != "local" else None
        other_f = pool.submit(second, [d for d in seconds if d != "local"])
        same, other = (same_f.result() if same_f else None), other_f.result()
    if "local" in seconds:
        other = second(["local"], other)
    if a == "local":
        same = look("local", 0, [])
    for word, r in (("again", same), ("second", other)):
        if r is not None:
            detail = r.get("why") if r["state"] == "unavailable" else None
            print(f"sight: {r['target']} {word}: {r['state']} ({detail or r.get('job') or r.get('why')})")

    verdict = sight_verdict(first, same, other)
    runs = [{k: r.get(k) for k in _RUN_FIELDS} for r in (first, same, other) if r is not None]
    verdict.update(signature=sig, signature_id=consensus.signature_id(sig), runs=runs)
    print(f"sight: {verdict['word']} ({verdict['why']})")
    if sig:
        print(f"sight: error: {sig}")
    note = consensus.describe_memory(consensus.recall(sig))
    if note:
        print(f"sight: {note}")
    if verdict["label"]:
        consensus.remember({"label": verdict["label"], "reason": verdict["why"], "signature": sig,
                            "signature_id": verdict["signature_id"], "cmd": list(cmd), "runs": runs, "via": "sight"})
    return verdict


def _sight_run(cwd: Path, key: str, reg: dict, cmd: list[str], first: dict, tags: list[str] | None,
               env_names: list[str] | None, timeout: int | None, no_local: bool, named: bool,
               result_file: Path | None) -> int:
    verdict = shared_sight(cwd, key, reg, cmd, first, tags, env_names, timeout, no_local, named)
    _write_result(result_file, {**first, "sight": verdict})
    return SIGHT_EXIT[verdict["word"]]


# --- asking the dusk ------------------------------------------------------------------------------

def compare(cwd: Path, cmd: list[str], tags: list[str] | None = None, env_names: list[str] | None = None,
            timeout: int | None = None) -> int:
    """Run cmd on your tree and on the dusk's last green tree at once."""
    key = project_key(cwd)
    reg = load_registry(key)
    dusk = dusk_of(reg)
    if dusk is None:
        raise ReplicaError("no dusk replica: orchard replica role <worker> dusk")
    if reg.get("dusk_tree") is None:
        print("compare: warning: the dusk was never blessed; it holds whatever tree it was given")
    token = fleet.new_token()
    scratch = local_dir(key) / "compare" / token
    scratch.mkdir(parents=True, exist_ok=True)
    lanes = {
        "day": (None, ["--pooled"]),
        "dusk": (dusk, ["--as-is", "--no-local", "--pooled"]),
    }
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            lane: ex.submit(_run_child, cwd, f"replica-compare-{token}-{lane}",
                            _child_argv(w, cmd, tags or [], env_names or [], timeout, scratch / f"{lane}.json", extra),
                            scratch / f"{lane}.json")
            for lane, (w, extra) in lanes.items()
        }
    day, dusk_r = futures["day"].result(), futures["dusk"].result()
    for lane, r in (("now", day), ("dusk", dusk_r)):
        print(f"  {lane:5} {r['state']:12} {r.get('replica') or r.get('target') or '-'} "
              f"({r.get('job') or r.get('why')})")
    if dusk_r.get("drifted"):
        print(f"compare: warning: the run changed {len(dusk_r['drifted'])} tracked file(s) in the dusk's copy; "
              "the next green check (or --bless) restores it")
    if dusk_r.get("tree") and reg.get("dusk_tree") and dusk_r["tree"] != reg["dusk_tree"]:
        print(f"compare: warning: the dusk's copy ({dusk_r['tree'][:12]}) is not the tree it was blessed at "
              f"({reg['dusk_tree'][:12]})")
        print("compare: no verdict while the dusk is away from its blessed tree; "
              "the next green check (or --bless) restores it")
        return 2
    d, k = day["state"], dusk_r["state"]
    if d == "succeeded":
        print("compare: green now" + ("; it was failing at the dusk tree, so your change fixed it" if k == "failed" else ""))
    elif d == "failed" and k == "succeeded":
        print("compare: your change broke it (it passes at the dusk tree)")
        print(f"compare: find the files: orchard replica blame -- {shlex.join(cmd)}")
    elif d == "failed" and k == "failed":
        print("compare: it fails at the dusk tree too, so it is not your change")
    elif d == "failed":
        print(f"compare: failing now; the dusk could not answer ({dusk_r.get('why') or k})")
    else:
        why = day.get("why")
        print(f"compare: the run did not finish ({d}" + (f": {why}" if why else "") + ")")
    return 0 if d == "succeeded" else 1


def _fetch_dusk_versions(w: Worker, key: str, arcs: list[str], manifest: dict, store: Path) -> None:
    """Copy the dusk's version of each file in arcs into store/<sha>, checked by sha."""
    store.mkdir(parents=True, exist_ok=True)
    need = [a for a in arcs if not (store / manifest[a][2]).exists()]
    if not need:
        return
    fd_in, arcs_path = tempfile.mkstemp(suffix=".json")
    fd_out, tgz_path = tempfile.mkstemp(suffix=".tgz")
    os.close(fd_out)
    try:
        with os.fdopen(fd_in, "w") as fh:
            json.dump(need, fh)
        proc = remote_call(w, "replica-pack", key, stdin_file=Path(arcs_path), stdout_file=Path(tgz_path), timeout=900)
        if proc.returncode != 0:
            raise ReplicaError(f"{w.name}: replica-pack failed ({proc.returncode}): {_stderr(proc)}")
        wanted = set(need)
        with tarfile.open(tgz_path, "r:gz") as tar:
            for member in tar:
                if member.name not in wanted or not member.isfile():
                    continue
                sha = manifest[member.name][2]
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != sha:
                    raise ReplicaError(f"{member.name} changed on the dusk while blame read it; run again")
                dest = store / sha
                dest.write_bytes(data)
                dest.chmod(0o755 if manifest[member.name][1] == "x" else 0o644)
                wanted.discard(member.name)
        if wanted:
            raise ReplicaError(f"the dusk did not send {sorted(wanted)[0]}")
    finally:
        for path in (arcs_path, tgz_path):
            with contextlib.suppress(OSError):
                os.unlink(path)


def ddmin(changed: list[str], test) -> list[str]:
    """Zeller's delta debugging over the changed files, a round at a time:
    test(subsets) runs every subset of one round at once and returns which
    fail. Returns a 1-minimal set of changes that still makes the check fail."""
    c = list(changed)
    n = 2
    while len(c) >= 2:
        size = len(c) / n
        chunks = [c[round(i * size):round((i + 1) * size)] for i in range(n)]
        subsets = [frozenset(ch) for ch in chunks if ch]
        complements = [frozenset(c) - s for s in subsets] if n > 2 else []
        verdicts = test(subsets + complements)
        failing = next((s for s, bad in zip(subsets, verdicts) if bad), None)
        if failing is not None:
            c, n = [a for a in c if a in failing], 2
            continue
        failing = next((s for s, bad in zip(complements, verdicts[len(subsets):]) if bad), None)
        if failing is not None:
            c, n = [a for a in c if a in failing], max(n - 1, 2)
            continue
        if n >= len(c):
            break
        n = min(len(c), 2 * n)
    return c


def blame(cwd: Path, cmd: list[str], tags: list[str] | None = None, env_names: list[str] | None = None,
          timeout: int | None = None) -> int:
    """Find the smallest set of your changed files that makes cmd fail, by
    running mixed trees (your files over the dusk's) across every day body."""
    key = project_key(cwd)
    reg = load_registry(key)
    dusk = dusk_of(reg)
    if dusk is None:
        raise ReplicaError("no dusk replica: orchard replica role <worker> dusk")
    w = worker_named(dusk)
    st = replica_state(w, key)
    if st is None or not st.get("exists"):
        raise ReplicaError(f"the dusk ({dusk}) did not answer or has no copy")
    dusk_tree, dmanifest = _remote_manifest(w, key, st)
    if reg.get("dusk_tree") is None:
        print("blame: warning: the dusk was never blessed; blaming against whatever tree it holds")
    elif dusk_tree != reg["dusk_tree"]:
        raise ReplicaError(f"the dusk's copy ({dusk_tree[:12]}) is not the tree it was blessed at "
                           f"({reg['dusk_tree'][:12]}); the next green check (or --bless) restores it")
    days, _probes = _usable_days(key, reg)
    if not days:
        raise ReplicaError("blame needs at least one day replica besides the dusk")
    local, _sources = local_manifest(cwd, key)
    changed = changed_arcs(local, dmanifest)
    if not changed:
        print("blame: your tree is the dusk's tree; nothing to blame")
        return 0
    token = fleet.new_token()
    work = local_dir(key) / "blame" / token
    store = local_dir(key) / "blame" / "store"
    _fetch_dusk_versions(w, key, [a for a in changed if a in dmanifest and dmanifest[a][0] == "F"], dmanifest, store)
    print(f"blame: {len(changed)} file(s) differ from the dusk's tree; testing on {', '.join(days)}")

    def dusk_version(arc: str):
        e = dmanifest.get(arc)
        if e is None:
            return None
        return ["L", e[1]] if e[0] == "L" else ["F", e[1], e[2], str(store / e[2])]

    memo: dict[frozenset, dict] = {}
    counter = iter(range(1 << 30))

    def run_trees(subsets: list[frozenset]) -> None:
        todo = [s for s in dict.fromkeys(subsets) if s not in memo]
        tasks = []
        for s in todo:
            i = next(counter)
            target = work / f"{i}.target.json"
            _save_json(target, {"overrides": {a: dusk_version(a) for a in changed if a not in s}})
            tasks.append({"id": i, "subset": s, "target": target, "bodies": set(days)})

        def launch(task: dict, body: str) -> dict:
            rf = work / f"{task['id']}-{slugify(body)}.result.json"
            rf.unlink(missing_ok=True)
            return _run_child(cwd, f"replica-blame-{token}-{task['id']}-{body}",
                              _child_argv(body, cmd, tags or [], env_names or [], timeout, rf,
                                          ["--target-file", str(task["target"]), "--no-local", "--pooled"]), rf)

        results, stranded = _run_pool(tasks, days, launch)
        if stranded:
            raise ReplicaError("blame: no body could run a tree: " + "; ".join(stranded[0].get("why") or []))
        for task in tasks:
            r = results[task["id"]]
            if r["state"] not in ("succeeded", "failed"):
                # A timeout or a lost body says nothing about the files: no verdict.
                raise ReplicaError(f"blame: a tree ended {r['state']} ({r.get('job') or r.get('why')}); no verdict")
            memo[task["subset"]] = r

    def test(subsets: list[frozenset]) -> list[bool]:
        run_trees(subsets)
        return [memo[s]["state"] == "failed" for s in subsets]

    everything, nothing = frozenset(changed), frozenset()
    bad_all, bad_none = test([everything, nothing])
    if not bad_all:
        print("blame: the check passes on your tree; nothing to blame")
        return 0
    if bad_none:
        print("blame: the check fails on the dusk's tree too; blame needs a tree where it passes")
        return 1
    culprits = ddmin(changed, test) if len(changed) > 1 else list(changed)
    print(f"blame: the failure needs these {len(culprits)} file(s) of yours together:")
    for arc in culprits:
        print(f"  {arc}")
    used = {r.get("replica") for r in memo.values() if r.get("replica")}
    print(f"blame: tested {len(memo)} trees on {len(used)} bod{'y' if len(used) == 1 else 'ies'}")
    return 1


# --- lifecycle ------------------------------------------------------------------------------------

def _set_role(reg: dict, name: str, role: str) -> None:
    if role not in ROLES:
        raise ReplicaError(f"role must be one of {', '.join(ROLES)}")
    entry = reg["replicas"][name]
    if role == "dusk":
        for other, r in reg["replicas"].items():
            if other != name and r.get("role") == "dusk":
                r["role"] = "day"
                print(f"replica: {other} is a day replica now")
        if entry.get("role") != "dusk":
            reg["dusk_tree"] = None
    elif entry.get("role") == "dusk":
        reg["dusk_tree"] = None
    entry["role"] = role


def up(cwd: Path, worker_name: str, role: str | None = None, prep: str | None = None,
       prep_key: list[str] | None = None, augment: list[str] | None = None,
       env_names: list[str] | None = None) -> int:
    """Create or refresh a replica on a body (Dawn)."""
    w = worker_named(worker_name)
    key = project_key(cwd)
    with edit_registry(key) as reg:
        reg["project"] = str(cwd)
        if prep is not None:
            reg["prep"] = prep or None
        if prep_key:
            reg["prep_keys"] = sorted(set(prep_key))
        entry = reg["replicas"].setdefault(w.name, {"role": "day", "augment": [], "created_at": time.time()})
        if augment:
            entry["augment"] = sorted(set(entry.get("augment") or []) | set(augment))
        if role:
            _set_role(reg, w.name, role)
        snapshot = json.loads(json.dumps(reg))
    is_blessed_dusk = snapshot["replicas"][w.name]["role"] == "dusk" and snapshot.get("dusk_tree")
    try:
        _up(cwd, w, key, snapshot, is_blessed_dusk, env_names)
    except (transport.RemoteError, subprocess.TimeoutExpired) as exc:
        raise ReplicaError(f"{w.name} went quiet ({exc}); run up again") from exc
    role_now = snapshot["replicas"][w.name]["role"]
    tags = snapshot["replicas"][w.name].get("augment") or []
    print(f"replica: {w.name} is up as a {role_now} replica" + (f", augmented with {', '.join(tags)}" if tags else ""))
    return 0


def _up(cwd: Path, w: Worker, key: str, snapshot: dict, is_blessed_dusk, env_names: list[str] | None) -> None:
    with worker_lock(key, w.name):
        st = replica_state(w, key)
        if st is None:
            raise ReplicaError(f"{w.name} did not answer")
        if is_blessed_dusk and st.get("exists"):
            # The dusk moves only on green: up refreshes its env and prep, not its tree.
            tree, manifest = _remote_manifest(w, key, st)
            print(f"dusk: {w.name} keeps tree {tree[:12]}; it moves only when checks pass (or --bless)")
        else:
            local, sources = local_manifest(cwd, key)
            res = sync(w, key, local, sources, st)
            tree, manifest = res["digest"], res["manifest"]
            print(f"replica: {w.name} holds tree {tree[:12]} ({res['sent']} file(s) sent, {_fmt_bytes(res['bytes'])})")
        _meta(w, key, {"project": cwd.name})
        job = _project_job(cwd, ["true"], env_names)
        venv = ensure_venv(w, key, st, job)
        ensure_prep(w, key, st, snapshot, manifest, venv, job.env_values)


def set_role(cwd: Path, name: str, role: str) -> None:
    key = project_key(cwd)
    with edit_registry(key) as reg:
        if name not in reg["replicas"]:
            raise ReplicaError(f"no replica on {name}")
        _set_role(reg, name, role)
    if role == "dusk":
        print(f"replica: {name} is the dusk; it remembers its current tree until checks pass (or --bless)")
    else:
        print(f"replica: {name} is a day replica")


def list_rows(cwd: Path) -> list[dict]:
    key = project_key(cwd)
    reg = load_registry(key)
    names = list(reg["replicas"])
    _workers, probes = _probe_all(names, key)
    local, _sources = local_manifest(cwd, key)
    here = receipts.digest_manifest(local)
    want_prep = prep_digest(reg, local)
    rows = []
    for name in names:
        entry = reg["replicas"][name]
        st = probes.get(name)
        row = {"name": name, "role": entry.get("role", "day"), "augment": entry.get("augment") or [],
               "status": "quiet" if st is None else ("up" if st.get("exists") else "missing")}
        if st and st.get("exists"):
            meta = st.get("meta") or {}
            row.update(digest=st["digest"], files=st.get("files"), busy=st.get("busy"),
                       drifted=st.get("drifted") or [], last_used=meta.get("last_used"),
                       current=st["digest"] == here,
                       dawn_pending=bool(want_prep) and meta.get("prep_digest") != want_prep)
            if row["role"] == "dusk":
                row["blessed"] = reg.get("dusk_tree") == st["digest"]
        rows.append(row)
    return rows


def down(cwd: Path, name: str) -> None:
    key = project_key(cwd)
    reg = load_registry(key)
    if name not in reg["replicas"]:
        raise ReplicaError(f"no replica on {name}")
    workers = {w.name: w for w in load_workers()}
    if name in workers:
        try:
            with worker_lock(key, name):
                proc = remote_call(workers[name], "replica-down", key, timeout=120)
        except (transport.RemoteError, subprocess.TimeoutExpired) as exc:
            raise ReplicaError(f"{name} did not answer ({exc}); its copy is still there") from exc
        if proc.returncode == 3:
            raise ReplicaBusy(f"{name} is running {_parse(proc).get('busy')}; try again when it ends")
        if proc.returncode != 0:
            raise ReplicaError(f"{name}: replica-down failed ({proc.returncode}): "
                               f"{_parse(proc).get('error') or _stderr(proc)}; it stays listed")
    else:
        print(f"note: {name} is not a configured worker; its copy (if any) goes when its agent's gc runs")
    with edit_registry(key) as fresh:
        entry = fresh["replicas"].pop(name, None) or {}
        if entry.get("role") == "dusk":
            fresh["dusk_tree"] = None
    _cache_path(key, name).unlink(missing_ok=True)
    print(f"replica: {name} is down")
