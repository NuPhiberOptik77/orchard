"""Put llama.cpp's engine or helper on a machine in one step, with an undo.

`orchard ai install` first says what it would fetch: the file, where from, its
size and the folder it goes in. Nothing is downloaded until you add --yes. The
release is pinned, and every file is checked against the digest written here
before it is unpacked; the binary must start before anything goes into the
folder. The folder must be new or empty, and each file that went into it is
written, with its digest, to a manifest on this machine, so `--undo` removes
only files that are still exactly what the install put there.
"""

from __future__ import annotations

import datetime
import json
import re
import os
import shlex
import signal
import subprocess
from pathlib import Path

from . import ai, config

RELEASE = "b12091"
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}"
# name: (bytes, sha256), from the release's own listing on GitHub.
ASSETS = {
    "llama-b12091-bin-macos-arm64.tar.gz": (16578212, "f1a9e3696a8d9c5950a611ca2a81f342119749d4a2cab4f6afe5d109ae1de910"),
    "llama-b12091-bin-macos-x64.tar.gz": (12314144, "ff1056ae7dbee0334a3318eb8d850593ab7ac82df711c58ff3a49a5ee62da36e"),
    "llama-b12091-bin-ubuntu-x64.tar.gz": (22963980, "0bb3f3a704e143b19b15617a43f0c9e6c179c1b126caf4a80d1f24bbbb125078"),
    "llama-b12091-bin-ubuntu-arm64.tar.gz": (18643938, "3ab4cd36348b44dd8cd1f331ad9de2d55223f89aae82558f89dd20a3d68402f7"),
    "llama-b12091-bin-ubuntu-cuda-12.8-x64.tar.gz":
        (187232843, "e35ff86b1696b7e2b61a11839f0f33420951f2bc6e7b53840cf7b621175d0fde"),
    "cudart-llama-b12091-bin-ubuntu-cuda-12.8-x64.tar.gz":
        (594371574, "d8da0f89b84b045eed31580b721206a5c6194fbf0f93272fa08f7c64054a8593"),
}
BINARY = {"engine": "llama-completion", "helper": "ggml-rpc-server"}
FOLDER = {"engine": "~/llm/engine", "helper": "~/llm/helper"}
INSTALL_DIR = config.ORCHARD_HOME / "install"
FOLDER_OK = re.compile(r"^(~/|/)[A-Za-z0-9._/-]+$")
FILE_OK = re.compile(r"^[A-Za-z0-9._+-]+(/[A-Za-z0-9._+-]+)*$")
HOST_OK = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]{0,63}@)?[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
HASH = "h() { (sha256sum \"$1\" 2>/dev/null || shasum -a 256 \"$1\") | cut -d' ' -f1; }"
PROBE = """uname -s; uname -m
if command -v nvidia-smi >/dev/null 2>&1; then echo nvidia; else echo none; fi
folder=%s
if [ ! -e "$folder" ]; then echo absent; elif [ -z "$(ls -A "$folder" 2>/dev/null)" ]; then echo empty; else echo full; fi
"""


class InstallError(Exception):
    """Nothing could be, or was, installed."""


class BadInput(InstallError):
    """A role, host or folder that isn't allowed."""


def _folder_expr(folder: str) -> str:
    """The folder for a shell on the target: ~ becomes that machine's home."""
    if not FOLDER_OK.match(folder) or ".." in folder.split("/"):
        raise BadInput(f"folder {folder!r}: use ~/name or /full/path, letters, digits, . _ - and /")
    folder = folder.rstrip("/") or "/"
    if folder in ("~", "/"):
        raise BadInput("give a folder of its own, not the home folder or /")
    return '"$HOME"/' + shlex.quote(folder[2:]) if folder.startswith("~/") else shlex.quote(folder)


def _sh(host: str, script: str, timeout: int) -> subprocess.CompletedProcess:
    """Run a script on that machine. On another machine, a script cut off by a
    timeout dies at its next line of output, before anything goes in the
    folder; here, it is stopped gently so its own clean-up runs."""
    if host:
        return ai._remote(host, "bash -s", timeout=timeout, stdin_text=script)
    proc = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", start_new_session=True)
    try:
        out, err = proc.communicate(script, timeout=timeout)
        return subprocess.CompletedProcess(["bash"], proc.returncode, out, err)
    except BaseException as exc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
        except ProcessLookupError:
            pass
        if isinstance(exc, subprocess.TimeoutExpired):
            return subprocess.CompletedProcess(["bash"], 255, "", "timed out")
        raise


def bundle(system: str, machine: str, nvidia: bool) -> tuple[list[str], str] | None:
    """The release files for a platform, and a plain name for it."""
    if system == "Darwin":
        return {"arm64": (["llama-b12091-bin-macos-arm64.tar.gz"], "macOS, Apple silicon"),
                "x86_64": (["llama-b12091-bin-macos-x64.tar.gz"], "macOS, Intel")}.get(machine)
    if system == "Linux" and machine == "x86_64":
        if nvidia:
            return (["llama-b12091-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b12091-bin-ubuntu-cuda-12.8-x64.tar.gz"],
                    "Linux x86_64 with an NVIDIA card")
        return ["llama-b12091-bin-ubuntu-x64.tar.gz"], "Linux x86_64"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return ["llama-b12091-bin-ubuntu-arm64.tar.gz"], "Linux arm64"
    return None


def manifest_path(host: str, role: str) -> Path:
    # "_here" can't be an ssh name, so this machine never shares a manifest with one.
    return INSTALL_DIR / f"{host or '_here'}-{role}.json"


def _check(role: str, host: str) -> None:
    if role not in BINARY:
        raise BadInput("--role is engine or helper")
    if host and not HOST_OK.match(host):
        raise BadInput(f"host {host!r} isn't an ssh name")


def plan(role: str, host: str = "", folder: str = "", run=None) -> dict:
    """What an install would do on that machine. Reads the machine; changes nothing."""
    _check(role, host)
    run = run or _sh
    folder = folder or FOLDER[role]
    done = run(host, PROBE % _folder_expr(folder), 30)
    lines = done.stdout.split()
    if done.returncode != 0 or len(lines) < 4:
        raise InstallError(f"couldn't read {host or 'this machine'}: {(done.stderr or done.stdout).strip()[:200]}")
    system, machine, gpu, state = lines[:4]
    found = bundle(system, machine, gpu == "nvidia")
    if not found:
        raise InstallError(f"no pinned llama.cpp release for {system} {machine}: build it from source, "
                           "see contrib/ai/README.md")
    files, platform = found
    return {"role": role, "host": host, "folder": folder, "platform": platform, "release": RELEASE,
            "folder_state": state, "binary": BINARY[role],
            "files": [{"name": n, "url": f"{BASE}/{n}", "bytes": ASSETS[n][0], "sha256": ASSETS[n][1]} for n in files],
            "installed": manifest_path(host, role).exists()}


def describe(p: dict) -> list[str]:
    where = p["host"] or "this machine"
    lines = [f"Install the {p['role']} on {where} ({p['platform']}), llama.cpp {p['release']}:"]
    for f in p["files"]:
        lines += [f"  file: {f['name']}, {f['bytes'] / 1e6:.1f} MB",
                  f"  from: {f['url']}", f"  checked against sha256 {f['sha256']}"]
    lines.append(f"  into: {p['folder']} ({'a new folder' if p['folder_state'] == 'absent' else p['folder_state']})")
    if p["role"] == "engine":
        lines.append("  For a model larger than memory, build the engine into its own folder with "
                     "contrib/ai/build-engine.sh instead: it adds two patches this release lacks.")
    return lines


def _install_script(p: dict) -> str:
    steps = []
    for f in p["files"]:
        name = shlex.quote(f["name"])
        steps.append(f"""curl -fsSL --retry 3 -o "$work/"{name} {shlex.quote(f['url'])}
[ "$(h "$work/"{name})" = {f['sha256']} ] || {{ echo "ORCHARD-DIGEST {f['name']}"; exit 4; }}
strip=$(tar -tzf "$work/"{name} | awk -F/ 'NF > 1 {{ n++ }} END {{ print (n == NR) ? 1 : 0 }}')
tar -xzf "$work/"{name} -C "$work/x" --strip-components="$strip"
rm -f "$work/"{name}
echo "ORCHARD-GOT {f['name']}\"""")
    # Everything happens in a folder of its own beside the target. Only when every
    # file matched and the binary started does that folder become the target, in
    # one rename; the file list comes from it, so nothing else can be counted.
    return f"""set -eu
{HASH}
folder={_folder_expr(p['folder'])}
parent=$(dirname "$folder")
made_parent=0
[ -d "$parent" ] || {{ mkdir -p "$parent"; made_parent=1; }}
done_ok=0
work=$(mktemp -d "$folder.orchard.XXXXXX")
trap 'rm -rf "$work"; [ "$done_ok" = 1 ] || [ "$made_parent" = 0 ] || rmdir "$parent" 2>/dev/null || true' EXIT
trap 'exit 1' HUP INT TERM PIPE
mkdir "$work/x"
{chr(10).join(steps)}
rc=0
"$work/x/"{shlex.quote(p['binary'])} --help >/dev/null 2>&1 </dev/null || rc=$?
echo "ORCHARD-CHECK $rc"
[ "$rc" -lt 126 ] || exit 5
cd "$work/x"
find . -mindepth 1 -type d | sed 's|^\\./|ORCHARD-DIR |'
find . -type l | sed 's|^\\./|ORCHARD-LINK |'
find . -type f | while IFS= read -r f; do echo "ORCHARD-FILE $(h "$f") ${{f#./}}"; done
cd /
created=1
if [ -d "$folder" ]; then created=0; rmdir "$folder" 2>/dev/null || {{ echo ORCHARD-NOT-EMPTY; exit 3; }}; fi
mv "$work/x" "$folder"
done_ok=1
echo "ORCHARD-CREATED $created"
"""


def _read_install(out: str) -> tuple[list[dict], list[str]]:
    files, dirs = [], []
    for line in out.splitlines():
        kind, _, rest = line.partition(" ")
        if kind == "ORCHARD-FILE":
            digest, _, path = rest.partition(" ")
            files.append({"path": path, "sha256": digest})
        elif kind == "ORCHARD-LINK":
            files.append({"path": rest, "link": True})
        elif kind == "ORCHARD-DIR":
            dirs.append(rest)
    return files, dirs


def install(p: dict, run=None) -> dict:
    """Download, check, unpack and start-test, then write the manifest."""
    run = run or _sh
    if p["installed"]:
        raise InstallError(f"Orchard already installed the {p['role']} there: "
                           f"undo it first with orchard ai install --role {p['role']}"
                           + (f" --host {p['host']}" if p["host"] else "") + " --undo")
    if p["folder_state"] == "full":
        raise InstallError(f"{p['folder']} already has files in it: pick a new --folder")
    total = sum(f["bytes"] for f in p["files"])
    done = run(p["host"], _install_script(p), 600 + total // 200_000)
    out = done.stdout
    if done.returncode != 0 or "ORCHARD-CREATED" not in out:
        why = {3: f"{p['folder']} already has files in it", 4: "a file didn't match its digest, so nothing was unpacked",
               5: f"{p['binary']} didn't start on that machine, so nothing went into the folder"}.get(done.returncode)
        if not why and p["host"]:
            why = (f"the link to {p['host']} broke or timed out; if it dropped late, the files may be in "
                   f"{p['folder']} with no record here, so look there before trying again")
        raise InstallError(f"{why or 'the install stopped before anything went into the folder'}: "
                           f"{(done.stderr or out).strip()[-300:]}")
    files, dirs = _read_install(out)
    record = {"role": p["role"], "host": p["host"], "folder": p["folder"], "release": p["release"],
              "platform": p["platform"], "assets": [f["name"] for f in p["files"]],
              "created_folder": "ORCHARD-CREATED 1" in out, "files": files, "dirs": dirs,
              "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    path = manifest_path(p["host"], p["role"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _safe(path: str) -> bool:
    return bool(FILE_OK.match(path)) and ".." not in path.split("/")


def undo(role: str, host: str = "", run=None) -> dict:
    """Remove the files the install put there that are still as it left them,
    the folders it made once they're empty, and the folder itself if it made it."""
    _check(role, host)
    run = run or _sh
    path = manifest_path(host, role)
    if not path.exists():
        raise InstallError(f"Orchard hasn't installed the {role} on {host or 'this machine'}, so there's nothing to undo")
    try:
        record = json.loads(path.read_text())
        files, dirs = record["files"], record.get("dirs", [])
        folder = _folder_expr(record["folder"])
    except (ValueError, KeyError, TypeError, BadInput) as exc:
        raise InstallError(f"{path} can't be read ({exc}); remove the files by hand") from exc
    removals = []
    files = [f for f in files if isinstance(f, dict) and _safe(str(f.get("path", "")))
             and (f.get("link") or re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))))]
    for f in files:
        q = "./" + shlex.quote(f["path"])
        if f.get("link"):
            removals.append(f"[ -L {q} ] && rm -f -- {q} && echo ORCHARD-REMOVED || true")
        else:
            removals.append(f"[ -f {q} ] && [ ! -L {q} ] && [ \"$(h {q})\" = {f['sha256']} ] "
                            f"&& rm -f -- {q} && echo ORCHARD-REMOVED || true")
    deepest = sorted((d for d in dirs if isinstance(d, str) and _safe(d)), key=lambda d: d.count("/"), reverse=True)
    script = f"""{HASH}
folder={folder}
if [ ! -e "$folder" ] && [ ! -L "$folder" ]; then echo ORCHARD-GONE; exit 0; fi
[ -d "$folder" ] && [ ! -L "$folder" ] && cd "$folder" || {{ echo "can't open $folder" >&2; exit 1; }}
{chr(10).join(removals)}
{chr(10).join(f"rmdir -- ./{shlex.quote(d)} 2>/dev/null || true" for d in deepest)}
left=$(ls -A . | head -5 | tr '\\n' ' ')
if [ {1 if record.get('created_folder') else 0} = 1 ] && [ -z "$left" ]; then cd / && rmdir "$folder" && echo ORCHARD-FOLDER-GONE; fi
echo "ORCHARD-LEFT $left"
"""
    done = run(host, script, 600)
    if done.returncode != 0:
        raise InstallError(f"couldn't undo, so the record of the install is kept: "
                           f"{(done.stderr or done.stdout).strip()[:200]}")
    path.unlink()
    left = next((line[len("ORCHARD-LEFT "):].strip() for line in done.stdout.splitlines()
                 if line.startswith("ORCHARD-LEFT ")), "")
    return {"removed": done.stdout.count("ORCHARD-REMOVED"), "listed": len(files), "folder": record["folder"],
            "folder_gone": "ORCHARD-FOLDER-GONE" in done.stdout, "gone_already": "ORCHARD-GONE" in done.stdout,
            "left": left}
