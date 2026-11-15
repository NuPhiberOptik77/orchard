"""Find machines on the home network that Orchard could use.

`orchard find` looks, for a few seconds, for machines that announce ssh over
Bonjour (a Mac with Remote Login on does; Linux with Avahi can). It opens no
port and starts no listener. For each one it says whether it is already an
Orchard worker, and whether the ssh keys Orchard already uses open it. Nobody
on the network can join this way: a machine is only suggested when your own
keys get in, and nothing is changed. Adding one stays your step.
"""

from __future__ import annotations

import concurrent.futures
import getpass
import re
import shutil
import subprocess
from typing import Callable

from . import config

LOOK_SECONDS = 3
RESOLVE_SECONDS = 3
SSH_SECONDS = 8
SERVICE = "_ssh._tcp"


def _run_for(argv: list[str], seconds: float) -> str:
    """What a command that never ends by itself printed in `seconds`."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=seconds).stdout
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or b""
        return out.decode(errors="replace") if isinstance(out, bytes) else out
    except OSError:
        return ""


def parse_browse(text: str) -> list[str]:
    """Instance names from `dns-sd -B` output, each once."""
    names = []
    for line in text.splitlines():
        match = re.search(r"\sAdd\s+\d+\s+\d+\s+\S+\s+_ssh\._tcp\.\s+(.+?)\s*$", line)
        if match and match[1] not in names:
            names.append(match[1])
    return names


def parse_lookup(text: str) -> tuple[str, int] | None:
    """Host and port from `dns-sd -L` output."""
    match = re.search(r"can be reached at (\S+?)\.?:(\d+)", text)
    return (match[1], int(match[2])) if match else None


def parse_avahi(text: str) -> list[dict]:
    """Resolved entries from `avahi-browse -rpt` output, one per host."""
    out = []
    for line in text.splitlines():
        parts = line.split(";")
        if len(parts) >= 9 and parts[0] == "=" and parts[4] == SERVICE:
            host = parts[6].rstrip(".")
            if not any(f["host"] == host for f in out):
                out.append({"name": parts[3].replace("\\032", " "), "host": host, "port": int(parts[8] or 22)})
    return out


def look(seconds: float = LOOK_SECONDS, run: Callable[[list[str], float], str] = _run_for) -> list[dict]:
    """Machines announcing ssh: name, host and port."""
    if shutil.which("dns-sd"):
        names = parse_browse(run(["dns-sd", "-B", SERVICE, "local."], seconds))

        def resolve(name: str) -> dict | None:
            found = parse_lookup(run(["dns-sd", "-L", name, SERVICE, "local."], RESOLVE_SECONDS))
            return {"name": name, "host": found[0], "port": found[1]} if found else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            found = [f for f in pool.map(resolve, names) if f]
        return [f for i, f in enumerate(found) if f["host"] not in {g["host"] for g in found[:i]}]
    if shutil.which("avahi-browse"):
        return parse_avahi(run(["avahi-browse", "-rpt", SERVICE], seconds + RESOLVE_SECONDS))
    return []


def _ssh_hostname(target: str) -> str:
    """The host ssh really connects to for a name, after ~/.ssh/config."""
    try:
        out = subprocess.run(["ssh", "-G", target], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return target
    match = re.search(r"^hostname (\S+)", out, re.M)
    return match[1] if match else target


def _this_host() -> str:
    try:
        name = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        name = ""
    return f"{name}.local" if name else ""


def keys_open(host: str, port: int, user: str) -> bool:
    """Your ssh keys log in, without a password prompt. Only `true` runs; the
    machine's host key isn't recorded, and nothing is forwarded."""
    argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_SECONDS}", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ForwardAgent=no",
            "-o", "ClearAllForwardings=yes", "-p", str(port), f"{user}@{host}", "true"]
    try:
        return subprocess.run(argv, capture_output=True, timeout=SSH_SECONDS + 5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _short(host: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", host.lower().removesuffix(".local")).strip("-")[:20] or "machine"


def find(seconds: float = LOOK_SECONDS, *, look_fn: Callable | None = None, hostname: Callable | None = None,
         opens: Callable | None = None, here: Callable | None = None, user: str | None = None) -> list[dict]:
    """Each machine found, and what it is to Orchard: this machine, a worker
    already, one your keys open, or one they don't."""
    look_fn, hostname = look_fn or look, hostname or _ssh_hostname
    opens, here = opens or keys_open, here or _this_host
    user = user or getpass.getuser()
    known = {}
    for w in config.load_workers():
        known[hostname(w.target).lower().rstrip(".")] = w.name
        known[w.host.lower().rstrip(".")] = w.name
    me = here().lower()
    found = look_fn(seconds)

    def judge(f: dict) -> dict:
        host = f["host"].lower().rstrip(".")
        if host == me:
            return f | {"status": "this machine"}
        if host in known:
            return f | {"status": "worker", "worker": known[host]}
        if opens(f["host"], f["port"], user):
            return f | {"status": "opens", "user": user, "suggest": _short(f["host"])}
        return f | {"status": "closed", "user": user}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(judge, found))


def report(found: list[dict], seconds: float) -> list[str]:
    lines = [f"Machines on this network that offer ssh (looked for {seconds:g} s):"]
    if not found:
        lines.append("  none answered. A Mac shows up once Remote Login is on (System Settings, General, Sharing).")
    for f in found:
        where = f["host"] + ("" if f["port"] == 22 else f":{f['port']}")
        if f["status"] == "this machine":
            lines.append(f"  {f['name']}: {where}, this machine")
        elif f["status"] == "worker":
            lines.append(f"  {f['name']}: {where}, already an Orchard worker ({f['worker']})")
        elif f["status"] == "opens":
            lines += [f"  {f['name']}: {where}, your ssh keys open it as {f['user']}. To add it, put this in "
                      f"{config.CONFIG_PATH}:",
                      "      [[workers]]", f'      name = "{f["suggest"]}"', f'      host = "{f["host"]}"',
                      "      weight = 0.2", "      tags = []"]
            if f["port"] != 22:
                lines.append(f"      (it answers ssh on port {f['port']}: set Port for it in ~/.ssh/config)")
        else:
            lines.append(f"  {f['name']}: {where}, found, but your ssh keys don't open it as {f['user']}: copy your "
                         f"key there (ssh-copy-id {f['user']}@{f['host']}) or set its user in ~/.ssh/config")
    lines.append("Machines that don't announce themselves, like most Windows and Linux ones, can be added by hand. "
                 "Nothing was changed.")
    return lines
