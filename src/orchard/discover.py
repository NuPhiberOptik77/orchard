"""Find machines on the home network that Orchard could use.

`orchard find` looks, for a few seconds, for machines that announce ssh over
Bonjour (a Mac with Remote Login on does; Linux with Avahi can). It opens no
port and starts no listener, and says whether each one is already an Orchard
worker. Only with --check-keys does it try your ssh keys, and only on machines
whose host key you already trust, so your keys never go to a machine you
haven't checked. Nothing is changed and nobody on the network can join: adding
a machine stays your step. What a machine announces is checked before it is
printed, since anything on the network can announce anything.
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
HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")


def _clean_name(name: str) -> str:
    """An announced name, safe to print: escapes decoded, control characters gone."""
    name = re.sub(r"\\(\d{3})", lambda m: chr(int(m[1])), name)
    return "".join(ch for ch in name if ch.isprintable())[:80]


def _host_ok(host: str) -> bool:
    return bool(HOST.match(host))


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
    return (match[1], int(match[2])) if match and _host_ok(match[1]) and 0 < int(match[2]) < 65536 else None


def parse_avahi(text: str) -> list[dict]:
    """Resolved entries from `avahi-browse -rpt` output, one per host."""
    out = []
    for line in text.splitlines():
        parts = line.split(";")
        if len(parts) >= 9 and parts[0] == "=" and parts[4] == SERVICE:
            host, port = parts[6].rstrip("."), parts[8] or "22"
            if _host_ok(host) and port.isdigit() and not any(f["host"] == host for f in out):
                out.append({"name": _clean_name(parts[3]), "host": host, "port": int(port)})
    return out


def look(seconds: float = LOOK_SECONDS, run: Callable[[list[str], float], str] = _run_for) -> list[dict]:
    """Machines announcing ssh: name, host and port."""
    if shutil.which("dns-sd"):
        names = parse_browse(run(["dns-sd", "-B", SERVICE, "local."], seconds))

        def resolve(name: str) -> dict | None:
            found = parse_lookup(run(["dns-sd", "-L", name, SERVICE, "local."], RESOLVE_SECONDS))
            return {"name": _clean_name(name), "host": found[0], "port": found[1]} if found else None

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


def keys_open(host: str, port: int, user: str) -> str:
    """"opens" if your ssh keys log in, "unverified" if the machine's host key
    isn't one you already trust (so no key was offered), else "closed". Only
    `true` runs; no password is tried and nothing is forwarded."""
    argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_SECONDS}", "-o", "StrictHostKeyChecking=yes",
            "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no", "-o", "ForwardAgent=no",
            "-o", "ForwardX11=no", "-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR",
            "-p", str(port), f"{user}@{host}", "true"]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=SSH_SECONDS + 5)
    except (OSError, subprocess.TimeoutExpired):
        return "closed"
    if done.returncode == 0:
        return "opens"
    return "unverified" if "host key" in (done.stderr or "").lower() else "closed"


def _short(host: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", host.lower().removesuffix(".local")).strip("-")[:20] or "machine"


def find(seconds: float = LOOK_SECONDS, *, check_keys: bool = False, look_fn: Callable | None = None,
         hostname: Callable | None = None, opens: Callable | None = None, here: Callable | None = None,
         user: str | None = None) -> list[dict]:
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
        if not check_keys:
            return f | {"status": "found"}
        status = opens(f["host"], f["port"], user)
        return f | {"status": status, "user": user} | ({"suggest": _short(f["host"])} if status == "opens" else {})

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
        elif f["status"] == "found":
            lines.append(f"  {f['name']}: {where}, not a worker; `orchard find --check-keys` tries your ssh keys "
                         "on it if you already trust its host key")
        elif f["status"] == "unverified":
            lines.append(f"  {f['name']}: {where}, its host key isn't one you trust yet, so no key was tried: "
                         f"connect once with ssh {f['user']}@{f['host']} and check the fingerprint it shows")
        else:
            lines.append(f"  {f['name']}: {where}, found, but your ssh keys don't open it as {f['user']}: copy your "
                         f"key there (ssh-copy-id {f['user']}@{f['host']}) or set its user in ~/.ssh/config")
    lines.append("Machines that don't announce themselves, like most Windows and Linux ones, can be added by hand. "
                 "Nothing was changed.")
    return lines
