"""Ask a large model spread across your own machines.

Running a model bigger than any one machine's memory takes a dozen steps by
hand: keep other jobs off the machine, start the GPU helper and its tunnel,
pause the sync apps, wait for memory, run the engine, watch it, and put
everything back. `orchard ai ask` does all of that for one question, and puts
everything back however the run ends.

The answer lands in one sealed record, like a job's receipt: what was asked,
what came back, which files answered it and how fast. Nothing leaves your
machines. Settings live in ~/.orchard/ai.toml, never in the repo. The engine is
llama.cpp, and the model files stay yours to download.
"""

from __future__ import annotations

import base64
import codecs
import fcntl
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Callable

from . import config, dispatcher, fleet, receipts
from .config import ORCHARD_HOME, STATE_DIR

AI_DIR = STATE_DIR / "ai"
CONFIG_PATH = ORCHARD_HOME / "ai.toml"
HOLD_NAME = "ai-hold"
EXIT_CODES = {"answered": 0, "failed": 1, "refused": 2, "stopped": 3}

GUARD_POLL = 20
LOW_READS = 3
UNREAD_READS = 6
PAUSED_READS = 2
CRITICAL_PRESSURE = 4
MEMORY_POLL = 10
HOLD_POLL = 5
HELPER_SETTLE = 4
TUNNEL_TRIES = 6
TUNNEL_SETTLE = 3
TUNNEL_GAP = 10
PAUSE_WAIT = 60
RESUME_TRIES = 5
RESUME_GAP = 5
KILL_WAIT = 60

_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_PROCESS_NAME = re.compile(r"^[\w .-]+$")
_SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$")

DEFAULTS = {
    "engine": {
        "binary": "llama-completion", "busy_pattern": r"^\./llama-|^\./ggml-rpc", "threads": 4, "context": 512,
        "layers": 99, "devices": "", "split": "", "overrides": "", "args": [], "env": {}, "words": 200,
        "cap_seconds": 2400, "worker": "", "worker_tags": [], "hold_wait_seconds": 900, "start_free_mb": 0,
        "stop_free_mb": 0, "settle_seconds": 300, "thinking_end": "",
    },
    "helper": {"binary": "ggml-rpc-server", "device": "", "port": 50052, "tunnel_port": 50072},
    "windows": {"pause": []},
}

TEMPLATE = """\
# Settings for `orchard ai`. They stay on this machine: never put them in a repo.
# Change the values marked CHANGE, then run `orchard ai setup` to check them.
# docs/ai.md explains each one and how to build the engine.

[engine]
host = "CHANGE"                                     # ssh name of the machine that holds the model
folder = "~/llm/engine"                             # holds the llama.cpp binaries
model = "~/llm/models/CHANGE-00001-of-00006.gguf"   # first file of the model
threads = 4
context = 512
layers = 99
words = 200                                         # most words in an answer
thinking_end = ""                                   # for a model that reasons first, the mark that ends it, like "</think>"
cap_seconds = 2400                                  # stop the engine after this long
devices = ""                                        # for example "CUDA0,RPC0"; empty lets the engine decide
split = ""                                          # for example "22,78"
overrides = ""                                      # for example "_exps|per_layer_token_embd=CPU"
args = ["--temp", "0", "-no-cnv", "--no-display-prompt", "--no-warmup"]
start_free_mb = 5000                                # wait for this much free memory before starting
stop_free_mb = 200                                  # stop the run after three reads under this
worker = ""                                         # Orchard worker on that machine, so jobs wait for the run
worker_tags = []                                    # for example ["os:linux"]

[engine.env]
# GGML_RPC_BATCH_FETCH = "1"

# Lend a GPU on this machine to the engine over an ssh tunnel.
# [helper]
# folder = "~/llm/helper"
# device = "MTL0"

# The engine machine runs Linux inside Windows: read free memory from Windows,
# and pause these apps during a run.
# [windows]
# host = "CHANGE"
# pause = ["OneDrive"]
"""


class ConfigError(Exception):
    """The settings file is missing or wrong."""


class Refused(Exception):
    """The question was never put to the model."""


class _Interrupted(Exception):
    pass


def load_settings(threads: int | None = None, split: str | None = None, overrides: str | None = None,
                  words: int | None = None) -> dict:
    """The settings file over the defaults, then the given values over both."""
    try:
        raw = tomllib.loads(CONFIG_PATH.read_text())
    except OSError:
        raise ConfigError(f"no settings at {CONFIG_PATH}: run `orchard ai setup`")
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{CONFIG_PATH}: {exc}")
    cfg = {section: {**values, **raw.get(section, {})} for section, values in DEFAULTS.items()}
    engine = cfg["engine"]
    for key in ("host", "folder", "model"):
        if not engine.get(key) or "CHANGE" in str(engine[key]):
            raise ConfigError(f"{CONFIG_PATH}: set engine.{key}")
    for name in cfg["windows"]["pause"]:
        if not _PROCESS_NAME.match(name):
            raise ConfigError(f"{CONFIG_PATH}: bad process name {name!r}")
    for key, value in (("threads", threads), ("split", split), ("overrides", overrides), ("words", words)):
        if value not in (None, ""):
            engine[key] = value
    return cfg


def has_helper(cfg: dict) -> bool:
    return bool(cfg["helper"].get("folder"))


def has_windows(cfg: dict) -> bool:
    return bool(cfg["windows"].get("host"))


def _rpath(path: str) -> str:
    """A path for a remote shell: quoted, with a leading ~ still expanding."""
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def model_pattern(model: str) -> str:
    """A glob for every file of a model, from its first file's name."""
    return _SHARD.sub("-*.gguf", model)


def engine_command(cfg: dict, question: str, words: int) -> list[str]:
    e = cfg["engine"]
    argv = [f"./{e['binary']}", "-m", e["model"], "-p", question, "-n", str(words),
            "-c", str(e["context"]), "-t", str(e["threads"])]
    if has_helper(cfg):
        argv += ["--rpc", f"127.0.0.1:{cfg['helper']['tunnel_port']}"]
    for flag, key in (("--device", "devices"), ("-ngl", "layers"), ("-ts", "split"), ("-ot", "overrides")):
        if e[key] != "":
            argv += [flag, str(e[key])]
    return argv + [str(a) for a in e["args"]]


def run_script(cfg: dict, question: str, words: int) -> str:
    """The script the engine machine runs. The engine's stdout is the answer.
    On the way out it asks the kernel to drop the model from memory."""
    e = cfg["engine"]
    argv = engine_command(cfg, question, words)
    # The model path goes through the shell like the folder does, so ~ works.
    command = " ".join(_rpath(a) if i == 2 else shlex.quote(a) for i, a in enumerate(argv))
    lines = [f"cd {_rpath(e['folder'])} || exit 127"]
    lines += [f"export {key}={shlex.quote(str(value))}" for key, value in e["env"].items()]
    lines += [
        "give_back() {",
        "python3 -I - >&2 <<'GIVE_BACK'",
        "import glob, os",
        f"paths = glob.glob(os.path.expanduser({model_pattern(e['model'])!r}))",
        "for path in paths:",
        "    fd = os.open(path, os.O_RDONLY)",
        "    if hasattr(os, 'posix_fadvise'):",
        "        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)",
        "    os.close(fd)",
        "print(f'gave back {len(paths)} model files')",
        "GIVE_BACK",
        "}",
        "trap give_back EXIT",
        f"perl -e 'alarm {int(e['cap_seconds'])}; exec @ARGV' {command} < /dev/null",
        "exit $?",
    ]
    return "\n".join(lines) + "\n"


_FINGERPRINT = """\
import glob, hashlib, json, os
os.chdir(os.path.expanduser({folder!r}))
engine = {{}}
for path in sorted({{{binary!r}, *glob.glob('libllama*'), *glob.glob('libggml*')}}):
    if os.path.isfile(path) and not os.path.islink(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        engine[path] = digest.hexdigest()
model = [dict(name=os.path.basename(p), bytes=os.path.getsize(p), modified=int(os.path.getmtime(p)))
         for p in sorted(glob.glob(os.path.expanduser({pattern!r})))]
print(json.dumps(dict(engine=engine, model=model)))
"""


def fingerprint_script(cfg: dict) -> str:
    """Which engine and model files answered: hashes for the engine, sizes and
    dates for the model, which is too big to hash on every question."""
    e = cfg["engine"]
    return _FINGERPRINT.format(folder=e["folder"], binary=e["binary"], pattern=model_pattern(e["model"]))


END_OF_TEXT = "[end of text]"


def split_answer(raw: str, thinking_end: str = "") -> tuple[str, str]:
    """(thinking, answer) from what the engine printed. A model that reasons
    first prints its reasoning, the mark, then its answer. No mark means the
    words ran out before it got to the answer."""
    text = raw.strip()
    if text.endswith(END_OF_TEXT):
        text = text[:-len(END_OF_TEXT)].rstrip()
    if not thinking_end:
        return "", text
    thinking, mark, answer = text.partition(thinking_end)
    return (thinking.strip(), answer.strip()) if mark else (text, "")


class AnswerStream:
    """What the engine prints, made fit to read as it arrives: the thinking is
    labelled, and the engine's end mark and trailing blanks never show."""

    def __init__(self, thinking_end: str, write: Callable[[str], None]) -> None:
        self.thinking_end, self.write = thinking_end, write
        self.thinking = bool(thinking_end)
        self.started = False
        self.fresh = True
        self.held = ""

    def _print(self, text: str) -> None:
        if not text:
            return
        if not self.started:
            self.write("--- thinking\n" if self.thinking else "--- answer\n")
            self.started = True
        self.write(text)

    def _safe_length(self, text: str) -> int:
        """How much can print now: not trailing blanks, which may be the end,
        and not what may yet turn out to be a mark."""
        marks = (END_OF_TEXT, self.thinking_end) if self.thinking else (END_OF_TEXT,)
        body = text.rstrip()
        for size in range(min(len(body), max(map(len, marks))), 0, -1):
            if any(mark.startswith(body[-size:]) for mark in marks):
                return len(body[:-size].rstrip())
        return len(body)

    def feed(self, chunk: str) -> None:
        text = self.held + chunk
        if self.thinking and self.thinking_end in text:
            before, _, text = text.partition(self.thinking_end)
            self._print(before.rstrip())
            self.write("\n--- answer\n" if self.started else "--- answer\n")
            self.started, self.thinking, self.fresh = True, False, True
        if self.fresh:
            text = text.lstrip()
            self.fresh = not text
        keep = self._safe_length(text)
        self._print(text[:keep])
        self.held = text[keep:]

    def finish(self) -> None:
        if self.held.strip() != END_OF_TEXT:
            self._print(self.held.rstrip())
        self.held = ""


def parse_timings(log: str) -> dict:
    out: dict = {}
    for line in log.splitlines():
        if match := re.search(r"prompt eval time\s*=\s*([\d.]+) ms", line):
            out["prompt_seconds"] = round(float(match[1]) / 1000, 1)
        elif match := re.search(r"eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) runs\s*\(\s*([\d.]+) ms per token", line):
            out["words"], out["ms_per_word"] = int(match[1]), float(match[2])
        elif match := re.search(r"load time\s*=\s*([\d.]+) ms", line):
            out["load_seconds"] = round(float(match[1]) / 1000, 1)
    return out


def memory_lines(log: str) -> list[str]:
    """Where the model sat, as the engine reports it on the way out. The engine
    only prints this at a verbose log level, so usually there's nothing to find."""
    return [line.split(":", 1)[1].strip() for line in log.splitlines()
            if "memory_breakdown" in line and "|" in line]


def parse_typeperf(text: str) -> int | None:
    rows = [row for row in text.replace("\x00", "").splitlines() if re.match(r'^"\d', row)]
    if not rows:
        return None
    match = re.search(r'"([\d.]+)"\s*$', rows[-1])
    return int(float(match[1])) if match else None


def _ps_names(names: list[str]) -> str:
    return ",".join(f"'{name}'" for name in names)


def hold_script(names: list[str], limit: int) -> str:
    """Pauses the apps until this ssh session ends, however it ends, or until
    the limit, which covers a link that vanishes without closing. Windows
    OpenSSH leaves this process running after the session ends and stdin never
    tells it, so it waits on the session's own sshd. No session found: no pause."""
    return f"""\
$limit = {int(limit)}
$ProgressPreference = 'SilentlyContinue'
Add-Type -Name N -Namespace S -MemberDefinition '[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h); [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
$names = {_ps_names(names)}
$session = $null; $id = $PID
foreach ($hop in 1..4) {{
  $id = (Get-CimInstance Win32_Process -Filter "ProcessId=$id").ParentProcessId
  $p = Get-Process -Id $id -ErrorAction SilentlyContinue
  if ($p -and $p.ProcessName -like 'sshd*') {{ $session = $p; break }}
}}
if (-not $session) {{ [Console]::Out.WriteLine('held none'); exit 1 }}
try {{
  Get-Process -Name $names -ErrorAction SilentlyContinue | ForEach-Object {{ [void][S.N]::NtSuspendProcess($_.Handle) }}
  Get-Process -Name $names -ErrorAction SilentlyContinue | ForEach-Object {{
    $t = $_.Threads
    $s = @($t | Where-Object {{ $_.ThreadState -eq 'Wait' -and $_.WaitReason -eq 'Suspended' }}).Count
    [Console]::Out.WriteLine("held $($_.ProcessName) suspended $s of $($t.Count)")
  }}
  [Console]::Out.WriteLine('held done')
  [Console]::Out.Flush()
  [void]$session.WaitForExit($limit * 1000)
}} finally {{
  foreach ($k in 1..6) {{ Get-Process -Name $names -ErrorAction SilentlyContinue | ForEach-Object {{ [void][S.N]::NtResumeProcess($_.Handle) }} }}
}}
"""


def resume_script(names: list[str]) -> str:
    """Brings the apps back and says how they stand. Safe at any time. It
    resumes several times because suspends stack."""
    return f"""\
$ProgressPreference = 'SilentlyContinue'
Add-Type -Name N -Namespace S -MemberDefinition '[DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
$names = {_ps_names(names)}
foreach ($k in 1..6) {{ Get-Process -Name $names -ErrorAction SilentlyContinue | ForEach-Object {{ [void][S.N]::NtResumeProcess($_.Handle) }} }}
Get-Process -Name $names -ErrorAction SilentlyContinue | ForEach-Object {{ $t = $_.Threads; $s = @($t | Where-Object {{ $_.ThreadState -eq 'Wait' -and $_.WaitReason -eq 'Suspended' }}).Count; "back $($_.ProcessName) suspended $s of $($t.Count)" }}
'back done'
"""


def _powershell(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    return f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"


def _ssh(host: str) -> list[str]:
    return ["ssh", "-T", "-o", "BatchMode=yes", host]


def _remote(host: str, command: str, timeout: int = 60, stdin_text: str | None = None) -> subprocess.CompletedProcess:
    argv = [*_ssh(host), command]
    try:
        return subprocess.run(argv, input=stdin_text, capture_output=True, text=True, errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL if stdin_text is None else None)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 255, "", "timed out")


def _local_pressure() -> int:
    """This Mac's memory pressure level; 0 where there's no such reading."""
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                  capture_output=True, text=True, timeout=10).stdout.strip() or 0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0


def _pkill_local(pattern: str) -> None:
    subprocess.run(["pkill", "-f", pattern], capture_output=True)


def _job_entry(job_id: str) -> dict | None:
    return next((entry for entry in dispatcher.load_entries() if entry.get("id") == job_id), None)


def _submit_hold(cfg: dict, seconds: int, cwd: Path) -> subprocess.CompletedProcess:
    e = cfg["engine"]
    argv = fleet.child_argv(e["worker"], HOLD_NAME, ["sleep", str(seconds)], tags=e["worker_tags"], timeout=seconds,
                            background=True, no_local=True)
    return subprocess.run(argv, capture_output=True, text=True, cwd=cwd, env=fleet.child_env(), timeout=300,
                          stdin=subprocess.DEVNULL)


def _cancel_job(job_id: str) -> bool:
    argv = [sys.executable, "-m", "orchard", "cancel", job_id]
    return subprocess.run(argv, capture_output=True, env=fleet.child_env(), timeout=120,
                          stdin=subprocess.DEVNULL).returncode == 0


def _stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class _Pool:
    """Everything one question starts, so that close() can put it all back."""

    def __init__(self, cfg: dict, folder: Path, say: Callable[[str], None]):
        self.cfg, self.folder, self.say = cfg, folder, say
        self.host = cfg["engine"]["host"]
        self.holder: str | None = None
        self.helper: subprocess.Popen | None = None
        self.tunnel: subprocess.Popen | None = None
        self.hold: subprocess.Popen | None = None
        self.engine: subprocess.Popen | None = None
        self.guard: threading.Thread | None = None
        self.guard_done = threading.Event()
        self.stop_reason: str | None = None
        self.lowest_free: int | None = None
        self.start_free: int | None = None
        self.paused: list[str] = []
        self.pause_seen = False
        self.resumed: bool | None = None
        self.fingerprint: dict = {}
        self.problems: list[str] = []

    def preflight(self) -> None:
        e = self.cfg["engine"]
        if has_helper(self.cfg) and _local_pressure() >= CRITICAL_PRESSURE:
            raise Refused("this machine is short of memory")
        script = (f"pgrep -af {shlex.quote(e['busy_pattern'])} | sed 's/^/busy /'\n"
                  f"[ -x {_rpath(e['folder'])}/{shlex.quote(e['binary'])} ] || echo missing engine\n"
                  f"[ -f {_rpath(e['model'])} ] || echo missing model\n"
                  "echo reached\n")
        out = _remote(self.host, "bash -s", 30, script).stdout
        if "reached" not in out:
            raise Refused(f"cannot reach {self.host}")
        if busy := [line[5:] for line in out.splitlines() if line.startswith("busy ")]:
            raise Refused(f"already running on {self.host}: {busy[0][:80]}")
        if missing := [line[8:] for line in out.splitlines() if line.startswith("missing ")]:
            raise Refused(f"{' and '.join(missing)} not found on {self.host}: run `orchard ai setup`")
        result = _remote(self.host, "python3 -I -", 120, fingerprint_script(self.cfg))
        try:
            self.fingerprint = json.loads(result.stdout)
        except json.JSONDecodeError:
            self.problems.append("could not record which files answered")

    def hold_machine(self) -> None:
        """A sleeping job on the machine, so the queue keeps other jobs off it.
        The engine isn't an Orchard job, so the dispatcher can't see it otherwise."""
        e = self.cfg["engine"]
        cwd = AI_DIR / "hold"
        cwd.mkdir(parents=True, exist_ok=True)
        result = _submit_hold(self.cfg, int(e["cap_seconds"]) + 1800, cwd)
        match = re.search(rf"{HOLD_NAME}-\d{{8}}-\d{{6}}-[0-9a-f]+", result.stdout + result.stderr)
        if result.returncode != 0 or not match:
            raise Refused(f"could not hold {e['worker']} in the queue: {(result.stdout + result.stderr).strip()[-200:]}")
        self.holder = match[0]
        self.say(f"holding {e['worker']} with job {self.holder}")
        end = time.time() + float(e["hold_wait_seconds"])
        while True:
            state = (_job_entry(self.holder) or {}).get("state")
            if state == "running":
                return
            if state not in (None, "queued", "paused"):
                raise Refused(f"the hold on {e['worker']} ended early: {state}")
            if time.time() >= end:
                raise Refused(f"{e['worker']} stayed busy for {int(e['hold_wait_seconds'])} s")
            time.sleep(HOLD_POLL)

    def start_helper(self) -> None:
        h = self.cfg["helper"]
        _pkill_local(f"^./{h['binary']}")
        argv = [f"./{h['binary']}", "-H", "127.0.0.1", "-p", str(h["port"]), "-c"]
        if h["device"]:
            argv += ["-d", h["device"]]
        with (self.folder / "helper.log").open("wb") as log:
            self.helper = subprocess.Popen(argv, cwd=os.path.expanduser(h["folder"]), stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
        time.sleep(HELPER_SETTLE)
        if self.helper.poll() is not None:
            raise Refused(f"the GPU helper stopped straight away: see {self.folder / 'helper.log'}")

    def start_tunnel(self) -> None:
        h = self.cfg["helper"]
        port = h["tunnel_port"]
        _pkill_local(f"^ssh .*-R {port}:")
        argv = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-R", f"{port}:127.0.0.1:{h['port']}", self.host]
        for attempt in range(TUNNEL_TRIES):
            self.tunnel = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
            time.sleep(TUNNEL_SETTLE)
            listening = _remote(self.host, f"ss -ltn | grep -c ':{port} '", 20).stdout.strip()
            if self.tunnel.poll() is None and listening not in ("", "0"):
                return
            _stop_process(self.tunnel)
            if attempt + 1 < TUNNEL_TRIES:
                time.sleep(TUNNEL_GAP)
        raise Refused(f"no tunnel to {self.host} after {TUNNEL_TRIES} tries")

    def pause_apps(self) -> None:
        win, names = self.cfg["windows"].get("host"), self.cfg["windows"]["pause"]
        limit = int(self.cfg["engine"]["cap_seconds"]) + 1200
        log_path = self.folder / "pause.log"
        with log_path.open("wb") as log:
            self.hold = subprocess.Popen([*_ssh(win), _powershell(hold_script(names, limit))],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        end = time.time() + PAUSE_WAIT
        while time.time() < end and self.hold.poll() is None:
            text = log_path.read_bytes().decode(errors="replace").replace("\x00", "").replace("\r", "")
            lines = text.splitlines()
            if "held done" in lines:
                self.pause_seen = True
                self.paused = [line[5:] for line in lines if line.startswith("held ") and " suspended " in line]
                break
            if "held none" in lines:
                break
            time.sleep(1)
        if not self.pause_seen:
            self.say("sync apps not paused: the run goes on, just slower")

    def free_mb(self) -> int | None:
        if has_windows(self.cfg):
            read = _remote(self.cfg["windows"]["host"], 'typeperf "\\Memory\\Available MBytes" -sc 1', 40)
            return parse_typeperf(read.stdout)
        read = _remote(self.host, "awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo", 40)
        return int(read.stdout) if read.stdout.strip().isdigit() else None

    def _note_free(self, free: int | None) -> None:
        if free is not None and (self.lowest_free is None or free < self.lowest_free):
            self.lowest_free = free

    def wait_for_memory(self) -> None:
        """Starting short of memory is what trips the guard during load."""
        e = self.cfg["engine"]
        need = e["start_free_mb"]
        if not need:
            return
        end = time.time() + float(e["settle_seconds"])
        while True:
            free = self.free_mb()
            if free is not None and free >= need:
                self.start_free = free
                return
            if time.time() >= end:
                seen = "an unknown amount" if free is None else f"{free} MB"
                raise Refused(f"{seen} free on the engine machine, and a run needs {need} MB")
            self.say(f"waiting for free memory: {'unknown' if free is None else free} of {need} MB")
            time.sleep(MEMORY_POLL)

    def _kill_engine(self) -> None:
        _remote(self.host, f"pkill -f {shlex.quote('^./' + self.cfg['engine']['binary'])}", 30)

    def _why_stop(self, reads: dict, deadline: float) -> str | None:
        stop_at = self.cfg["engine"]["stop_free_mb"]
        if stop_at:
            free = self.free_mb()
            self._note_free(free)
            reads["unread"] = reads["unread"] + 1 if free is None else 0
            reads["low"] = reads["low"] + 1 if free is not None and free < stop_at else 0
            if reads["low"] >= LOW_READS:
                return f"under {stop_at} MB free on the engine machine"
            if reads["unread"] >= UNREAD_READS:
                return f"could not read free memory {UNREAD_READS} times running"
        if has_helper(self.cfg) and _local_pressure() >= CRITICAL_PRESSURE:
            return "this machine is short of memory"
        if self.holder:
            entry = _job_entry(self.holder) or {}
            # The dispatcher marks a pause with paused_since and leaves the state running.
            in_use = entry.get("state") == "paused" or entry.get("paused_since")
            reads["paused"] = reads["paused"] + 1 if in_use else 0
            if reads["paused"] >= PAUSED_READS:
                return f"the machine is in use: {entry.get('pause_reason') or entry.get('note') or 'paused'}"
        if time.time() > deadline:
            return "passed its time limit"
        return None

    def _watch(self, deadline: float) -> None:
        """Runs beside the engine: backs off if memory runs low, or if the
        machine's user comes back and Orchard pauses the hold."""
        reads = {"low": 0, "unread": 0, "paused": 0}
        while not self.guard_done.wait(GUARD_POLL):
            why = self._why_stop(reads, deadline)
            if why is None:
                continue
            self.stop_reason = why
            self.say(f"stopping the run: {why}")
            self._kill_engine()
            try:
                self.engine.wait(KILL_WAIT)
            except subprocess.TimeoutExpired:
                self.engine.kill()
            return

    def run_engine(self, question: str, words: int, echo: Callable[[str], None] | None) -> int:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        script = run_script(self.cfg, question, words)
        with (self.folder / "engine.log").open("wb") as log, (self.folder / "answer.txt").open("wb") as answer:
            self.engine = subprocess.Popen([*_ssh(self.host), "bash -s"], stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=log)
            self.engine.stdin.write(script.encode())
            self.engine.stdin.close()
            deadline = time.time() + float(self.cfg["engine"]["cap_seconds"]) + 120
            self.guard = threading.Thread(target=self._watch, args=(deadline,), daemon=True)
            self.guard.start()
            while chunk := os.read(self.engine.stdout.fileno(), 4096):
                answer.write(chunk)
                answer.flush()
                if echo and (text := decoder.decode(chunk)):
                    echo(text)
        return self.engine.wait()

    def _step(self, name: str, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as exc:  # one failed step mustn't stop the rest of the cleanup
            self.problems.append(f"{name}: {exc}")

    def _end_pause(self) -> None:
        _stop_process(self.hold)
        win, names = self.cfg["windows"]["host"], self.cfg["windows"]["pause"]
        for attempt in range(RESUME_TRIES):
            out = _remote(win, _powershell(resume_script(names)), 60).stdout.replace("\x00", "")
            if "back done" in out and not re.search(r"suspended [1-9]", out):
                self.resumed = True
                return
            if attempt + 1 < RESUME_TRIES:
                time.sleep(RESUME_GAP)
        self.resumed = False
        self.problems.append("SYNC APPS NOT CONFIRMED BACK: run `orchard ai resume`")

    def close(self) -> None:
        self.guard_done.set()
        if self.engine is not None and self.engine.poll() is None:
            self._step("engine", self._kill_engine)
            self._step("engine", lambda: self.engine.wait(KILL_WAIT))
            self._step("engine", self.engine.kill)
        if self.guard is not None:
            self._step("guard", lambda: self.guard.join(5))
        if self.hold is not None:
            self._step("pause", self._end_pause)
        self._step("tunnel", lambda: _stop_process(self.tunnel))
        self._step("helper", lambda: _stop_process(self.helper))
        if self.holder and not _cancel_job(self.holder):
            self.problems.append(f"could not cancel hold job {self.holder}")


def _new_id() -> str:
    return f"ai-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def _save(record: dict) -> None:
    AI_DIR.mkdir(parents=True, exist_ok=True)
    path = AI_DIR / f"{record['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def ask(question: str, *, threads: int | None = None, split: str | None = None, overrides: str | None = None,
        words: int | None = None, pause: bool = True, hold: bool = True, say: Callable[[str], None] = lambda _: None,
        echo: Callable[[str], None] | None = None) -> dict:
    """Put one question to the pool. Raises ConfigError on bad settings; every
    other outcome, a refusal included, comes back as a saved record."""
    cfg = load_settings(threads, split, overrides, words)
    e = cfg["engine"]
    AI_DIR.mkdir(parents=True, exist_ok=True)
    rid = _new_id()
    folder = AI_DIR / rid
    folder.mkdir()
    record: dict = {
        "id": rid, "kind": "ai", "state": "running", "question": question, "started_at": time.time(),
        "words_asked": int(e["words"]),
        "settings": {key: e[key] for key in ("threads", "context", "layers", "devices", "split", "overrides", "args", "env")},
        "machines": {"engine": e["host"], "helper": has_helper(cfg), "windows": cfg["windows"].get("host", "")},
    }
    _save(record)
    pool = _Pool(cfg, folder, say)
    state, reason, rc = "failed", None, None

    def interrupt(signum, _frame):
        raise _Interrupted(signal.Signals(signum).name)

    saved = {sig: signal.signal(sig, interrupt) for sig in _SIGNALS}
    lock = (AI_DIR / ".lock").open("w")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused("another question is running: one at a time")
        pool.preflight()
        if hold and e["worker"]:
            pool.hold_machine()
        if has_helper(cfg):
            say("starting the GPU helper and its tunnel")
            pool.start_helper()
            pool.start_tunnel()
        if pause and has_windows(cfg) and cfg["windows"]["pause"]:
            pool.pause_apps()
        pool.wait_for_memory()
        say("loading the model and answering; a big model takes minutes to start")
        stream = AnswerStream(e["thinking_end"], echo) if echo else None
        rc = pool.run_engine(question, int(e["words"]), stream.feed if stream else None)
        if stream:
            stream.finish()
        if pool.stop_reason:
            state, reason = "stopped", pool.stop_reason
        elif rc == 0:
            state = "answered"
        else:
            state, reason = "failed", f"the engine exited {rc}"
    except Refused as exc:
        state, reason = "refused", str(exc)
    except _Interrupted as exc:
        state, reason = "stopped", f"interrupted by {exc}"
    except Exception as exc:  # the record and the cleanup still have to happen
        state, reason = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        for sig in _SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        pool.close()
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        lock.close()
    log = _read(folder / "engine.log")
    thinking, answer = split_answer(_read(folder / "answer.txt") or "", e["thinking_end"])
    if state == "answered" and e["thinking_end"] and not answer:
        state, reason = "failed", "the word limit ran out before the answer: ask again with a higher --words"
    record.update({
        "state": state, "reason": reason, "exit": rc, "ended_at": time.time(),
        "thinking": thinking, "answer": answer, "timings": parse_timings(log), "memory": memory_lines(log),
        "start_free_mb": pool.start_free, "lowest_free_mb": pool.lowest_free, "paused": pool.paused,
        "resumed": pool.resumed, "hold_job": pool.holder, "files_used": pool.fingerprint, "problems": pool.problems,
        "command": engine_command(cfg, question, int(e["words"])),
        "logs": {name: _digest(folder / name) for name in ("answer.txt", "engine.log")},
    })
    record = receipts.sign(record)
    _save(record)
    return record


def resume_apps() -> bool:
    """Bring the paused apps back and confirm it. Safe at any time."""
    cfg = load_settings()
    if not has_windows(cfg) or not cfg["windows"]["pause"]:
        return True
    pool = _Pool(cfg, AI_DIR, lambda _: None)
    pool._end_pause()
    return bool(pool.resumed)


def report(record: dict) -> list[str]:
    timings = record.get("timings") or {}
    took = record.get("ended_at", 0) - record.get("started_at", 0)
    lines = [f"{record['state']} in {took:.0f} s" + (f": {record['reason']}" if record.get("reason") else "")]
    if "ms_per_word" in timings:
        lines.append(f"{timings['ms_per_word']:.1f} ms per word, {timings.get('words', '?')} words, "
                     f"prompt {timings.get('prompt_seconds', '?')} s")
    if record.get("lowest_free_mb") is not None:
        lines.append(f"lowest free memory {record['lowest_free_mb']} MB")
    if record.get("resumed") is not None:
        lines.append("sync apps back" if record["resumed"] else "SYNC APPS NOT CONFIRMED BACK")
    lines += [f"problem: {problem}" for problem in record.get("problems") or []]
    lines.append(f"record {record['id']}")
    return lines


def find(prefix: str) -> dict | None:
    matches = sorted(AI_DIR.glob(f"{prefix}*.json"), reverse=True) if AI_DIR.exists() else []
    if not matches:
        return None
    return json.loads(matches[0].read_text())


def check(record: dict) -> list[str]:
    """What's wrong with a kept record: the seal, then the files it names."""
    problems = [] if receipts.verify(record) else ["seal does not verify"]
    folder = AI_DIR / record["id"]
    for name, digest in (record.get("logs") or {}).items():
        if digest is not None and _digest(folder / name) != digest:
            problems.append(f"{name} changed or missing")
    return problems


def recent(limit: int = 20) -> list[dict]:
    if not AI_DIR.exists():
        return []
    out = []
    for path in sorted(AI_DIR.glob("ai-*.json"), reverse=True)[:limit]:
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


_PROBE = """\
import glob, json, os, platform, shutil, subprocess
facts = dict(system=platform.system(), arch=platform.machine(), threads=os.cpu_count(),
             memory_mb=os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // (1 << 20),
             wsl='microsoft' in platform.release().lower(), gpus=[])
for tool in ('nvidia-smi', '/usr/lib/wsl/lib/nvidia-smi'):
    if shutil.which(tool):
        out = subprocess.run([tool, '-L'], capture_output=True, text=True).stdout
        facts['gpus'] = [line.split(' (UUID')[0] for line in out.splitlines() if line.startswith('GPU')]
        break
if not facts['gpus'] and shutil.which('system_profiler'):
    out = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True).stdout
    facts['gpus'] = [line.split(':', 1)[1].strip() for line in out.splitlines() if 'Chipset Model' in line]
folder = os.path.expanduser({folder!r})
facts['engine'] = os.access(os.path.join(folder, {binary!r}), os.X_OK)
shards = glob.glob(os.path.expanduser({pattern!r})) if {pattern!r} else []
facts['model_files'], facts['model_gb'] = len(shards), round(sum(map(os.path.getsize, shards)) / 1e9, 1)
place = shards[0] if shards else (folder if os.path.isdir(folder) else os.path.expanduser('~'))
facts['disk_free_gb'] = round(shutil.disk_usage(place).free / 1e9, 1)
print(json.dumps(facts))
"""

_PROBE_WINDOWS = """\
$ProgressPreference = 'SilentlyContinue'
Get-CimInstance Win32_VideoController | ForEach-Object { "gpu $($_.Name)" }
"memory_mb $([int]((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1024))"
"disk_free_gb $([math]::Round((Get-PSDrive C).Free / 1e9, 1))"
"""


def _row(rows: list[dict], machine: str, what: str, ok: bool, detail: str) -> None:
    rows.append({"machine": machine, "what": what, "ok": ok, "detail": detail})


def _facts_rows(rows: list[dict], machine: str, facts: dict) -> None:
    _row(rows, machine, "machine", True,
         f"{facts['system']} {facts['arch']}, {facts['threads']} threads, {facts['memory_mb']} MB")
    _row(rows, machine, "GPUs", True, "; ".join(facts["gpus"]) or "none found")


def probe(cfg: dict) -> list[dict]:
    """Check each machine in the settings. Reads only: it changes nothing."""
    rows: list[dict] = []
    e = cfg["engine"]
    pattern = model_pattern(e["model"])
    script = _PROBE.format(folder=e["folder"], binary=e["binary"], pattern=pattern)
    result = _remote(e["host"], "python3 -I -", 120, script)
    try:
        facts = json.loads(result.stdout)
    except json.JSONDecodeError:
        _row(rows, e["host"], "reachable", False, (result.stderr or "no answer").strip()[-120:])
    else:
        _facts_rows(rows, e["host"], facts)
        _row(rows, e["host"], "engine", facts["engine"], f"{e['folder']}/{e['binary']}")
        _row(rows, e["host"], "model", facts["model_files"] > 0,
             f"{facts['model_files']} files, {facts['model_gb']} GB (downloading it is yours to do)")
        note = " (virtual disk: the Windows figure below is the real limit)" if facts["wsl"] else ""
        _row(rows, e["host"], "free disk", True, f"{facts['disk_free_gb']} GB{note}")
    if has_helper(cfg):
        h = cfg["helper"]
        script = _PROBE.format(folder=h["folder"], binary=h["binary"], pattern="")
        local = subprocess.run([sys.executable, "-I", "-"], input=script, capture_output=True, text=True)
        try:
            facts = json.loads(local.stdout)
        except json.JSONDecodeError:
            _row(rows, "this machine", "helper", False, local.stderr.strip()[-120:])
        else:
            _facts_rows(rows, "this machine", facts)
            _row(rows, "this machine", "helper", facts["engine"], f"{h['folder']}/{h['binary']}")
    if has_windows(cfg):
        win = cfg["windows"]["host"]
        read = _remote(win, _powershell(_PROBE_WINDOWS), 60)
        lines = read.stdout.replace("\x00", "").replace("\r", "").splitlines()
        if not any(line.startswith("memory_mb ") for line in lines):
            _row(rows, win, "reachable", False, (read.stderr or "no answer").strip()[-120:])
        else:
            for line in lines:
                key, _, value = line.partition(" ")
                _row(rows, win, {"gpu": "GPU", "memory_mb": "memory MB", "disk_free_gb": "free disk GB"}.get(key, key),
                     True, value)
    if e["worker"]:
        known = e["worker"] in {worker.name for worker in config.load_workers()}
        _row(rows, e["worker"], "Orchard worker", known, "known" if known else "not found: check `orchard workers`")
    else:
        _row(rows, e["host"], "Orchard worker", True, "not set, so jobs can land on the machine during a run")
    return rows


def write_template() -> bool:
    """Write a settings file to fill in. Never overwrites one."""
    if CONFIG_PATH.exists():
        return False
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(TEMPLATE)
    return True
