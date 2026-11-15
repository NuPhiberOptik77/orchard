"""Keep a model ready on your own machines for the tools you already use.

`orchard ai serve` listens on this machine's loopback only. The first question
loads the model the way `orchard ai ask` does: it holds the engine machine in
Orchard's queue, starts the GPU helper and its tunnel, pauses the sync apps and
waits for memory. Then the model stays loaded, so later questions skip all of
that. Everything is put back when nobody has asked for a while, when someone
starts using the machine, or when memory runs low; the next question loads it
again.

The engine is llama.cpp's own server. It listens on a socket file on the engine
machine, reached through the ssh link Orchard already uses, so no network port
opens anywhere. OpenAI Chat, OpenAI Responses and Claude Messages requests pass
straight through to it, and Ollama requests are translated. Every answer leaves
a sealed record, like `orchard ai ask`.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import http.client
import importlib.metadata
import json
import os
import re
import secrets
import select
import shlex
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterator

from . import ai, receipts
from .config import ORCHARD_HOME

SOCKET_DIR = ORCHARD_HOME / "run"
HEALTH_POLL = 2
SOCKET_WAIT = 60
FORWARD_WAIT = 20
KEEPALIVE_GAP = 10
RETRY_AFTER = 60
MAX_BODY = 16 << 20
try:
    VERSION = importlib.metadata.version("orchard")
except importlib.metadata.PackageNotFoundError:
    VERSION = "0"

# Flags that only mean something to the one-question engine.
_COMPLETION_ONLY = {"-no-cnv", "--no-conversation", "-cnv", "--conversation", "-st", "--single-turn",
                    "--no-display-prompt", "-i", "--interactive", "-if", "--interactive-first",
                    "--verbose-prompt", "-mli", "--multiline-input", "--simple-io"}
_COMPLETION_VALUE = {"-p", "--prompt", "-f", "--file", "-sys", "--system-prompt", "-sysf", "--system-prompt-file",
                     "-n", "--predict", "--n-predict", "-r", "--reverse-prompt", "--in-prefix", "--in-suffix"}
_LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
_PASS_THROUGH = {"/v1/chat/completions": "openai-chat", "/v1/completions": "openai-completions",
                 "/v1/responses": "openai-responses", "/v1/messages": "claude-messages",
                 "/v1/messages/count_tokens": "claude-count-tokens"}
_OLLAMA_OPTIONS = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "seed": "seed",
                   "num_predict": "max_tokens", "stop": "stop", "repeat_penalty": "repeat_penalty", "min_p": "min_p",
                   "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty"}


class NotLoaded(Exception):
    """The model could not be loaded for this question."""

    def __init__(self, reason: str, refused: bool = True):
        super().__init__(reason)
        self.refused = refused


def load_settings() -> dict:
    """`orchard ai`'s settings, with the [server] section checked."""
    cfg = ai.load_settings()
    s = cfg["server"]
    for key in ("port", "context"):
        if not isinstance(s[key], int) or isinstance(s[key], bool) or s[key] < 0:
            raise ai.ConfigError(f"{ai.settings_path()}: server.{key} must be a whole number")
    for key in ("keep_alive_seconds", "max_loaded_seconds"):
        if not isinstance(s[key], (int, float)) or isinstance(s[key], bool) or s[key] < 0:
            raise ai.ConfigError(f"{ai.settings_path()}: server.{key} must be a number of seconds")
    if s["port"] > 65535:
        raise ai.ConfigError(f"{ai.settings_path()}: server.port must be under 65536")
    if not isinstance(s["origins"], list) or not all(isinstance(o, str) for o in s["origins"]):
        raise ai.ConfigError(f"{ai.settings_path()}: server.origins must be a list of web origins")
    if not s["socket"] or not str(s["socket"]).startswith(("/", "~/")) or not str(s["socket"]).endswith(".sock"):
        raise ai.ConfigError(f"{ai.settings_path()}: server.socket must be a full path ending in .sock")
    return cfg


def model_name(cfg: dict) -> str:
    """What tools see as the model's name: the setting, or the model's file name."""
    if cfg["server"]["name"]:
        return str(cfg["server"]["name"])
    base = os.path.basename(cfg["engine"]["model"])
    return re.sub(r"(-\d{5}-of-\d{5})?\.gguf$", "", base)


def server_args(args: list) -> list[str]:
    """The engine's settings, less the flags that only the one-question engine takes."""
    out, skip = [], False
    for arg in map(str, args):
        if skip:
            skip = False
        elif arg in _COMPLETION_VALUE:
            skip = True
        elif arg not in _COMPLETION_ONLY:
            out.append(arg)
    return out


SOCKET_ARG = "\0socket"


def server_command(cfg: dict) -> list[str]:
    e, s = cfg["engine"], cfg["server"]
    argv = [f"./{s['binary']}", "-m", e["model"], "--host", SOCKET_ARG, "-c", str(s["context"] or e["context"]),
            "-t", str(e["threads"]), "-np", "1", "--no-webui", "--cache-ram", "0",
            "--timeout", str(int(e["cap_seconds"]) + 60), "--alias", model_name(cfg)]
    if rpc := ai.rpc_list(cfg):
        argv += ["--rpc", rpc]
    for flag, key in (("--device", "devices"), ("-ngl", "layers"), ("-ts", "split"), ("-ot", "overrides")):
        if e[key] != "":
            argv += [flag, str(e[key])]
    return argv + server_args(e["args"]) + [str(a) for a in s["args"]]


def server_script(cfg: dict) -> str:
    """The script the engine machine runs. It starts the engine on a socket
    file, says where, and stops the engine as soon as this ssh session ends,
    however it ends, so a lost link never leaves the model loaded."""
    e, s = cfg["engine"], cfg["server"]

    def word(i: int, arg: str) -> str:
        if arg == SOCKET_ARG:
            return '"$sock"'
        return ai._rpath(arg) if i == 2 else shlex.quote(arg)

    command = " ".join(word(i, a) for i, a in enumerate(server_command(cfg)))
    lines = [f"cd {ai._rpath(e['folder'])} || exit 127"]
    lines += [f"export {key}={shlex.quote(str(value))}" for key, value in e["env"].items()]
    lines += ["umask 077", f"sock={ai._rpath(str(s['socket']))}", 'mkdir -p "$(dirname "$sock")"', 'rm -f "$sock"']
    lines += ai.give_back_lines(cfg)
    lines += [
        "exec 3<&0",
        f"{command} < /dev/null 3<&- 1>&2 &",
        "pid=$!",
        'echo "socket $sock"',
        # A builtin read, so killing the watcher leaves nothing behind.
        "( while read -r _ <&3; do :; done; kill $pid 2> /dev/null ) &",
        "watch=$!",
        "wait $pid",
        "rc=$?",
        "kill $watch 2> /dev/null",
        'rm -f "$sock"',
        "give_back",
        "exit $rc",
    ]
    return "\n".join(lines) + "\n"


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.unix_path)
        self.sock = sock


class _ServerPool(ai._Pool):
    """One loading of the model, with the engine's socket forwarded here."""

    def __init__(self, cfg: dict, folder: Path, say: Callable[[str], None]):
        super().__init__(cfg, folder, say)
        self.forward: subprocess.Popen | None = None
        self.local_socket = SOCKET_DIR / "ai-engine.sock"
        self.remote_socket: str | None = None

    def start_server(self) -> None:
        log = (self.folder / "engine.log").open("wb")
        self.engine = subprocess.Popen([*ai._ssh(self.host), "bash -c " + shlex.quote(server_script(self.cfg))],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        log.close()
        end = time.time() + SOCKET_WAIT
        line = b""
        while time.time() < end and not line.endswith(b"\n"):
            ready, _, _ = select.select([self.engine.stdout], [], [], 1)
            if ready:
                byte = os.read(self.engine.stdout.fileno(), 1)
                if not byte:
                    break
                line += byte
        text = line.decode(errors="replace").strip()
        if not text.startswith("socket /"):
            raise NotLoaded(f"the engine did not start on {self.host}: see {self.folder / 'engine.log'}", False)
        self.remote_socket = text[len("socket "):]

    def start_forward(self) -> None:
        SOCKET_DIR.mkdir(parents=True, exist_ok=True)
        ai._pkill_local(f"^ssh .*-L {self.local_socket}:")  # a link left by a server that was killed
        with contextlib.suppress(FileNotFoundError):
            self.local_socket.unlink()
        argv = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-o", "StreamLocalBindUnlink=yes", "-o", "StreamLocalBindMask=0177",
                "-L", f"{self.local_socket}:{self.remote_socket}", self.host]
        self.forward = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
        end = time.time() + FORWARD_WAIT
        while time.time() < end:
            if why := self.alive():
                raise NotLoaded(f"{why}: see {self.folder / 'engine.log'}", False)
            if os.path.lexists(self.local_socket):
                return
            time.sleep(0.1)
        raise NotLoaded(f"no link to the engine on {self.host}", False)

    def health(self) -> int | None:
        """200 when the model is loaded, 503 while it loads, None if unreachable."""
        conn = _UnixConnection(str(self.local_socket), 10)
        try:
            conn.request("GET", "/health")
            return conn.getresponse().status
        except OSError:
            return None
        finally:
            conn.close()

    def alive(self) -> str | None:
        """Why the loaded engine can no longer answer, or None."""
        if self.engine is not None and self.engine.poll() is not None:
            return "the engine stopped by itself"
        if self.forward is not None and self.forward.poll() is not None:
            return "the link to the engine closed"
        return None

    def close(self) -> None:
        # Ending the session stops the engine on the far side straight away.
        if self.engine is not None and self.engine.stdin and not self.engine.stdin.closed:
            self._step("engine", self.engine.stdin.close)
        super().close()
        self._step("link", lambda: ai._stop_process(self.forward))
        with contextlib.suppress(OSError):
            self.local_socket.unlink()


def _session_id() -> str:
    return f"ais-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class Engine:
    """The model's life: idle, loading, loaded, and back. One question at a time."""

    def __init__(self, cfg: dict, say: Callable[[str], None]):
        self.cfg = cfg
        self.server_cfg = copy.deepcopy(cfg)
        self.server_cfg["engine"]["binary"] = cfg["server"]["binary"]
        self.say = say
        self.name = model_name(cfg)
        self.state = "idle"
        self.pool: _ServerPool | None = None
        self.session: dict | None = None
        self.loaded_at: float | None = None
        self.change = threading.RLock()
        self.turn = threading.Lock()
        self.count_lock = threading.Lock()
        self.waiting = 0
        self.last_used = time.time()
        self.request_started: float | None = None
        self.active: http.client.HTTPConnection | None = None
        self.cancel_reason: str | None = None
        self.stop_reason: str | None = None
        self.guard: threading.Thread | None = None
        self.guard_done = threading.Event()
        self.flock = None

    @property
    def keep_alive(self) -> float:
        return float(self.cfg["server"]["keep_alive_seconds"])

    @contextlib.contextmanager
    def use(self) -> Iterator[bool]:
        """Wait for the engine, load it if needed, and hold it for one question.
        Yields whether this question had to load the model."""
        with self.count_lock:
            self.waiting += 1
        try:
            with self.turn:
                fresh = self.ensure_loaded()
                self.request_started, self.cancel_reason = time.time(), None
                try:
                    yield fresh
                finally:
                    self.request_started, self.active = None, None
                    self.last_used = time.time()
        finally:
            with self.count_lock:
                self.waiting -= 1
            self.last_used = time.time()

    def connect(self) -> http.client.HTTPConnection:
        conn = _UnixConnection(str(self.pool.local_socket), float(self.cfg["engine"]["cap_seconds"]) + 60)
        self.active = conn
        return conn

    def ensure_loaded(self) -> bool:
        with self.change:
            if self.state == "loaded" and self.pool is not None:
                why = self.pool.alive()
                if why is None:
                    return False
                self.unload(why)
            self._load()
            return True

    def _load(self) -> None:
        e, s = self.cfg["engine"], self.cfg["server"]
        sid = _session_id()
        folder = ai.AI_DIR / sid
        folder.mkdir(parents=True)
        longest = int(s["max_loaded_seconds"])
        self.session = {
            "id": sid, "kind": "ai-session", "state": "loading", "model": self.name, "started_at": time.time(),
            "settings": {key: e[key] for key in ("threads", "layers", "devices", "split", "overrides", "args", "env")}
            | {"context": s["context"] or e["context"], "keep_alive_seconds": s["keep_alive_seconds"]},
            "machines": {"engine": e["host"], "helper": ai.has_helper(self.cfg),
                         "windows": self.cfg["windows"].get("host", ""),
                         "far_helpers": [h["host"] for h in self.cfg.get("far_helpers", [])]},
            "command": server_command(self.server_cfg), "answers": [],
        }
        if self.cfg.get("layout"):
            self.session["layout"] = self.cfg["layout"]
        ai._save(self.session)
        self.state, self.stop_reason = "loading", None
        self.pool = _ServerPool(self.server_cfg, folder, self.say)
        self.say(f"loading {self.name} for the first question; a big model takes minutes")
        self.flock = (ai.AI_DIR / ".lock").open("w")
        try:
            try:
                fcntl.flock(self.flock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ai.Refused("another question is running: one at a time")
            self.pool.preflight()
            if e["worker"]:
                self.pool.hold_machine(longest + 1800)
            if ai.has_helper(self.cfg):
                self.say("starting the GPU helper and its tunnel")
                self.pool.start_helper()
                self.pool.start_tunnel()
            if self.cfg.get("far_helpers"):
                self.say("starting the helpers on " + ", ".join(h["host"] for h in self.cfg["far_helpers"]))
                self.pool.start_far_helpers()
            if ai.has_windows(self.cfg) and self.cfg["windows"]["pause"]:
                self.pool.pause_apps(longest + 1200)
            self.pool.wait_for_memory()
            # Each loading has its own guard and its own stop signal, so a guard
            # still finishing a slow check can never run beside the next one.
            self.guard_done = threading.Event()
            self.guard = threading.Thread(target=self._watch, args=(self.guard_done, self.pool), daemon=True)
            self.guard.start()
            self.pool.start_server()
            self.pool.start_forward()
            self._wait_until_loaded()
        except ai.Refused as exc:
            self._give_back("refused", str(exc))
            raise NotLoaded(str(exc))
        except NotLoaded as exc:
            self._give_back("failed", self.stop_reason or str(exc))
            raise NotLoaded(self.stop_reason or str(exc), exc.refused and not self.stop_reason)
        except Exception as exc:  # the cleanup still has to happen
            self._give_back("failed", f"{type(exc).__name__}: {exc}")
            raise NotLoaded(f"{type(exc).__name__}: {exc}", False)
        self.loaded_at = time.time()
        self.state = "loaded"
        self.session.update({"state": "loaded", "loaded_at": self.loaded_at,
                             "load_seconds": round(self.loaded_at - self.session["started_at"], 1),
                             "start_free_mb": self.pool.start_free, "files_used": self.pool.fingerprint})
        ai._save(self.session)
        self.say(f"loaded in {self.session['load_seconds']:.0f} s; it stays loaded until "
                 f"{self.keep_alive:.0f} s pass without a question")

    def _wait_until_loaded(self) -> None:
        end = time.time() + float(self.cfg["engine"]["cap_seconds"])
        while True:
            if self.stop_reason:
                raise NotLoaded(self.stop_reason, False)
            if why := self.pool.alive():
                raise NotLoaded(f"{why}: see {self.pool.folder / 'engine.log'}", False)
            if self.pool.health() == 200:
                return
            if time.time() > end:
                raise NotLoaded("the model took too long to load", False)
            time.sleep(HEALTH_POLL)

    def _why_unload(self, reads: dict) -> tuple[str, bool] | None:
        """Why to put everything back now, and whether it's urgent (a question
        in progress is stopped) or can wait until nobody is waiting."""
        if why := self.pool._why_stop(reads, float("inf")):
            return why, True
        if self.pool.holder:
            entry = ai._job_entry(self.pool.holder) or {}
            if entry and entry.get("state") not in ("queued", "running", "paused"):
                return f"the hold on {self.cfg['engine']['worker']} ended: {entry.get('state')}", True
        if self.state != "loaded":
            return None
        if why := self.pool.alive():
            return why, True
        started = self.request_started
        if started and time.time() - started > float(self.cfg["engine"]["cap_seconds"]):
            self._cancel("the question passed its time limit")
        if self.waiting == 0 and time.time() - self.last_used > self.keep_alive:
            return f"nobody asked for {self.keep_alive:.0f} s", False
        if self.waiting == 0 and time.time() - self.loaded_at > float(self.cfg["server"]["max_loaded_seconds"]):
            return "loaded for the longest allowed; the next question loads it again", False
        return None

    def _cancel(self, reason: str) -> None:
        """Stop the question in progress but keep the model loaded."""
        conn = self.active
        if conn is not None and conn.sock is not None and self.cancel_reason is None:
            self.cancel_reason = reason
            self.say(f"stopping the question: {reason}")
            with contextlib.suppress(OSError):
                conn.sock.shutdown(socket.SHUT_RDWR)

    def _watch(self, done: threading.Event, pool: _ServerPool) -> None:
        reads = {"low": 0, "unread": 0, "paused": 0}
        while not done.wait(ai.GUARD_POLL):
            found = self._why_unload(reads)
            if done.is_set() or self.pool is not pool:
                return
            if found is None:
                continue
            why, urgent = found
            if not urgent and not self.turn.acquire(blocking=False):
                continue
            try:
                if done.is_set() or self.pool is not pool:
                    return
                self.stop_reason = why
                self.unload(why, from_guard=True)
            finally:
                if not urgent:
                    self.turn.release()
            return

    def unload(self, reason: str, from_guard: bool = False) -> None:
        """Put everything back. Safe to call at any time, from any thread."""
        if self.state == "loading" and self.pool is not None:
            # The loader holds `change` and cleans up once it sees the stop.
            self.stop_reason = self.stop_reason or reason
            self.say(f"stopping the load: {reason}")
            self.pool.stopping.set()
            if self.pool.engine is not None:
                self.pool._kill_engine()
            if from_guard:
                return
        with self.change:
            if self.state != "loaded":
                return
            self.stop_reason = self.stop_reason or reason
            self.say(f"unloading: {reason}")
            self._give_back("served", reason)

    def _give_back(self, state: str, reason: str) -> None:
        self.guard_done.set()
        if self.guard is not None and self.guard is not threading.current_thread():
            self.guard.join(5)
        self.state = "unloading"
        if self.pool is not None:
            try:
                self.pool.close()
            except Exception as exc:  # the record and the idle state still have to happen
                self.pool.problems.append(f"{type(exc).__name__}: {exc}")
        if self.flock is not None:
            self.flock.close()
            self.flock = None
        if self.session is not None and self.pool is not None:
            self.session.update({
                "state": state, "reason": reason, "ended_at": time.time(),
                "lowest_free_mb": self.pool.lowest_free, "start_free_mb": self.pool.start_free,
                "paused": self.pool.paused, "resumed": self.pool.resumed, "hold_job": self.pool.holder,
                "files_used": self.pool.fingerprint, "problems": self.pool.problems,
                "logs": {"engine.log": ai._digest(self.pool.folder / "engine.log")},
            })
            self.session = receipts.sign(self.session)
            ai._save(self.session)
            for problem in self.pool.problems:
                self.say(f"problem: {problem}")
            if state == "served":
                self.say("everything put back")
        self.state, self.loaded_at = "idle", None

    def status(self) -> dict:
        out = {"status": self.state, "model": self.name, "waiting": self.waiting}
        if self.session and self.state in ("loading", "loaded"):
            out["session"] = self.session["id"]
        if self.loaded_at:
            out["loaded_for_seconds"] = round(time.time() - self.loaded_at)
            out["idle_for_seconds"] = 0 if self.waiting else round(time.time() - self.last_used)
        return out

    def note_answer(self, record_id: str) -> None:
        if self.session is not None and "answers" in self.session and "seal" not in self.session:
            self.session["answers"].append(record_id)


# What came back, in whichever API's shape.

def stream_piece(obj: dict) -> tuple[str, str]:
    """(thinking, answer) text in one streamed event of any of the APIs."""
    if obj.get("choices"):
        choice = obj["choices"][0] or {}
        delta = choice.get("delta") or {}
        return delta.get("reasoning_content") or "", delta.get("content") or choice.get("text") or ""
    kind = obj.get("type")
    if kind == "content_block_delta":
        delta = obj.get("delta") or {}
        return delta.get("thinking") or "", delta.get("text") or ""
    if kind == "response.output_text.delta":
        return "", obj.get("delta") or ""
    if kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
        return obj.get("delta") or "", ""
    return "", ""


def whole_texts(obj: dict) -> tuple[str, str]:
    """(thinking, answer) in a whole reply of any of the APIs."""
    if obj.get("choices"):
        choice = obj["choices"][0] or {}
        message = choice.get("message") or {}
        return message.get("reasoning_content") or "", message.get("content") or choice.get("text") or ""
    if obj.get("type") == "message":
        blocks = obj.get("content") or []
        return ("".join(b.get("thinking") or "" for b in blocks if b.get("type") == "thinking"),
                "".join(b.get("text") or "" for b in blocks if b.get("type") == "text"))
    if obj.get("object") == "response":
        thinking, answer = [], []
        for item in obj.get("output") or []:
            parts = [part.get("text") or "" for part in item.get("content") or []]
            if item.get("type") == "reasoning":
                thinking += parts
            elif item.get("type") == "message":
                answer += parts
        return "".join(thinking), "".join(answer)
    return "", ""


def engine_timings(obj: dict) -> dict:
    t = obj.get("timings") or {}
    out = {}
    if "predicted_n" in t:
        out["words"] = int(t["predicted_n"])
    if "predicted_per_token_ms" in t:
        out["ms_per_word"] = round(float(t["predicted_per_token_ms"]), 1)
    if "prompt_ms" in t:
        out["prompt_seconds"] = round(float(t["prompt_ms"]) / 1000, 1)
    if "prompt_n" in t:
        out["prompt_words"] = int(t["prompt_n"])
    return out


class SSE:
    """Reads server-sent events as they arrive and keeps the text they carry."""

    def __init__(self) -> None:
        self.buffer = b""
        self.thinking: list[str] = []
        self.answer: list[str] = []
        self.timings: dict = {}
        self.raw_timings: dict = {}
        self.finish: str | None = None
        self.usage: dict = {}
        self.error: str | None = None
        self.first_word_at: float | None = None

    def feed(self, data: bytes) -> list[dict]:
        self.buffer += data
        events = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            thinking, answer = stream_piece(obj)
            if (thinking or answer) and self.first_word_at is None:
                self.first_word_at = time.time()
            self.thinking.append(thinking)
            self.answer.append(answer)
            self.timings |= engine_timings(obj)
            self.raw_timings |= obj.get("timings") or {}
            if obj.get("usage"):
                self.usage = obj["usage"]
            if obj.get("choices") and (obj["choices"][0] or {}).get("finish_reason"):
                self.finish = obj["choices"][0]["finish_reason"]
            if obj.get("error"):
                self.error = str(obj["error"])
            events.append(obj)
        return events


def question_text(api: str, req: dict) -> str:
    """The question in plain words, for the record and `orchard ai list`."""
    def flat(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(part.get("text") or "" for part in content if isinstance(part, dict))
        return ""

    if api in ("openai-completions", "ollama-generate"):
        return flat(req.get("prompt") or "")
    if api == "openai-responses":
        given = req.get("input")
        if isinstance(given, list) and given:
            last = given[-1]
            return flat(last.get("content") if isinstance(last, dict) else last)
        return flat(given or "")
    users = [m for m in req.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
    return flat(users[-1].get("content")) if users else ""


# Ollama's API, over the engine's OpenAI one.

def _iso(when: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(when)) + f".{int(when % 1 * 1e6):06d}Z"


def ollama_to_openai(req: dict, generate: bool) -> dict:
    """An Ollama chat or generate request as an OpenAI chat request."""
    if generate:
        messages = ([{"role": "system", "content": req["system"]}] if req.get("system") else [])
        messages.append({"role": "user", "content": req.get("prompt") or "", **({"images": req["images"]}
                                                                                 if req.get("images") else {})})
    else:
        messages = list(req.get("messages") or [])
    out_messages = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        if message.get("images"):
            raise ValueError("this model takes text only")
        item = {"role": message.get("role") or "user", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            item["tool_calls"] = [{"id": f"call_{i}_{j}", "type": "function",
                                   "function": {"name": (call.get("function") or {}).get("name", ""),
                                                "arguments": json.dumps((call.get("function") or {}).get("arguments")
                                                                        or {})}}
                                  for j, call in enumerate(message["tool_calls"])]
        if message.get("tool_name"):
            item["name"] = message["tool_name"]
        out_messages.append(item)
    body: dict = {"messages": out_messages, "stream": bool(req.get("stream", True))}
    for key, value in (req.get("options") or {}).items():
        if key in _OLLAMA_OPTIONS:
            body[_OLLAMA_OPTIONS[key]] = value
    fmt = req.get("format")
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": fmt}}
    if req.get("tools"):
        body["tools"] = req["tools"]
    if req.get("think") is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def ollama_tool_calls(calls: list[dict]) -> list[dict]:
    out = []
    for call in calls:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            arguments = {}
        out.append({"function": {"name": function.get("name", ""), "arguments": arguments}})
    return out


def ollama_counts(timings: dict, usage: dict, took: float, load: float) -> dict:
    t = timings or {}
    return {
        "total_duration": int(took * 1e9), "load_duration": int(load * 1e9),
        "prompt_eval_count": int(t.get("prompt_n", usage.get("prompt_tokens", 0))),
        "prompt_eval_duration": int(float(t.get("prompt_ms", 0)) * 1e6),
        "eval_count": int(t.get("predicted_n", usage.get("completion_tokens", 0))),
        "eval_duration": int(float(t.get("predicted_ms", 0)) * 1e6),
    }


def _done_reason(finish: str | None) -> str:
    return "length" if finish == "length" else "stop"


class Answer:
    """The sealed record of one question put through the server."""

    def __init__(self, engine: Engine, api: str, body: bytes, req: dict):
        self.engine = engine
        self.id = ai._new_id()
        self.folder = ai.AI_DIR / self.id
        self.folder.mkdir(parents=True)
        (self.folder / "request.json").write_bytes(body)
        self.raw = (self.folder / "answer.txt").open("wb")
        self.started = time.time()
        self.sse: SSE | None = None  # a streamed answer's reader, for when the first word came
        self.record: dict = {
            "id": self.id, "kind": "ai", "via": "server", "api": api, "state": "running",
            "question": question_text(api, req), "model_asked": str(req.get("model") or ""),
            "started_at": self.started, "machines": copy.deepcopy(engine.session["machines"])
            if engine.session else {},
        }

    def write(self, data: bytes) -> None:
        self.raw.write(data)

    def finish(self, state: str, reason: str | None, thinking: str, answer: str, timings: dict,
               fresh: bool = False) -> dict:
        if "seal" in self.record:  # already finished: a tool hanging up afterwards changes nothing
            return self.record
        self.raw.close()
        engine = self.engine
        session = engine.session or {}
        if not thinking and engine.cfg["engine"]["thinking_end"] and engine.cfg["engine"]["thinking_end"] in answer:
            thinking, answer = ai.split_answer(answer, engine.cfg["engine"]["thinking_end"])
        self.record.update({
            "state": state, "reason": reason, "ended_at": time.time(), "thinking": thinking.strip(),
            "answer": answer.strip(), "timings": timings, "session": session.get("id"),
            "loaded_for_this_question": fresh, "load_seconds": session.get("load_seconds") if fresh else 0,
            "first_word_seconds": round(first - self.started, 1) if (first := self.sse and self.sse.first_word_at)
            else None,
            "files_used": session.get("files_used") or {}, "settings": session.get("settings") or {},
            "problems": [],
            "logs": {name: ai._digest(self.folder / name) for name in ("request.json", "answer.txt")},
        })
        if session.get("id"):
            self.record["machines"] = session.get("machines", {})
            engine.note_answer(self.id)
        self.record = receipts.sign(self.record)
        ai._save(self.record)
        speed = f", {timings['ms_per_word']:.0f} ms per word" if timings.get("ms_per_word") else ""
        engine.say(f"{state} {self.id} in {self.record['ended_at'] - self.started:.0f} s{speed}"
                   + (f": {reason}" if reason else ""))
        return self.record


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_port = False  # a second server on the same port must fail, not share it

    def __init__(self, cfg: dict, port: int, say: Callable[[str], None], engine: Engine | None = None):
        self.cfg = cfg
        self.say = say
        self.engine = engine or Engine(cfg, say)
        self.started = time.time()
        super().__init__(("127.0.0.1", port), _Handler)
        self.port = self.server_address[1]
        self.origins = {f"http://{host}:{self.port}" for host in _LOOPBACK} | set(cfg["server"]["origins"])

    def close(self, reason: str = "the server stopped") -> None:
        self.shutdown()
        self.server_close()
        self.engine.unload(reason)


class _Handler(BaseHTTPRequestHandler):
    server: Server
    protocol_version = "HTTP/1.1"
    server_version = f"orchard-ai/{VERSION}"

    def log_message(self, *_args) -> None:
        pass

    # The gate: this machine's own programs only, never a web page.

    def _allowed(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if host:
            name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
            if name not in _LOOPBACK:
                self._error(403, "only this machine's own programs may use this server")
                return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.origins:
            self._error(403, "web pages may not use this server: add the origin to server.origins to allow it")
            return False
        if (self.headers.get("Sec-Fetch-Site") or "none") not in ("none", "same-origin"):
            self._error(403, "web pages may not use this server")
            return False
        return True

    def do_HEAD(self) -> None:
        if self._allowed():
            self._send(200, b"", "text/plain")

    def do_OPTIONS(self) -> None:
        self._error(403, "web pages may not use this server")

    def do_DELETE(self) -> None:
        self._error(404, "not found")

    def do_GET(self) -> None:
        if not self._allowed():
            return
        engine, name = self.server.engine, self.server.engine.name
        if self.path == "/":
            self._send(200, b"Orchard AI is running", "text/plain")
        elif self.path in ("/health", "/v1/health"):
            self._json(200, engine.status())
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": name, "object": "model",
                                                         "created": int(self.server.started), "owned_by": "orchard"}]})
        elif self.path == "/api/version":
            self._json(200, {"version": VERSION})
        elif self.path == "/api/tags":
            self._json(200, {"models": [self._ollama_model()]})
        elif self.path == "/api/ps":
            models = []
            if engine.state == "loaded":
                expires = engine.last_used + engine.keep_alive
                models.append({**self._ollama_model(), "expires_at": _iso(expires), "size_vram": 0})
            self._json(200, {"models": models})
        else:
            self._error(404, f"not found: {self.path}")

    def do_POST(self) -> None:
        if not self._allowed():
            return
        path = self.path.split("?", 1)[0]
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            self._error(411, "send a Content-Length")
            return
        if int(length) > MAX_BODY:
            self._error(413, "the request is too large")
            return
        body = self.rfile.read(int(length))
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            self._error(400, "the request is not JSON")
            return
        if not isinstance(req, dict):
            self._error(400, "the request must be a JSON object")
            return
        if path in _PASS_THROUGH:
            self._relay(_PASS_THROUGH[path], path, body, req)
        elif path == "/api/chat":
            self._ollama(req, body, generate=False)
        elif path == "/api/generate":
            self._ollama(req, body, generate=True)
        elif path == "/api/show":
            self._json(200, self._ollama_show())
        else:
            self._error(404, f"not found: {path}")

    # Replies.

    def _send(self, status: int, data: bytes, ctype: str, extra: tuple = ()) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.close_connection = True
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, status: int, obj: dict, extra: tuple = ()) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json", extra)

    def _error(self, status: int, message: str, api: str = "", extra: tuple = ()) -> None:
        if api.startswith("ollama"):
            body = {"error": message}
        elif api.startswith("claude"):
            kind = "overloaded_error" if status == 503 else "invalid_request_error"
            body = {"type": "error", "error": {"type": kind, "message": message}}
        else:
            body = {"error": {"message": message, "type": "orchard_error", "code": status}}
        self._json(status, body, extra)

    def _start_stream(self, ctype: str, record_id: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.send_header("X-Orchard-Record", record_id)
        self.end_headers()
        self.close_connection = True

    def _chunk(self, data: bytes) -> None:
        if data:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

    def _end_stream(self) -> None:
        with contextlib.suppress(OSError):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def _stream_error(self, api: str, message: str) -> None:
        if api.startswith("claude"):
            event = {"type": "error", "error": {"type": "overloaded_error", "message": message}}
            data = b"event: error\ndata: " + json.dumps(event).encode() + b"\n\n"
        elif api.startswith("ollama"):
            data = json.dumps({"error": message}).encode() + b"\n"
        else:
            data = b"data: " + json.dumps({"error": {"message": message, "type": "orchard_error"}}).encode() + b"\n\n"
        with contextlib.suppress(OSError):
            self._chunk(data)

    @contextlib.contextmanager
    def _keepalive(self, active: bool) -> Iterator[Callable[[], None]]:
        """While the model loads or a question waits its turn, a streamed reply
        gets a comment every few seconds so the tool knows it's still alive.
        Yields the call that stops the beats before the answer is written."""
        done = threading.Event()
        lock = threading.Lock()

        def beat() -> None:
            while not done.wait(KEEPALIVE_GAP):
                with lock:
                    if done.is_set():
                        return
                    try:
                        self._chunk(b": orchard is getting the model ready\n\n")
                    except OSError:
                        return

        def stop() -> None:
            with lock:
                done.set()

        if active:
            threading.Thread(target=beat, daemon=True).start()
        try:
            yield stop
        finally:
            stop()

    # Passing a question to the engine.

    def _not_loaded(self, answer: Answer, api: str, exc: NotLoaded, streaming: bool) -> None:
        state = "refused" if exc.refused else "failed"
        answer.finish(state, str(exc), "", "", {})
        if streaming:
            self._stream_error(api, str(exc))
            self._end_stream()
        else:
            self._error(503, str(exc), api, (("Retry-After", str(RETRY_AFTER)), ("X-Orchard-Record", answer.id)))

    def _broken(self, answer: Answer, sse: SSE | None, why: str, fresh: bool, timings: dict | None = None) -> None:
        engine = self.server.engine
        reason = engine.cancel_reason or engine.stop_reason or why
        thinking = "".join(sse.thinking) if sse else ""
        text = "".join(sse.answer) if sse else ""
        answer.finish("stopped", reason, thinking, text, timings or (sse.timings if sse else {}), fresh)

    def _relay(self, api: str, path: str, body: bytes, req: dict) -> None:
        engine = self.server.engine
        streaming = bool(req.get("stream"))
        answer = Answer(engine, api, body, req)
        if streaming:
            self._start_stream("text/event-stream", answer.id)
        sse = answer.sse = SSE() if streaming else None
        fresh = False
        try:
            with self._keepalive(streaming) as quiet, engine.use() as fresh:
                conn = engine.connect()
                conn.request("POST", path, body, {"Content-Type": "application/json"})
                resp = conn.getresponse()
                quiet()
                if not streaming:
                    data = resp.read()
                    answer.write(data)
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        obj = {}
                    thinking, text = whole_texts(obj) if isinstance(obj, dict) else ("", "")
                    ok = resp.status == 200
                    record = answer.finish("answered" if ok else "failed",
                                           None if ok else f"the engine said {resp.status}", thinking, text,
                                           engine_timings(obj) if isinstance(obj, dict) else {}, fresh)
                    with contextlib.suppress(OSError):  # the answer is kept even if the tool hung up
                        self._send(resp.status, data, resp.getheader("Content-Type") or "application/json",
                                   (("X-Orchard-Record", record["id"]),))
                    return
                if resp.status != 200:
                    data = resp.read()
                    answer.write(data)
                    answer.finish("failed", f"the engine said {resp.status}", "", "", {}, fresh)
                    self._stream_error(api, data.decode(errors="replace")[:500])
                    self._end_stream()
                    return
                while chunk := resp.read1(65536):
                    answer.write(chunk)
                    sse.feed(chunk)
                    try:
                        self._chunk(chunk)
                    except OSError:
                        conn.close()
                        self._broken(answer, sse, "the tool hung up", fresh)
                        return
                state = "answered" if not sse.error else "failed"
                answer.finish(state, sse.error, "".join(sse.thinking), "".join(sse.answer), sse.timings, fresh)
                self._end_stream()
        except NotLoaded as exc:
            self._not_loaded(answer, api, exc, streaming)
        except (OSError, http.client.HTTPException) as exc:
            self._broken(answer, sse, f"lost the engine: {type(exc).__name__}", fresh)
            if streaming:
                self._stream_error(api, answer.record.get("reason") or "lost the engine")
                self._end_stream()
            else:
                with contextlib.suppress(OSError):
                    self._error(502, answer.record.get("reason") or "lost the engine", api,
                                (("X-Orchard-Record", answer.id),))

    def _ollama(self, req: dict, body: bytes, generate: bool) -> None:
        engine = self.server.engine
        api = "ollama-generate" if generate else "ollama-chat"
        try:
            openai_body = ollama_to_openai(req, generate)
        except ValueError as exc:
            self._error(400, str(exc), api)
            return
        streaming = openai_body["stream"]
        name = engine.name
        if generate and not (req.get("prompt") or "").strip() and not req.get("system"):
            # An empty generate is how Ollama tools ask for the model to be loaded.
            try:
                with engine.use():
                    pass
            except NotLoaded as exc:
                self._error(503, str(exc), api, (("Retry-After", str(RETRY_AFTER)),))
                return
            self._json(200, {"model": name, "created_at": _iso(time.time()), "response": "", "done": True,
                             "done_reason": "load"})
            return
        answer = Answer(engine, api, body, req)
        field = "response" if generate else "message"

        def piece(thinking: str, text: str) -> dict:
            if generate:
                return {"response": text, **({"thinking": thinking} if thinking else {})}
            return {"message": {"role": "assistant", "content": text, **({"thinking": thinking} if thinking else {})}}

        fresh = False
        sse = None
        try:
            with engine.use() as fresh:
                conn = engine.connect()
                conn.request("POST", "/v1/chat/completions", json.dumps(openai_body).encode(),
                             {"Content-Type": "application/json"})
                resp = conn.getresponse()
                if resp.status != 200 or not streaming:
                    data = resp.read()
                    answer.write(data)
                    if resp.status != 200:
                        answer.finish("failed", f"the engine said {resp.status}", "", "", {}, fresh)
                        self._error(resp.status, data.decode(errors="replace")[:500], api,
                                    (("X-Orchard-Record", answer.id),))
                        return
                    obj = json.loads(data)
                    thinking, text = whole_texts(obj)
                    choice = (obj.get("choices") or [{}])[0]
                    out = {"model": name, "created_at": _iso(time.time()), **piece(thinking, text), "done": True,
                           "done_reason": _done_reason(choice.get("finish_reason"))}
                    calls = (choice.get("message") or {}).get("tool_calls")
                    if calls and not generate:
                        out["message"]["tool_calls"] = ollama_tool_calls(calls)
                    out |= ollama_counts(obj.get("timings") or {}, obj.get("usage") or {},
                                         time.time() - answer.started, engine.session.get("load_seconds", 0)
                                         if fresh else 0)
                    record = answer.finish("answered", None, thinking, text, engine_timings(obj), fresh)
                    with contextlib.suppress(OSError):  # the answer is kept even if the tool hung up
                        self._json(200, out, (("X-Orchard-Record", record["id"]),))
                    return
                self._start_stream("application/x-ndjson", answer.id)
                sse = answer.sse = SSE()
                calls: dict[int, dict] = {}
                while chunk := resp.read1(65536):
                    answer.write(chunk)
                    lines = b""
                    for event in sse.feed(chunk):
                        thinking, text = stream_piece(event)
                        for call in ((event.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls") or []:
                            slot = calls.setdefault(call.get("index", 0), {"function": {"name": "", "arguments": ""}})
                            function = call.get("function") or {}
                            slot["function"]["name"] += function.get("name") or ""
                            slot["function"]["arguments"] += function.get("arguments") or ""
                        if thinking or text:
                            line = {"model": name, "created_at": _iso(time.time()), **piece(thinking, text),
                                    "done": False}
                            lines += json.dumps(line).encode() + b"\n"
                    try:
                        self._chunk(lines)
                    except OSError:
                        conn.close()
                        self._broken(answer, sse, "the tool hung up", fresh)
                        return
                if calls and not generate:
                    message = {"role": "assistant", "content": "",
                               "tool_calls": ollama_tool_calls([calls[i] for i in sorted(calls)])}
                    self._chunk(json.dumps({"model": name, "created_at": _iso(time.time()), "message": message,
                                            "done": False}).encode() + b"\n")
                final = {"model": name, "created_at": _iso(time.time()), **piece("", ""), "done": True,
                         "done_reason": _done_reason(sse.finish)}
                final |= ollama_counts(sse.raw_timings, sse.usage, time.time() - answer.started,
                                       engine.session.get("load_seconds", 0) if fresh else 0)
                self._chunk(json.dumps(final).encode() + b"\n")
                answer.finish("answered" if not sse.error else "failed", sse.error, "".join(sse.thinking),
                              "".join(sse.answer), sse.timings, fresh)
                self._end_stream()
        except NotLoaded as exc:
            self._not_loaded(answer, api, exc, False)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            self._broken(answer, sse, f"lost the engine: {type(exc).__name__}", fresh)
            with contextlib.suppress(OSError):
                if sse is not None:
                    self._stream_error(api, answer.record.get("reason") or "lost the engine")
                    self._end_stream()
                else:
                    self._error(502, answer.record.get("reason") or "lost the engine", api,
                                (("X-Orchard-Record", answer.id),))

    # Ollama's descriptions of the one model.

    def _ollama_model(self) -> dict:
        name = self.server.engine.name
        size = sum(f.get("bytes", 0) for f in ((self.server.engine.session or {}).get("files_used") or {})
                   .get("model", []))
        # Ollama tools show a digest. Hashing a model this big at every start would
        # take too long, so this one names the model setting, not the files' bytes.
        digest = hashlib.sha256(str(self.server.cfg["engine"]["model"]).encode()).hexdigest()
        return {"name": name, "model": name, "modified_at": _iso(self.server.started), "size": size, "digest": digest,
                "details": {"format": "gguf", "family": "", "families": None, "parameter_size": "",
                            "quantization_level": ""}}

    def _ollama_show(self) -> dict:
        capabilities = ["completion"]
        if self.server.cfg["engine"]["thinking_end"]:
            capabilities.append("thinking")
        return {"modelfile": "", "parameters": "", "template": "", "details": self._ollama_model()["details"],
                "model_info": {}, "capabilities": capabilities, "modified_at": _iso(self.server.started)}


def clear_leftovers(say: Callable[[str], None]) -> list[str]:
    """A server killed outright (no chance to put things back) leaves its hold
    job queued for hours. When no question holds the lock, no hold is live, so
    any still waiting or running is a leftover: cancel it."""
    ai.AI_DIR.mkdir(parents=True, exist_ok=True)
    with (ai.AI_DIR / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return []
        stale = [entry["id"] for entry in ai.dispatcher.load_entries()
                 if str(entry.get("id", "")).startswith(ai.HOLD_NAME) and entry.get("state") in ("queued", "running", "paused")]
        cancelled = [job for job in stale if ai._cancel_job(job)]
    for job in cancelled:
        say(f"cancelled {job}, left by an earlier run that was stopped without cleaning up")
    return cancelled


def serve(port: int | None = None, keep_alive: int | None = None, say: Callable[[str], None] = lambda _: None,
          model: str | None = None) -> int:
    """Run until stopped, then put everything back. Raises ai.ConfigError on
    bad settings or a port that's taken. `model` names a listed model to lay
    out across the machines, read once at the start."""
    cfg = load_settings()
    if model:
        from . import ai_auto
        cfg, layout = ai_auto.apply(cfg, model, say)
        cfg["server"]["name"], cfg["layout"] = layout["model"], layout
    if port is not None:
        cfg["server"]["port"] = port
    if keep_alive is not None:
        cfg["server"]["keep_alive_seconds"] = keep_alive
    try:
        server = Server(cfg, int(cfg["server"]["port"]), say)
    except OSError as exc:
        raise ai.ConfigError(f"can't listen on 127.0.0.1:{cfg['server']['port']}: {exc.strerror or exc}")
    clear_leftovers(say)
    stop = threading.Event()
    why = {"reason": "the server stopped"}

    def on_signal(signum, _frame) -> None:
        why["reason"] = f"the server stopped ({signal.Signals(signum).name})"
        stop.set()

    saved = {sig: signal.signal(sig, on_signal) for sig in ai._SIGNALS}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.port}"
    say(f"ready at {base} for {server.engine.name}")
    say(f"  OpenAI tools: base URL {base}/v1")
    say(f"  Claude tools: base URL {base}")
    say(f"  Ollama tools: OLLAMA_HOST=127.0.0.1:{server.port}")
    say(f"nothing loads until the first question; it unloads after {server.engine.keep_alive:.0f} s without one")
    try:
        while not stop.wait(1):
            pass
    finally:
        for sig in ai._SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        server.close(why["reason"])
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    return 0
