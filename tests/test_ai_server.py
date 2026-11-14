import fcntl
import http.client
import json
import os
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from orchard import ai, ai_server, cli, dispatcher, receipts

runner = CliRunner()

# Stands in for ssh, so no test reaches a real machine. A tunnel, a link to the
# engine's socket and a pause stay up until killed, and say so in events.log.
# Everything else runs here in a shell, which finds the fake pkill first on PATH.
SSH = r'''#!/usr/bin/env python3
import base64, os, signal, sys, time
home = os.environ["AI_STUB"]
def note(line):
    with open(f"{home}/events.log", "a") as log:
        log.write(line + "\n")
def stay_up(up, down, then=lambda: None):
    def finish(*_):
        then()
        note(down)
        os._exit(0)
    signal.signal(signal.SIGTERM, finish)
    note(up)
def next_free():
    with open(f"{home}/free.txt") as handle:
        values = handle.read().split()
    if len(values) > 1:
        with open(f"{home}/free.txt", "w") as handle:
            handle.write("\n".join(values[1:]))
    return values[0]
args, mode, spec, host, words = sys.argv[1:], None, None, None, []
while args:
    arg = args.pop(0)
    if arg in ("-T", "-N"):
        continue
    if arg == "-o":
        args.pop(0)
    elif arg == "-R":
        args.pop(0)
        mode = "tunnel"
    elif arg == "-L":
        spec, mode = args.pop(0), "link"
    else:
        host, words = arg, args
        break
if host == "gone":
    sys.exit(255)
if mode == "tunnel":
    stay_up("tunnel up", "tunnel down")
    time.sleep(600)
    sys.exit(0)
if mode == "link":
    local, remote = spec.split(":", 1)
    os.symlink(remote, local)
    stay_up("link up", "link down", lambda: os.path.lexists(local) and os.unlink(local))
    time.sleep(600)
    sys.exit(0)
command = " ".join(words)
if command.startswith("ss "):
    print(1)
elif command.startswith("typeperf"):
    free = next_free()
    if free.isdigit():
        print('"(PDH-CSV 4.0)","Available MBytes"')
        print(f'"11/06/2026 06:00:00.000","{free}.000000"')
elif command.startswith("awk"):
    free = next_free()
    if free.isdigit():
        print(free)
elif command.startswith("pkill"):
    note("engine killed")
    try:
        with open(f"{home}/engine.pid") as handle:
            os.kill(int(handle.read()), signal.SIGTERM)
    except (OSError, ValueError):
        pass
elif command.startswith("powershell"):
    script = base64.b64decode(command.split()[-1]).decode("utf-16-le")
    if "NtSuspendProcess" in script:
        stay_up("pause on", "pause off")
        print("held OneDrive suspended 3 of 3\nheld done", flush=True)
        time.sleep(600)
    elif "NtResumeProcess" in script:
        note("apps resumed")
        print("back OneDrive suspended 0 of 3\nback done")
else:
    os.execvp("bash", ["bash", "-c", command])
'''

# Stands in for llama.cpp's server: listens on the socket file it's given and
# answers in each API's shape.
ENGINE = r'''#!/usr/bin/env python3
import json, os, signal, socketserver, sys, time
from http.server import BaseHTTPRequestHandler
home = os.environ["AI_STUB"]
args = sys.argv[1:]
path = args[args.index("--host") + 1]
def note(line):
    with open(f"{home}/events.log", "a") as log:
        log.write(line + "\n")
def flag(name, default=None):
    try:
        with open(f"{home}/{name}") as handle:
            return handle.read()
    except OSError:
        return default
with open(f"{home}/engine.pid", "w") as handle:
    handle.write(str(os.getpid()))
with open(f"{home}/server.args", "w") as handle:
    handle.write("\n".join(args))
with open(f"{home}/engine.env", "w") as handle:
    handle.write("\n".join(f"{k}={v}" for k, v in os.environ.items()))
note("engine started")
loading = [int(flag("loading", "0"))]
def stop(*_):
    note("engine stopped")
    try:
        os.unlink(path)
    except OSError:
        pass
    os._exit(0)
signal.signal(signal.SIGTERM, stop)
THINKING = "Light scatters."
TEXT = "The sky scatters blue."
TIMINGS = {"prompt_n": 12, "prompt_ms": 1500.0, "predicted_n": 5, "predicted_ms": 15000.0,
           "predicted_per_token_ms": 3000.0}
CALL = {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\": \"Accra\"}"}}

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *_):
        pass
    def reply(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def events(self, items):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for name, obj in items:
            if name == "pause":
                time.sleep(0.3)
                continue
            head = f"event: {name}\n".encode() if name else b""
            body = b"[DONE]" if obj is None else json.dumps(obj).encode()
            self.wfile.write(head + b"data: " + body + b"\n\n")
            self.wfile.flush()
    def do_GET(self):
        if self.path == "/health":
            if loading[0] > 0:
                loading[0] -= 1
                self.reply(503, {"error": {"message": "Loading model"}})
            else:
                self.reply(200, {"status": "ok"})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(f"{home}/requests.log", "a") as log:
            log.write(json.dumps({"path": self.path, "body": body}) + "\n")
        note(f"begin {self.path}")
        try:
            self.answer(body)
        finally:
            note(f"end {self.path}")
    def answer(self, body):
        if flag("slow"):
            time.sleep(float(flag("slow")))
        if flag("status"):
            return self.reply(int(flag("status")), {"error": {"message": "engine trouble"}})
        words = ["The ", "sky ", "scatters ", "blue."]
        stream = body.get("stream")
        if self.path == "/v1/chat/completions":
            tools = bool(body.get("tools"))
            if not stream:
                message = {"role": "assistant", "content": "" if tools else TEXT, "reasoning_content": THINKING}
                if tools:
                    message["tool_calls"] = [CALL]
                return self.reply(200, {"choices": [{"index": 0, "message": message,
                                                     "finish_reason": "tool_calls" if tools else "stop"}],
                                        "usage": {"prompt_tokens": 12, "completion_tokens": 5}, "timings": TIMINGS})
            items = [(None, {"choices": [{"delta": {"role": "assistant", "content": None}}]}), ("pause", None),
                     (None, {"choices": [{"delta": {"reasoning_content": THINKING}}]})]
            if tools:
                items += [(None, {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                           "function": {"name": "weather", "arguments": "{\"city\": "}}]}}]}),
                          (None, {"choices": [{"delta": {"tool_calls": [{"index": 0,
                           "function": {"arguments": "\"Accra\"}"}}]}}]})]
            else:
                items += [(None, {"choices": [{"delta": {"content": w}}]}) for w in words]
            items += [(None, {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": TIMINGS}), (None, None)]
            return self.events(items)
        if self.path == "/v1/messages":
            if not stream:
                return self.reply(200, {"type": "message", "role": "assistant", "content": [
                    {"type": "thinking", "thinking": THINKING}, {"type": "text", "text": TEXT}]})
            items = [("message_start", {"type": "message_start"}),
                     ("content_block_delta", {"type": "content_block_delta",
                                              "delta": {"type": "thinking_delta", "thinking": THINKING}})]
            items += [("content_block_delta", {"type": "content_block_delta",
                                               "delta": {"type": "text_delta", "text": w}}) for w in words]
            return self.events(items + [("message_stop", {"type": "message_stop"})])
        if self.path == "/v1/responses":
            return self.reply(200, {"object": "response", "output": [
                {"type": "reasoning", "content": [{"text": THINKING}]},
                {"type": "message", "content": [{"type": "output_text", "text": TEXT}]}]})
        self.reply(404, {"error": "no such path"})

class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

Server(path, Handler).serve_forever()
'''

HELPER = '''#!/bin/bash
trap 'kill $child; echo "helper down" >> "$AI_STUB/events.log"; exit 0' TERM
echo "helper up" >> "$AI_STUB/events.log"
sleep 600 & child=$!
wait $child
'''


def _script(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture(autouse=True)
def stub(tmp_path, monkeypatch):
    """A fake world: ssh, the engine's server, the helper, and short waits."""
    home = tmp_path / "stub"
    tools = home / "bin"
    tools.mkdir(parents=True)
    _script(tools / "ssh", SSH)
    _script(tools / "pkill", "#!/bin/sh\nexit 0\n")
    engine = tmp_path / "engine"
    engine.mkdir()
    _script(engine / "llama-server", ENGINE)
    (engine / "libllama.so").write_text("lib")
    helper = tmp_path / "helper"
    helper.mkdir()
    _script(helper / "ggml-rpc-server", HELPER)
    models = tmp_path / "models"
    models.mkdir()
    for number in (1, 2):
        (models / f"m-0000{number}-of-00002.gguf").write_text("weights")
    (home / "free.txt").write_text("6000")
    (home / "events.log").write_text("")
    # Socket paths have to be short, so they live in their own temporary folders.
    sockets = Path(tempfile.mkdtemp(prefix="oai"))
    monkeypatch.setenv("AI_STUB", str(home))
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(ai, "AI_DIR", tmp_path / "ai")
    monkeypatch.setattr(ai, "CONFIG_PATH", tmp_path / "ai.toml")
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(receipts, "_key", lambda: b"test key")
    monkeypatch.setattr(ai, "_local_pressure", lambda: 0)
    monkeypatch.setattr(ai, "_pkill_local", lambda _pattern: None)
    monkeypatch.setattr(ai_server, "SOCKET_DIR", sockets / "run")
    for name, value in (("GUARD_POLL", 0.05), ("MEMORY_POLL", 0.01), ("HOLD_POLL", 0.01), ("HELPER_SETTLE", 0.1),
                        ("TUNNEL_SETTLE", 0.3), ("TUNNEL_GAP", 0), ("RESUME_TRIES", 2), ("RESUME_GAP", 0),
                        ("PAUSE_WAIT", 10), ("KILL_WAIT", 10)):
        monkeypatch.setattr(ai, name, value)
    for name, value in (("HEALTH_POLL", 0.05), ("KEEPALIVE_GAP", 0.05), ("SOCKET_WAIT", 10), ("FORWARD_WAIT", 10)):
        monkeypatch.setattr(ai_server, name, value)
    yield SimpleNamespace(home=home, engine=engine, helper=helper, models=models, sockets=sockets)
    shutil.rmtree(sockets, ignore_errors=True)


def write_settings(stub, *, helper=False, windows=False, worker="", start=0, stop=0, keep=600, engine_extra="",
                   server_extra=""):
    text = f'''
[engine]
host = "box"
folder = "{stub.engine}"
model = "{stub.models}/m-00001-of-00002.gguf"
threads = 4
devices = "CUDA0,RPC0"
split = "22,78"
overrides = "_exps=CPU"
args = ["--temp", "0", "-no-cnv", "--no-display-prompt", "-ub", "32", "--jinja", "-cnv", "-st"]
start_free_mb = {start}
stop_free_mb = {stop}
settle_seconds = 0
worker = "{worker}"
{engine_extra}

[engine.env]
GGML_RPC_BATCH_FETCH = "1"

[server]
socket = "{stub.sockets}/engine.sock"
keep_alive_seconds = {keep}
{server_extra}
'''
    if helper:
        text += f'\n[helper]\nfolder = "{stub.helper}"\ndevice = "MTL0"\n'
    if windows:
        text += '\n[windows]\nhost = "winbox"\npause = ["OneDrive"]\n'
    ai.CONFIG_PATH.write_text(text)


def events(stub):
    return (stub.home / "events.log").read_text().split("\n")[:-1]


def requests(stub):
    path = stub.home / "requests.log"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def serve(stub):
    """Start a server on a free port; put everything back afterwards."""
    started = []

    def start(**settings):
        write_settings(stub, **settings)
        said = []
        server = ai_server.Server(ai_server.load_settings(), 0, said.append)
        server.said = said
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return server

    yield start
    for server in started:
        server.close()


def call(server, method, path, body=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    conn.request(method, path, data, {"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    out = resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    conn.close()
    return out


def chat(server, text="Why is the sky blue?", **extra):
    return call(server, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": text}], **extra})


def wait_for(check, seconds=10):
    end = time.time() + seconds
    while time.time() < end:
        if check():
            return True
        time.sleep(0.05)
    return False


def record_of(headers):
    record = ai.find(headers["x-orchard-record"])
    assert record is not None
    assert ai.check(record) == []
    return record


# Loading once and answering.

def test_the_first_question_loads_the_model_and_the_next_one_reuses_it(serve, stub):
    server = serve()
    assert server.engine.state == "idle" and "engine started" not in events(stub)
    status, headers, body = chat(server)
    assert status == 200, body
    assert json.loads(body)["choices"][0]["message"]["content"] == "The sky scatters blue."
    first = record_of(headers)
    status, headers, body = chat(server, "And at sunset?")
    assert status == 200
    second = record_of(headers)
    assert events(stub).count("engine started") == 1
    assert server.engine.state == "loaded"
    assert first["via"] == "server" and first["api"] == "openai-chat"
    assert first["question"] == "Why is the sky blue?" and second["question"] == "And at sunset?"
    assert first["answer"] == "The sky scatters blue." and first["thinking"] == "Light scatters."
    assert first["timings"]["ms_per_word"] == 3000.0 and first["timings"]["words"] == 5
    assert first["loaded_for_this_question"] is True and second["loaded_for_this_question"] is False
    assert first["session"] == second["session"] and first["session"].startswith("ais-")


def test_the_engine_runs_on_a_socket_file_with_server_settings(serve, stub):
    server = serve(helper=True)
    assert chat(server)[0] == 200
    args = (stub.home / "server.args").read_text().split("\n")
    assert args[args.index("--host") + 1] == f"{stub.sockets}/engine.sock"
    for flag, value in (("-np", "1"), ("--cache-ram", "0"), ("--alias", "m"), ("-ts", "22,78"),
                        ("--rpc", "127.0.0.1:50072"), ("-ub", "32"), ("-t", "4")):
        assert args[args.index(flag) + 1] == value
    assert "--no-webui" in args and "--jinja" in args
    for flag in ("-no-cnv", "--no-display-prompt", "-cnv", "-st"):
        assert flag not in args
    assert "GGML_RPC_BATCH_FETCH=1" in (stub.home / "engine.env").read_text()
    assert os.stat(f"{stub.sockets}/engine.sock").st_mode & 0o077 == 0


def test_a_streamed_answer_passes_through_and_is_recorded(serve, stub):
    server = serve()
    status, headers, body = chat(server, stream=True)
    assert status == 200 and headers["content-type"] == "text/event-stream"
    assert body.count(b"data: ") == 8 and body.rstrip().endswith(b"data: [DONE]")
    record = record_of(headers)
    assert record["answer"] == "The sky scatters blue." and record["thinking"] == "Light scatters."
    assert record["timings"]["ms_per_word"] == 3000.0
    # The engine's opening chunk carries no words, so it doesn't count as the first word.
    assert record["first_word_seconds"] >= 0.3
    assert record_of(chat(server)[1])["first_word_seconds"] is None


def test_claude_messages_and_openai_responses_pass_through(serve, stub):
    server = serve()
    status, headers, body = call(server, "POST", "/v1/messages", {
        "model": "claude-like", "max_tokens": 50, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Why blue?"}]}]})
    assert status == 200 and json.loads(body)["content"][1]["text"] == "The sky scatters blue."
    record = record_of(headers)
    assert record["api"] == "claude-messages" and record["question"] == "Why blue?"
    assert record["thinking"] == "Light scatters." and record["model_asked"] == "claude-like"
    status, headers, body = call(server, "POST", "/v1/responses", {"input": "Why blue?"})
    assert status == 200
    record = record_of(headers)
    assert record["api"] == "openai-responses" and record["answer"] == "The sky scatters blue."
    status, headers, body = call(server, "POST", "/v1/messages", {
        "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "Why blue?"}]})
    assert status == 200 and b"event: content_block_delta" in body
    assert record_of(headers)["answer"] == "The sky scatters blue."


def test_engine_errors_come_back_to_the_tool(serve, stub):
    server = serve()
    (stub.home / "status").write_text("500")
    status, headers, body = chat(server)
    assert status == 500 and b"engine trouble" in body
    assert record_of(headers)["state"] == "failed"


# Ollama.

def test_ollama_chat_is_translated(serve, stub):
    server = serve()
    status, headers, body = call(server, "POST", "/api/chat", {
        "model": "m:latest", "stream": False, "messages": [{"role": "user", "content": "Why blue?"}],
        "options": {"temperature": 0.2, "num_predict": 64, "num_ctx": 9999}, "format": "json"})
    assert status == 200, body
    reply = json.loads(body)
    assert reply["message"] == {"role": "assistant", "content": "The sky scatters blue.", "thinking": "Light scatters."}
    assert reply["done"] is True and reply["done_reason"] == "stop" and reply["model"] == "m"
    assert reply["eval_count"] == 5 and reply["prompt_eval_count"] == 12 and reply["eval_duration"] == 15_000_000_000
    sent = requests(stub)[-1]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["body"]["temperature"] == 0.2 and sent["body"]["max_tokens"] == 64
    assert sent["body"]["response_format"] == {"type": "json_object"} and "num_ctx" not in json.dumps(sent["body"])
    record = record_of(headers)
    assert record["api"] == "ollama-chat" and record["answer"] == "The sky scatters blue."


def test_ollama_chat_streams_lines_of_json(serve, stub):
    server = serve()
    status, headers, body = call(server, "POST", "/api/chat", {
        "model": "m", "messages": [{"role": "user", "content": "Why blue?"}]})
    assert status == 200 and headers["content-type"] == "application/x-ndjson"
    lines = [json.loads(line) for line in body.splitlines()]
    assert all(not line["done"] for line in lines[:-1]) and lines[-1]["done"] is True
    assert "".join(line["message"]["content"] for line in lines) == "The sky scatters blue."
    assert lines[0]["message"]["thinking"] == "Light scatters."
    assert lines[-1]["eval_count"] == 5 and lines[-1]["done_reason"] == "stop"
    assert record_of(headers)["answer"] == "The sky scatters blue."


def test_ollama_generate_and_an_empty_generate_loads_the_model(serve, stub):
    server = serve()
    status, _, body = call(server, "POST", "/api/generate", {"model": "m", "prompt": ""})
    assert status == 200 and json.loads(body)["done_reason"] == "load"
    assert server.engine.state == "loaded" and "begin /v1/chat/completions" not in events(stub)
    status, headers, body = call(server, "POST", "/api/generate", {
        "model": "m", "prompt": "Why blue?", "system": "Be brief.", "stream": False})
    assert status == 200 and json.loads(body)["response"] == "The sky scatters blue."
    sent = requests(stub)[-1]["body"]["messages"]
    assert sent == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Why blue?"}]
    assert record_of(headers)["question"] == "Why blue?"


def test_ollama_tool_calls_come_back_as_objects(serve, stub):
    server = serve()
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    status, _, body = call(server, "POST", "/api/chat", {
        "model": "m", "stream": False, "tools": tools, "messages": [{"role": "user", "content": "Weather?"}]})
    assert status == 200
    assert json.loads(body)["message"]["tool_calls"] == [{"function": {"name": "weather",
                                                                      "arguments": {"city": "Accra"}}}]
    status, _, body = call(server, "POST", "/api/chat", {
        "model": "m", "tools": tools, "messages": [{"role": "user", "content": "Weather?"}]})
    lines = [json.loads(line) for line in body.splitlines()]
    calls = [line["message"]["tool_calls"] for line in lines if line["message"].get("tool_calls")]
    assert calls == [[{"function": {"name": "weather", "arguments": {"city": "Accra"}}}]]
    status, _, _ = call(server, "POST", "/api/chat", {"model": "m", "stream": False, "messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "weather",
                                                                          "arguments": {"city": "Accra"}}}]},
        {"role": "tool", "content": "Sunny", "tool_name": "weather"}]})
    sent = requests(stub)[-1]["body"]["messages"]
    assert sent[1]["tool_calls"][0]["function"] == {"name": "weather", "arguments": '{"city": "Accra"}'}
    assert sent[2] == {"role": "tool", "content": "Sunny", "name": "weather"}


def test_ollama_images_are_refused_plainly(serve, stub):
    server = serve()
    status, _, body = call(server, "POST", "/api/chat", {"model": "m", "messages": [
        {"role": "user", "content": "What is this?", "images": ["aGk="]}]})
    assert status == 400 and json.loads(body) == {"error": "this model takes text only"}
    assert "engine started" not in events(stub)


def test_listings_answer_without_loading_anything(serve, stub):
    server = serve()
    assert call(server, "GET", "/")[2] == b"Orchard AI is running"
    assert call(server, "HEAD", "/")[0] == 200
    assert json.loads(call(server, "GET", "/v1/models")[2])["data"][0]["id"] == "m"
    tags = json.loads(call(server, "GET", "/api/tags")[2])["models"]
    assert tags[0]["name"] == "m" and len(tags[0]["digest"]) == 64
    assert json.loads(call(server, "GET", "/api/version")[2])["version"]
    assert json.loads(call(server, "GET", "/api/ps")[2]) == {"models": []}
    show = json.loads(call(server, "POST", "/api/show", {"model": "m"})[2])
    assert show["capabilities"] == ["completion"]
    assert json.loads(call(server, "GET", "/health")[2])["status"] == "idle"
    assert "engine started" not in events(stub)
    chat(server)
    assert json.loads(call(server, "GET", "/api/ps")[2])["models"][0]["name"] == "m"
    assert json.loads(call(server, "GET", "/health")[2])["status"] == "loaded"


# Putting everything back.

def test_unloads_when_nobody_asks_and_puts_everything_back_in_order(serve, stub, monkeypatch):
    cancelled = hold_job(monkeypatch, [{"state": "running"}])
    server = serve(helper=True, windows=True, worker="box", keep=0.3)
    status, headers, _ = chat(server)
    assert status == 200
    assert wait_for(lambda: server.engine.state == "idle")
    log = events(stub)
    assert log[:3] == ["helper up", "tunnel up", "pause on"]
    assert sorted(log[3:5]) == ["engine started", "link up"]
    assert log.index("engine stopped") < log.index("apps resumed") < log.index("tunnel down") < log.index("helper down")
    assert "link down" in log
    assert cancelled == ["ai-hold-20261106-062057-31b6"]
    session = ai.find(record_of(headers)["session"])
    assert receipts.verify(session) and session["state"] == "served"
    assert session["reason"] == "nobody asked for 0 s" and session["answers"] == [headers["x-orchard-record"]]
    assert session["resumed"] is True and session["paused"] == ["OneDrive suspended 3 of 3"]
    assert not list(ai_server.SOCKET_DIR.glob("*.sock"))
    assert chat(server)[0] == 200
    assert events(stub).count("engine started") == 2


def test_backs_off_when_someone_starts_using_the_machine(serve, stub, monkeypatch):
    hold = {"state": "running"}
    hold_job(monkeypatch, [])
    monkeypatch.setattr(ai, "_job_entry", lambda _job_id: dict(hold))
    server = serve(worker="box")
    status, headers, _ = chat(server)
    assert status == 200
    hold.update(paused_since=1, pause_reason="owner active")
    assert wait_for(lambda: server.engine.state == "idle")
    session = ai.find(record_of(headers)["session"])
    assert "the machine is in use: owner active" in session["reason"]
    assert "engine stopped" in events(stub)


def test_stops_when_memory_runs_low(serve, stub):
    (stub.home / "free.txt").write_text("6000 6000 100 100 100")
    server = serve(start=5000, stop=200)
    status, headers, _ = chat(server)
    assert status == 200
    assert wait_for(lambda: server.engine.state == "idle")
    session = ai.find(record_of(headers)["session"])
    assert session["reason"] == "under 200 MB free on the engine machine" and session["lowest_free_mb"] == 100


def test_a_question_in_progress_is_stopped_when_memory_runs_low(serve, stub):
    (stub.home / "free.txt").write_text("6000 6000 6000 100 100 100")
    (stub.home / "slow").write_text("5")
    server = serve(start=5000, stop=200)
    status, headers, body = chat(server)
    assert status == 502
    record = record_of(headers)
    assert record["state"] == "stopped" and record["reason"] == "under 200 MB free on the engine machine"


def test_refused_while_the_machine_stays_busy(serve, stub, monkeypatch):
    hold_job(monkeypatch, [{"state": "queued"}])
    server = serve(worker="box", engine_extra="hold_wait_seconds = 0")
    status, headers, body = chat(server)
    assert status == 503 and headers["retry-after"] == "60"
    assert "box stayed busy" in json.loads(body)["error"]["message"]
    record = record_of(headers)
    assert record["state"] == "refused" and "engine started" not in events(stub)
    assert server.engine.state == "idle"


def test_a_load_that_fails_says_so_and_cleans_up(serve, stub):
    (stub.engine / "llama-server").write_text("#!/bin/sh\necho broken >&2\nexit 3\n")
    server = serve(helper=True)
    status, headers, body = chat(server)
    assert status == 503 and "engine stopped by itself" in json.loads(body)["error"]["message"]
    assert record_of(headers)["state"] == "failed"
    assert wait_for(lambda: "helper down" in events(stub))
    assert server.engine.state == "idle"


def test_an_engine_that_died_is_loaded_again(serve, stub):
    server = serve()
    assert chat(server)[0] == 200
    os.kill(int((stub.home / "engine.pid").read_text()), 15)
    assert wait_for(lambda: server.engine.state == "idle" or server.engine.pool.alive())
    assert chat(server)[0] == 200
    assert events(stub).count("engine started") == 2


def test_closing_the_server_puts_everything_back(serve, stub):
    server = serve(helper=True)
    status, headers, _ = chat(server)
    session_id = record_of(headers)["session"]
    server.close()
    assert "engine stopped" in events(stub) and "helper down" in events(stub)
    assert ai.find(session_id)["reason"] == "the server stopped"


# One at a time.

def test_a_second_question_waits_its_turn(serve, stub):
    server = serve()
    (stub.home / "slow").write_text("0.4")
    results = []
    threads = [threading.Thread(target=lambda: results.append(chat(server)[0])) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [200, 200]
    turns = [line for line in events(stub) if line.startswith(("begin", "end"))]
    assert turns == ["begin /v1/chat/completions", "end /v1/chat/completions"] * 2
    assert events(stub).count("engine started") == 1


def test_ask_is_refused_while_the_server_holds_the_model(serve, stub):
    server = serve()
    assert chat(server)[0] == 200
    record = ai.ask("hi")
    assert record["state"] == "refused" and "one at a time" in record["reason"]


def test_the_server_waits_while_ask_runs(serve, stub):
    server = serve()
    ai.AI_DIR.mkdir(parents=True, exist_ok=True)
    with (ai.AI_DIR / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status, headers, body = chat(server)
    assert status == 503 and "one at a time" in body.decode()
    assert chat(server)[0] == 200


def test_a_question_over_its_time_limit_stops_but_the_model_stays(serve, stub):
    server = serve(engine_extra="cap_seconds = 1")
    assert chat(server)[0] == 200
    (stub.home / "slow").write_text("4")
    status, headers, _ = chat(server)
    assert status == 502
    record = record_of(headers)
    assert record["state"] == "stopped" and record["reason"] == "the question passed its time limit"
    assert server.engine.state == "loaded" and events(stub).count("engine started") == 1


def test_a_stream_hears_from_the_server_while_the_model_loads(serve, stub):
    (stub.home / "loading").write_text("10")
    server = serve()
    status, headers, body = chat(server, stream=True)
    assert status == 200 and body.startswith(b": orchard is getting the model ready")
    assert record_of(headers)["answer"] == "The sky scatters blue."


# Only this machine's own programs.

@pytest.mark.parametrize("headers", [{"Host": "evil.example"}, {"Host": "evil.example:6724"},
                                     {"Origin": "https://evil.example"}, {"Origin": "null"},
                                     {"Sec-Fetch-Site": "cross-site"}])
def test_web_pages_and_other_names_are_turned_away(serve, stub, headers):
    server = serve()
    status, _, _ = call(server, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]},
                        headers)
    assert status == 403
    assert "engine started" not in events(stub) and not ai.AI_DIR.exists()


def test_a_listed_origin_and_the_loopback_names_are_let_in(serve, stub):
    server = serve(server_extra='origins = ["vscode-webview://tool"]')
    assert server.server_address[0] == "127.0.0.1"
    for headers in ({"Origin": "vscode-webview://tool"}, {"Host": f"localhost:{server.port}"},
                    {"Origin": f"http://127.0.0.1:{server.port}"}):
        assert call(server, "GET", "/v1/models", headers=headers)[0] == 200
    assert call(server, "OPTIONS", "/v1/chat/completions")[0] == 403


def test_bad_requests_are_refused_before_the_engine(serve, stub):
    server = serve()
    assert call(server, "POST", "/v1/chat/completions", raw=b"not json")[0] == 400
    assert call(server, "POST", "/v1/chat/completions", raw=b"[1]")[0] == 400
    assert call(server, "POST", "/v1/embeddings", {"input": "x"})[0] == 404
    assert "engine started" not in events(stub)


# The pieces.

def test_the_engine_stops_when_its_ssh_session_ends(stub):
    write_settings(stub)
    cfg = ai_server.load_settings()
    cfg["engine"]["binary"] = "llama-server"
    proc = subprocess.Popen(["bash", "-c", ai_server.server_script(cfg)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout.readline().decode().strip() == f"socket {stub.sockets}/engine.sock"
    assert wait_for(lambda: os.path.exists(f"{stub.sockets}/engine.sock"))
    proc.stdin.close()
    proc.wait(10)
    assert "engine stopped" in events(stub)
    assert not os.path.exists(f"{stub.sockets}/engine.sock")


def test_the_cli_says_when_the_port_is_taken(stub):
    write_settings(stub)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        result = runner.invoke(cli.app, ["ai", "serve", "--port", str(taken.getsockname()[1])])
    assert result.exit_code == 2 and "can't listen on 127.0.0.1" in result.output


@pytest.mark.parametrize("line, message", [("port = -1", "server.port"), ("port = 70000", "server.port"),
                                           ('origins = "x"', "server.origins"),
                                           ("keep_alive_seconds = -5", "server.keep_alive_seconds")])
def test_bad_server_settings_are_named(stub, line, message):
    write_settings(stub, server_extra=line)
    text = ai.CONFIG_PATH.read_text()
    if line.startswith("keep_alive_seconds"):
        text = text.replace("keep_alive_seconds = 600\n", "")
    ai.CONFIG_PATH.write_text(text)
    with pytest.raises(ai.ConfigError, match=message):
        ai_server.load_settings()


def test_a_relative_engine_socket_is_refused(stub):
    write_settings(stub)
    ai.CONFIG_PATH.write_text(ai.CONFIG_PATH.read_text().replace(f'socket = "{stub.sockets}/engine.sock"',
                                                                 'socket = "relative.sock"'))
    with pytest.raises(ai.ConfigError, match="server.socket"):
        ai_server.load_settings()


def test_model_name_comes_from_the_file_or_the_setting(stub):
    write_settings(stub)
    cfg = ai_server.load_settings()
    assert ai_server.model_name(cfg) == "m"
    cfg["engine"]["model"] = "~/llm/models/Big-Model-Q4.gguf"
    assert ai_server.model_name(cfg) == "Big-Model-Q4"
    cfg["server"]["name"] = "big"
    assert ai_server.model_name(cfg) == "big"


def hold_job(monkeypatch, states):
    """A hold job whose queue entry moves through the given entries. Returns the
    list of jobs cancelled."""
    entries = iter(states)
    last = {}

    def entry(_job_id):
        nonlocal last
        last = next(entries, last)
        return last

    cancelled = []
    monkeypatch.setattr(ai, "_submit_hold", lambda *_a: subprocess.CompletedProcess(
        [], 0, "queued: ai-hold-20261106-062057-31b6 (worker: box)\n", ""))
    monkeypatch.setattr(ai, "_job_entry", entry)
    monkeypatch.setattr(ai, "_cancel_job", lambda job_id: cancelled.append(job_id) or True)
    return cancelled
