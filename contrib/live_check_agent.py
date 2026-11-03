#!/usr/bin/env python3
"""Write code with Claude while `orchard live` checks each finished function.

The model writes files through a `write_file` tool. Its input streams in as it
is generated (eager input streaming), and this script hands the file text to
`orchard live` piece by piece. When a check fails, the stream is stopped right
there and the failure goes back to the model as the next message, so it fixes
the function instead of writing the rest of the file on top of it.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python contrib/live_check_agent.py --root . --check "python -m pytest -x -q" \\
        "Add a slugify() helper to src/text.py, with tests"

Server-side refusal fallbacks are on (FALLBACKS = "default"): if the model
declines a request, the API reruns it on Anthropic's recommended fallback model
inside the same call. Set FALLBACKS = None to turn that off.

Exit codes: 0 the model finished, 1 it stopped early (refusal, out of tokens,
too many turns).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import queue
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path

MODEL = "claude-opus-5-5"
MAX_TOKENS = 64000
FALLBACKS = "default"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
TAIL_CHARS = 4000
SHOWN_CHARS = 8000
RETRIES = 2
CHECK_TIMEOUT = 300  # seconds before `orchard live` kills one check
CLOSE_GRACE = 30     # and how much longer the last check may take to report
_GOES_ON = ("end_turn", "tool_use", "pause_turn", "stop_sequence")  # anything else: stop

TOOLS = [
    {
        "name": "write_file",
        "description": "Create or replace a file. Always give the complete contents. "
                       "Paths are relative to the project root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "file path relative to the project root"},
                "contents": {"type": "string", "description": "the complete new contents of the file"},
            },
            "required": ["path", "contents"],
        },
        "eager_input_streaming": True,
    },
    {
        "name": "read_file",
        "description": "Read a file. Paths are relative to the project root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "file path relative to the project root"}},
            "required": ["path"],
        },
    },
]

SYSTEM = """You are changing a software project. Use read_file to look at files and \
write_file to create or replace them, always with the complete file.

While you write a file, `{check}` runs each time a function or class in it is \
finished. If it fails you are stopped at that point and shown the failure; \
nothing from the stopped reply is kept, so fix the problem and write the files \
again in full. Order your code so each finished function can already pass: \
helpers before the code that uses them."""


def make_client():
    import anthropic  # here, so the tests and --help work without it

    return anthropic.Anthropic()


class _Fields:
    """The string fields of a JSON object that arrives in pieces."""

    ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", '"': '"', "\\": "\\"}

    def __init__(self):
        self.values: dict[str, str] = {}
        self.closed: set[str] = set()
        self.state = "start"
        self.buf: list[str] = []
        self.key = ""
        self.esc = ""         # "\\", or "\\u" plus the hex digits so far
        self.high = None      # a high surrogate waiting for its low half
        self.nest = 0         # depth inside a skipped object or array
        self.skip_str = self.skip_esc = False

    def feed(self, chunk: str) -> None:
        for c in chunk:
            self._char(c)
        if self.state == "value":
            self.values[self.key] = "".join(self.buf)

    def _char(self, c: str) -> None:
        state = self.state
        if state == "start":
            if c == "{":
                self.state = "key?"
        elif state == "key?":
            if c == '"':
                self.state, self.buf = "key", []
        elif state == "key":
            if self.esc:
                self.buf.append(self.ESCAPES.get(c, c))
                self.esc = ""
            elif c == "\\":
                self.esc = "\\"
            elif c == '"':
                self.key, self.state = "".join(self.buf), "colon"
            else:
                self.buf.append(c)
        elif state == "colon":
            if c == ":":
                self.state = "value?"
        elif state == "value?":
            if c == '"':
                self.state, self.buf = "value", []
                self.values[self.key] = ""
            elif not c.isspace():
                self.state, self.nest = "skip", 0
                self._skip(c)
        elif state == "value":
            self._string(c)
        elif state == "skip":
            self._skip(c)

    def _string(self, c: str) -> None:
        if self.esc == "\\":
            if c == "u":
                self.esc = "\\u"
                return
            self.esc = ""
            self._put(self.ESCAPES.get(c, c))
        elif self.esc:
            self.esc += c
            if len(self.esc) < 6:
                return
            try:
                code = int(self.esc[2:], 16)
            except ValueError:
                code = 0xFFFD
            self.esc = ""
            if 0xD800 <= code < 0xDC00:
                if self.high is not None:
                    self.buf.append("�")
                self.high = code
            elif 0xDC00 <= code < 0xE000 and self.high is not None:
                self.buf.append(chr(0x10000 + ((self.high - 0xD800) << 10) + (code - 0xDC00)))
                self.high = None
            else:
                self._put("�" if 0xD800 <= code < 0xE000 else chr(code))
        elif c == "\\":
            self.esc = "\\"
        elif c == '"':
            self._put("")
            self.values[self.key] = "".join(self.buf)
            self.closed.add(self.key)
            self.state = "key?"
        else:
            self._put(c)

    def _put(self, text: str) -> None:
        if self.high is not None:  # a high surrogate with no low half after it
            self.buf.append("�")
            self.high = None
        if text:
            self.buf.append(text)

    def _skip(self, c: str) -> None:
        if self.skip_str:
            if self.skip_esc:
                self.skip_esc = False
            elif c == "\\":
                self.skip_esc = True
            elif c == '"':
                self.skip_str = False
        elif c == '"':
            self.skip_str = True
        elif c in "[{":
            self.nest += 1
        elif c in "]}" and self.nest:
            self.nest -= 1
        elif c == "," and not self.nest:
            self.state = "key?"


_PLAIN = re.compile(r"[\w.@+/-]+")  # the paths `orchard live` accepts


def _inside(root: Path, rel) -> Path | None:
    if (not isinstance(rel, str) or not _PLAIN.fullmatch(rel)
            or any(part.startswith("-") for part in Path(rel).parts)):
        return None
    try:
        p = (root / rel).resolve()
    except (OSError, ValueError):
        return None
    return p if p.is_relative_to(root) and p != root else None


class _Live:
    """One `orchard live` process for one model reply."""

    def __init__(self, orchard: list[str], check: str, root: Path, timeout: int = CHECK_TIMEOUT):
        self.argv = [*orchard, "live", "--timeout", str(timeout), "--", *shlex.split(check)]
        self.root, self.timeout = root, timeout
        self._start()

    def _start(self) -> None:
        self.proc = subprocess.Popen(self.argv, cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.events: queue.Queue = queue.Queue()
        self.failure: dict | None = None
        self.results: dict[str, dict] = {}  # path -> its check on the finished file
        self.seen = 0  # events read, to tell a slow check from a stuck one
        self.pump = threading.Thread(target=self._pump, args=(self.proc, self.events), daemon=True)
        self.pump.start()

    def restart(self) -> None:
        """A fresh process: what the declined model wrote, and its failure, are gone."""
        self.stop()
        self._start()

    def _pump(self, proc, events: queue.Queue) -> None:
        for line in proc.stdout:
            self.seen += 1
            try:
                events.put(json.loads(line))
            except ValueError:
                pass
        events.put(None)

    def send(self, item: dict) -> None:
        if self.failure is not None or self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(json.dumps(item) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass  # it exited; its failure is on the way

    def poll(self) -> dict | None:
        """The first failed check so far."""
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                return self.failure
            if ev is None:
                return self.failure
            kind = ev.get("event")
            if kind == "check":
                if not ev.get("ok") and self.failure is None:
                    self.failure = ev
                if ev.get("final"):
                    self.results[ev["path"]] = ev
            elif kind in ("speculate", "error"):
                print(f"[orchard live] {json.dumps(ev)}", file=sys.stderr)

    def close(self) -> dict | None:
        """Let the last checks finish; the first failure, if any."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        while True:
            seen = self.seen
            try:
                code = self.proc.wait(timeout=self.timeout + CLOSE_GRACE)
                break
            except subprocess.TimeoutExpired:
                if self.seen == seen:  # not even a check has ended: stuck
                    self.stop()
                    raise RuntimeError("orchard live stopped answering") from None
        self.pump.join(timeout=5)
        failure = self.poll()
        if code not in (0, 1) or (code == 1 and failure is None):
            raise RuntimeError(f"orchard live exited {code}")
        return failure

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.pump.join(timeout=5)


def _on_event(event, blocks: dict, live: _Live, root: Path, out) -> None:
    kind = event.type
    if kind == "content_block_start":
        block = event.content_block
        info = {"type": block.type}
        if block.type == "tool_use" and block.name == "write_file":
            info.update(fields=_Fields(), path=None, sent=0)
        elif block.type == "fallback":
            live.restart()  # the declined model's files, and any failure in them, are written again
        blocks[event.index] = info
    elif kind == "content_block_delta":
        delta, info = event.delta, blocks.get(event.index, {})
        if delta.type == "text_delta":
            out.write(delta.text)
            out.flush()
            live.send({"think": delta.text})
        elif delta.type == "thinking_delta":
            live.send({"think": delta.thinking})
        elif delta.type == "input_json_delta" and "fields" in info:
            _feed_write(info, delta.partial_json, live, root)
    elif kind == "content_block_stop":
        info = blocks.get(event.index, {})
        if info.get("path"):
            live.send({"path": info["path"], "done": True})


def _feed_write(info: dict, partial: str, live: _Live, root: Path) -> None:
    fields = info["fields"]
    fields.feed(partial)
    if info["path"] is None:
        if "path" not in fields.closed:
            return
        path = fields.values["path"]
        info["path"] = path if _inside(root, path) else ""  # "" = never checked; refused at the end
        if not info["path"]:
            return
        live.send({"path": path, "reset": True})
    text = fields.values.get("contents", "")
    if info["path"] and len(text) > info["sent"]:
        live.send({"path": info["path"], "text": text[info["sent"]:]})
        info["sent"] = len(text)


def _stream(client, system: str, messages: list, effort: str, live: _Live, root: Path, out):
    """One reply: (final message, None), or (None, failure) if a check failed first."""
    extra = {"betas": [FALLBACK_BETA], "fallbacks": FALLBACKS} if FALLBACKS else {}
    blocks: dict = {}
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
        tools=TOOLS,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": effort},
        **extra,
    ) as stream:
        for event in stream:
            _on_event(event, blocks, live, root, out)
            failure = live.poll()
            if failure:
                return None, failure, blocks  # leaving the block closes the stream
        return stream.get_final_message(), None, blocks


def _written(blocks: dict, path: str) -> str:
    for info in reversed(list(blocks.values())):  # after a fallback, the newest copy
        if info.get("path") == path:
            return info["fields"].values.get("contents", "")
    return ""


def _failure_text(failure: dict, check: str, blocks: dict) -> str:
    path, line, unit = failure.get("path"), failure.get("line") or 0, failure.get("unit")
    where = f"{path} at line {line}" + (f", just after `{unit}`" if unit else "")
    parts = [f"I stopped you while you were writing {where}: `{check}` failed."]
    if failure.get("why"):
        parts.append(failure["why"])
    parts += ["", "```", (failure.get("tail") or "")[-TAIL_CHARS:], "```"]
    shown = "".join(_written(blocks, path).splitlines(keepends=True)[:line])
    if shown:
        cut = len(shown) > SHOWN_CHARS
        parts += ["", f"What you had written of {path}" + (" (the end of it)" if cut else "") + ":",
                  "```", shown[-SHOWN_CHARS:].rstrip("\n"), "```"]
    parts += ["", "Nothing from that reply was applied. Fix the problem and write every file again in full."]
    return "\n".join(parts)


def _add_user_text(messages: list, text: str) -> None:
    last = messages[-1]
    if last["role"] != "user":
        messages.append({"role": "user", "content": text})
        return
    content = last["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    last["content"] = [*content, {"type": "text", "text": text}]


def _check_note(path: str, live: _Live) -> tuple[str, bool]:
    failure = live.failure
    if failure and failure.get("path") == path:
        tail = (failure.get("tail") or "")[-TAIL_CHARS:]
        return (f"Written, but the check failed at line {failure.get('line')}"
                + (f" ({failure['why']})" if failure.get("why") else "") + f":\n{tail}"), True
    ev = live.results.get(path)
    if ev and ev.get("ok"):
        return "Written; the check passed.", False
    if failure:
        return f"Written; not checked, because the check had already failed in {failure.get('path')}.", False
    return "Written; not checked.", False


def _run_tool(block, root: Path, live: _Live, streamed: dict) -> dict:
    result = {"type": "tool_result", "tool_use_id": block.id}
    data = block.input
    need = {"write_file": ("path", "contents"), "read_file": ("path",)}.get(block.name)
    if need is None:
        return {**result, "is_error": True, "content": f"unknown tool {block.name}"}
    if not isinstance(data, dict) or any(not isinstance(data.get(k), str) for k in need):
        return {**result, "is_error": True, "content": json.dumps({"INVALID_JSON": json.dumps(data)})}
    target = _inside(root, data["path"])
    if target is None:
        return {**result, "is_error": True, "content": f"{data['path']} is not a plain relative path inside the project"}
    if block.name == "read_file":
        try:
            return {**result, "content": target.read_text(encoding="utf-8")}
        except (OSError, UnicodeDecodeError) as exc:
            return {**result, "is_error": True, "content": str(exc)}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data["contents"], encoding="utf-8")
    except OSError as exc:
        return {**result, "is_error": True, "content": str(exc)}
    note, is_error = _check_note(data["path"], live)
    if streamed.get(data["path"]) != data["contents"]:
        note, is_error = "Written; not checked (what streamed in differs from the final input).", False
    return {**result, "content": note, **({"is_error": True} if is_error else {})}


def run(task: str, root, check: str, client=None, orchard: list[str] | None = None,
        effort: str = "medium", max_turns: int = 20, out=None, timeout: int = CHECK_TIMEOUT) -> int:
    root = Path(root).resolve()
    out = out or sys.stdout
    client = client or make_client()
    orchard = orchard or ["orchard"]
    system = SYSTEM.format(check=check)
    messages: list = [{"role": "user", "content": task}]
    for _ in range(max_turns):
        for attempt in range(RETRIES + 1):
            live = _Live(orchard, check, root, timeout)
            try:
                msg, failure, blocks = _stream(client, system, messages, effort, live, root, out)
                break
            except ValueError as exc:  # the SDK could not parse a streamed tool input
                live.stop()
                if attempt == RETRIES or isinstance(exc, UnicodeError):  # ours, not the SDK's: retrying won't help
                    raise
            except BaseException:
                live.stop()
                raise
        if failure:
            live.stop()
            out.write(f"\n[check failed in {failure.get('path')} line {failure.get('line')}; asking for a fix]\n")
            _add_user_text(messages, _failure_text(failure, check, blocks))
            continue
        if msg.stop_reason not in _GOES_ON:  # refusal, max_tokens, a full context window...
            live.stop()
            out.write(f"\n[stopped: {msg.stop_reason}]\n")
            return 1
        live.close()
        content = list(msg.content)
        cut = max((i for i, b in enumerate(content) if b.type == "fallback"), default=-1)
        kept = [b for i, b in enumerate(content) if i > cut or b.type in ("text", "fallback")]
        messages.append({"role": "assistant", "content": kept})
        if msg.stop_reason == "pause_turn":
            continue
        tools = [b for i, b in enumerate(content) if i > cut and b.type == "tool_use"]
        if not tools:
            return 0
        streamed = {info["path"]: info["fields"].values.get("contents")
                    for info in blocks.values() if info.get("path")}
        messages.append({"role": "user", "content": [_run_tool(b, root, live, streamed) for b in tools]})
    out.write(f"\n[stopped after {max_turns} turns]\n")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", help="what to build or change")
    ap.add_argument("--root", default=".", help="project folder (default: here)")
    ap.add_argument("--check", required=True, help="command run at each finished function, e.g. 'python -m pytest -x -q'")
    ap.add_argument("--orchard", default="orchard", help="how to start orchard (default: orchard)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=CHECK_TIMEOUT, help="seconds before one check is killed")
    args = ap.parse_args(argv)
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(errors="replace")  # a model's text the terminal can't show
    return run(args.task, args.root, args.check, orchard=shlex.split(args.orchard),
               effort=args.effort, max_turns=args.max_turns, timeout=args.timeout)


if __name__ == "__main__":
    sys.exit(main())
