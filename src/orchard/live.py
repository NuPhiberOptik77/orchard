"""Checking alongside the AI: run a check each time a function is finished,
while the model is still writing the rest of the file. See docs/agent-native.md
section 6.

`orchard live -- <check>` reads JSON lines on stdin:

  {"path": "src/a.py", "reset": true}   start (or restart) writing a file
  {"path": "src/a.py", "text": "..."}   the next piece of it
  {"path": "src/a.py", "done": true}    the file is finished
  {"think": "..."}                      the model's reasoning or prose

It keeps a shadow copy of the project. When a file reaches a boundary (a
Python def or class closes, a brace block returns to depth 0, or the file is
done) the finished part goes into the copy and the check runs there. One JSON
line per check goes to stdout, and the first failure ends it with exit 1: the
caller's signal to stop the model and hand it the failure.

The check may use {path} and {unit} (the file, and the function that just
closed); the same values are in ORCHARD_LIVE_PATH and ORCHARD_LIVE_UNIT, and
ORCHARD_LIVE_FINAL is 1 for the finished file. A path must be a plain relative
path inside the project (letters, digits, _ . @ + - and /, no part starting
with -), so {path} is safe in an argv or a shell string. While an existing .py
file is being rewritten, the definitions the new text has not reached yet are
taken from the old file, so a test suite still imports.

Checks the model names in backticks that match a registered speculate check
are queued once on an idle machine, so their answer may be waiting.
"""
from __future__ import annotations

import ast
import collections
import contextlib
import io
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import speculate
from .engine import check_shipping_size, iter_shipped

BRACE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".go", ".java", ".js", ".jsx", ".ts",
                  ".tsx", ".kt", ".rs", ".swift", ".scala", ".php", ".dart"}
LINKS = (".venv", "venv", "node_modules")  # ignored, so never shipped; linked instead
TAIL_LINES = 30
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_UNPARSABLE = (SyntaxError, ValueError, MemoryError, RecursionError)  # what ast.parse can raise
_LINES = re.compile(r"\r\n|\r|\n")  # the line breaks ast counts; splitlines() has more
_SPAN = re.compile(r"`([^`\n]+)`")
_PLAIN = re.compile(r"[\w.@+/-]+")  # no spaces, quotes or shell metacharacters
_PLACEHOLDER = re.compile(r"\{(path|unit)\}")


def _kind(rel: str) -> str:
    suffix = Path(rel).suffix.lower()
    if suffix == ".py":
        return "py"
    return "brace" if suffix in BRACE_SUFFIXES else "plain"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _module_end(tree: ast.Module) -> int:
    return max((n.end_lineno or 0) for n in tree.body) if tree.body else 0


_RUST_CHAR = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'")
_REGEX_SUFFIXES = {".js", ".jsx", ".ts", ".tsx"}
_TRIPLE_SUFFIXES = {".java", ".kt", ".scala", ".swift", ".cs", ".dart"}
_RAW_TRIPLE_SUFFIXES = {".kt", ".scala", ".cs"}  # no escapes inside """..."""
_BEFORE_REGEX = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_WORDS = {"return", "typeof", "case", "delete", "void", "in", "of", "instanceof", "new", "throw",
                "yield", "await"}


def _regex_end(text: str, i: int) -> int | None:
    """Where a JS regex literal starting at text[i] ends; i + 1 if the line
    ends first (a division after all); None if the text ends first."""
    j, in_class = i + 1, False
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "\n":
            return i + 1
        if in_class:
            in_class = c != "]"
        elif c == "[":
            in_class = True
        elif c == "/":
            return j + 1
        j += 1
    return None


class _Braces:
    """Incremental brace depth for C-like files. A boundary is the end of the
    line where a statement ends at depth 0 after a closing brace. Text that may
    still turn out to be a comment, a string or a regex stops the scan until
    more arrives."""

    def __init__(self, suffix: str = ""):
        self.rust = suffix == ".rs"
        self.opens = '"`' if self.rust else '"\'`'  # Rust: 'a is a lifetime, '}' a char
        self.regex = suffix in _REGEX_SUFFIXES
        self.triples = ({'"""', "'''"} if suffix == ".dart" else {'"""'}) if suffix in _TRIPLE_SUFFIXES else set()
        self.raw = suffix in _RAW_TRIPLE_SUFFIXES
        self.pos = self.depth = self.nest = 0  # nest: open ( and [
        self.quote: str | None = None
        self.esc = self.line_comment = self.block = self.pending = False
        self.last = self.word = ""  # the last code character, and the word it ends
        self.gap = False            # whitespace since the last code character
        self.unit: str | None = None

    def _code(self, c: str) -> None:
        word = c.isalnum() or c in "_$"
        self.word = (self.word + c if self.word and not self.gap else c) if word else ""
        self.last, self.gap = c, False

    def _newline(self, i: int, found: list) -> None:
        if self.pending and self.depth == 0 and self.nest == 0:
            if self.last in ("}", ")", "]", ";"):
                found.append((i + 1, self.unit))
                self.pending = False
            elif self.last != ",":  # a trailing comma: the statement goes on
                self.pending = False
        self.gap = True

    def feed(self, text: str) -> list[tuple[int, str | None]]:
        found: list = []
        i, n = self.pos, len(text)
        while i < n:
            c = text[i]
            if self.line_comment:
                if c == "\n":
                    self.line_comment = False
                    self._newline(i, found)
                i += 1
            elif self.block:
                close = text.find("*/", i)
                if close == -1:
                    i = n - 1 if text.endswith("*") else n
                    break
                self.block = False
                i = close + 2
            elif self.quote:
                q = self.quote
                if self.esc:
                    self.esc = False
                elif c == "\\" and not (len(q) == 3 and self.raw):
                    self.esc = True
                elif len(q) == 3:
                    if text.startswith(q, i):
                        self.quote = None
                        self._code('"')
                        i += 3
                        continue
                    if c == q[0] and n - i < 3:
                        break
                elif c == q:
                    self.quote = None
                    self._code('"')
                elif c == "\n" and q != "`":
                    self.quote = None
                    self._newline(i, found)
                i += 1
            elif c == "/":
                if i + 1 >= n:
                    break
                if text[i + 1] in "/*":
                    self.line_comment, self.block = text[i + 1] == "/", text[i + 1] == "*"
                    i += 2
                    continue
                if self.regex and (not self.last or self.last in _BEFORE_REGEX or self.word in _REGEX_WORDS):
                    close = _regex_end(text, i)
                    if close is None:
                        break
                    if close > i + 1:
                        self._code('"')
                        i = close
                        continue
                self._code(c)
                i += 1
            elif c == "'" and self.rust:
                m = _RUST_CHAR.match(text, i)
                if m:
                    self._code('"')
                    i = m.end()
                    continue
                if n - i < 12 and "\n" not in text[i:]:
                    break
                self._code(c)  # a lifetime
                i += 1
            elif c in self.opens:
                triple = c * 3
                if triple in self.triples:
                    if text.startswith(triple, i):
                        self.quote = triple
                        i += 3
                        continue
                    if triple.startswith(text[i:]):
                        break
                self.quote = c
                i += 1
            else:
                if c == "{":
                    if self.depth == 0:
                        self.unit = _brace_unit(text[max(0, i - 200):i])
                    self.depth += 1
                    self.pending = False
                elif c == "}":
                    self.depth = max(0, self.depth - 1)
                    self.pending = self.depth == 0
                elif c in "([":
                    self.nest += 1
                elif c in ")]":
                    self.nest = max(0, self.nest - 1)
                if c == "\n":
                    self._newline(i, found)
                elif c in " \t\r":
                    self.gap = True
                else:
                    self._code(c)
                i += 1
        self.pos = i
        return found


_ARROW = re.compile(r"(\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|\w+)(?:\s*:\s*[^=]+?)?\s*=>")
_IMPL = re.compile(r"\bimpl(?:\s*<[^>]*>)?\s+(?:[\w:]+(?:<[^>]*>)?\s+for\s+)?(\w+)")


def _brace_unit(before: str) -> str | None:
    head = re.split(r"[;}]", before)[-1]
    for pattern in (_ARROW, _IMPL):
        named = pattern.findall(head)
        if named:
            return named[-1]
    named = re.findall(r"\b(?:class|struct|interface|enum|impl|trait|object|namespace)\s+(\w+)", head)
    if named:
        return named[-1]
    calls = re.findall(r"(\w+)\s*(?:<[^(){};]*>)?\s*\(", head)  # also name<T>(
    calls = [c for c in calls if c not in ("if", "for", "while", "switch", "catch", "func", "function")]
    if calls:
        return calls[-1]
    words = re.findall(r"\w+", head)
    return words[-1] if words else None


class _File:
    def __init__(self, rel: str, original: str | None):
        self.rel = rel
        self.original = original
        self.kind = _kind(rel)
        self.reset()

    def reset(self) -> None:
        self.text = ""
        self.scan = 0           # py: where the next candidate line search starts
        self.closed: set = set()  # py: (line, col) of defs known to be closed
        self.braces = _Braces(Path(self.rel).suffix.lower()) if self.kind == "brace" else None
        self.done = False
        self.ready: tuple[int, str | None, bool] | None = None  # (offset, unit, final) to check

    def boundary(self) -> tuple[int, str | None] | None:
        if self.kind == "brace":
            found = self.braces.feed(self.text)
            return found[-1] if found else None
        if self.kind == "py":
            return self._py_boundary()
        return None

    def _py_boundary(self) -> tuple[int, str | None] | None:
        # Candidates: starts of lines (after `scan`) whose first character is
        # known and is not a comment. Newest first: the newest one that parses
        # sees every def closed so far, because a def closed earlier now has a
        # statement after it.
        text = self.text
        starts = []
        at = text.find("\n", self.scan)
        while at != -1:
            start = at + 1
            nl = text.find("\n", start)
            line = text[start:] if nl == -1 else text[start:nl]
            stripped = line.lstrip(" \t")
            if stripped and not stripped.startswith("#"):
                starts.append((start, _indent(line)))
                self.scan = start
            elif nl == -1:
                break
            else:
                self.scan = start
            at = nl
        for start, indent in reversed(starts):
            try:
                tree = ast.parse(text[:start])
            except _UNPARSABLE:
                continue
            end = _module_end(tree)
            closed, unit, last = set(), None, -1
            for node in ast.walk(tree):
                if isinstance(node, _DEFS) and ((node.end_lineno or 0) < end or indent <= node.col_offset):
                    closed.add((node.lineno, node.col_offset))
                    if (node.end_lineno or 0) > last:
                        unit, last = node.name, node.end_lineno or 0
            if closed - self.closed:
                self.closed |= closed
                return start, unit
            return None
        return None

    def content(self, offset: int, final: bool) -> str:
        prefix = self.text if final else self.text[:offset]
        if final or self.kind != "py" or not self.original:
            return prefix
        # The next line is indented: the model is still inside the block.
        inside = offset < len(self.text) and self.text[offset] in " \t"
        return _fill(prefix, self.original, inside)


def _defined(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, _DEFS):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _bound(node: ast.Import | ast.ImportFrom) -> set[str]:
    return {a.asname or a.name.split(".")[0] for a in node.names if a.name != "*"}


def _missing(body: list, have: set[str], lines: list[str]) -> list[list[str]]:
    """The lines of each definition or simple assignment in `body` whose names
    are not in `have`."""
    parts = []
    for node in body:
        if isinstance(node, _DEFS):
            keep = node.name not in have
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            keep = all(isinstance(t, ast.Name) for t in targets) and not (_defined(ast.Module([node], [])) & have)
        else:
            keep = False
        if keep:
            first = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
            parts.append(lines[first - 1:node.end_lineno])
    return parts


def _class_rest(new: ast.Module, old: ast.Module, prefix: str, lines: list[str]) -> list[str]:
    """When the prefix stops inside a class the old file also has, that old
    class's members the prefix has not reached yet, indented to match."""
    cls = new.body[-1] if new.body else None
    if not isinstance(cls, ast.ClassDef):
        return []
    was = next((n for n in old.body if isinstance(n, ast.ClassDef) and n.name == cls.name), None)
    if was is None:
        return []
    new_indent = _LINES.split(prefix)[cls.body[0].lineno - 1][:cls.body[0].col_offset]
    old_indent = lines[was.body[0].lineno - 1][:was.body[0].col_offset]
    parts = []
    for chunk in _missing(was.body, _defined(ast.Module(cls.body, [])), lines):
        if new_indent != old_indent:
            chunk = [new_indent + ln[len(old_indent):] if ln.startswith(old_indent) else ln for ln in chunk]
        parts.append("\n".join(chunk))
    return parts


def _fill(prefix: str, original: str, inside: bool = False) -> str:
    """The prefix, then the old file's imports and the top-level definitions
    the prefix has not reached yet. With `inside`, first the rest of the class
    the prefix stops in."""
    try:
        old, new = ast.parse(original), ast.parse(prefix)
    except _UNPARSABLE:
        return prefix
    have = _defined(new)
    imported = set().union(*(_bound(n) for n in new.body if isinstance(n, (ast.Import, ast.ImportFrom))))
    taken = have | imported  # an old binding appended after the prefix would replace these
    lines = _LINES.split(original)
    members = _class_rest(new, old, prefix, lines) if inside else []
    parts = []
    for node in old.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue  # only legal at the top, and the prefix has its own
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # `import a.b` again is harmless next to the prefix's `import a.c`;
            # `from x import name` or an alias would rebind the prefix's name.
            plain = isinstance(node, ast.Import) and all(a.asname is None for a in node.names)
            if not _bound(node) & (have if plain else taken):
                first = node.lineno
                parts.append("\n".join(lines[first - 1:node.end_lineno]))
    parts += ["\n".join(chunk) for chunk in _missing(old.body, taken, lines)]
    if not parts and not members:
        return prefix
    out = prefix.rstrip("\n") + "\n"
    if members:
        out += "\n" + "\n\n".join(members) + "\n"
    if parts:
        out += "\n\n" + "\n\n".join(parts) + "\n"
    return out


def make_shadow(root: Path, links: list[str] | tuple = ()) -> Path:
    """A copy of what a run job would ship, under the same size cap (ValueError)."""
    check_shipping_size(root, "run")
    root_real = root.resolve()
    shadow = Path(tempfile.mkdtemp(prefix="orchard-live-"))
    shadow_real = shadow.resolve()
    try:
        for full, arc in iter_shipped(root):
            dest = shadow / arc
            # A tracked folder since replaced by a symlink: its files would be
            # read from, or written to, somewhere outside. mkdir would follow
            # a link already copied into the shadow, so check before it.
            if any(p.is_symlink() for p in dest.parents if p != shadow and p.is_relative_to(shadow)):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.parent.resolve().is_relative_to(shadow_real):
                continue
            try:
                if full.is_symlink():
                    os.symlink(os.readlink(full), dest)
                elif full.is_file() and full.resolve().is_relative_to(root_real):
                    shutil.copy2(full, dest)
            except (FileExistsError, shutil.SameFileError):
                continue
        for name in (*LINKS, *links):
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or not (root / rel).exists() or (shadow / rel).exists():
                continue
            (shadow / rel).parent.mkdir(parents=True, exist_ok=True)
            (shadow / rel).symlink_to((root / rel).resolve(), target_is_directory=(root / rel).is_dir())
    except BaseException:
        shutil.rmtree(shadow, ignore_errors=True)
        raise
    return shadow


class Live:
    def __init__(self, cmd: list[str], root: Path, shadow: Path, out, timeout: int | None = None,
                 speculate_on: bool = True):
        self.cmd, self.root, self.shadow, self.out, self.timeout = cmd, root, shadow, out, timeout
        self.root_real = root.resolve()
        self.shadow_real = shadow.resolve()
        self.files: dict[str, _File] = {}
        self.order: list[str] = []
        self.checks = 0
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []
        self.think = ""
        self.asked: set[tuple] = set()
        try:
            self.registered = speculate.load(root) if speculate_on else []
        except Exception:
            self.registered = []

    def emit(self, event: dict) -> None:
        with self.lock:
            self.out.write(json.dumps(event) + "\n")
            self.out.flush()

    def _target(self, rel: str) -> Path | None:
        p = Path(rel)
        if (not _PLAIN.fullmatch(rel) or p.is_absolute() or ".." in p.parts
                or any(part.startswith("-") for part in p.parts)):
            return None
        try:
            dest = (self.shadow / p).resolve()
        except (OSError, ValueError):
            return None
        return dest if dest.is_relative_to(self.shadow_real) and dest != self.shadow_real else None

    def _file(self, rel: str) -> _File | None:
        if rel in self.files:
            return self.files[rel]
        src = self.root / rel
        try:  # the original is read from the project, which may hold ignored links out of it
            inside = self._target(rel) is not None and src.resolve().is_relative_to(self.root_real)
        except (OSError, ValueError):
            inside = False
        if not inside:
            self.emit({"event": "error", "path": rel, "why": "path is not a plain relative path inside the project"})
            return None
        try:
            original = src.read_text(encoding="utf-8") if src.is_file() else None
        except (OSError, UnicodeDecodeError):
            original = None
        self.files[rel] = f = _File(rel, original)
        return f

    def apply(self, item) -> None:
        if not isinstance(item, dict):
            self.emit({"event": "error", "why": "each line must be a JSON object"})
            return
        if isinstance(item.get("think"), str):
            self._mentions(item["think"])
        rel = item.get("path")
        if not isinstance(rel, str):
            return
        f = self._file(rel)
        if f is None:
            return
        if item.get("reset"):
            f.reset()
        if isinstance(item.get("text"), str) and item["text"]:
            f.text += item["text"]
            f.done = False
            found = f.boundary()
            if found:
                f.ready = (found[0], found[1], False)
        if item.get("done"):
            f.done = True
            f.ready = (len(f.text), None, True)
        if f.ready and rel not in self.order:
            self.order.append(rel)

    def check_ready(self) -> int:
        while self.order:
            f = self.files[self.order.pop(0)]
            if f.ready is None:  # reset after its boundary arrived
                continue
            offset, unit, final = f.ready
            f.ready = None
            dest = self._target(f.rel)
            if dest is None:
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_symlink():
                    dest.unlink()
                dest.write_text(f.content(offset, final), encoding="utf-8")
            except (OSError, ValueError) as exc:  # a folder of that name, a name too long...
                self.emit({"event": "error", "path": f.rel, "why": f"could not write it: {exc}"})
                continue
            event = self._run(f, offset, unit, final)
            self.emit(event)
            if not event["ok"]:
                return 1
        return 0

    def _run(self, f: _File, offset: int, unit: str | None, final: bool) -> dict:
        values = {"path": f.rel, "unit": unit or ""}
        argv = [_PLACEHOLDER.sub(lambda m: values[m.group(1)], a) for a in self.cmd]
        env = dict(os.environ, ORCHARD_LIVE_PATH=f.rel, ORCHARD_LIVE_UNIT=unit or "",
                   ORCHARD_LIVE_FINAL="1" if final else "0")
        # The shadow first, so an editable install or a PYTHONPATH pointing at
        # the project does not import the real files instead of the new ones.
        paths = [str(p) for p in (self.shadow / "src", self.shadow) if p.is_dir()]
        env["PYTHONPATH"] = os.pathsep.join(paths + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        checked = f.text if final else f.text[:offset]
        event = {"event": "check", "path": f.rel, "line": checked.count("\n") + (0 if checked.endswith("\n") or not checked else 1),
                 "unit": unit, "final": final}
        self.checks += 1
        t0 = time.monotonic()
        why = None
        try:
            proc = subprocess.Popen(argv, cwd=self.shadow, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, errors="replace", env=env, start_new_session=True)
        except OSError as exc:
            event.update(ok=False, exit=None, seconds=0.0, why=str(exc), tail="")
            return event
        # Read in a thread and wait for the check itself: a background child
        # holding the pipe open must not stretch the check, or its timeout.
        tail: collections.deque = collections.deque(maxlen=TAIL_LINES)
        reader = threading.Thread(target=_drain, args=(proc.stdout, tail), daemon=True)
        reader.start()
        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            why = f"timed out after {self.timeout}s"
        except BaseException:  # stopped (Ctrl-C, SIGTERM): the check is in its own session
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)  # whatever it left behind
        proc.wait()
        reader.join(timeout=5)
        if not reader.is_alive():
            proc.stdout.close()
        ok = why is None and proc.returncode == 0
        event.update(ok=ok, exit=proc.returncode, seconds=round(time.monotonic() - t0, 3))
        if not ok:
            event["tail"] = "\n".join(tail)
            if why:
                event["why"] = why
        return event

    def _mentions(self, text: str) -> None:
        if not self.registered:
            return
        buf = self.think + text
        end = 0
        for m in _SPAN.finditer(buf):
            end = m.end()
            self._maybe_speculate(m.group(1))
        rest = buf[end:]
        tick = rest.rfind("`")
        self.think = rest[tick:][-500:] if tick >= 0 and "\n" not in rest[tick:] else ""

    def _maybe_speculate(self, span: str) -> None:
        try:
            words = shlex.split(span.strip())
        except ValueError:
            return
        for check in self.registered:
            key = tuple(check["cmd"])
            if words == check["cmd"] and key not in self.asked:
                self.asked.add(key)
                t = threading.Thread(target=self._speculate, args=(check,), daemon=True)
                self.threads.append(t)
                t.start()

    def _speculate(self, check: dict) -> None:
        # As `orchard speculate` does: no second job for an answered or
        # running check, and no more than its cap in flight.
        event = {"event": "speculate", "cmd": check["cmd"]}
        try:
            with speculate._locked(self.root):
                rows, _stale = speculate.survey(self.root, [check])
                row = rows[0][0]
                if row["state"] != "to queue":
                    event["state"] = row["state"]
                    if row.get("job"):
                        event["job"] = row["job"]
                elif len(speculate.own_entries(self.root)) >= speculate.MAX_IN_FLIGHT:
                    event.update(state="held", why=f"{speculate.MAX_IN_FLIGHT} speculative jobs already in flight")
                else:
                    job, why = speculate.submit(self.root, check)
                    event.update({"job": job} if job else {"why": why})
        except Exception as exc:  # a failed submit must never stop the checks
            event["why"] = str(exc)
        self.emit(event)

    def finish(self) -> None:
        for t in self.threads:
            t.join(timeout=30)


def _drain(stream, tail: collections.deque) -> None:
    with contextlib.suppress(OSError, ValueError):
        for line in stream:
            tail.append(line.rstrip("\n"))


def _reader(inp, q: queue.Queue) -> None:
    try:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                q.put(json.loads(line))
            except ValueError:
                q.put(("error", f"not JSON: {line[:80]}"))
    except Exception as exc:  # undecodable bytes, a closed pipe: end, never hang
        q.put(("error", f"could not read input: {exc}"))
    finally:
        q.put(None)


def serve(cmd: list[str], root: Path, inp=None, out=None, timeout: int | None = None,
          links: list[str] | tuple = (), speculate_on: bool = True) -> int:
    """Run until stdin ends (0) or a check fails (1)."""
    if inp is None:
        inp = sys.stdin
        with contextlib.suppress(AttributeError, ValueError, io.UnsupportedOperation):
            inp.reconfigure(errors="replace")
    out = sys.stdout if out is None else out
    old_term = None
    if threading.current_thread() is threading.main_thread():
        # A caller stopping the model sends SIGTERM; unwind so the check is
        # killed and the shadow removed.
        old_term = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    shadow = live = None
    ended = False
    try:
        shadow = make_shadow(root, links)
        live = Live(cmd, root, shadow, out, timeout, speculate_on)
        q: queue.Queue = queue.Queue()
        threading.Thread(target=_reader, args=(inp, q), daemon=True).start()
        while True:
            # Take everything that arrived while the last check ran: only the
            # newest boundary of each file is checked.
            items = [q.get()]
            with contextlib.suppress(queue.Empty):
                while items[-1] is not None:
                    items.append(q.get_nowait())
            for item in items:
                if isinstance(item, tuple):
                    live.emit({"event": "error", "why": item[1]})
                elif item is not None:
                    live.apply(item)
            if live.check_ready():
                ended = True
                return 1
            if items[-1] is None:
                live.emit({"event": "done", "checks": live.checks})
                ended = True
                return 0
    finally:
        # A caller sends SIGTERM as soon as it reads the failure; hold it
        # until the shadow is gone, or it would stop rmtree halfway.
        held: list[int] = []
        if old_term is not None:
            signal.signal(signal.SIGTERM, lambda signum, frame: held.append(signum))
        try:
            if shadow is not None:
                shutil.rmtree(shadow, ignore_errors=True)
        finally:
            if old_term is not None:
                signal.signal(signal.SIGTERM, old_term)
        if held:
            sys.exit(128 + held[0])
        if ended:
            live.finish()  # let a queued speculate report its job; not when stopped
