"""orchard live: boundaries, the shadow copy, checks, and the Claude harness
(driven by a fake client; nothing here calls the API)."""
import contextlib
import copy
import importlib.util
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import orchard
from orchard import live, speculate

PY = sys.executable
SRC = str(Path(orchard.__file__).resolve().parent.parent)
AGENT_PATH = Path(__file__).resolve().parent.parent / "contrib" / "live_check_agent.py"


def _boundaries(rel, text, original=None):
    """(lines checked, unit) for each boundary, feeding one character at a time."""
    f = live._File(rel, original)
    found = []
    for c in text:
        f.text += c
        b = f.boundary()
        if b:
            found.append((f.text[:b[0]].count("\n"), b[1]))
    return found


@contextlib.contextmanager
def _session(root, cmd, timeout=None):
    out = io.StringIO()
    shadow = live.make_shadow(root)
    try:
        yield live.Live(cmd, root, shadow, out, timeout, speculate_on=False), out
    finally:
        shutil.rmtree(shadow, ignore_errors=True)


def _feed(lv, rel, text):
    for c in text:
        lv.apply({"path": rel, "text": c})
        if lv.check_ready():
            return 1
    return 0


def _events(out):
    return [json.loads(line) for line in out.getvalue().splitlines()]


# --- boundaries ---------------------------------------------------------------

def test_python_boundaries_are_the_defs_that_just_closed():
    text = (
        "import os\n\n"
        "def alpha():\n    return 1\n\n\n"
        "def beta(x):\n    if x:\n        return 2\n    return 3\n\n"
        "class Gamma:\n    def one(self):\n        return 1\n\n    def two(self):\n        return 2\n\n"
        "x = alpha()\n"
    )
    assert _boundaries("m.py", text) == [(6, "alpha"), (11, "beta"), (15, "one"), (18, "Gamma")]


def test_python_one_liners_and_decorators():
    assert _boundaries("m.py", "def f(): return 1\ndef g(): return 2\n") == [(1, "f")]
    text = "@dec\ndef f():\n    pass\n\n@dec\ndef g():\n    pass\n"
    assert _boundaries("m.py", text) == [(4, "f")]


def test_python_prefix_that_does_not_parse_is_no_boundary():
    in_string = 'def a():\n    s = """\nx = 1\n"""\n    return s\n\ny = 2\n'
    assert _boundaries("m.py", in_string) == [(6, "a")]
    in_call = "def a():\n    return foo(\n1,\n    )\n\nb = 1\n"
    assert _boundaries("m.py", in_call) == [(5, "a")]


def test_brace_boundaries_skip_strings_and_comments():
    js = ('function add(a, b) {\n  return a + b;\n}\n\n'
          'class Box {\n  get() { return "}"; }\n}\n'
          "// a } in a comment\nconst s = '{';\n")
    assert [u for _, u in _boundaries("a.js", js)] == ["add", "Box"]


def test_rust_lifetimes_are_not_quotes():
    rs = "fn first<'a>(x: &'a str) -> &'a str {\n    x\n}\n\nstruct S;\nimpl S {\n    fn go(&self) {}\n}\n"
    assert [u for _, u in _boundaries("a.rs", rs)] == ["first", "S"]


def test_plain_files_only_check_when_done():
    assert _boundaries("notes.txt", "a\n}\n\nb\n") == []


def test_fill_keeps_old_definitions_the_new_text_has_not_reached():
    original = ("from __future__ import annotations\nimport os\n\nX = 1\n\n"
                "def a():\n    return 1\n\n\ndef b():\n    return 2\n")
    prefix = "from __future__ import annotations\n\ndef a():\n    return 10\n"
    filled = live._fill(prefix, original)
    assert filled.startswith(prefix.rstrip("\n"))
    assert filled.count("__future__") == 1
    assert "import os" in filled and "X = 1" in filled and "def b():" in filled
    assert "return 1\n" not in filled
    ns = {}
    exec(compile(filled, "m.py", "exec"), ns)
    assert ns["a"]() == 10 and ns["b"]() == 2


def test_braces_in_char_regex_and_text_block_literals_are_not_code():
    assert _boundaries("a.rs", "fn is_close(c: char) -> bool {\n    c == '}'\n}\n") == [(3, "is_close")]
    assert _boundaries("a.js", 'function f(s) {\n  return s.replace(/}/g, "");\n}\n') == [(3, "f")]
    assert _boundaries("a.js", "const half = n / 2;\nfunction g() {\n  return 1 / 2 / 3;\n}\n") == [(4, "g")]
    assert _boundaries("A.java", 'class A {\n  String s = """\n    }\n  """;\n}\n') == [(5, "A")]


def test_a_callback_argument_is_not_the_end_of_the_statement():
    js = "setTimeout(() => {\n  run();\n},\n1000);\n"
    assert [n for n, _ in _boundaries("a.js", js)] == [4]


def test_brace_units_name_arrow_functions_and_generic_impls():
    assert live._brace_unit("const add = (a, b) => ") == "add"
    assert live._brace_unit("export const handler = async (event) => ") == "handler"
    assert live._brace_unit("impl<T: Clone> Stack<T> ") == "Stack"
    assert live._brace_unit("impl<T> Display for Wrapper<T> ") == "Wrapper"


def test_fill_never_rebinds_a_name_the_new_text_defines():
    original = "import os\nfrom json import loads\n\ndef other():\n    return os.sep\n"
    filled = live._fill("import os.path\n\ndef loads(s):\n    return s\n", original)
    assert "from json import loads" not in filled and "import os\n" in filled and "def other" in filled
    ns = {}
    exec(compile(filled, "m.py", "exec"), ns)
    assert ns["loads"]("x") == "x" and ns["other"]() == os.sep


def test_fill_counts_lines_the_way_ast_does():
    original = "import os\n\x0c\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n"
    ns = {}
    exec(compile(live._fill("def a():\n    return 10\n", original), "m.py", "exec"), ns)
    assert ns["b"]() == 2


def test_fill_keeps_the_rest_of_a_class_being_rewritten(tmp_path):
    (tmp_path / "mod.py").write_text("class C:\n    def one(self):\n        return 1\n\n"
                                     "    def two(self):\n        return 2\n")
    check = [PY, "-c", "import mod; assert mod.C().one() == 10 and mod.C().two() == 2"]
    with _session(tmp_path, check) as (lv, out):
        _feed(lv, "mod.py", "class C:\n    def one(self):\n        return 10\n\n    def three(self):\n")
    first = _events(out)[0]
    assert first["ok"] and first["unit"] == "one"


def test_a_parser_that_gives_up_is_no_boundary_not_a_crash(monkeypatch):
    def give_up(*a, **k):
        raise RecursionError("too deep")
    monkeypatch.setattr(live.ast, "parse", give_up)
    assert live._fill("x = 1\n", "y = 2\n") == "x = 1\n"
    assert _boundaries("m.py", "def a():\n    pass\n\nx = 1\n") == []


# --- checks -------------------------------------------------------------------

def test_first_failing_function_stops_with_its_output(tmp_path):
    text = "def a():\n    return 1/0\n\ndef b():\n    return 2\n"
    with _session(tmp_path, [PY, "-c", "import mod; mod.a()"]) as (lv, out):
        assert _feed(lv, "mod.py", text) == 1
    (ev,) = _events(out)
    assert ev["event"] == "check" and not ev["ok"] and ev["exit"] == 1
    assert (ev["unit"], ev["line"], ev["final"]) == ("a", 3, False)
    assert "ZeroDivisionError" in ev["tail"]


def test_rewrite_borrows_old_definitions_until_the_file_is_done(tmp_path):
    (tmp_path / "mod.py").write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n")
    check = [PY, "-c", "import mod; assert mod.a() == 10; assert mod.b() == 2"]
    with _session(tmp_path, check) as (lv, out):
        assert _feed(lv, "mod.py", "def a():\n    return 10\n\n\ndef c():\n    return 3\n") == 0
        lv.apply({"path": "mod.py", "done": True})
        assert lv.check_ready() == 1  # b() is really gone now
    first, final = _events(out)
    assert first["ok"] and first["unit"] == "a"
    assert not final["ok"] and final["final"] and "AttributeError" in final["tail"]
    assert (tmp_path / "mod.py").read_text().count("return 1\n") == 1  # the project is untouched


def test_placeholders_and_env(tmp_path):
    script = ("import os, sys; print(*sys.argv[1:], os.environ['ORCHARD_LIVE_PATH'], "
              "os.environ['ORCHARD_LIVE_UNIT'], os.environ['ORCHARD_LIVE_FINAL']); sys.exit(3)")
    with _session(tmp_path, [PY, "-c", script, "{path}", "{unit}"]) as (lv, out):
        assert _feed(lv, "pkg/mod.py", "def a():\n    pass\n\nx = 1\n") == 1
    (ev,) = _events(out)
    assert ev["exit"] == 3
    assert ev["tail"].strip() == "pkg/mod.py a pkg/mod.py a 0"


def test_only_the_newest_boundary_of_a_burst_is_checked(tmp_path):
    with _session(tmp_path, [PY, "-c", "pass"]) as (lv, out):
        lv.apply({"path": "mod.py", "text": "def a():\n    pass\n\ndef b():\n    pass\n\nx = 1\n"})
        lv.apply({"path": "mod.py", "done": True})
        assert lv.check_ready() == 0
    (ev,) = _events(out)
    assert ev["ok"] and ev["final"] and lv.checks == 1


def test_timeout_is_a_failure_with_a_reason(tmp_path):
    with _session(tmp_path, [PY, "-c", "import time; time.sleep(30)"], timeout=1) as (lv, out):
        lv.apply({"path": "notes.txt", "text": "hi"})
        lv.apply({"path": "notes.txt", "done": True})
        t0 = time.monotonic()
        assert lv.check_ready() == 1
        assert time.monotonic() - t0 < 15
    (ev,) = _events(out)
    assert not ev["ok"] and ev["why"] == "timed out after 1s" and ev["line"] == 1


def test_a_background_child_holding_the_pipe_does_not_stretch_the_check(tmp_path):
    with _session(tmp_path, ["sh", "-c", "sleep 6 & exit 0"], timeout=2) as (lv, out):
        lv.apply({"path": "notes.txt", "text": "hi"})
        lv.apply({"path": "notes.txt", "done": True})
        t0 = time.monotonic()
        assert lv.check_ready() == 0
        assert time.monotonic() - t0 < 4
    (ev,) = _events(out)
    assert ev["ok"]


def test_checks_import_the_shadow_not_the_real_project(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("")
    monkeypatch.setenv("PYTHONPATH", str(root / "src"))  # as an editable install would
    check = [PY, "-c", "import os, sys, pkg; "
             "sys.exit(0 if os.path.realpath(pkg.__file__).startswith(os.path.realpath(os.getcwd())) else 1)"]
    with _session(root, check) as (lv, out):
        lv.apply({"path": "notes.txt", "text": "hi"})
        lv.apply({"path": "notes.txt", "done": True})
        assert lv.check_ready() == 0
    assert _events(out)[0]["ok"]


def test_cli_timeout_zero_means_no_timeout(tmp_path):
    env = dict(os.environ, PYTHONPATH=SRC, HOME=str(tmp_path / "home"))
    lines = json.dumps({"path": "n.txt", "text": "hi"}) + "\n" + json.dumps({"path": "n.txt", "done": True}) + "\n"
    p = subprocess.run([PY, "-m", "orchard", "live", "--timeout", "0", "--no-speculate", "--", PY, "-c", "pass"],
                       cwd=tmp_path, env=env, input=lines, capture_output=True, text=True, timeout=60)
    evs = [json.loads(x) for x in p.stdout.splitlines()]
    assert p.returncode == 0 and evs[0]["ok"], p.stdout + p.stderr


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_serve_rejects_bad_lines_and_outside_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    root = tmp_path / "proj"
    root.mkdir()
    lines = ["not json", "[1]", json.dumps({"path": "../x.py", "text": "a"}),
             json.dumps({"path": "../x.py", "done": True}), json.dumps({"path": "/etc/x.py", "text": "a"}),
             json.dumps({"path": "ok.txt", "text": "hi"}), json.dumps({"path": "ok.txt", "done": True})]
    out = io.StringIO()
    rc = live.serve([PY, "-c", "pass"], root, inp=io.StringIO("\n".join(lines) + "\n"), out=out,
                    speculate_on=False)
    assert rc == 0
    evs = _events(out)
    whys = [e.get("why") for e in evs if e["event"] == "error"]
    bad = "path is not a plain relative path inside the project"
    assert whys == ["not JSON: not json", "each line must be a JSON object", bad, bad, bad]
    assert [e for e in evs if e["event"] == "check"][0]["ok"]
    assert evs[-1] == {"event": "done", "checks": 1}
    assert os.listdir(tmp_path / "tmp") == []  # an escape from the shadow would have left tmp/x.py


def test_input_that_cannot_be_decoded_ends_serve_instead_of_hanging(tmp_path):
    raw = io.TextIOWrapper(io.BytesIO(b'{"path": "a.txt", "text": "\xff"}\n'), encoding="utf-8")
    out = io.StringIO()
    assert live.serve([PY, "-c", "pass"], tmp_path, inp=raw, out=out, speculate_on=False) == 0
    evs = _events(out)
    assert evs[0]["event"] == "error" and evs[0]["why"].startswith("could not read input")
    assert evs[-1]["event"] == "done"


def test_a_project_file_linked_outside_is_never_read_or_written(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "conf.py").write_text("TOKEN = 'outside-secret'\n")
    root = tmp_path / "proj"
    root.mkdir()
    (root / "link.py").symlink_to(private / "conf.py")
    check = [PY, "-c", "import sys; print(open(sys.argv[1]).read()); sys.exit(1)", "{path}"]
    lines = [{"path": "link.py", "text": "def a():\n    pass\n\nx = 1\n"}, {"path": "link.py", "done": True}]
    out = io.StringIO()
    rc = live.serve(check, root, inp=io.StringIO("".join(json.dumps(x) + "\n" for x in lines)), out=out,
                    speculate_on=False)
    assert rc == 0 and "outside-secret" not in out.getvalue()
    assert [e["event"] for e in _events(out)] == ["error", "error", "done"]
    assert (private / "conf.py").read_text() == "TOKEN = 'outside-secret'\n"


def test_sigterm_during_cleanup_still_removes_the_shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    real = shutil.rmtree

    def interrupted(path, *a, **k):
        if Path(path).name.startswith("orchard-live-"):
            os.kill(os.getpid(), signal.SIGTERM)
        return real(path, *a, **k)
    monkeypatch.setattr(live.shutil, "rmtree", interrupted)
    lines = json.dumps({"path": "n.txt", "text": "hi"}) + "\n" + json.dumps({"path": "n.txt", "done": True}) + "\n"
    with pytest.raises(SystemExit) as stopped:
        live.serve([PY, "-c", "pass"], tmp_path, inp=io.StringIO(lines), out=io.StringIO(), speculate_on=False)
    assert stopped.value.code == 128 + signal.SIGTERM
    assert os.listdir(tmp_path / "tmp") == []


def test_paths_that_could_become_options_or_shell_code_are_refused(tmp_path):
    outside, root = tmp_path / "outside", tmp_path / "proj"
    outside.mkdir()
    root.mkdir()
    with _session(root, [PY, "-c", "pass"]) as (lv, out):
        (lv.shadow / "out").symlink_to(outside, target_is_directory=True)
        for rel in ("-x.py", "--junitxml=/tmp/r.xml", "src/-k", "a b.py", "a;b.py", "a$(id).py", "t{unit}.py",
                    "a\u0000b.py", "~/x.py", "../x.py", "/abs.py", "", "out/x.py"):
            assert lv._target(rel) is None, rel
        assert lv._target("src/a_b-c.d@2+x.py") == lv.shadow_real / "src/a_b-c.d@2+x.py"


def test_original_is_never_read_through_an_ignored_link_out_of_the_project(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "conf.py").write_text("import os\nTOKEN = 'outside-secret'\n")
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init", "-q")
    (root / ".gitignore").write_text("vendor\n")
    (root / "vendor").symlink_to(private, target_is_directory=True)
    check = [PY, "-c", "import sys; print(open(sys.argv[1]).read()); sys.exit(1)", "{path}"]
    lines = [{"path": "vendor/conf.py", "text": "def a():\n    pass\n\nx = 1\n"}, {"path": "vendor/conf.py", "done": True}]
    out = io.StringIO()
    rc = live.serve(check, root, inp=io.StringIO("".join(json.dumps(x) + "\n" for x in lines)), out=out,
                    speculate_on=False)
    assert rc == 0 and "outside-secret" not in out.getvalue()
    assert [e["event"] for e in _events(out)] == ["error", "error", "done"]  # the text line and the done line


def test_a_path_that_cannot_be_written_is_an_error_not_a_crash(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "README").write_text("hi\n")
    lines = []
    for rel in ("src", "README/x.py", "a" * 300 + ".py"):
        lines += [{"path": rel, "text": "x"}, {"path": rel, "done": True}]
    out = io.StringIO()
    rc = live.serve([PY, "-c", "pass"], tmp_path, inp=io.StringIO("".join(json.dumps(x) + "\n" for x in lines)),
                    out=out, speculate_on=False)
    evs = _events(out)
    assert rc == 0 and evs[-1] == {"event": "done", "checks": 0}
    assert all(e["event"] == "error" for e in evs[:-1]) and len(evs) == 4


def test_shadow_keeps_the_shipping_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "data.bin").write_bytes(b"x" * 4096)
    monkeypatch.setenv("ORCHARD_MAX_TAR_MB", "0.001")
    with pytest.raises(ValueError, match="cap"):
        live.make_shadow(tmp_path / "proj")
    assert os.listdir(tmp_path / "tmp") == []


def test_tracked_folder_replaced_by_a_link_is_not_copied_through(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    root, elsewhere = tmp_path / "proj", tmp_path / "elsewhere"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("x = 1\n")
    (root / "pkg" / "sub" / "b.py").write_text("z = 3\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    shutil.rmtree(root / "pkg")
    elsewhere.mkdir()
    (elsewhere / "a.py").write_text("y = 2\n")
    (root / "pkg").symlink_to(elsewhere, target_is_directory=True)
    shadow = live.make_shadow(root)
    try:
        assert (shadow / "pkg").is_symlink()
        assert sorted(os.listdir(elsewhere)) == ["a.py"] and (elsewhere / "a.py").read_text() == "y = 2\n"
        assert not (elsewhere / "sub").exists()
    finally:
        shutil.rmtree(shadow)


def test_sigterm_kills_the_running_check_and_removes_the_shadow(tmp_path):
    root, tmp, mark = tmp_path / "proj", tmp_path / "tmp", tmp_path / "pid"
    root.mkdir()
    tmp.mkdir()
    env = dict(os.environ, PYTHONPATH=SRC, HOME=str(tmp_path / "home"), TMPDIR=str(tmp), MARK=str(mark))
    check = "import os, time; open(os.environ['MARK'], 'w').write(str(os.getpid())); time.sleep(60)"
    p = subprocess.Popen([PY, "-m", "orchard", "live", "--no-speculate", "--", PY, "-c", check], cwd=root, env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    p.stdin.write(json.dumps({"path": "n.txt", "text": "hi"}) + "\n" + json.dumps({"path": "n.txt", "done": True}) + "\n")
    p.stdin.flush()
    deadline = time.monotonic() + 30
    while not (mark.exists() and mark.read_text()) and time.monotonic() < deadline:
        time.sleep(0.1)
    pid = int(mark.read_text())
    p.terminate()
    assert p.wait(timeout=15) != 0
    assert [n for n in os.listdir(tmp) if n.startswith("orchard-live-")] == []
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_backticked_registered_check_is_speculated_once(tmp_path, monkeypatch):
    check = {"cmd": ["make", "test"], "tags": [], "env": {}, "timeout": None}
    submitted = []
    _gate(monkeypatch, check)
    monkeypatch.setattr(speculate, "submit", lambda root, c: (submitted.append(c), ("job123", None))[1])
    lines = [{"think": "I will run `make"}, {"think": " test` now, then `make test` again and `make lint`"}]
    out = io.StringIO()
    rc = live.serve([PY, "-c", "pass"], tmp_path, inp=io.StringIO("".join(json.dumps(x) + "\n" for x in lines)),
                    out=out)
    assert rc == 0 and submitted == [check]
    assert {"event": "speculate", "cmd": ["make", "test"], "job": "job123"} in _events(out)


def _gate(monkeypatch, check, state="to queue", job=None, in_flight=0):
    row = {"state": state, **({"job": job} if job else {})}
    monkeypatch.setattr(speculate, "load", lambda root: [check])
    monkeypatch.setattr(speculate, "_locked", lambda root: contextlib.nullcontext())
    monkeypatch.setattr(speculate, "survey", lambda root, checks: ([(row, checks[0])], []))
    monkeypatch.setattr(speculate, "own_entries", lambda root: [{}] * in_flight)


@pytest.mark.parametrize("state,job,in_flight,expect", [
    ("answered", "old1", 0, {"state": "answered", "job": "old1"}),
    ("in flight", "run2", 0, {"state": "in flight", "job": "run2"}),
    ("to queue", None, 3, {"state": "held", "why": "3 speculative jobs already in flight"}),
])
def test_speculate_from_live_keeps_the_dedupe_and_the_cap(tmp_path, monkeypatch, state, job, in_flight, expect):
    check = {"cmd": ["make", "test"], "tags": [], "env": {}, "timeout": None}
    _gate(monkeypatch, check, state, job, in_flight)
    monkeypatch.setattr(speculate, "submit", lambda root, c: pytest.fail("submitted"))
    out = io.StringIO()
    live.serve([PY, "-c", "pass"], tmp_path, inp=io.StringIO(json.dumps({"think": "run `make test`"}) + "\n"),
               out=out)
    assert {"event": "speculate", "cmd": ["make", "test"], **expect} in _events(out)


# --- the Claude harness -------------------------------------------------------

def _agent():
    spec = importlib.util.spec_from_file_location("live_check_agent", AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fields_decoder_handles_pieces_split_anywhere():
    agent = _agent()
    value = 'a "q" \\ back\n\ttab \u00e9 \U0001f600 \u2028 end'
    raw = json.dumps({"path": "x.py", "n": 3, "nested": {"k": [1, "}", {"s": "\\\""}]}, "contents": value})
    for size in (1, 2, 3, 5, 11):
        f = agent._Fields()
        for i in range(0, len(raw), size):
            f.feed(raw[i:i + size])
            assert value.startswith(f.values.get("contents", ""))
        assert f.values == {"path": "x.py", "contents": value}
        assert f.closed == {"path", "contents"}
    lone = agent._Fields()
    lone.feed('{"contents": "a\\ud83dz", "path": "b\\ud83d"}')
    assert lone.values == {"contents": "a\ufffdz", "path": "b\ufffd"}


def test_tool_input_is_validated_before_running(tmp_path):
    agent = _agent()
    root = tmp_path.resolve()
    bad = agent._run_tool(NS(id="t1", name="write_file", input={"path": 3}), root, None, {})
    assert bad["is_error"] and "INVALID_JSON" in json.loads(bad["content"])
    out = agent._run_tool(NS(id="t2", name="write_file", input={"path": "../x", "contents": ""}), root, None, {})
    assert out["is_error"] and "inside the project" in out["content"]
    assert not (tmp_path.parent / "x").exists()
    for rel in ("--junitxml=/tmp/r.xml", "a\u0000b.py", "a b.py"):
        assert agent._inside(root, rel) is None, rel
    (root / "src").mkdir()
    out = agent._run_tool(NS(id="t3", name="write_file", input={"path": "src", "contents": "x"}), root, None, {})
    assert out["is_error"]


def _start(i, type_, **kw):
    return NS(type="content_block_start", index=i, content_block=NS(type=type_, **kw))


def _delta(i, dtype, slow=False, **kw):
    return NS(type="content_block_delta", index=i, delta=NS(type=dtype, **kw), slow=slow)


def _stop(i):
    return NS(type="content_block_stop", index=i)


def _write(i, tid, path, contents, slow_from=None):
    raw = json.dumps({"path": path, "contents": contents})
    evs = [_start(i, "tool_use", id=tid, name="write_file")]
    for k in range(0, len(raw), 7):
        evs.append(_delta(i, "input_json_delta", slow=slow_from is not None and k >= slow_from,
                          partial_json=raw[k:k + 7]))
    return evs + [_stop(i)]


def _tool(tid, path, contents):
    return NS(type="tool_use", id=tid, name="write_file", input={"path": path, "contents": contents})


class _Stream:
    def __init__(self, events, final):
        self.events, self.final, self.sent = events, final, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for ev in self.events:
            if getattr(ev, "slow", False):
                time.sleep(0.05)
            self.sent += 1
            yield ev

    def get_final_message(self):
        return self.final


class _Client:
    def __init__(self, *streams):
        self.streams, self.requests = list(streams), []
        self.beta = NS(messages=self)

    def stream(self, **kw):
        self.requests.append({**kw, "messages": copy.deepcopy(kw["messages"])})
        return self.streams.pop(0)


def _texts(message):
    content = message["content"]
    return [content] if isinstance(content, str) else [b.get("text", "") for b in content if isinstance(b, dict)]


def test_harness_stops_the_stream_at_the_first_failure_and_asks_for_a_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", SRC)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "proj"
    root.mkdir()
    broken = "def a():\n    return 1/0\n\ndef b():\n    return 2\n" + "# filler\n" * 400
    good = "def a():\n    return 1\n\ndef b():\n    return 2\n"
    raw = json.dumps({"path": "mod.py", "contents": broken})
    first = [_start(0, "text"), _delta(0, "text_delta", text="Writing mod.py."), _stop(0),
             *_write(1, "t1", "mod.py", broken, slow_from=raw.index("# filler"))]
    second = _write(0, "t2", "mod.py", good)
    cut = _Stream(first, NS(stop_reason="tool_use", content=[_tool("t1", "mod.py", broken)]))
    client = _Client(
        cut,
        _Stream(second, NS(stop_reason="tool_use", content=[_tool("t2", "mod.py", good)])),
        _Stream([_start(0, "text"), _delta(0, "text_delta", text="Done."), _stop(0)],
                NS(stop_reason="end_turn", content=[NS(type="text", text="Done.")])),
    )
    agent = _agent()
    check = f"{shlex.quote(PY)} -c 'import mod; mod.a()'"
    rc = agent.run("write mod.py", root, check, client=client, orchard=[PY, "-m", "orchard"], out=io.StringIO())

    assert rc == 0
    assert (root / "mod.py").read_text() == good
    assert client.streams == []
    assert client.requests[0]["tools"][0]["eager_input_streaming"] is True
    assert cut.sent < len(first) - 50  # stopped well before the filler ran out
    retry = client.requests[1]["messages"]
    assert [m["role"] for m in retry] == ["user"]
    note = "\n".join(_texts(retry[0]))
    assert "I stopped you while you were writing mod.py at line 3, just after `a`" in note
    assert "ZeroDivisionError" in note and "Nothing from that reply was applied" in note
    assert "# filler" not in note
    final = client.requests[2]["messages"]
    assert [m["role"] for m in final] == ["user", "assistant", "user"]
    (result,) = final[2]["content"]
    assert result["tool_use_id"] == "t2" and result["content"] == "Written; the check passed."
    for req in client.requests:
        assert req["model"] == agent.MODEL and req["fallbacks"] == "default"
        assert req["betas"] == ["server-side-fallback-2026-07-01"]


QUIET = [PY, "-c", "import sys\nfor _ in sys.stdin: pass"]  # speaks the protocol, reports nothing


def test_harness_drops_what_came_before_a_fallback(tmp_path):
    root = tmp_path
    events = [_start(0, "thinking"), _delta(0, "thinking_delta", thinking="hm"), _stop(0),
              *_write(1, "old", "a.txt", "old"), _start(2, "fallback"), _stop(2),
              *_write(3, "new", "a.txt", "new")]
    content = [NS(type="thinking", thinking="hm"), _tool("old", "a.txt", "old"), NS(type="fallback"),
               _tool("new", "a.txt", "new")]
    client = _Client(_Stream(events, NS(stop_reason="tool_use", content=content)),
                     _Stream([], NS(stop_reason="end_turn", content=[])))
    agent = _agent()
    assert agent.run("x", root, "true", client=client, orchard=QUIET, out=io.StringIO()) == 0
    assert (root / "a.txt").read_text() == "new"
    _, assistant, results = client.requests[1]["messages"]
    assert [getattr(b, "type", None) for b in assistant["content"]] == ["fallback", "tool_use"]
    assert assistant["content"][1].id == "new"
    assert [r["tool_use_id"] for r in results["content"]] == ["new"]
    assert results["content"][0]["content"] == "Written; not checked."


def test_harness_never_runs_tools_from_a_truncated_or_refused_reply(tmp_path):
    agent = _agent()
    for reason in ("max_tokens", "refusal", "model_context_window_exceeded"):
        client = _Client(_Stream(_write(0, "t1", "a.txt", "x"),
                                 NS(stop_reason=reason, content=[_tool("t1", "a.txt", "x")])))
        assert agent.run("x", tmp_path, "true", client=client, orchard=QUIET, out=io.StringIO()) == 1
        assert not (tmp_path / "a.txt").exists()


def test_harness_written_text_is_the_newest_copy_of_a_file():
    agent = _agent()
    blocks = {1: {"path": "a.py", "fields": NS(values={"contents": "old"})},
              3: {"path": "a.py", "fields": NS(values={"contents": "new"})}}
    assert agent._written(blocks, "a.py") == "new"


def test_harness_restarts_orchard_live_at_a_fallback(tmp_path):
    agent = _agent()
    mark = tmp_path / "started"
    fake = [PY, "-c", "import json, os, sys\n"
            f"m = {str(mark)!r}\n"
            "if not os.path.exists(m):\n"
            "    open(m, 'w').close()\n"
            "    print(json.dumps({'event': 'check', 'ok': False, 'path': 'a.py', 'final': True}), flush=True)\n"
            "for _ in sys.stdin: pass"]
    lv = agent._Live(fake, "true", tmp_path, timeout=7)
    assert lv.argv[len(fake):len(fake) + 4] == ["live", "--timeout", "7", "--"]
    deadline = time.monotonic() + 15
    while lv.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    first = lv.proc
    assert lv.failure and lv.results
    agent._on_event(_start(2, "fallback"), {}, lv, tmp_path, io.StringIO())
    assert lv.proc is not first and lv.failure is None and lv.results == {}
    assert lv.close() is None


def test_harness_gives_up_on_an_orchard_live_that_stops_answering(tmp_path, monkeypatch):
    agent = _agent()
    monkeypatch.setattr(agent, "CLOSE_GRACE", 0)
    lv = agent._Live([PY, "-c", "import time; time.sleep(60)"], "true", tmp_path, timeout=1)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="stopped answering"):
        lv.close()
    assert time.monotonic() - t0 < 15 and lv.proc.poll() is not None


def test_harness_does_not_retry_its_own_encoding_errors(tmp_path):
    agent = _agent()

    class Broken(_Stream):
        def __iter__(self):
            raise UnicodeEncodeError("ascii", "\u2603", 0, 1, "terminal can't show it")
            yield

    client = _Client(Broken([], None), Broken([], None))
    with pytest.raises(UnicodeEncodeError):
        agent.run("x", tmp_path, "true", client=client, orchard=QUIET, out=io.StringIO())
    assert len(client.requests) == 1
