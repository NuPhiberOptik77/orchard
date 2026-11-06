import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import threading
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from orchard import ai, cli, config, dispatcher, receipts

runner = CliRunner()

# Stands in for ssh, so no test reaches a real machine. A tunnel and a pause stay
# up until killed, like the real ones, and say so in events.log. Everything else
# runs here in a shell, which finds the fake pgrep and pkill first on PATH.
SSH = r'''#!/usr/bin/env python3
import base64, os, signal, sys, time
home = os.environ["AI_STUB"]
def note(line):
    with open(f"{home}/events.log", "a") as log:
        log.write(line + "\n")
def stay_up(up, down):
    def finish(*_):
        note(down)
        os._exit(0)
    signal.signal(signal.SIGTERM, finish)
    note(up)
    return finish
def next_free():
    with open(f"{home}/free.txt") as handle:
        values = handle.read().split()
    if len(values) > 1:
        with open(f"{home}/free.txt", "w") as handle:
            handle.write("\n".join(values[1:]))
    return values[0]
args, tunnel, host, words = sys.argv[1:], False, None, []
while args:
    arg = args.pop(0)
    if arg == "-T":
        continue
    if arg == "-N":
        tunnel = True
    elif arg in ("-o", "-R"):
        args.pop(0)
    else:
        host, words = arg, args
        break
if host == "gone":
    sys.exit(255)
if tunnel:
    stay_up("tunnel up", "tunnel down")
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
    with open(f"{home}/engine.pid") as handle:
        os.kill(int(handle.read()), signal.SIGTERM)
elif command.startswith("powershell"):
    script = base64.b64decode(command.split()[-1]).decode("utf-16-le")
    if "NtSuspendProcess" in script:
        stay_up("pause on", "pause off")
        print("held OneDrive suspended 3 of 3\nheld done", flush=True)
        time.sleep(600)
    elif "NtResumeProcess" in script:
        note("apps resumed")
        stuck = os.path.exists(f"{home}/stuck")
        print(f"back OneDrive suspended {3 if stuck else 0} of 3\nback done")
    else:
        print("gpu Some GPU\nmemory_mb 16000\ndisk_free_gb 60")
else:
    os.execvp("bash", ["bash", "-c", command])
'''

PGREP = '''#!/bin/bash
[ -e "$AI_STUB/busy" ] && echo "123 ./llama-completion -m model"
exit 0
'''

ENGINE = '''#!/bin/bash
echo $$ > "$AI_STUB/engine.pid"
printf '%s\\n' "$@" > "$AI_STUB/engine.args"
env > "$AI_STUB/engine.env"
echo "engine started" >> "$AI_STUB/events.log"
printf '%s' "${AI_ENGINE_TEXT:-The sky scatters blue.}"
if [ -e "$AI_STUB/slow" ]; then exec sleep 600; fi
cat >&2 <<'LOG'
21.39.292.854 I common_perf_print:        load time =  423533.39 ms
21.39.294.150 I common_perf_print: prompt eval time =  12500.26 ms /    40 tokens (  312.48 ms per token,     3.20 tokens per second)
21.39.294.198 I common_perf_print:        eval time =  93000.00 ms /    31 runs   ( 3000.00 ms per token,     0.33 tokens per second)
LOG
exit "$(cat "$AI_STUB/engine.exit" 2>/dev/null || echo 0)"
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
    """A fake world: ssh, the engine, the helper, and short waits."""
    home = tmp_path / "stub"
    tools = home / "bin"
    tools.mkdir(parents=True)
    _script(tools / "ssh", SSH)
    _script(tools / "pgrep", PGREP)
    _script(tools / "pkill", "#!/bin/sh\nexit 0\n")
    engine = tmp_path / "engine"
    engine.mkdir()
    _script(engine / "llama-completion", ENGINE)
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
    monkeypatch.setenv("AI_STUB", str(home))
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(ai, "AI_DIR", tmp_path / "ai")
    monkeypatch.setattr(ai, "CONFIG_PATH", tmp_path / "ai.toml")
    monkeypatch.setattr(dispatcher, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(receipts, "_key", lambda: b"test key")
    monkeypatch.setattr(ai, "_local_pressure", lambda: 0)
    monkeypatch.setattr(ai, "_pkill_local", lambda _pattern: None)
    for name, value in (("GUARD_POLL", 0.05), ("MEMORY_POLL", 0.01), ("HOLD_POLL", 0.01), ("HELPER_SETTLE", 0.1),
                        ("TUNNEL_SETTLE", 0.3), ("TUNNEL_GAP", 0), ("RESUME_TRIES", 2), ("RESUME_GAP", 0),
                        ("PAUSE_WAIT", 10), ("KILL_WAIT", 10)):
        monkeypatch.setattr(ai, name, value)
    return SimpleNamespace(home=home, engine=engine, helper=helper, models=models)


def write_settings(stub, *, helper=False, windows=False, worker="", start=0, stop=0, extra=""):
    text = f'''
[engine]
host = "box"
folder = "{stub.engine}"
model = "{stub.models}/m-00001-of-00002.gguf"
threads = 4
words = 8
devices = "CUDA0,RPC0"
split = "22,78"
overrides = "_exps=CPU"
args = ["--temp", "0"]
start_free_mb = {start}
stop_free_mb = {stop}
settle_seconds = 0
worker = "{worker}"
{extra}

[engine.env]
GGML_RPC_BATCH_FETCH = "1"
'''
    if helper:
        text += f'\n[helper]\nfolder = "{stub.helper}"\ndevice = "MTL0"\n'
    if windows:
        text += '\n[windows]\nhost = "winbox"\npause = ["OneDrive"]\n'
    ai.CONFIG_PATH.write_text(text)


def events(stub):
    return (stub.home / "events.log").read_text().split("\n")[:-1]


def invoke(*args):
    return runner.invoke(cli.app, ["ai", *args])


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


def test_answers_and_puts_everything_back_in_order(stub, monkeypatch):
    write_settings(stub, helper=True, windows=True, worker="box", start=5000, stop=200)
    cancelled = hold_job(monkeypatch, [{"state": "running"}])
    started = []
    for name in ("start_helper", "start_tunnel"):
        def spy(pool, name=name, original=getattr(ai._Pool, name)):
            started.append(name)
            original(pool)
        monkeypatch.setattr(ai._Pool, name, spy)
    result = invoke("ask", "why is the sky blue?")
    assert result.exit_code == 0, result.output
    assert "--- answer\nThe sky scatters blue.\n--- end" in result.output
    assert "3000.0 ms per word, 31 words, prompt 12.5 s" in result.output
    assert "sync apps back" in result.output
    seen = events(stub)
    # the fake helper and tunnel log once their own processes get going, which on
    # a busy machine can be in either order, so their order is read from the calls
    assert started == ["start_helper", "start_tunnel"]
    assert "helper up" in seen and "tunnel up" in seen
    assert seen.index("pause on") < seen.index("engine started")
    assert seen.index("engine started") < seen.index("pause off") < seen.index("apps resumed")
    assert seen.index("apps resumed") < seen.index("tunnel down") < seen.index("helper down")
    assert cancelled == ["ai-hold-20261106-062057-31b6"]
    record = ai.recent()[0]
    assert record["state"] == "answered" and record["answer"] == "The sky scatters blue."
    assert record["start_free_mb"] == 6000 and record["paused"] == ["OneDrive suspended 3 of 3"]
    assert record["hold_job"] == cancelled[0]
    assert "llama-completion" in record["files_used"]["engine"]
    assert [row["name"] for row in record["files_used"]["model"]] == ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
    assert record["timings"] == {"load_seconds": 423.5, "prompt_seconds": 12.5, "words": 31, "ms_per_word": 3000.0}
    shown = invoke("show", record["id"])
    assert shown.exit_code == 0 and "seal: verified" in shown.output


def test_passes_the_settings_to_the_engine(stub):
    write_settings(stub, helper=True)
    assert invoke("ask", "hello there", "--threads", "8").exit_code == 0
    args = (stub.home / "engine.args").read_text().split("\n")[:-1]
    assert args[args.index("-p") + 1] == "hello there"
    assert args[args.index("-t") + 1] == "8"
    assert args[args.index("-n") + 1] == "8"
    assert args[args.index("--rpc") + 1] == "127.0.0.1:50072"
    assert args[args.index("--device") + 1] == "CUDA0,RPC0"
    assert args[args.index("-ts") + 1] == "22,78"
    assert args[args.index("-ot") + 1] == "_exps=CPU"
    assert args[-2:] == ["--temp", "0"]
    assert "GGML_RPC_BATCH_FETCH=1" in (stub.home / "engine.env").read_text()


def test_env_var_changes_a_setting_so_ab_can_drive_it(stub, monkeypatch):
    write_settings(stub)
    monkeypatch.setenv("ORCHARD_AI_THREADS", "6")
    assert invoke("ask", "hi").exit_code == 0
    args = (stub.home / "engine.args").read_text().split("\n")
    assert args[args.index("-t") + 1] == "6"


def test_a_question_with_shell_characters_arrives_intact(stub):
    write_settings(stub)
    question = """it's "quoted" $HOME `x`; rm -rf /"""
    assert invoke("ask", question).exit_code == 0
    args = (stub.home / "engine.args").read_text().split("\n")
    assert args[args.index("-p") + 1] == question


def test_refuses_when_the_engine_is_already_running(stub):
    write_settings(stub, helper=True)
    (stub.home / "busy").write_text("")
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and "already running on box" in result.output
    assert events(stub) == []


@pytest.mark.parametrize("missing, word", [("engine", "engine"), ("model", "model")])
def test_refuses_when_files_are_missing(stub, missing, word):
    write_settings(stub)
    target = stub.engine / "llama-completion" if missing == "engine" else stub.models / "m-00001-of-00002.gguf"
    target.unlink()
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and f"{word} not found on box" in result.output
    assert ai.recent()[0]["state"] == "refused"


def test_refuses_an_unreachable_machine(stub):
    write_settings(stub)
    ai.CONFIG_PATH.write_text(ai.CONFIG_PATH.read_text().replace('host = "box"', 'host = "gone"'))
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and "cannot reach gone" in result.output


def test_refuses_a_second_question_at_once(stub):
    write_settings(stub, helper=True)
    ai.AI_DIR.mkdir()
    with (ai.AI_DIR / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = invoke("ask", "hi")
    assert result.exit_code == 2 and "one at a time" in result.output
    assert events(stub) == []


def test_refuses_when_this_machine_is_short_of_memory(stub, monkeypatch):
    write_settings(stub, helper=True)
    monkeypatch.setattr(ai, "_local_pressure", lambda: 4)
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and "short of memory" in result.output


def test_stops_when_memory_runs_low_and_brings_the_apps_back(stub):
    write_settings(stub, windows=True, start=5000, stop=200)
    (stub.home / "slow").write_text("")
    (stub.home / "free.txt").write_text("6000 900 100 100 100")
    result = invoke("ask", "hi")
    assert result.exit_code == 3, result.output
    assert "under 200 MB free" in result.output and "lowest free memory 100 MB" in result.output
    assert "sync apps back" in result.output
    seen = events(stub)
    assert seen.index("engine started") < seen.index("engine killed") < seen.index("apps resumed")


def test_one_low_read_does_not_stop_the_run(stub):
    write_settings(stub, stop=200)
    (stub.home / "slow").write_text("")
    (stub.home / "free.txt").write_text("100 900 100 900 100 100 100")
    assert invoke("ask", "hi").exit_code == 3
    assert ai.recent()[0]["lowest_free_mb"] == 100
    assert (stub.home / "free.txt").read_text().split() == ["100"]


def test_stops_when_free_memory_cannot_be_read(stub, monkeypatch):
    monkeypatch.setattr(ai, "UNREAD_READS", 2)
    write_settings(stub, windows=True, start=5000, stop=200)
    (stub.home / "slow").write_text("")
    (stub.home / "free.txt").write_text("6000 unreadable")
    result = invoke("ask", "hi")
    assert result.exit_code == 3 and "could not read free memory" in result.output


def test_stops_when_this_machine_runs_short_mid_run(stub, monkeypatch):
    write_settings(stub, helper=True)
    (stub.home / "slow").write_text("")
    reads = iter([0])
    monkeypatch.setattr(ai, "_local_pressure", lambda: next(reads, 4))
    result = invoke("ask", "hi")
    assert result.exit_code == 3 and "this machine is short of memory" in result.output
    assert "helper down" in events(stub) and "tunnel down" in events(stub)


def test_refuses_to_start_short_of_memory_and_ends_the_pause(stub):
    write_settings(stub, windows=True, start=5000)
    (stub.home / "free.txt").write_text("300")
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and "300 MB free on the engine machine" in result.output
    seen = events(stub)
    assert "engine started" not in seen and seen.index("pause off") < seen.index("apps resumed")


def test_waits_for_memory_to_come_back(stub):
    write_settings(stub, start=5000)
    ai.CONFIG_PATH.write_text(ai.CONFIG_PATH.read_text().replace("settle_seconds = 0", "settle_seconds = 30"))
    (stub.home / "free.txt").write_text("300 300 6000")
    assert invoke("ask", "hi").exit_code == 0
    assert ai.recent()[0]["start_free_mb"] == 6000


@pytest.mark.parametrize("paused", [{"state": "paused", "note": "user active"},
                                    {"state": "running", "paused_since": 1.0, "note": "user active"}])
def test_backs_off_when_the_machine_comes_into_use(stub, monkeypatch, paused):
    write_settings(stub, worker="box")
    (stub.home / "slow").write_text("")
    cancelled = hold_job(monkeypatch, [{"state": "running"}, paused])
    result = invoke("ask", "hi")
    assert result.exit_code == 3 and "the machine is in use: user active" in result.output
    assert "engine killed" in events(stub) and cancelled


def test_reports_an_engine_failure_and_cleans_up(stub):
    write_settings(stub, helper=True, windows=True)
    (stub.home / "engine.exit").write_text("7")
    result = invoke("ask", "hi")
    assert result.exit_code == 1 and "the engine exited 7" in result.output
    assert {"apps resumed", "tunnel down", "helper down"} <= set(events(stub))


def test_a_crash_still_leaves_a_record_and_cleans_up(stub, monkeypatch):
    write_settings(stub, windows=True)

    def crash(*_args):
        raise OSError("the disk went away")

    monkeypatch.setattr(ai._Pool, "wait_for_memory", crash)
    result = invoke("ask", "hi")
    assert result.exit_code == 1 and "the disk went away" in result.output
    assert ai.recent()[0]["state"] == "failed" and "apps resumed" in events(stub)


def test_a_signal_mid_run_cleans_up_everything(stub, monkeypatch):
    write_settings(stub, helper=True, windows=True, worker="box")
    (stub.home / "slow").write_text("")
    cancelled = hold_job(monkeypatch, [{"state": "running"}])
    done = threading.Event()

    def signal_once_started():
        while not done.wait(0.02):
            if "engine started" in events(stub):
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=signal_once_started, daemon=True).start()
    try:
        result = invoke("ask", "hi")
    finally:
        done.set()
    assert result.exit_code == 3 and "interrupted by SIGTERM" in result.output
    seen = events(stub)
    assert {"engine killed", "apps resumed", "tunnel down", "helper down"} <= set(seen)
    assert cancelled


def test_no_pause_leaves_the_apps_alone(stub):
    write_settings(stub, windows=True)
    assert invoke("ask", "hi", "--no-pause").exit_code == 0
    assert "pause on" not in events(stub) and "apps resumed" not in events(stub)
    assert ai.recent()[0]["resumed"] is None


def test_without_windows_it_reads_linux_memory(stub):
    write_settings(stub, start=5000)
    (stub.home / "free.txt").write_text("7000")
    assert invoke("ask", "hi").exit_code == 0
    assert ai.recent()[0]["start_free_mb"] == 7000


def test_says_when_the_apps_are_not_confirmed_back(stub):
    write_settings(stub, windows=True)
    (stub.home / "stuck").write_text("")
    result = invoke("ask", "hi")
    assert result.exit_code == 0
    assert "SYNC APPS NOT CONFIRMED BACK" in result.output and "orchard ai resume" in result.output
    assert ai.recent()[0]["resumed"] is False
    assert invoke("resume").exit_code == 1
    (stub.home / "stuck").unlink()
    back = invoke("resume")
    assert back.exit_code == 0 and "sync apps back" in back.output


def test_no_apps_running_is_not_a_failed_pause(stub, monkeypatch):
    write_settings(stub, windows=True)
    tools = stub.home / "bin"
    (tools / "ssh").write_text(SSH.replace('print("held OneDrive suspended 3 of 3\\nheld done", flush=True)',
                                           'print("held done", flush=True)'))
    said = []
    record = ai.ask("hi", say=said.append)
    assert record["paused"] == [] and record["resumed"] is True
    assert not any("not paused" in line for line in said)


def test_waits_for_the_hold_to_start_then_cancels_it(stub, monkeypatch):
    write_settings(stub, worker="box")
    cancelled = hold_job(monkeypatch, [{"state": "queued"}, {"state": "queued"}, {"state": "running"}])
    assert invoke("ask", "hi").exit_code == 0
    assert cancelled == ["ai-hold-20261106-062057-31b6"]


def test_refuses_when_the_machine_stays_busy(stub, monkeypatch):
    write_settings(stub, worker="box", extra="hold_wait_seconds = 0")
    cancelled = hold_job(monkeypatch, [{"state": "queued"}])
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and "stayed busy" in result.output
    assert cancelled and "engine started" not in events(stub)


def test_refuses_when_the_hold_ends_early(stub, monkeypatch):
    write_settings(stub, worker="box")
    hold_job(monkeypatch, [{"state": "failed"}])
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and "ended early: failed" in result.output


def test_no_hold_starts_no_job(stub, monkeypatch):
    write_settings(stub, worker="box")
    cancelled = hold_job(monkeypatch, [{"state": "failed"}])
    assert invoke("ask", "hi", "--no-hold").exit_code == 0
    assert cancelled == []


def test_a_hold_that_cannot_be_cancelled_is_reported(stub, monkeypatch):
    write_settings(stub, worker="box")
    hold_job(monkeypatch, [{"state": "running"}])
    monkeypatch.setattr(ai, "_cancel_job", lambda _job: False)
    result = invoke("ask", "hi")
    assert result.exit_code == 0 and "could not cancel hold job" in result.output


def test_the_hold_is_opt_in_and_never_falls_back_here(stub, monkeypatch):
    write_settings(stub, worker="box")
    cfg = ai.load_settings()
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ai._submit_hold(cfg, 60, stub.home)
    assert "--no-local" in seen["argv"] and "-b" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--name") + 1] == ai.HOLD_NAME
    assert seen["argv"][-2:] == ["sleep", "60"]


def test_setup_writes_a_template_once(stub):
    result = invoke("setup")
    assert result.exit_code == 2 and "fill it in" in result.output
    template = ai.CONFIG_PATH.read_text()
    assert "CHANGE" in template
    assert invoke("setup").exit_code == 2  # still has CHANGE in it
    assert ai.CONFIG_PATH.read_text() == template
    assert ai.write_template() is False


def test_setup_checks_each_machine(stub, monkeypatch):
    write_settings(stub, helper=True, windows=True, worker="box")
    monkeypatch.setattr(config, "load_workers", lambda: [SimpleNamespace(name="box")])
    result = invoke("setup")
    assert result.exit_code == 0, result.output
    assert "box: model: 2 files" in result.output
    assert "this machine: helper:" in result.output
    assert "winbox: GPU: Some GPU" in result.output
    assert "box: Orchard worker: known" in result.output


def test_setup_says_what_is_wrong(stub, monkeypatch):
    write_settings(stub, worker="nobody")
    (stub.engine / "llama-completion").unlink()
    monkeypatch.setattr(config, "load_workers", lambda: [SimpleNamespace(name="box")])
    result = invoke("setup")
    assert result.exit_code == 1
    assert re.search(r"NO +box: engine:", result.output) and re.search(r"NO +nobody: Orchard worker", result.output)


def test_setup_says_when_a_machine_is_out_of_reach(stub):
    write_settings(stub)
    ai.CONFIG_PATH.write_text(ai.CONFIG_PATH.read_text().replace('host = "box"', 'host = "gone"'))
    result = invoke("setup")
    assert result.exit_code == 1 and re.search(r"NO +gone: reachable", result.output)


def test_show_catches_a_changed_answer(stub):
    write_settings(stub)
    invoke("ask", "hi")
    record = ai.recent()[0]
    (ai.AI_DIR / record["id"] / "answer.txt").write_text("something else")
    result = invoke("show", record["id"])
    assert result.exit_code == 1 and "answer.txt changed or missing" in result.output


def test_show_catches_a_changed_record(stub):
    write_settings(stub)
    invoke("ask", "hi")
    record = ai.recent()[0]
    path = ai.AI_DIR / f"{record['id']}.json"
    path.write_text(json.dumps({**record, "answer": "forged"}))
    result = invoke("show", record["id"])
    assert result.exit_code == 1 and "DOES NOT VERIFY" in result.output


def test_show_and_list(stub):
    assert invoke("show", "nothing").exit_code == 1
    write_settings(stub)
    invoke("ask", "why is the sky blue?")
    listed = invoke("list")
    assert "answered" in listed.output and "3000 ms/word" in listed.output and "why is the sky blue?" in listed.output


def test_json_output_is_the_whole_record(stub):
    write_settings(stub)
    result = invoke("ask", "hi", "--json")
    assert result.exit_code == 0
    record = json.loads(result.output[result.output.index("{"):])
    assert record["answer"] == "The sky scatters blue." and receipts.verify(record)


def test_empty_question_and_no_settings_are_refused(stub):
    assert invoke("ask", "  ").exit_code == 2
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and "no settings at" in result.output


@pytest.mark.parametrize("old, new, message", [
    ('host = "box"', 'host = "CHANGE"', "set engine.host"),
    ('threads = 4', 'threads = [', "ai.toml"),
])
def test_bad_settings_are_named(stub, old, new, message):
    write_settings(stub)
    ai.CONFIG_PATH.write_text(ai.CONFIG_PATH.read_text().replace(old, new))
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and message in result.output


def test_a_bad_process_name_is_refused_before_it_reaches_powershell(stub):
    write_settings(stub, windows=True)
    ai.CONFIG_PATH.write_text(ai.CONFIG_PATH.read_text().replace('"OneDrive"', '''"x'; Stop-Computer; '"'''))
    result = invoke("ask", "hi")
    assert result.exit_code == 2 and "bad process name" in result.output


def test_model_pattern_covers_every_file():
    assert ai.model_pattern("~/m/big-00001-of-00006.gguf") == "~/m/big-*.gguf"
    assert ai.model_pattern("~/m/small.gguf") == "~/m/small.gguf"


def test_remote_paths_keep_home_and_quote_the_rest():
    assert ai._rpath("~/a b/c") == """"$HOME"/'a b/c'"""
    assert ai._rpath("/x/y") == "/x/y"


def test_parse_typeperf_reads_windows_output():
    text = '"(PDH-CSV 4.0)","x"\r\n"11/06/2026 06:00:00.000","1234.000000"\r\n\x00'
    assert ai.parse_typeperf(text) == 1234
    assert ai.parse_typeperf("Error: no such counter") is None


def test_memory_lines_are_empty_at_the_normal_log_level():
    assert ai.memory_lines("I common_perf_print: load time = 1 ms") == []
    assert ai.memory_lines("I llama_memory_breakdown_print: | - RPC0 | 12124 |") == ["| - RPC0 | 12124 |"]


def test_pause_script_names_the_apps_and_a_limit():
    script = ai.hold_script(["OneDrive", "FileCoAuth"], 123)
    assert "$limit = 123" in script and "$names = 'OneDrive','FileCoAuth'" in script
    assert "held none" in script and "finally" in script


def test_run_script_gives_memory_back_on_the_way_out(stub):
    write_settings(stub)
    script = ai.run_script(ai.load_settings(), "q", 4)
    assert "trap give_back EXIT" in script and "POSIX_FADV_DONTNEED" in script
    assert "m-*.gguf" in script and "alarm 2400" in script


REASONED = ("Need two sentences. Mention scattering.</think>The sky is blue because air scatters blue light most. "
            "It reaches us from all directions. [end of text]\n\n")
THOUGHT = "Need two sentences. Mention scattering."
ANSWER = "The sky is blue because air scatters blue light most. It reaches us from all directions."


@pytest.mark.parametrize("raw, mark, want", [
    ("Just the answer.\n", "", ("", "Just the answer.")),
    ("Just the answer. [end of text]\n\n", "", ("", "Just the answer.")),
    (REASONED, "</think>", (THOUGHT, ANSWER)),
    ("Still reasoning about", "</think>", ("Still reasoning about", "")),
])
def test_split_answer(raw, mark, want):
    assert ai.split_answer(raw, mark) == want


def _streamed(mark, *chunks):
    seen = []
    stream = ai.AnswerStream(mark, seen.append)
    for chunk in chunks:
        stream.feed(chunk)
    stream.finish()
    return "".join(seen)


def test_stream_prints_the_same_wherever_the_chunks_break():
    expected = f"--- thinking\n{THOUGHT}\n--- answer\n{ANSWER}"
    for cut in range(len(REASONED) + 1):
        assert _streamed("</think>", REASONED[:cut], REASONED[cut:]) == expected, cut
    assert _streamed("</think>", *REASONED) == expected


def test_stream_without_thinking_keeps_lookalikes_and_drops_only_the_end_mark():
    assert _streamed("", "\nAn answer. [end of text]\n") == "--- answer\nAn answer."
    assert _streamed("", "See [end of the", " road] now") == "--- answer\nSee [end of the road] now"
    assert _streamed("", "Ends with [end") == "--- answer\nEnds with [end"
    assert _streamed("", "") == ""


def test_a_model_that_reasons_first_has_its_thinking_kept_apart(stub, monkeypatch):
    monkeypatch.setenv("AI_ENGINE_TEXT", REASONED)
    write_settings(stub, extra='thinking_end = "</think>"')
    result = invoke("ask", "why is the sky blue?")
    assert result.exit_code == 0, result.output
    shown = f"--- thinking\n{THOUGHT}\n--- answer\n{ANSWER}\n--- end"
    assert shown in result.output
    record = ai.recent()[0]
    assert record["thinking"] == THOUGHT and record["answer"] == ANSWER and receipts.verify(record)
    assert (ai.AI_DIR / record["id"] / "answer.txt").read_text() == REASONED
    assert shown in invoke("show", record["id"]).output


def test_running_out_of_words_while_thinking_is_not_an_answer(stub, monkeypatch):
    monkeypatch.setenv("AI_ENGINE_TEXT", "Still reasoning about")
    write_settings(stub, extra='thinking_end = "</think>"')
    result = invoke("ask", "why is the sky blue?")
    assert result.exit_code == 1, result.output
    assert "word limit ran out" in result.output and "--- answer" not in result.output
    record = ai.recent()[0]
    assert record["state"] == "failed" and record["answer"] == "" and record["thinking"] == "Still reasoning about"
