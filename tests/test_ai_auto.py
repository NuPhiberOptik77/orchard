import json
import subprocess

import pytest
from typer.testing import CliRunner

from orchard import ai, ai_auto, ai_pick, cli, receipts

runner = CliRunner()
GIB = 1024


@pytest.fixture(autouse=True)
def _no_real_records(tmp_path, monkeypatch):
    monkeypatch.setattr(ai, "AI_DIR", tmp_path / "ai")
    monkeypatch.setattr(receipts, "KEY_PATH", tmp_path / "receipt.key")


def machine(name, ram_gb, *, gpu=None, vram_gb=0, speed=20.0, system="Darwin", target="same"):
    gpus = []
    if gpu == "apple":
        gpus.append({"name": "Apple GPU", "kind": "apple", "memory_mb": None})
    elif gpu == "nvidia":
        gpus.append({"name": "GeForce", "kind": "nvidia", "memory_mb": vram_gb * GIB})
    return ai_pick.Machine(name=name, target=name if target == "same" else target, reachable=True, system=system,
                           arch="x86_64", threads=8, wsl=system == "Linux", ram_mb=ram_gb * GIB,
                           free_mb=ram_gb * GIB // 2, chip="", gpus=gpus, disk_free_gb=100.0, copy_gbs=speed)


def pool():
    # big-box runs the engine (8 GB, a 2 GB card); this Mac lends its GPU; old-mac is far.
    box = machine("big-box", 8, gpu="nvidia", vram_gb=2, speed=8.0, system="Linux")
    mac = machine("this Mac", 16, gpu="apple", speed=40.0, target=None)
    old = machine("old-mac", 8, speed=8.5)
    box["part"], mac["part"], old["part"] = "engine", "helper", "far"
    return [box, mac, old]


def model(mid="mid", gb=18.0, layers=30, quants=None):
    quants = quants or [{"quant": "Q4_K_M", "files": [f"{mid}-Q4_K_M.gguf"], "bytes": int(gb * 1e9)}]
    return {"id": mid, "name": mid.title(), "repo": f"maker/{mid}-GGUF", "layers": layers, "quants": quants,
            "params_b": 30.0, "scores": [], "good_at": []}


def settings(far=True, helper=True):
    cfg = {section: dict(values) for section, values in ai.DEFAULTS.items() if section != "far_helper"}
    cfg["engine"].update(host="big-box", folder="~/llm/engine", model="~/llm/models/old/other.gguf")
    if helper:
        cfg["helper"]["folder"] = "~/llm/helper"
    cfg["far_helpers"] = [{**ai.DEFAULTS["far_helper"], "host": "old-mac", "folder": "~/llm/helper"}] if far else []
    return cfg


def fake_remote(paths):
    def remote(host, command, timeout, stdin_text=None):
        lines = []
        for line in stdin_text.splitlines():
            index = line.split("=", 1)[0].removeprefix('echo "')
            name = line.split("-name ", 1)[1].split(" ", 1)[0].strip("'")
            lines.append(f"{index}={paths.get(name, '')}")
        return subprocess.CompletedProcess([], 0, "\n".join(lines) + "\n", "")
    return remote


def reader(machines):
    by_name = {m["name"]: m for m in machines}
    return lambda name, target: ai_pick.Machine(**{k: v for k, v in by_name[name].items() if k != "part"})


# Finding the model and its file.

def test_a_model_is_found_by_id_name_or_a_start_only_it_has():
    cat = {"models": [model("gemma-4-26b-a4b"), model("gemma-4-31b"), model("qwen3.6-35b-a3b")]}
    assert ai_auto.find_model("gemma-4-26b-a4b", cat)["id"] == "gemma-4-26b-a4b"
    assert ai_auto.find_model("QWEN3.6", cat)["id"] == "qwen3.6-35b-a3b"
    with pytest.raises(ai.ConfigError, match="could be gemma-4-26b-a4b, gemma-4-31b"):
        ai_auto.find_model("gemma", cat)
    with pytest.raises(ai.ConfigError, match="isn't on Orchard's list"):
        ai_auto.find_model("nope", cat)


def test_the_file_is_looked_for_on_the_engine_machine():
    m = model(quants=[{"quant": "Q4_K_M", "files": ["dir/mid-Q4_K_M.gguf"], "bytes": 1},
                      {"quant": "IQ2_M", "files": ["mid-IQ2_M.gguf"], "bytes": 1}])
    seen = []
    remote = fake_remote({"mid-IQ2_M.gguf": "/home/u/llm/models/picks/mid-IQ2_M.gguf"})
    found = ai_auto.find_files(settings(), m, lambda *a: seen.append(a) or remote(*a))
    assert found == {"IQ2_M": "/home/u/llm/models/picks/mid-IQ2_M.gguf"}
    assert seen[0][0] == "big-box"
    assert '"$HOME"/llm/models "$HOME"/llm/models/old' in seen[0][3]


# Laying it out.

def test_the_card_fills_first_then_a_much_faster_helper_then_the_engine_then_the_rest():
    layout = ai_auto.lay_out(model(), model()["quants"][0], pool())
    # 0.66 GB a layer: the card 2, this Mac 18 (of 12 GB), big-box's own memory 7 (of 5 GB), old-mac the last 3
    assert [(d["machine"]["name"], d["device"], d["layers"]) for d in layout] == [
        ("big-box", "CUDA0", 2), ("this Mac", "RPC", 18), ("big-box", "CPU", 7), ("old-mac", "RPC", 3)]


def test_a_far_helper_goes_before_the_engine_only_when_twice_as_fast():
    box, mac, old = pool()
    old["copy_gbs"] = 16.0
    order = [(d["machine"]["name"], d["device"]) for d in ai_auto.lay_out(model(), model()["quants"][0], [box, mac, old])]
    assert order.index(("old-mac", "RPC")) < order.index(("big-box", "CPU"))


def test_an_engine_on_a_mac_puts_its_share_on_its_graphics_chip():
    mac = machine("studio", 32, gpu="apple", speed=60.0)
    mac["part"] = "engine"
    layout = ai_auto.lay_out(model(), model()["quants"][0], [mac])
    assert [(d["device"], d["layers"]) for d in layout] == [("MTL0", 30)]


def test_a_model_too_big_for_the_pool_is_refused_with_the_numbers():
    with pytest.raises(ai.ConfigError, match="has 40 layers of about 0.61 GB each, and these machines have room for 35"):
        ai_auto.lay_out(model(gb=22.1, layers=40), model(gb=22.1, layers=40)["quants"][0], pool())
    with pytest.raises(ai.ConfigError, match="doesn't know how many layers"):
        ai_auto.lay_out(model(layers=None), model()["quants"][0], pool())


# Settings for the run.

def test_the_run_gets_the_file_devices_split_and_layers_and_only_the_helpers_it_uses():
    small = model(gb=8.0)
    cat = {"models": [small]}
    cfg, layout = ai_auto.apply(settings(), "mid", catalogue=cat, read=reader(pool()),
                                remote=fake_remote({"mid-Q4_K_M.gguf": "/m/mid-Q4_K_M.gguf"}))
    e = cfg["engine"]
    # 0.29 GB a layer: the card 5, this Mac the other 25; old-mac isn't needed
    assert (e["model"], e["devices"], e["split"], e["layers"]) == ("/m/mid-Q4_K_M.gguf", "CUDA0,RPC0", "5,25", 30)
    assert cfg["far_helpers"] == [] and ai.has_helper(cfg)
    assert layout["not_started"] == ["old-mac"]
    argv = ai.engine_command(cfg, "q", 10)
    assert argv[argv.index("--rpc") + 1] == "127.0.0.1:50072"
    assert ai_auto.describe(layout) == ("Mid (Q4_K_M, 30 layers): big-box 5 layers (1.5 GB, all on its graphics chip); "
                                        "this Mac 25 layers (7.3 GB); not needed: old-mac")


def test_without_this_machines_helper_a_far_helper_becomes_rpc0():
    box, mac, old = pool()
    old["copy_gbs"] = 40.0
    cfg, layout = ai_auto.apply(settings(helper=False), "mid", catalogue={"models": [model(gb=8.0)]},
                                read=reader([box, old]), remote=fake_remote({"mid-Q4_K_M.gguf": "/m/a.gguf"}))
    assert cfg["engine"]["devices"] == "CUDA0,RPC0" and [h["host"] for h in cfg["far_helpers"]] == ["old-mac"]
    argv = ai.engine_command(cfg, "q", 10)
    assert argv[argv.index("--rpc") + 1] == "127.0.0.1:50073"


def test_the_largest_file_you_have_that_fits_is_used():
    m = model(quants=[{"quant": "Q8_0", "files": ["mid-Q8_0.gguf"], "bytes": int(40e9)},
                      {"quant": "Q4_K_M", "files": ["mid-Q4_K_M.gguf"], "bytes": int(18e9)},
                      {"quant": "IQ2_M", "files": ["mid-IQ2_M.gguf"], "bytes": int(9e9)}])
    files = {"mid-Q8_0.gguf": "/m/q8.gguf", "mid-Q4_K_M.gguf": "/m/q4.gguf", "mid-IQ2_M.gguf": "/m/q2.gguf"}
    cfg, layout = ai_auto.apply(settings(), "mid", catalogue={"models": [m]}, read=reader(pool()),
                                remote=fake_remote(files))
    assert layout["quant"] == "Q4_K_M" and cfg["engine"]["model"] == "/m/q4.gguf"


def test_a_model_not_downloaded_says_what_to_get_and_where():
    with pytest.raises(ai.ConfigError, match=r"isn't on big-box under ~/llm/models or ~/llm/models/old: download "
                                             r"mid-Q4_K_M.gguf or a larger file \(18.0 GB and up\) from "
                                             r"huggingface.co/maker/mid-GGUF"):
        ai_auto.apply(settings(), "mid", catalogue={"models": [model()]}, read=reader(pool()), remote=fake_remote({}))


def test_a_machine_that_ran_short_of_memory_gets_fewer_layers():
    box, mac, old = pool()
    machines = [box, mac, old]
    record = receipts.sign({"id": "ai-20261114-110000-bbbb", "kind": "ai", "state": "stopped", "started_at": 1.0,
                            "reason": "this machine is short of memory",
                            "command": ["./llama-completion", "-m", "/m/mid-Q4_K_M.gguf"],
                            "settings": {"layers": 18, "devices": "CUDA0,RPC0", "split": "2,16", "overrides": ""},
                            "machines": {"engine": "big-box", "helper": True, "far_helpers": []}})
    ai.AI_DIR.mkdir(parents=True)
    (ai.AI_DIR / "ai-20261114-110000-bbbb.json").write_text(json.dumps(record))
    _, layout = ai_auto.apply(settings(), "mid", catalogue={"models": [model(gb=12.0)]}, read=reader(machines),
                              remote=fake_remote({"mid-Q4_K_M.gguf": "/m/mid-Q4_K_M.gguf"}))
    mac_layers = next(p["layers"] for p in layout["parts"] if p["machine"] == "this Mac")
    # It held 16 layers of 0.4 GB, 6.4 GB; 80% of that is 5.1 GB: 11 layers of 0.44 GB, not the 27 it would take
    assert mac_layers == 11


def test_model_with_a_hand_split_or_overrides_is_refused(monkeypatch, tmp_path):
    cfg = settings()
    cfg["engine"]["overrides"] = "exps=CPU"
    with pytest.raises(ai.ConfigError, match="leave out overrides"):
        ai_auto.apply(cfg, "mid", catalogue={"models": [model()]})
    monkeypatch.setattr(ai, "load_settings", lambda *a, **k: settings())
    with pytest.raises(ai.ConfigError, match="leave out --split and --overrides"):
        ai.ask("q", split="1,1", model="mid")


def test_ai_plan_prints_the_layout_and_the_engine_flags(monkeypatch):
    monkeypatch.setattr(ai, "load_settings", lambda *a, **k: settings())
    real = ai_auto.apply
    monkeypatch.setattr(ai_auto, "apply", lambda cfg, name, say, announce=True: real(
        cfg, name, say, catalogue={"models": [model(gb=8.0)]}, read=reader(pool()),
        remote=fake_remote({"mid-Q4_K_M.gguf": "/m/mid-Q4_K_M.gguf"}), announce=announce))
    result = runner.invoke(cli.app, ["ai", "plan", "mid"])
    assert result.exit_code == 0, result.output
    assert "Mid (Q4_K_M, 30 layers): big-box 5 layers" in result.stdout
    assert "engine flags: --device CUDA0,RPC0 -ngl 30 -ts 5,25" in result.stdout
    result = runner.invoke(cli.app, ["ai", "plan", "mid", "--json"])
    data = json.loads(result.stdout)
    assert data["devices"] == "CUDA0,RPC0" and data["parts"][0]["machine"] == "big-box"
