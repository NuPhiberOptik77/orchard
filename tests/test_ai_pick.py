import json

import pytest
from typer.testing import CliRunner

from orchard import ai, ai_pick, cli, cli_agentic, receipts

runner = CliRunner()
GIB = 1024


@pytest.fixture(autouse=True)
def _no_real_records(tmp_path, monkeypatch):
    # Earlier runs are read from Orchard's records: keep the real ones out.
    monkeypatch.setattr(ai, "AI_DIR", tmp_path / "ai")
    monkeypatch.setattr(receipts, "KEY_PATH", tmp_path / "receipt.key")


def machine(name, ram_gb, *, gpu=None, vram_gb=0, disk=100.0, speed=20.0, wsl=False, system="Darwin"):
    gpus = []
    if gpu == "apple":
        gpus.append({"name": "Apple M5 GPU", "kind": "apple", "memory_mb": None})
    elif gpu == "nvidia":
        gpus.append({"name": "GeForce GTX 1050", "kind": "nvidia", "memory_mb": vram_gb * GIB})
    return ai_pick.Machine(name=name, target=None if name == "this Mac" else name, reachable=True, system=system,
                           arch="arm64" if gpu == "apple" else "x86_64", threads=8, wsl=wsl, ram_mb=ram_gb * GIB,
                           free_mb=ram_gb * GIB // 2, chip="", gpus=gpus, disk_free_gb=disk, copy_gbs=speed)


def fleet():
    return [machine("this Mac", 16, gpu="apple", speed=40.0, disk=50.0),
            machine("big-box", 8, gpu="nvidia", vram_gb=2, system="Linux", wsl=True, disk=55.0),
            machine("old-mac", 8, disk=17.0)]


def model(mid, score, gb, *, active=None, params=30.0, good_at=("coding",), quants=None, **extra):
    quants = quants or [{"quant": "Q4_K_M", "files": [f"{mid}-Q4_K_M.gguf"], "bytes": int(gb * 1e9)}]
    return {"id": mid, "name": mid.title(), "maker": "m", "repo": f"maker/{mid}-GGUF", "params_b": params,
            "active_b": active, "context": 32768, "licence": "apache-2.0", "gated": False, "llama_cpp": True,
            "good_at": list(good_at), "quants": quants, "scores": [{"source": "idx", "benchmark": "Index", "score": score}],
            "notes": "", **extra}


def catalogue(*models):
    return {"as_of": "2026-11-14", "primary": "idx",
            "sources": [{"id": "idx", "task": "general", "name": "Some Index", "url": "https://example.org",
                         "read_on": "2026-11-14", "what": "general"}],
            "models": list(models)}


# Room on each machine.

def test_room_keeps_a_quarter_and_counts_graphics_memory():
    mac, box, old = fleet()
    assert mac.room_gb == 12.0
    assert old.room_gb == 4.0  # a Mac keeps at least 4 GB for itself
    assert box.room_gb == 7.6  # 6 GB of memory and 80% of a 2 GB card
    assert machine("tiny", 4, system="Linux").room_gb == 2.0  # at least 2 GB stays with the machine
    assert machine("tiny-mac", 4).room_gb == 0.0


def test_the_fewest_machines_share_in_proportion_to_their_room():
    mac, box, old = fleet()
    split = ai_pick.plan_memory(10.0, [old, box, mac])
    assert [(m["name"], gb) for m, gb in split] == [("this Mac", 10.0)]
    split = ai_pick.plan_memory(19.6, [old, box, mac])
    assert [m["name"] for m, _ in split] == ["this Mac", "big-box"]
    assert [gb for _, gb in split] == [12.0, 7.6]
    assert ai_pick.plan_memory(26.0, [old, box, mac]) is None


# Fast, slow, or not here.

def test_a_model_that_fits_in_memory_is_fast_with_a_split_and_an_estimate():
    row = ai_pick.judge(model("mid", 50, 18.0, active=3.0), {"quant": "Q4_K_M", "files": [], "bytes": int(18e9)}, fleet())
    assert row["group"] == "fast" and row["need_gb"] == 20.8
    assert [m["name"] for m, _ in row["split"]] == ["this Mac", "big-box", "old-mac"]
    assert row["words_per_second"] > 0


def test_a_big_mixture_of_experts_runs_from_disk_and_a_big_dense_model_does_not():
    big_moe = model("moe", 70, 45.0, params=300.0, active=13.0)
    row = ai_pick.judge(big_moe, big_moe["quants"][0], fleet())
    assert row["group"] == "slow" and row["home"]["name"] == "big-box"
    assert "reads from disk" in row["why"]
    dense = model("dense", 70, 45.0, params=70.0)
    row = ai_pick.judge(dense, dense["quants"][0], fleet())
    assert row["group"] == "no" and "your machines have 23.6 GB together" in row["why"]


def test_a_model_bigger_than_any_disk_is_turned_away_with_the_reason():
    huge = model("huge", 80, 240.0, params=700.0, active=30.0)
    row = ai_pick.judge(huge, huge["quants"][0], fleet())
    assert row["group"] == "no" and "needs 240.0 GB of disk" in row["why"] and "(big-box)" in row["why"]


def test_missing_files_and_unsupported_models_are_turned_away():
    gone = model("gone", 90, 5.0)
    gone["quants"][0]["missing"] = True
    assert "no longer on Hugging Face" in ai_pick.judge(gone, gone["quants"][0], fleet())["why"]
    odd = model("odd", 90, 5.0, llama_cpp=False)
    assert "can't run this kind" in ai_pick.judge(odd, odd["quants"][0], fleet())["why"]


# Ranking.

def test_the_strongest_fast_model_is_the_best_pick():
    result = ai_pick.pick(fleet(), catalogue(
        model("small", 30, 5.0), model("strong", 60, 18.0, active=3.0), model("huge", 90, 240.0, params=700.0, active=30.0),
        model("moe", 75, 45.0, params=300.0, active=13.0)))
    assert result["best"]["model"]["id"] == "strong"
    assert [r["model"]["id"] for r in result["groups"]["fast"]] == ["strong", "small"]
    assert [r["model"]["id"] for r in result["groups"]["slow"]] == ["moe"]
    assert [r["model"]["id"] for r in result["groups"]["no"]] == ["huge"]


def test_without_a_fast_model_the_best_is_the_strongest_slow_one():
    result = ai_pick.pick(fleet(), catalogue(model("moe", 75, 45.0, params=300.0, active=13.0)))
    assert result["best"]["group"] == "slow"


def test_the_largest_quantization_that_runs_fast_is_chosen():
    two = model("two", 60, 0, quants=[{"quant": "Q8_0", "files": ["a"], "bytes": int(40e9)},
                                      {"quant": "Q4_K_M", "files": ["b"], "bytes": int(20e9)},
                                      {"quant": "Q2_K", "files": ["c"], "bytes": int(11e9)}])
    assert ai_pick.pick(fleet(), catalogue(two))["best"]["quant"]["quant"] == "Q4_K_M"


def test_only_models_good_at_what_was_asked():
    result = ai_pick.pick(fleet(), catalogue(model("coder", 60, 5.0), model("talker", 70, 5.0, good_at=("chat",))))
    assert result["best"]["model"]["id"] == "talker"
    result = ai_pick.pick(fleet(), catalogue(model("coder", 60, 5.0), model("talker", 70, 5.0, good_at=("chat",))),
                          use="coding")
    assert result["best"]["model"]["id"] == "coder"


def test_unreachable_machines_are_left_out_of_the_sums():
    gone = ai_pick.Machine(name="old-mac", target="old-mac", reachable=False, why="timed out")
    result = ai_pick.pick([fleet()[0], gone], catalogue(model("mid", 50, 14.0)))
    assert result["best"] is None or result["best"]["group"] != "fast"
    text = "\n".join(ai_pick.report(result, None, []))
    assert "old-mac    not reachable: timed out" in text


# The look-up on Hugging Face.

def test_refresh_takes_current_sizes_and_marks_missing_files():
    models = [model("a", 50, 5.0), model("b", 40, 6.0), model("c", 30, 7.0)]
    files = {"maker/a-GGUF": {"a-Q4_K_M.gguf": 5_100_000_000}, "maker/b-GGUF": {"other.gguf": 1}}

    def get_files(repo):
        if repo == "maker/c-GGUF":
            raise OSError("unreachable")
        return files[repo]

    seen = ai_pick.refresh(models, get_files)
    assert seen == {"repos": 3, "reached": 2, "checked": 1, "missing": 1}
    assert models[0]["quants"][0]["bytes"] == 5_100_000_000 and models[0]["quants"][0]["checked"]
    assert models[1]["quants"][0]["missing"] is True
    assert models[2]["quants"][0]["bytes"] == 7_000_000_000 and "checked" not in models[2]["quants"][0]


def test_new_models_are_popular_gguf_repos_not_on_the_list():
    lists = {"unsloth": [{"id": "unsloth/Known-GGUF", "downloads": 9}, {"id": "unsloth/New-GGUF", "downloads": 5},
                         {"id": "unsloth/not-a-quant", "downloads": 99}],
             "bartowski": [{"id": "bartowski/Fresh-GGUF", "downloads": 7, "lastModified": "2026-11-10T00:00:00Z"}]}

    def get(url):
        author = url.split("author=")[1].split("&")[0]
        if author == "ggml-org":
            raise OSError("down")
        return lists[author]

    lists["unsloth"] += [{"id": "unsloth/Picture-GGUF", "downloads": 50, "pipeline_tag": "text-to-image"}]
    lists["bartowski"] += [{"id": "bartowski/maker_Known-GGUF", "downloads": 40}]
    new = ai_pick.new_models({"unsloth/Known-GGUF"}, get)
    assert [m["repo"] for m in new] == ["bartowski/Fresh-GGUF", "unsloth/New-GGUF"]
    assert new[0]["modified"] == "2026-11-10"


# Reading a real machine.

def test_the_probe_reads_this_machine():
    m = ai_pick.read_machine("this Mac", None)
    assert m["reachable"] and m["ram_mb"] > 0 and m["threads"] > 0 and m["copy_gbs"] > 0
    assert m["free_mb"] is None or 0 <= m["free_mb"] <= m["ram_mb"]
    assert m.room_gb > 0


@pytest.mark.skipif(__import__("sys").platform != "darwin", reason="macOS only")
def test_on_a_mac_free_memory_is_the_figure_macos_itself_uses():
    # What macOS can free by compressing idle apps counts as free, as its memory pressure does.
    import subprocess
    m = ai_pick.read_machine("this Mac", None)
    level = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True).stdout)
    assert abs(m["free_mb"] - m["ram_mb"] * level / 100) <= m["ram_mb"] * 0.05


def test_a_machine_that_does_not_answer_is_marked(monkeypatch):
    monkeypatch.setattr(ai, "_ssh", lambda host: ["false"])
    m = ai_pick.read_machine("old-mac", "old-mac")
    assert m["reachable"] is False


def test_linux_in_windows_says_its_disk_is_virtual(monkeypatch):
    box = machine("big-box", 8, wsl=True, system="Linux")
    monkeypatch.setattr(ai_pick, "read_machine", lambda name, target: box)
    monkeypatch.setattr(ai_pick.config, "load_workers", lambda: [])
    monkeypatch.setattr(ai, "CONFIG_PATH", ai.CONFIG_PATH.parent / "no-such-ai.toml")
    machines = ai_pick.read_machines()
    assert "virtual disk" in machines[0]["disk_note"]


# The report and the command.

def test_the_report_says_where_it_runs_and_where_the_numbers_come_from():
    result = ai_pick.pick(fleet(), catalogue(model("strong", 60, 18.0, active=3.0)))
    text = "\n".join(ai_pick.report(result, {"repos": 1, "reached": 1, "checked": 1, "missing": 0},
                                    [{"repo": "unsloth/New-GGUF", "downloads": 1200, "modified": ""}]))
    assert "Best pick\n  Strong (Q4_K_M, 18.0 GB file)" in text
    assert "held in memory: this Mac" in text and "(estimate)" in text
    assert "scores: general 60 (Index)" in text
    assert "made for coding; takes in 32K tokens at once (about 25K words); 30B parameters, 3.0B at work per word" in text
    assert "general: Some Index, https://example.org, read 2026-11-14" in text
    assert "Ranked for a bit of everything by Some Index: 1 of the 1 models have a score." in text
    assert "Sizes checked on Hugging Face just now: 1 files" in text
    assert "huggingface.co/unsloth/New-GGUF (1,200 downloads in the last month)" in text


def test_the_command_prints_json_and_says_when_nothing_runs(monkeypatch):
    monkeypatch.setattr(ai_pick, "read_machines", lambda only=None: fleet())
    monkeypatch.setattr(ai_pick, "builtin_list", lambda: catalogue(model("strong", 60, 18.0, active=3.0)))
    result = runner.invoke(cli.app, ["ai", "pick", "--offline", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["best"]["id"] == "strong" and data["best"]["group"] == "fast" and data["online"] is None
    assert [s["machine"] for s in data["best"]["split"]] == ["this Mac", "big-box", "old-mac"]
    monkeypatch.setattr(ai_pick, "builtin_list", lambda: catalogue(model("dense", 70, 300.0, params=400.0)))
    assert runner.invoke(cli.app, ["ai", "pick", "--offline"]).exit_code == 1


# The list that ships.

def test_the_built_in_list_is_complete_and_sourced():
    data = ai_pick.builtin_list()
    sources = {s["id"] for s in data["sources"]}
    assert data["primary"] in sources and data["as_of"]
    ids = [m["id"] for m in data["models"]]
    assert len(ids) == len(set(ids)) >= 10
    for m in data["models"]:
        assert m["repo"] and m["name"] and m["quants"], m["id"]
        for quant in m["quants"]:
            assert quant["files"] and quant["bytes"] > 0 and all(f.endswith(".gguf") for f in quant["files"]), m["id"]
        for score in m.get("scores") or []:
            assert score["source"] in sources and isinstance(score["score"], (int, float)), m["id"]
        if "vision" in m["good_at"]:
            assert m["pictures"]["file"].endswith(".gguf") and m["pictures"]["bytes"] > 0, m["id"]
    for source in data["sources"]:
        assert source["task"] in ai_pick.USES and source["url"].startswith("https://") and source["read_on"], source
    # Every use but general has its own leaderboard, and it scores most of the list.
    for use in ai_pick.USES:
        ranker = ai_pick.ranking_source(data, use)
        assert ranker and not ranker.get("self_reported"), use
        if use != "vision":
            assert sum(ai_pick.score_of(m, ranker["id"]) is not None for m in data["models"]) >= 12, use


def test_the_best_pick_is_the_strongest_that_is_comfortable_to_use():
    result = ai_pick.pick(fleet(), catalogue(model("strong", 60, 18.0), model("quick", 50, 18.0, active=3.0)))
    assert [r["model"]["id"] for r in result["groups"]["fast"]] == ["strong", "quick"]
    assert result["groups"]["fast"][0]["words_per_second"] < ai_pick.COMFORT_WPS
    assert result["best"]["model"]["id"] == "quick"
    text = "\n".join(ai_pick.report(result, None, []))
    assert "should answer at 2 or more words a second" in text
    alone = ai_pick.pick(fleet(), catalogue(model("strong", 60, 18.0)))
    assert alone["best"]["model"]["id"] == "strong"  # nothing comfortable: still the strongest that runs


def test_a_model_the_engine_may_not_run_is_not_offered():
    unsure = model("unsure", 99, 5.0, llama_cpp=None)
    row = ai_pick.judge(unsure, unsure["quants"][0], fleet())
    assert row["group"] == "no" and "isn't confirmed" in row["why"]


def test_the_model_orchard_ai_already_uses_is_marked(stub_settings):
    models = [model("big", 70, 300.0, params=700.0, active=16.0,
                    quants=[{"quant": "IQ2", "files": ["IQ2/big-00001-of-00002.gguf", "IQ2/big-00002-of-00002.gguf"],
                             "bytes": int(300e9)}])]
    assert ai_pick.mark_owned(models) == "big" and models[0]["quants"][0]["owned_on"] == "box"
    machines = fleet()
    machines[1]["target"] = "box"  # big-box holds it, with too little disk left for a second copy
    result = ai_pick.pick(machines, catalogue(*models))
    assert result["best"]["group"] == "slow" and result["best"]["home"]["name"] == "big-box"
    assert "already on your machines" in "\n".join(ai_pick.report(result, None, []))


@pytest.fixture
def stub_settings(tmp_path, monkeypatch):
    path = tmp_path / "ai.toml"
    path.write_text('[engine]\nhost = "box"\nfolder = "~/e"\nmodel = "~/m/big-00001-of-00002.gguf"\n')
    monkeypatch.setattr(ai, "CONFIG_PATH", path)
    return path


def test_a_squeezed_model_is_flagged_and_not_the_best_pick():
    squeezed = model("squeezed", 80, 0, params=36.0, active=4.0,
                     quants=[{"quant": "IQ2_M", "files": ["s"], "bytes": int(13e9)}])
    full = model("full", 60, 0, params=35.0, active=3.0, quants=[{"quant": "Q4_K_M", "files": ["f"], "bytes": int(19e9)}])
    result = ai_pick.pick(fleet(), catalogue(squeezed, full))
    assert result["best"]["model"]["id"] == "full"
    text = "\n".join(ai_pick.report(result, None, []))
    assert "squeezed to 2.9 bits a weight: weaker than its score" in text
    assert ai_pick.pick(fleet(), catalogue(squeezed))["best"]["model"]["id"] == "squeezed"  # better than nothing


# What it's for.

def coding_catalogue(*models, higher_is_better=True):
    cat = catalogue(*models)
    cat["sources"].append({"id": "code", "task": "coding", "short": "CodeBench", "name": "CodeBench (someone)",
                           "url": "https://example.org/code", "read_on": "2026-11-13", "scale": "fraction",
                           "higher_is_better": higher_is_better})
    return cat


def with_score(m, source, score):
    m["scores"].append({"source": source, "benchmark": source, "score": score})
    return m


def test_a_use_is_named_by_word_or_number():
    assert ai_pick.resolve_use("1") == "coding" and ai_pick.resolve_use(" Images ") == "vision"
    assert ai_pick.resolve_use("maths") == "reasoning" and ai_pick.resolve_use("everything") == "general"
    for bad in ("7", "0", "cooking", ""):
        with pytest.raises(ValueError):
            ai_pick.resolve_use(bad)


def test_the_question_takes_a_number_and_enter_means_everything():
    said = []
    answers = iter(["cooking\n", "3\n"])
    assert ai_pick.ask_use(lambda prompt: next(answers), said.append) == "chat"
    assert said[0] == "What will you use it for?" and "  5 reading pictures" in said and "isn't one Orchard knows" in said[-1]
    assert ai_pick.ask_use(lambda prompt: "\n", said.append) == "general"

    def closed(prompt):
        raise EOFError

    assert ai_pick.ask_use(closed, said.append) == "general"


def test_a_task_ranks_by_its_own_leaderboard_and_says_who_has_no_score():
    unscored = model("unscored", 90, 5.0)  # strongest in general, no coding score
    weak = with_score(model("weak", 50, 5.0), "code", 0.4)
    strong = with_score(model("strong", 70, 5.0, good_at=("chat",)), "code", 0.6)
    result = ai_pick.pick(fleet(), coding_catalogue(unscored, weak, strong), use="coding")
    assert [r["model"]["id"] for r in result["groups"]["fast"]] == ["strong", "weak", "unscored"]
    assert result["best"]["model"]["id"] == "strong" and result["overall"]["model"]["id"] == "unscored"
    text = "\n".join(ai_pick.report(result, None, []))
    assert "Best pick for coding\n  Strong" in text
    assert "scores: coding 60.0% (CodeBench), general 70 (Index)" in text
    assert "the strongest at coding by CodeBench of those here that should answer at 2 or more words a second" in text
    assert "Strongest overall here, by general score\n  Unscored" in text
    assert "scores: no coding score yet, general 90 (Index)" in text
    assert "Ranked for coding by CodeBench (someone): 2 of the 3 models have a score; the rest come after them" in text
    assert "coding: CodeBench (someone), https://example.org/code, read 2026-11-13" in text
    # Ranked generally, the same three go by their general score.
    general = ai_pick.pick(fleet(), coding_catalogue(unscored, weak, strong))
    assert general["best"]["model"]["id"] == "unscored" and general["overall"] is None


def test_lower_is_better_leaderboards_rank_the_other_way():
    first = with_score(model("first", 50, 5.0), "code", 2)
    tenth = with_score(model("tenth", 90, 5.0), "code", 10)
    result = ai_pick.pick(fleet(), coding_catalogue(first, tenth, higher_is_better=False), use="coding")
    assert result["best"]["model"]["id"] == "first"
    assert "(lower is better)" in "\n".join(ai_pick.report(result, None, []))


def test_without_a_leaderboard_for_the_task_models_made_for_it_come_first():
    result = ai_pick.pick(fleet(), catalogue(model("talker", 90, 5.0, good_at=("chat",)), model("coder", 60, 5.0)),
                          use="coding")
    assert result["best"]["model"]["id"] == "coder"
    text = "\n".join(ai_pick.report(result, None, []))
    assert "No coding leaderboard on the list yet: ranked by general strength, models made for it first." in text


def test_pictures_need_a_model_that_reads_them_and_room_for_its_picture_file():
    seer = model("seer", 60, 10.0, good_at=("chat", "vision"), pictures={"file": "mmproj-F16.gguf", "bytes": int(2e9)})
    blind = model("blind", 90, 5.0)
    result = ai_pick.pick(fleet(), catalogue(seer, blind), use="vision")
    assert [r["model"]["id"] for g in result["groups"].values() for r in g] == ["seer"]
    assert result["best"]["need_gb"] == 14.0  # 10 GB file, a tenth more, 1 GB working memory, 2 GB picture file
    text = "\n".join(ai_pick.report(result, None, []))
    assert "made for chat, pictures" in text
    assert "to read pictures it also needs mmproj-F16.gguf (2.0 GB) from the same page, counted in the memory above" in text
    assert ai_pick.pick(fleet(), catalogue(seer))["best"]["need_gb"] == 12.0  # not counted for other uses


def test_the_picture_file_is_found_in_the_repo():
    files = {"m-Q4.gguf": 9, "mmproj-F32.gguf": 4, "mmproj-BF16.gguf": 2, "mmproj-F16.gguf": 2, "x/mmproj-F16.gguf_file": 1}
    assert ai_pick.picture_file(files) == {"file": "mmproj-F16.gguf", "bytes": 2}
    assert ai_pick.picture_file({"mmproj-model.gguf": 3}) == {"file": "mmproj-model.gguf", "bytes": 3}
    assert ai_pick.picture_file({"m-Q4.gguf": 9}) is None
    seer = model("seer", 60, 10.0, good_at=("vision",))
    ai_pick.refresh([seer], lambda repo: {"seer-Q4_K_M.gguf": 10, "mmproj-F16.gguf": 7})
    assert seer["pictures"] == {"file": "mmproj-F16.gguf", "bytes": 7}


# What the Checker found.

def test_no_machine_answering_is_said_not_crashed():
    gone = ai_pick.Machine(name="old-mac", target="old-mac", reachable=False, why="timed out")
    for machines in ([], [gone]):
        result = ai_pick.pick(machines, catalogue(model("mid", 50, 5.0), model("moe", 75, 45.0, params=300.0, active=13.0)))
        assert result["best"] is None
        assert all(r["why"] == "no machine answered, so nothing could be judged" for r in result["groups"]["no"])
        assert "Nothing on the list runs on these machines." in "\n".join(ai_pick.report(result, None, []))


def test_the_copy_you_have_runs_from_disk_and_a_model_that_wont_run_names_its_smallest_file():
    big = model("big", 70, 0, params=700.0, active=16.0,
                quants=[{"quant": "Q4", "files": ["q4"], "bytes": int(40e9)},
                        {"quant": "IQ2", "files": ["iq2"], "bytes": int(30e9), "owned": True, "owned_on": "big-box"}])
    assert ai_pick.pick(fleet(), catalogue(big))["best"]["quant"]["quant"] == "IQ2"
    dense = model("dense", 70, 0, params=100.0,
                  quants=[{"quant": "Q8", "files": ["a"], "bytes": int(100e9)}, {"quant": "Q4", "files": ["b"], "bytes": int(60e9)}])
    row = ai_pick.pick(fleet(), catalogue(dense))["groups"]["no"][0]
    assert row["quant"]["quant"] == "Q4" and "needs 67.0 GB of memory" in row["why"]


def test_a_repo_hugging_face_says_is_gone_is_missing_not_unreachable():
    import urllib.error

    def get_files(repo):
        raise urllib.error.HTTPError(repo, 401, "Unauthorized", {}, None)

    models = [model("gone", 50, 5.0)]
    assert ai_pick.refresh(models, get_files) == {"repos": 1, "reached": 1, "checked": 0, "missing": 1}
    assert models[0]["quants"][0]["missing"] is True


# The command, by use.

def test_the_command_asks_what_it_is_for_when_someone_is_there(monkeypatch):
    unscored = model("unscored", 90, 5.0)
    strong = with_score(model("strong", 70, 5.0), "code", 0.6)
    monkeypatch.setattr(ai_pick, "read_machines", lambda only=None: fleet())
    monkeypatch.setattr(ai_pick, "builtin_list", lambda: coding_catalogue(unscored, strong))
    monkeypatch.setattr(cli_agentic, "_can_ask", lambda: True)
    result = runner.invoke(cli.app, ["ai", "pick", "--offline"], input="1\n")
    assert result.exit_code == 0, result.output
    assert "Best pick for coding\n  Strong" in result.stdout
    result = runner.invoke(cli.app, ["ai", "pick", "--offline", "--for", "coding", "--json"])
    data = json.loads(result.stdout)
    assert data["use"] == "coding" and data["ranked_by"] == "code" and data["scored"] == 1 and data["listed"] == 2
    assert data["best"]["id"] == "strong" and data["best"]["use_score"] == 0.6 and data["overall"]["id"] == "unscored"
    assert {"source": "code", "benchmark": "code", "score": 0.6, "task": "coding"} in data["best"]["scores"]
    assert data["best"]["context_tokens"] == 32768 and data["best"]["good_at"] == ["coding"]
    monkeypatch.setattr(cli_agentic, "_can_ask", lambda: False)
    assert "Best pick\n" in runner.invoke(cli.app, ["ai", "pick", "--offline"]).stdout  # no one to ask: everything
    bad = runner.invoke(cli.app, ["ai", "pick", "--offline", "--for", "cooking"])
    assert bad.exit_code == 2


def test_the_command_says_when_no_machine_matches(monkeypatch):
    monkeypatch.setattr(ai_pick.config, "load_workers", lambda: [])
    monkeypatch.setattr(ai_pick, "builtin_list", lambda: catalogue(model("mid", 50, 5.0)))
    result = runner.invoke(cli.app, ["ai", "pick", "--offline", "--for", "general", "--machines", "nobody"])
    assert result.exit_code == 1 and "Nothing on the list runs" in result.stdout


# What the Checker found in the by-use ranking.

def test_only_a_model_with_a_general_score_is_called_strongest_overall():
    unscored = model("unscored", 0, 5.0, params=8.0, good_at=("vision",), pictures={"file": "p.gguf", "bytes": 1})
    unscored["scores"] = []
    squeezed = model("squeezed", 80, 0, params=36.0, active=4.0, good_at=("vision",),
                     quants=[{"quant": "IQ2_M", "files": ["s"], "bytes": int(13e9)}], pictures={"file": "p.gguf", "bytes": 1})
    result = ai_pick.pick(fleet(), catalogue(unscored, squeezed), use="vision")
    assert result["best"]["model"]["id"] == "unscored"
    assert result["overall"]["model"]["id"] == "squeezed"  # never the unscored one
    alone = ai_pick.pick(fleet(), catalogue(unscored), use="vision")
    assert alone["overall"] is None
    text = "\n".join(ai_pick.report(alone, None, []))
    assert "Strongest overall" not in text
    assert ("no pictures leaderboard on the list yet, so this is the strongest of them made for reading pictures, "
            "with no general score either") in text


def test_a_pick_without_a_score_for_the_use_says_how_it_was_chosen():
    coder = model("coder", 60, 5.0, params=8.0)
    talker = model("talker", 90, 5.0, params=8.0, good_at=("chat",))
    scored_but_slow = with_score(model("dense", 99, 18.0), "code", 0.9)
    result = ai_pick.pick(fleet(), coding_catalogue(coder, talker, scored_but_slow), use="coding")
    assert result["best"]["model"]["id"] == "coder"
    text = "\n".join(ai_pick.report(result, None, []))
    assert ("none of the models here that run well at full strength has a coding score yet, so this is the "
            "strongest of them made for coding, by general score") in text


def test_no_question_when_the_output_is_piped(monkeypatch):
    class Stream:
        def __init__(self, tty):
            self.tty = tty

        def isatty(self):
            return self.tty

    monkeypatch.setattr(cli_agentic.sys, "stdin", Stream(True))
    monkeypatch.setattr(cli_agentic.sys, "stderr", Stream(True))
    monkeypatch.setattr(cli_agentic.sys, "stdout", Stream(False))
    assert cli_agentic._can_ask() is False
    monkeypatch.setattr(cli_agentic.sys, "stdout", Stream(True))
    assert cli_agentic._can_ask() is True


# Right now, and what earlier runs showed.

NOON = __import__("time").mktime((2026, 11, 14, 12, 0, 0, 0, 0, -1))


def test_room_now_is_whats_free_less_a_margin_and_never_more_than_room():
    mac = machine("this Mac", 16, gpu="apple")
    mac["free_mb"] = 3 * GIB
    assert mac.room_now_gb == 2.0
    mac["free_mb"] = 15 * GIB
    assert mac.room_now_gb == 12.0
    mac["cap_gb"] = 5.0
    assert (mac.room_gb, mac.full_room_gb, mac.room_now_gb) == (5.0, 12.0, 5.0)
    mac["free_mb"] = None
    assert mac.room_now_gb is None
    box = machine("big-box", 8, gpu="nvidia", vram_gb=2, system="Linux", wsl=True)
    box["free_mb"] = 2 * GIB
    assert box.room_now_gb == 2.6  # 1 GB of what's free, and the card's own memory


def busy_fleet():
    mac, box, old = fleet()
    mac["free_mb"] = 3 * GIB
    return [mac, box, old]


def test_a_pick_that_does_not_fit_now_names_the_busy_machine_and_the_best_that_does():
    big, small = model("big", 60, 18.0, active=3.0), model("small", 30, 3.0, params=5.0)
    result = ai_pick.pick(busy_fleet(), catalogue(big, small))
    assert result["best"]["model"]["id"] == "big" and result["best"]["fits_now"] is False
    assert result["best_now"]["model"]["id"] == "small"
    text = "\n".join(ai_pick.report(result, None, []))
    assert "room for 12.0 GB of model (2.0 GB of it free right now)" in text
    assert "right now: busy (this Mac has 2.0 GB free for its 10.6 GB share" in text
    assert "it should fit when this Mac, big-box and old-mac are quieter" in text
    assert "doesn't fit" not in text and "won't fit" not in text
    assert "The picks are for when your machines are quiet." in text
    assert "Best that fits right now\n  Small (Q4_K_M, 3.0 GB file)" in text
    assert "held in memory: big-box 4.3 GB; about" in text  # where it would sit right now
    assert "Together: room for 23.6 GB of model in memory, 9.6 GB of it free right now." in text
    data = ai_pick.as_json(result, None, [])
    assert data["best"]["fits_now"] is False and data["best_now"]["id"] == "small"
    assert data["best"]["short_now"][0] == {"machine": "this Mac", "share_gb": 10.6, "free_gb": 2.0}
    assert data["machines"][0]["room_now_gb"] == 2.0


def test_when_the_best_pick_fits_now_nothing_else_is_offered():
    result = ai_pick.pick(fleet(), catalogue(model("small", 30, 3.0, params=5.0)))
    assert result["best"]["fits_now"] is True and result["best_now"] is None
    assert "right now: fits\n" in "\n".join(ai_pick.report(result, None, []))


def save_record(rid, **fields):
    ai.AI_DIR.mkdir(parents=True, exist_ok=True)
    record = receipts.sign({"id": rid, "kind": "ai", "started_at": NOON, **fields})
    (ai.AI_DIR / f"{rid}.json").write_text(json.dumps(record))
    return record


def layered(mid="mid", gb=18.0, layers=30):
    return model(mid, 50, gb, active=3.0, layers=layers)


def run_of(rid, state, *, layers=18, devices="CUDA0,RPC0", split="2,16", far=(), reason=None, ms=None,
           overrides="", file="mid-Q4_K_M.gguf", **extra):
    return save_record(rid, state=state, reason=reason, timings={"ms_per_word": ms} if ms else {},
                       command=["./llama-completion", "-m", f"~/models/{file}", "-p", "hi"],
                       settings={"layers": layers, "devices": devices, "split": split, "overrides": overrides},
                       machines={"engine": "big-box", "helper": True, "far_helpers": list(far)}, **extra)


def test_how_much_each_machine_held_comes_from_the_layers():
    mid = layered()
    record = {"settings": {"layers": 28, "devices": "CUDA0,RPC0,RPC1", "split": "2,16,10", "overrides": ""},
              "machines": {"engine": "big-box", "helper": True, "far_helpers": ["old-mac"]}}
    held = ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac")
    # 0.6 GB a layer; -ngl 28 sends 27 layers and the output layer: big-box keeps 3 and
    # its card takes 2, this Mac 16, old-mac 10 (one of them the output layer)
    assert held == {"big-box": 3.0, "this Mac": 9.6, "old-mac": 6.0}
    record["settings"]["overrides"] = "exps=CPU"
    assert ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac") is None
    record["settings"].update(overrides="", split="")
    assert ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac") is None


def test_an_answered_run_shows_its_measured_speed_next_to_the_estimate():
    run_of("ai-20261114-100000-aaaa", "answered", ms=250.0)
    result = ai_pick.pick(fleet(), catalogue(layered()), past=ai_pick.history(fleet(), catalogue(layered())))
    text = "\n".join(ai_pick.report(result, None, []))
    assert "(estimate)" in text
    assert ("ran here: 4.0 words a second on big-box and this Mac, 14 Nov "
            "(record ai-20261114-100000-aaaa, measured)") in text
    run = ai_pick.as_json(result, None, [])["best"]["runs"][0]
    assert run["held"] == {"big-box": 9.0, "this Mac": 9.6} and run["words_per_second"] == 4.0


def test_a_machine_that_ran_short_of_memory_counts_less_room_after():
    run_of("ai-20261114-110000-bbbb", "stopped", reason="this machine is short of memory")
    machines = fleet()
    past = ai_pick.history(machines, catalogue(layered()))
    assert past["caps"]["this Mac"]["gb"] == 7.7  # 80% of the 9.6 GB it held
    result = ai_pick.pick(machines, catalogue(layered()), past=past)
    assert machines[0].room_gb == 7.7
    text = "\n".join(ai_pick.report(result, None, []))
    assert ("counted at 7.7 GB, not 12.0: it ran short of memory holding about 9.6 GB on 14 Nov "
            "(record ai-20261114-110000-bbbb)") in text
    assert "tried here: stopped, this machine is short of memory, 14 Nov (record ai-20261114-110000-bbbb)" in text


def test_a_run_that_answered_holding_more_lifts_the_limit():
    run_of("ai-20261114-110000-bbbb", "stopped", reason="this machine is short of memory")
    run_of("ai-20261114-120000-cccc", "answered", ms=300.0, layers=20, split="2,18")  # this Mac held 10.8
    past = ai_pick.history(fleet(), catalogue(layered()))
    assert past["caps"]["this Mac"]["gb"] == 10.8


def test_a_far_helper_or_the_engine_running_short_is_read_too():
    run_of("ai-20261114-110000-bbbb", "stopped", layers=28, devices="CUDA0,RPC0,RPC1", split="2,16,10",
           far=["old-mac"], reason="old-mac is short of memory")
    run_of("ai-20261114-120000-cccc", "stopped", reason="under 1500 MB free on the engine machine")
    caps = ai_pick.history(fleet(), catalogue(layered()))["caps"]
    assert caps["old-mac"]["gb"] == 4.8 and caps["big-box"]["gb"] == 7.2  # 80% of 6.0 and of 9.0
    machines = fleet()
    ai_pick.pick(machines, catalogue(layered()), past={"runs": {}, "caps": caps})
    assert machines[2].room_gb == 4.0 and "cap_gb" not in machines[2]  # already below the limit
    assert machines[1].room_gb == 7.2


def test_unsealed_refused_and_hand_split_runs_change_nothing():
    tampered = run_of("ai-20261114-090000-dddd", "answered", ms=100.0)
    tampered["timings"]["ms_per_word"] = 10.0
    (ai.AI_DIR / "ai-20261114-090000-dddd.json").write_text(json.dumps(tampered))
    run_of("ai-20261114-100000-eeee", "refused", reason="this machine is short of memory")
    run_of("ai-20261114-110000-ffff", "stopped", reason="this machine is short of memory", overrides="exps=CPU")
    past = ai_pick.history(fleet(), catalogue(layered()))
    assert past["caps"] == {}
    assert [r["id"] for r in past["runs"][("mid", "Q4_K_M")]] == ["ai-20261114-110000-ffff"]


def test_a_served_answer_is_read_through_its_session():
    save_record("ais-20261114-100000-1111", kind="ai-session", state="stopped", reason="idle",
                command=["./llama-server", "-m", "~/models/mid-Q4_K_M.gguf"],
                settings={"layers": 18, "devices": "CUDA0,RPC0", "split": "2,16", "overrides": ""},
                machines={"engine": "big-box", "helper": True, "far_helpers": []})
    save_record("ai-20261114-100500-2222", via="server", session="ais-20261114-100000-1111", state="answered",
                timings={"ms_per_word": 500.0}, machines={"engine": "big-box", "helper": True, "far_helpers": []},
                settings={"layers": 18, "devices": "CUDA0,RPC0", "split": "2,16", "overrides": ""})
    runs = ai_pick.history(fleet(), catalogue(layered()))["runs"][("mid", "Q4_K_M")]
    assert [(r["id"], r["words_per_second"]) for r in runs] == [("ai-20261114-100500-2222", 2.0)]


def test_the_stop_reasons_read_here_are_the_ones_orchard_ai_writes():
    import inspect
    source = inspect.getsource(ai._Pool._why_stop)
    assert '"this machine is short of memory"' in source
    assert 'f"{h[\'host\']} is short of memory"' in source
    assert 'f"under {stop_at} MB free on the engine machine"' in source
    who = {"machines": {"engine": "big-box"}}
    assert ai_pick.short_of_memory("this machine is short of memory", who, str, "this Mac") == "this Mac"
    assert ai_pick.short_of_memory("old-mac is short of memory", who, str, "this Mac") == "old-mac"
    assert ai_pick.short_of_memory("under 1500 MB free on the engine machine", who, str, "this Mac") == "big-box"
    assert ai_pick.short_of_memory("old-mac is in use: a job", who, str, "this Mac") is None


def test_linux_in_windows_counts_what_windows_has_free(monkeypatch):
    box = machine("big-box", 8, wsl=True, system="Linux")
    box["free_mb"] = 7 * GIB
    monkeypatch.setattr(ai_pick, "read_machine", lambda name, target: box)
    monkeypatch.setattr(ai_pick.config, "load_workers", lambda: [])
    monkeypatch.setattr(ai, "load_settings", lambda *a, **k: {"engine": {"host": "big-box"}, "windows": {"host": "win"}})
    monkeypatch.setattr(ai, "has_windows", lambda cfg: True)
    monkeypatch.setattr(ai_pick, "windows_disk", lambda host: 40.0)
    seen = []
    monkeypatch.setattr(ai, "_remote", lambda host, cmd, timeout: seen.append((host, cmd)) or type(
        "R", (), {"stdout": '"(PDH-CSV 4.0)","\\\\WIN\\Memory\\Available MBytes"\n"11/14/2026 12:00:00.000","2048.000000"\n'})())
    machines = ai_pick.read_machines()
    assert machines[0]["free_mb"] == 2048 and machines[0]["disk_free_gb"] == 40.0
    assert seen == [("win", ai_pick.WINDOWS_FREE)]


def test_the_best_that_fits_now_is_judged_where_it_would_sit_now():
    # Right now only old-mac, slow at 1 GB/s, has room: the bigger model would crawl
    # there, so the smaller one that stays comfortable wins.
    mac, box, old = fleet()
    mac["free_mb"], box["free_mb"], old["copy_gbs"] = 1 * GIB, 1 * GIB, 1.0
    big = model("big", 90, 18.0, active=3.0)
    slow_now = model("mid", 60, 2.0, params=4.0)
    quick_now = model("small", 30, 0.5, params=1.0)
    result = ai_pick.pick([mac, box, old], catalogue(big, slow_now, quick_now))
    assert result["best"]["model"]["id"] == "big"
    assert result["best_now"]["model"]["id"] == "small"
    mid = next(r for r in result["groups"]["fast"] if r["model"]["id"] == "mid")
    assert mid["words_per_second_now"] < 2.0 < mid["words_per_second"]
    text = "\n".join(ai_pick.report(result, None, []))
    assert "right now: fits, held as old-mac 2.1 GB, big-box 1.1 GB; about 0.7 words a second there" in text
    assert "Best that fits right now\n  Small (Q4_K_M, 0.5 GB file)" in text


def test_an_answer_holding_the_same_does_not_lift_the_limit():
    run_of("ai-20261114-100000-aaaa", "answered", ms=250.0)  # this Mac held 9.6
    run_of("ai-20261114-110000-bbbb", "stopped", reason="this machine is short of memory")  # also 9.6
    assert ai_pick.history(fleet(), catalogue(layered()))["caps"]["this Mac"]["gb"] == 7.7


def test_engine_flags_that_move_parts_or_odd_layer_counts_are_not_measured():
    mid = layered()
    record = {"settings": {"layers": 18, "devices": "CUDA0,RPC0", "split": "2,16", "overrides": "",
                           "args": ["--n-cpu-moe", "30"]},
              "machines": {"engine": "big-box", "helper": True, "far_helpers": []}}
    assert ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac") is None
    record["settings"]["args"] = ["-ot=exps=CPU"]
    assert ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac") is None
    record["settings"].update(args=[], layers=-1)
    assert ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac") is None
    record["settings"].update(args=["-ngl", "10"], layers=18)
    assert ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac") is None
    # A served model's extra flags are only in its command: written twice, the later one wins.
    record["settings"]["args"] = []
    served = ["./llama-server", "-m", "m.gguf", "-ngl", "18", "-ts", "2,16", "--device", "CUDA0,RPC0", "-ts", "1,1"]
    assert ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac", served) is None
    assert ai_pick.held_gb(record, mid, mid["quants"][0], lambda h: h, "this Mac", served[:-2]) is not None


def test_room_now_never_goes_below_zero_and_a_model_is_not_shown_twice():
    tiny = machine("tiny", 1, system="Linux")
    assert tiny.room_gb < 0 and tiny.room_now_gb == 0.0
    mac, box, old = fleet()
    mac["free_mb"] = 1 * GIB
    coder = model("coder", 50, 18.0, active=3.0, scores=[{"source": "code", "score": 0.5}])
    small = model("small", 30, 3.0, params=5.0)
    cat = catalogue(coder, small)
    cat["sources"].append({"id": "code", "task": "coding", "name": "Code Bench", "url": "https://example.org/c",
                           "read_on": "2026-11-14", "scale": "fraction"})
    result = ai_pick.pick([mac, box, old], cat, "coding")
    assert result["best_now"]["model"]["id"] == "small" and result["overall"]["model"]["id"] == "small"
    text = "\n".join(ai_pick.report(result, None, []))
    assert text.count("  Small (Q4_K_M") == 1 and "(also the strongest overall here, by general score)" in text


def test_a_model_one_machine_can_hold_goes_on_the_fastest_that_can():
    roomy_slow = machine("big-box", 16, system="Linux", speed=8.0)
    small_fast = machine("this Mac", 8, gpu="apple", speed=40.0)
    split = ai_pick.plan_memory(3.0, [roomy_slow, small_fast])
    assert [(m["name"], gb) for m, gb in split] == [("this Mac", 3.0)]
    split = ai_pick.plan_memory(10.0, [roomy_slow, small_fast])
    assert [(m["name"], gb) for m, gb in split] == [("big-box", 10.0)]
