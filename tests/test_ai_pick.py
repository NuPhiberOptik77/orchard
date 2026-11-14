import json

import pytest
from typer.testing import CliRunner

from orchard import ai, ai_pick, cli, cli_agentic

runner = CliRunner()
GIB = 1024


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
    assert mac.room_gb == 12.0 and old.room_gb == 6.0
    assert box.room_gb == 7.6  # 6 GB of memory and 80% of a 2 GB card
    assert machine("tiny", 4).room_gb == 2.0  # at least 2 GB stays with the machine


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
    assert row["group"] == "no" and "your machines have 25.6 GB together" in row["why"]


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
    full = model("full", 60, 0, params=35.0, active=3.0, quants=[{"quant": "Q4_K_M", "files": ["f"], "bytes": int(21e9)}])
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
