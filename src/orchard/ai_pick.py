"""Which of the strongest open models your machines can run, and how.

`orchard ai pick` reads each machine Orchard knows (this one and every worker):
its memory, graphics memory, free disk and how fast it reads memory. It then
goes through a dated list of open models and sorts them into three groups:

- fast: the model fits in memory across your machines, with room to spare
- slow: too big for memory, but built so it can run reading from disk
  (a mixture of experts), for background work
- won't run here, and why

Room on each machine is its memory less what the machine keeps for itself, and
less again where an earlier run there ran short of memory: Orchard's own sealed
records of past runs say how much that machine was holding. Each pick also says
whether it fits right now, from what each machine has free at this moment, and
shows what earlier runs of the same file measured next to the estimate.

It asks what the model is for (coding, agents, chat and writing, reasoning and
maths, reading pictures, or a bit of everything) and, within each group, puts
the strongest at that first: by a public leaderboard for that task where the
list has one, by a general one otherwise, each named with its date. Every model
shows what it can do, how much it takes in at once and all its scores. The split
preview says which machine would hold how much. Speeds are
rough estimates from each machine's measured memory speed; a short test run
gives the real number.

The list ships with Orchard. Unless told to stay offline, it also asks Hugging
Face for each model's current file sizes and for popular new GGUF models, so new
models show up between releases. Nothing about you or your machines is sent.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Callable

from . import ai, config, receipts

GB = 1e9
HF = "https://huggingface.co"
HF_TIMEOUT = 15
NEW_FROM = ("unsloth", "bartowski", "ggml-org")
NEW_LIMIT = 6
PROBE_TIMEOUT = 90
USER_AGENT = "orchard-ai-pick"

# What a machine keeps for itself: a quarter of its memory, and at least 2 GB;
# a Mac at least 4 GB, as macOS and its background work take that much (an 8 GB
# Mac ran short of memory holding 5.5 GB of a model).
RESERVE_SHARE = 0.25
RESERVE_MIN_MB = 2048
RESERVE_MIN_MB_MAC = 4096
# What fits right now leaves this much of what's free untouched.
NOW_MARGIN_GB = 1.0
# Where a run ran short of memory holding a share, count at most this part of it.
SHORT_SHARE = 0.8
# Engine flags that move parts of a model by hand, so the split alone doesn't say who held what.
PLACEMENT_FLAGS = {"-ot", "--override-tensor", "-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"}
# Flags Orchard writes once from its settings; written again, the later one wins.
POSITION_FLAGS = [{"-ngl", "--n-gpu-layers", "--gpu-layers"}, {"-ts", "--tensor-split"},
                  {"-dev", "--device"}, {"-sm", "--split-mode"}, {"-mg", "--main-gpu"}]


def _moved_by_hand(command) -> bool:
    """The engine was told where to put parts beyond Orchard's own settings."""
    words = [str(w).split("=", 1)[0] for w in (command if isinstance(command, list) else str(command or "").split())]
    return bool(set(words) & PLACEMENT_FLAGS) or any(sum(w in group for w in words) > 1 for group in POSITION_FLAGS)
# Of a graphics card's own memory, what a model can use.
VRAM_SHARE = 0.8
# Memory a model needs beyond its file: a tenth more, and 1 GB for its working memory.
NEED_SHARE = 1.10
NEED_EXTRA_GB = 1.0
# A mixture of experts reads this share of itself per word or less: it can run from disk.
DISK_ACTIVE_SHARE = 0.2
DISK_SHARE = 0.9
# The best pick should be comfortable to use, not just the strongest.
COMFORT_WPS = 2.0
# Squeezed below this many bits per weight, a model is weaker than its score says.
FULL_BITS = 3.5
# A rough rule for English text: three words to every four tokens.
WORDS_PER_TOKEN = 0.75

# What a model can be for, in the order the question lists them.
USES = {
    "coding": "coding",
    "agents": "agents and tool use",
    "chat": "chat and writing",
    "reasoning": "reasoning and maths",
    "vision": "reading pictures",
    "general": "a bit of everything",
}
ALIASES = {"code": "coding", "agent": "agents", "tools": "agents", "tool": "agents", "writing": "chat",
           "maths": "reasoning", "math": "reasoning", "images": "vision", "image": "vision",
           "pictures": "vision", "everything": "general", "all": "general"}
# How a task or a capability reads in a sentence.
WORD = {"vision": "pictures"}

# Read-only, and runs on Python 3.8, which an older Mac's command-line tools have.
PROBE = r"""
import json, os, platform, re, shutil, subprocess, time
def run(argv):
    try:
        return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True, timeout=20).stdout
    except Exception:
        return ""
f = {"system": platform.system(), "arch": platform.machine(), "threads": os.cpu_count(),
     "wsl": "microsoft" in platform.release().lower(), "gpus": []}
f["ram_mb"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1 << 20)
if f["system"] == "Linux":
    m = re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read())
    f["free_mb"] = int(m.group(1)) // 1024 if m else None
    m = re.search(r"model name\s*:\s*(.+)", open("/proc/cpuinfo").read())
    f["chip"] = m.group(1).strip() if m else ""
else:
    out = run(["vm_stat"])
    m = re.search(r"page size of (\d+)", out)
    page = int(m.group(1)) if m else 4096
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        m = re.search(key + r":\s+(\d+)", out)
        pages += int(m.group(1)) if m else 0
    f["free_mb"] = pages * page // (1 << 20) if out else None
    f["chip"] = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
for tool in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
    if shutil.which(tool) or os.path.exists(tool):
        for line in run([tool, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).splitlines():
            name, _, mb = line.rpartition(",")
            if name and mb.strip().isdigit():
                f["gpus"].append({"name": name.strip(), "kind": "nvidia", "memory_mb": int(mb)})
        break
if f["system"] == "Darwin" and f["arch"] == "arm64":
    f["gpus"].append({"name": (f["chip"] or "Apple") + " GPU", "kind": "apple", "memory_mb": None})
elif f["system"] == "Darwin":
    for line in run(["system_profiler", "SPDisplaysDataType"]).splitlines():
        if "Chipset Model" in line:
            f["gpus"].append({"name": line.split(":", 1)[1].strip(), "kind": "other", "memory_mb": None})
f["disk_free_gb"] = round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9, 1)
# Copy between two buffers already in memory, so page faults don't count.
src, dst = bytearray(64 << 20), bytearray(64 << 20)
dst[:] = src
start = time.perf_counter()
for _ in range(16):
    dst[:] = src
f["copy_gbs"] = round(16 * len(src) / 1e9 / (time.perf_counter() - start), 1)
print(json.dumps(f))
"""

WINDOWS_DISK = "$ProgressPreference = 'SilentlyContinue'; [math]::Round((Get-PSDrive C).Free / 1e9, 1)"
WINDOWS_FREE = 'typeperf "\\Memory\\Available MBytes" -sc 1'


class Machine(dict):
    """One machine's readings, and what it can give a model."""

    @property
    def vram_gb(self) -> float:
        """Most of a graphics card's own memory, which the machine itself doesn't use."""
        return sum(g["memory_mb"] or 0 for g in self["gpus"] if g["kind"] == "nvidia") * VRAM_SHARE / 1024

    @property
    def full_room_gb(self) -> float:
        """Its RAM less what the machine keeps for itself, plus graphics memory."""
        ram = self["ram_mb"]
        floor = RESERVE_MIN_MB_MAC if self.get("system") == "Darwin" else RESERVE_MIN_MB
        return round((ram - max(floor, ram * RESERVE_SHARE)) / 1024 + self.vram_gb, 1)

    @property
    def room_gb(self) -> float:
        """Memory a model can have here when the machine is quiet: its full room,
        or less where a run here ran short of memory (`cap_gb`)."""
        cap = self.get("cap_gb")
        return self.full_room_gb if cap is None else min(self.full_room_gb, cap)

    @property
    def room_now_gb(self) -> float | None:
        """What it can give a model at this moment: what's free less a margin,
        plus graphics memory, and never more than its room. None if unread."""
        free = self.get("free_mb")
        if free is None:
            return None
        return round(max(0.0, min(self.room_gb, max(0.0, free / 1024 - NOW_MARGIN_GB) + self.vram_gb)), 1)

    @property
    def speed_gbs(self) -> float:
        return float(self.get("copy_gbs") or 1.0)


def read_machine(name: str, target: str | None) -> Machine:
    """Run the probe here (target None) or over ssh. Reads only."""
    argv = [sys.executable, "-I", "-"] if target is None else [*ai._ssh(target), "python3 -I -"]
    try:
        out = subprocess.run(argv, input=PROBE, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        facts = json.loads(out.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        return Machine(name=name, target=target, reachable=False, why=str(exc)[:120] or "no answer")
    return Machine(name=name, target=target, reachable=True, **facts)


def windows_disk(host: str) -> float | None:
    out = ai._remote(host, ai._powershell(WINDOWS_DISK), 60).stdout.replace("\x00", "").strip()
    try:
        return float(out.splitlines()[-1])
    except (ValueError, IndexError):
        return None


def windows_free(host: str) -> int | None:
    """Free memory on Windows, in MB: Linux inside it can only grow into that."""
    return ai.parse_typeperf(ai._remote(host, WINDOWS_FREE, 40).stdout)


def read_machines(only: list[str] | None = None) -> list[Machine]:
    """This machine and every Orchard worker, read side by side."""
    wanted = [("this Mac" if sys.platform == "darwin" else "this machine", None)]
    wanted += [(w.name, w.target) for w in config.load_workers()]
    if only:
        wanted = [(n, t) for n, t in wanted if n in only or (t is None and "this" in only)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(wanted) or 1) as pool:
        machines = list(pool.map(lambda pair: read_machine(*pair), wanted))
    # Linux inside Windows sees a virtual disk; the real limit is the Windows drive.
    try:
        cfg = ai.load_settings()
    except ai.ConfigError:
        cfg = None
    for m in machines:
        if m.get("wsl") and cfg and ai.has_windows(cfg) and m["target"] == cfg["engine"]["host"]:
            real = windows_disk(cfg["windows"]["host"])
            if real is not None:
                m["disk_free_gb"], m["disk_note"] = real, "the Windows drive"
            free = windows_free(cfg["windows"]["host"])
            if free is not None:
                m["free_mb"] = free if m.get("free_mb") is None else min(m["free_mb"], free)
        elif m.get("wsl"):
            m["disk_note"] = "a virtual disk: the Windows drive may have less"
    return machines


# The list of models.

def builtin_list() -> dict:
    return json.loads(resources.files("orchard").joinpath("ai_models.json").read_text())


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HF_TIMEOUT) as resp:
        return json.loads(resp.read())


def hf_files(repo: str) -> dict[str, int]:
    """Every file in a Hugging Face repo, with its size in bytes."""
    data = _get_json(f"{HF}/api/models/{urllib.parse.quote(repo)}/tree/main?recursive=true")
    return {item["path"]: int((item.get("lfs") or {}).get("size") or item.get("size") or 0)
            for item in data if item.get("type") == "file"}


def picture_file(files: dict[str, int]) -> dict | None:
    """The engine reads pictures through a separate small file in the model's
    repo (its name has "mmproj" in it). F16 if there is one, else BF16, else the smallest."""
    found = {path: size for path, size in files.items()
             if "mmproj" in path.rsplit("/", 1)[-1].lower() and path.endswith(".gguf")}
    if not found:
        return None

    def rank(path: str) -> tuple:
        name = path.lower()
        return (0 if name.endswith("-f16.gguf") else 1 if name.endswith("-bf16.gguf") else 2, found[path])

    path = min(found, key=rank)
    return {"file": path, "bytes": found[path]}


def refresh(models: list[dict], get_files: Callable[[str], dict[str, int]] = hf_files) -> dict:
    """Check each listed file on Hugging Face and take its current size. A model
    whose files are gone is marked, not dropped. Returns what was checked."""
    repos = sorted({m["repo"] for m in models})
    found: dict[str, dict[str, int] | None] = {}

    def fetch(repo: str) -> None:
        try:
            found[repo] = get_files(repo)
        except urllib.error.HTTPError as exc:
            found[repo] = {} if exc.code in (401, 404) else None
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            found[repo] = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, repos))
    checked = missing = 0
    for model in models:
        files = found.get(model["repo"])
        if files is None:
            continue
        if "vision" in (model.get("good_at") or []):
            model["pictures"] = picture_file(files) or model.get("pictures")
        for quant in model["quants"]:
            sizes = [files.get(path) for path in quant["files"]]
            if all(size for size in sizes):
                quant["bytes"], quant["checked"] = sum(sizes), True
                checked += 1
            else:
                quant["missing"] = True
                missing += 1
    reached = sum(1 for v in found.values() if v is not None)
    return {"repos": len(repos), "reached": reached, "checked": checked, "missing": missing}


def new_models(known: set[str], get: Callable[[str], object] = _get_json) -> list[dict]:
    """Popular GGUF repos from well-known publishers that the list doesn't have yet."""
    def base(repo: str) -> str:
        name = repo.split("/")[-1]
        name = name.split("_", 1)[1] if "_" in name else name  # bartowski's Maker_Model-GGUF
        return re.sub(r"-gguf$", "", name.lower())

    known_bases = {base(repo) for repo in known}
    out = []
    for author in NEW_FROM:
        try:
            items = get(f"{HF}/api/models?author={author}&filter=gguf&pipeline_tag=text-generation"
                        "&sort=downloads&direction=-1&limit=30")
        except (urllib.error.URLError, OSError, ValueError):
            continue
        for item in items or []:
            repo = item.get("id") or item.get("modelId") or ""
            text_model = item.get("pipeline_tag") in (None, "text-generation", "image-text-to-text")
            if repo and base(repo) not in known_bases and repo.lower().endswith("gguf") and text_model:
                out.append({"repo": repo, "downloads": item.get("downloads") or 0,
                            "modified": (item.get("lastModified") or "")[:10]})
    out.sort(key=lambda m: m["downloads"], reverse=True)
    return out[:NEW_LIMIT]


# What fits, where, and how fast.

def need_gb(quant: dict) -> float:
    return quant["bytes"] / GB * NEED_SHARE + NEED_EXTRA_GB


def active_share(model: dict) -> float:
    if model.get("active_b") and model.get("params_b"):
        return min(1.0, model["active_b"] / model["params_b"])
    return 1.0


def plan_memory(need: float, machines: list[Machine],
                room_of: Callable[[Machine], float] = lambda m: m.room_gb) -> list[tuple[Machine, float]] | None:
    """The fewest machines that hold `need` GB between them, most room first,
    each given a share in proportion to its room. None if they can't."""
    ranked = sorted(machines, key=lambda m: (room_of(m), m.speed_gbs), reverse=True)
    chosen: list[Machine] = []
    for machine in ranked:
        if room_of(machine) <= 0:
            continue
        chosen.append(machine)
        room = sum(room_of(m) for m in chosen)
        if room >= need:
            return [(m, round(need * room_of(m) / room, 1)) for m in chosen]
    return None


def now_room(m: Machine) -> float:
    """Room right now; a machine whose free memory couldn't be read counts its room."""
    now = m.room_now_gb
    return m.room_gb if now is None else now


def right_now(row: dict, machines: list[Machine]) -> dict:
    """Whether a model that fits in memory fits at this moment, and if not, which
    machines are short and by how much."""
    if row["group"] != "fast":
        return row
    split = plan_memory(row["need_gb"], machines, now_room)
    short = [(m, share, now_room(m)) for m, share in row["split"] if share > now_room(m)]
    speed = estimate_words_per_second(row["model"], row["quant"], split) if split else None
    return row | {"fits_now": split is not None, "split_now": split, "short_now": [] if split else short,
                  "words_per_second_now": speed}


def estimate_words_per_second(model: dict, quant: dict, split: list[tuple[Machine, float]]) -> float:
    """Each word reads the model's active weights once, each machine its share,
    at that machine's measured memory speed. A rough guide only: it leaves out
    the network between machines and what graphics chips add."""
    per_word = 0.0
    total = sum(share for _, share in split) or 1.0
    active = quant["bytes"] / GB * active_share(model)
    for machine, share in split:
        per_word += active * share / total / machine.speed_gbs
    return round(1 / per_word, 1) if per_word else 0.0


def judge(model: dict, quant: dict, machines: list[Machine], extra_gb: float = 0.0) -> dict:
    """Where one model fits: fast (memory), slow (disk) or not here, and why.
    `extra_gb` is memory for something loaded beside it, such as its picture file."""
    need = round(need_gb(quant) + extra_gb, 1)
    out = {"model": model, "quant": quant, "need_gb": need}
    if quant.get("missing"):
        return out | {"group": "no", "why": "its files are no longer on Hugging Face under that name"}
    if model.get("llama_cpp") is False:
        return out | {"group": "no", "why": "the engine can't run this kind of model yet"}
    if model.get("llama_cpp") is None:
        return out | {"group": "no", "why": "the engine's support for this kind of model isn't confirmed yet"}
    if not machines:
        return out | {"group": "no", "why": "no machine answered, so nothing could be judged"}
    split = plan_memory(need, machines)
    if split:
        return right_now(out | {"group": "fast", "split": split,
                                "words_per_second": estimate_words_per_second(model, quant, split)}, machines)
    room = round(sum(max(m.room_gb, 0) for m in machines), 1)
    size = round(quant["bytes"] / GB, 1)
    home = max(machines, key=lambda m: (m.get("disk_free_gb") or 0, m.room_gb))
    holder = next((m for m in machines if quant.get("owned_on") and m.get("target") == quant["owned_on"]), None)
    if active_share(model) <= DISK_ACTIVE_SHARE:
        if holder is not None:
            return out | {"group": "slow", "home": holder,
                          "why": f"needs {need} GB of memory and your machines have {room} GB, so it reads from disk"}
        if size <= (home.get("disk_free_gb") or 0) * DISK_SHARE:
            return out | {"group": "slow", "home": home,
                          "why": f"needs {need} GB of memory and your machines have {room} GB, so it reads from disk"}
        return out | {"group": "no", "why": f"needs {size} GB of disk; the most free on one machine is "
                                            f"{home.get('disk_free_gb')} GB ({home['name']})"}
    return out | {"group": "no", "why": f"needs {need} GB of memory and your machines have {room} GB together"}


def bits_per_weight(model: dict, quant: dict) -> float | None:
    if not model.get("params_b"):
        return None
    return round(quant["bytes"] * 8 / (model["params_b"] * 1e9), 1)


def squeezed(row: dict) -> bool:
    bits = bits_per_weight(row["model"], row["quant"])
    return bits is not None and bits < FULL_BITS


def score_of(model: dict, source_id: str | None) -> float | None:
    for score in model.get("scores") or []:
        if score.get("source") == source_id and isinstance(score.get("score"), (int, float)):
            return float(score["score"])
    return None


def strength(model: dict, primary: str) -> float:
    score = score_of(model, primary)
    return float("-inf") if score is None else score


def resolve_use(text: str | None) -> str:
    """'coding', '1', 'images' and the like, to one of USES. ValueError if unknown."""
    word = (text or "").strip().lower()
    if word.isdigit() and 1 <= int(word) <= len(USES):
        return list(USES)[int(word) - 1]
    word = ALIASES.get(word, word)
    if word in USES:
        return word
    raise ValueError(f"'{text}' isn't one Orchard knows; pick one of: {', '.join(USES)}")


def ask_use(read: Callable[[str], str], write: Callable[[str], None]) -> str:
    """Ask what the model is for. Enter means a bit of everything."""
    write("What will you use it for?")
    for number, label in enumerate(USES.values(), 1):
        write(f"  {number} {label}")
    for _ in range(3):
        try:
            answer = read(f"Pick a number [{len(USES)}]: ")
        except EOFError:
            break
        try:
            return resolve_use(answer.strip() or str(len(USES)))
        except ValueError as exc:
            write(str(exc))
    return "general"


def ranking_source(catalogue: dict, use: str) -> dict | None:
    """The leaderboard that ranks this use: the general one, or for a task the
    independent one that scores the most listed models."""
    sources = catalogue.get("sources", [])
    if use == "general":
        return next((s for s in sources if s["id"] == catalogue.get("primary")), None)

    def reach(source: dict) -> tuple:
        scored = sum(1 for m in catalogue["models"] if score_of(m, source["id"]) is not None)
        return (scored > 0, not source.get("self_reported"), scored)

    candidates = [s for s in sources if s.get("task") == use and reach(s)[0]]
    return max(candidates, key=reach) if candidates else None


def rank_key(model: dict, use: str, ranker: dict | None, primary: str | None) -> tuple:
    """Higher is stronger. For a task: models with that task's score first, by it;
    the rest after them, those made for the task first, then by general strength."""
    general = strength(model, primary)
    if use == "general":
        return (general,)
    task = score_of(model, ranker["id"]) if ranker else None
    if task is None:
        return (0, use in (model.get("good_at") or []), general)
    return (1, task if ranker.get("higher_is_better", True) else -task, general)


def _best(groups: dict) -> dict | None:
    comfortable = [r for r in groups["fast"] if r["words_per_second"] >= COMFORT_WPS]
    full = [r for r in comfortable if not squeezed(r)]
    return (full or comfortable or groups["fast"] or groups["slow"] or [None])[0]


def pick(machines: list[Machine], catalogue: dict, use: str = "general", past: dict | None = None) -> dict:
    """Every model judged, the strongest at `use` first in each group. Reading
    pictures needs a model that can; any model can be asked to do the rest.
    `past` is what earlier runs showed (see `history`)."""
    past = past or {"runs": {}, "caps": {}}
    ready = [m for m in machines if m.get("reachable")]
    for m in ready:
        cap = past["caps"].get(m["name"])
        if cap and cap["gb"] < m.full_room_gb:
            m["cap_gb"], m["cap_why"] = cap["gb"], cap["why"]
    primary = catalogue.get("primary")
    ranker = ranking_source(catalogue, use)
    rows = []
    for model in catalogue["models"]:
        if use == "vision" and "vision" not in (model.get("good_at") or []):
            continue
        # The largest quantization that runs fast; else the copy you already have,
        # then the largest that runs from disk; if none runs, the smallest, so the
        # reason names the least it would take.
        pictures = (model.get("pictures") or {}).get("bytes", 0) / GB if use == "vision" else 0.0
        verdicts = [judge(model, quant, ready, pictures) for quant in model["quants"]]
        order = {"fast": 0, "slow": 1, "no": 2}
        verdicts.sort(key=lambda v: (order[v["group"]], v["group"] == "slow" and not v["quant"].get("owned"),
                                     v["quant"]["bytes"] if v["group"] == "no" else -v["quant"]["bytes"]))
        rows.append(verdicts[0] | {"runs": past["runs"].get((model["id"], verdicts[0]["quant"]["quant"]), [])})

    def grouped(for_use: str, among: list[dict]) -> dict:
        ranked = sorted(among, key=lambda r: rank_key(r["model"], for_use, ranker, primary), reverse=True)
        return {g: [r for r in ranked if r["group"] == g] for g in ("fast", "slow", "no")}

    groups = grouped(use, rows)
    best = _best(groups)
    # When the best pick doesn't fit at this moment, the best that does.
    best_now = None
    if best and best.get("fits_now") is False:
        now = [r | {"words_per_second": r["words_per_second_now"], "split": r["split_now"]}
               for r in groups["fast"] if r.get("fits_now")]
        best_now = _best({"fast": now, "slow": []})
    # For a task, also the strongest overall, when it isn't the same model. Only
    # a model with a general score can be called that.
    scored = [r for r in rows if score_of(r["model"], primary) is not None]
    overall = _best(grouped("general", scored)) if use != "general" else None
    covered = sum(1 for r in rows if ranker and score_of(r["model"], ranker["id"]) is not None)
    return {"machines": machines, "groups": groups, "best": best, "primary": primary, "use": use,
            "ranker": ranker, "overall": None if overall is best else overall, "best_now": best_now,
            "covered": covered, "listed": len(rows),
            "sources": {s["id"]: s for s in catalogue.get("sources", [])}, "as_of": catalogue.get("as_of")}


# What earlier runs showed.

def _model_file(command) -> str | None:
    parts = command if isinstance(command, list) else str(command or "").split()
    for flag, value in zip(parts, parts[1:]):
        if flag in ("-m", "--model"):
            return str(value).rsplit("/", 1)[-1]
    return None


def held_gb(record: dict, model: dict, quant: dict, name_of: Callable[[str | None], str],
            here: str, command=None) -> dict[str, float] | None:
    """About how much of the model each machine held in a run, from its layers:
    the engine's graphics card and each helper take their part of the layers
    sent to devices, and the engine keeps the rest. None when the run moved
    parts by hand or left the split to the engine."""
    settings, who = record.get("settings") or {}, record.get("machines") or {}
    blocks = model.get("layers")
    devices = [d.strip() for d in str(settings.get("devices") or "").split(",") if d.strip()]
    try:
        parts = [float(x) for x in str(settings.get("split") or "").split(",") if x.strip()]
        sent = min(int(settings.get("layers") or 0), blocks or 0)
    except ValueError:
        return None
    args = {str(a).split("=", 1)[0] for a in settings.get("args") or []}
    if (not blocks or settings.get("overrides") or args & (PLACEMENT_FLAGS | set().union(*POSITION_FLAGS))
            or _moved_by_hand(command) or not devices or sent < 0 or len(parts) != len(devices) or sum(parts) <= 0):
        return None
    helpers = ([here] if who.get("helper") else []) + [name_of(h) for h in who.get("far_helpers") or []]
    engine = name_of(who.get("engine"))
    per_layer = quant["bytes"] / GB / blocks
    out: dict[str, float] = {engine: (blocks - sent) * per_layer}
    for device, part in zip(devices, parts):
        index = device[3:]
        if device.upper().startswith("RPC") and not (index.isdigit() and int(index) < len(helpers)):
            return None
        holder = helpers[int(index)] if device.upper().startswith("RPC") else engine
        out[holder] = out.get(holder, 0.0) + sent * part / sum(parts) * per_layer
    return {name: round(gb, 1) for name, gb in out.items() if gb > 0}


def short_of_memory(reason: str | None, record: dict, name_of: Callable[[str | None], str],
                    here: str) -> str | None:
    """Which machine a run stopped for, when it stopped for want of memory."""
    reason = reason or ""
    if reason == "this machine is short of memory":
        return here
    if reason.startswith("under ") and reason.endswith(" free on the engine machine"):
        return name_of((record.get("machines") or {}).get("engine"))
    if match := re.fullmatch(r"(\S+) is short of memory", reason):
        return name_of(match[1])
    return None


def _day(seconds) -> str:
    try:
        return time.strftime("%d %b", time.localtime(float(seconds))).lstrip("0")
    except (TypeError, ValueError, OverflowError):
        return "?"


def history(machines: list[Machine], catalogue: dict, folder: Path | None = None) -> dict:
    """What Orchard's sealed records of earlier runs show: for each listed file,
    its runs; for each machine, the most a model may count on there. Only
    records whose seal checks out count."""
    folder = ai.AI_DIR if folder is None else folder
    here = next((m["name"] for m in machines if m.get("target") is None), "this machine")

    def name_of(host: str | None) -> str:
        return next((m["name"] for m in machines if host and host in (m.get("target"), m["name"])), host or "?")

    files = {quant["files"][0].rsplit("/", 1)[-1]: (model, quant)
             for model in catalogue["models"] for quant in model["quants"] if quant.get("files")}
    records = []
    for path in sorted(folder.glob("ai*.json")) if folder.is_dir() else []:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and receipts.verify(record):
            records.append(record)
    sessions = {r.get("id"): r for r in records if r.get("kind") == "ai-session"}
    runs: dict[tuple, list] = {}
    short: dict[str, tuple] = {}
    managed: dict[str, float] = {}
    for record in records:
        if record.get("state") == "refused":
            continue
        session = sessions.get(record.get("session")) or {}
        found = files.get(_model_file(record.get("command") or session.get("command")) or "")
        if not found:
            continue
        model, quant = found
        held = held_gb(record if record.get("settings") else session, model, quant, name_of, here,
                       record.get("command") or session.get("command"))
        speed = (record.get("timings") or {}).get("ms_per_word")
        who = record.get("machines") or {}
        used = [name_of(who.get("engine"))] + ([here] if who.get("helper") else []) \
            + [name_of(h) for h in who.get("far_helpers") or []]
        if record.get("kind") == "ai":
            runs.setdefault((model["id"], quant["quant"]), []).append({
                "id": record.get("id"), "day": _day(record.get("started_at")), "state": record.get("state"),
                "reason": record.get("reason"), "words_per_second": round(1000 / speed, 1) if speed else None,
                "machines": used, "held": held})
        ran_short = short_of_memory(record.get("reason"), record, name_of, here)
        if ran_short and held and ran_short in held:
            if ran_short not in short or held[ran_short] < short[ran_short][0]:
                short[ran_short] = (held[ran_short], _day(record.get("started_at")), record.get("id"))
        elif record.get("state") == "answered" and held:
            for name, gb in held.items():
                managed[name] = max(managed.get(name, 0.0), gb)
    caps = {}
    for name, (gb, day, rid) in short.items():
        # An answer holding more than it ran short with says the shortage was the moment's, not the machine's.
        lifted = managed.get(name, 0.0)
        cap = round(lifted if lifted > gb else gb * SHORT_SHARE, 1)
        caps[name] = {"gb": cap, "why": f"it ran short of memory holding about {gb} GB on {day} (record {rid})"}
    for found_runs in runs.values():
        found_runs.sort(key=lambda r: r["id"] or "", reverse=True)
    return {"runs": runs, "caps": caps}


# Saying it.

def _word(key: str) -> str:
    return WORD.get(key, key)


def _score_value(value: float, source: dict) -> str:
    """A share of tasks solved reads as a percentage; anything else as it was published."""
    return f"{value * 100:.1f}%" if source.get("scale") == "fraction" else f"{value:g}"


def _scores_text(model: dict, sources: dict, use: str, ranker: dict | None) -> str:
    """Every score the model has, the one it's ranked by first, each with its task."""
    first = ranker["id"] if ranker else None
    scores = sorted(model.get("scores") or [], key=lambda s: s.get("source") != first)
    parts = []
    for score in scores:
        source = sources.get(score.get("source"), {})
        name = source.get("short") or score.get("benchmark") or score.get("source")
        parts.append(f"{_word(source.get('task') or 'general')} {_score_value(score['score'], source)} ({name})")
    if use != "general" and ranker and score_of(model, ranker["id"]) is None:
        parts.insert(0, f"no {_word(use)} score yet")
    return ", ".join(parts) or "none yet"


def _billions(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f}T"
    return f"{value:.0f}B" if value >= 10 else f"{value:.1f}B"


def _abilities_text(model: dict) -> str:
    """What it can do, how much it takes in at once, and how big it is."""
    parts = []
    can = [_word(tag) for tag in model.get("good_at") or []]
    parts.append("made for " + (", ".join(can) if can else "general use"))
    tokens = model.get("context")
    if tokens:
        size = f"{tokens / (1 << 20):g}M" if tokens >= 1 << 20 else f"{tokens / 1024:.0f}K"
        words = tokens * WORDS_PER_TOKEN
        about = f"{words / 1e6:.1f}M" if words >= 1e6 else f"{words / 1e3:.0f}K"
        parts.append(f"takes in {size} tokens at once (about {about} words)")
    if model.get("params_b"):
        size = f"{_billions(model['params_b'])} parameters"
        if model.get("active_b"):
            size += f", {_billions(model['active_b'])} at work per word"
        parts.append(size)
    return "; ".join(parts)


def _pictures_text(model: dict) -> str:
    pictures = model.get("pictures")
    if not pictures:
        return "to read pictures it also needs its mmproj file from the same page"
    return (f"to read pictures it also needs {pictures['file']} ({pictures['bytes'] / GB:.1f} GB) "
            "from the same page, counted in the memory above")


def _machine_line(m: Machine) -> str:
    if not m.get("reachable"):
        return f"  {m['name']:<10} not reachable: {m.get('why', '')}"
    gpus = "; ".join(g["name"] + (f" {g['memory_mb'] / 1024:.0f} GB" if g["memory_mb"] else "") for g in m["gpus"])
    kind = "Linux in Windows" if m.get("wsl") else {"Darwin": "macOS", "Linux": "Linux"}.get(m["system"], m["system"])
    disk = f"{m['disk_free_gb']} GB free disk" + (f" ({m['disk_note']})" if m.get("disk_note") else "")
    now = "" if m.room_now_gb is None else f" ({m.room_now_gb} GB of it free right now)"
    line = (f"  {m['name']:<10} {kind} {m['arch']}, {m['ram_mb'] / 1024:.0f} GB memory"
            f"{', ' + gpus if gpus else ''}: room for {m.room_gb} GB of model{now}, {disk}, "
            f"reads memory at {m.speed_gbs} GB/s")
    if m.get("cap_why"):
        line += f"\n             counted at {m['cap_gb']} GB, not {m.full_room_gb}: {m['cap_why']}"
    return line


def _now_text(row: dict) -> str | None:
    if "fits_now" not in row:
        return None
    if row["fits_now"]:
        moved = [(m["name"], gb) for m, gb in row["split_now"]] != [(m["name"], gb) for m, gb in row["split"]]
        where = ", ".join(f"{m['name']} {gb} GB" for m, gb in row["split_now"])
        return "    right now: fits" + (f", held as {where}; about {row['words_per_second_now']} words a second there"
                                       if moved else "")
    if not row["short_now"]:
        return "    right now: doesn't fit, with what your machines have free"
    short = "; ".join(f"{m['name']} has {now} GB free for its {share} GB share" for m, share, now in row["short_now"])
    names = [m["name"] for m, _, _ in row["short_now"]]
    names = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
    return f"    right now: doesn't fit ({short}); it fits when {names} {'is' if len(row['short_now']) == 1 else 'are'} quieter"


def _runs_text(row: dict) -> list[str]:
    """What earlier sealed runs of this file measured, next to the estimate."""
    runs = row.get("runs") or []
    lines = []
    answered = [r for r in runs if r["state"] == "answered" and r["words_per_second"]]
    if answered:
        fast = max(answered, key=lambda r: r["words_per_second"])
        lines.append(f"    ran here: {fast['words_per_second']} words a second on {' and '.join(fast['machines'])}, "
                     f"{fast['day']} (record {fast['id']}, measured)")
    other = [r for r in runs if r["state"] != "answered"]
    if other and (not answered or other[0]["id"] > answered[0]["id"]):
        last = other[0]
        lines.append(f"    tried here: {last['state']}" + (f", {last['reason']}" if last.get("reason") else "")
                     + f", {last['day']} (record {last['id']})")
    return lines


def _row_lines(row: dict, result: dict) -> list[str]:
    model, quant = row["model"], row["quant"]
    size = round(quant["bytes"] / GB, 1)
    owned = ", already on your machines" if quant.get("owned") else ""
    bits = bits_per_weight(model, quant)
    squeeze = f", squeezed to {bits} bits a weight: weaker than its scores" if squeezed(row) else ""
    lines = [f"  {model['name']} ({quant['quant']}, {size} GB file{owned}{squeeze})",
             f"    scores: {_scores_text(model, result['sources'], result['use'], result['ranker'])}"]
    if row["group"] == "fast":
        split = ", ".join(f"{m['name']} {share} GB" for m, share in row["split"])
        lines.append(f"    held in memory: {split}; about {row['words_per_second']} words a second (estimate)")
        if now := _now_text(row):
            lines.append(now)
    elif row["group"] == "slow":
        lines.append(f"    {row['why']}: the engine sits on {row['home']['name']}; expect seconds per word")
    else:
        lines.append(f"    {row['why']}")
    lines += _runs_text(row)
    lines.append(f"    {_abilities_text(model)}")
    if result["use"] == "vision":
        lines.append(f"    {_pictures_text(model)}")
    lines.append(f"    licence {model.get('licence') or 'unknown'}"
                 + (", needs a sign-in to download" if model.get("gated") else "")
                 + f"; from huggingface.co/{model['repo']}")
    return lines


def _why_best(best: dict, result: dict) -> str:
    use, ranker = result["use"], result["ranker"]
    comfortable = best["group"] == "fast" and best["words_per_second"] >= COMFORT_WPS
    speed = (f" of those here that should answer at {COMFORT_WPS:.0f} or more words a second"
             + ("" if squeezed(best) else f", kept at {FULL_BITS} or more bits a weight")) if comfortable else " here"
    general = score_of(best["model"], result["primary"]) is not None
    # The models the pick was chosen from, which then have no score of that kind.
    among = ("here that run well at full strength" if comfortable and not squeezed(best)
             else "here that run well" if comfortable else "that run here")
    if use == "general":
        return f"    the strongest{speed}" if general else f"    none of the models {among} has a general score yet"
    if ranker and score_of(best["model"], ranker["id"]) is not None:
        return f"    the strongest at {USES[use]} by {ranker.get('short') or ranker['name']}{speed}"
    why = (f"none of the models {among} has a {_word(use)} score yet" if ranker
           else f"no {_word(use)} leaderboard on the list yet")
    made = f" made for {USES[use]}" if use in (best["model"].get("good_at") or []) else ""
    how = "by general score" if general else "with no general score either"
    return f"    {why}, so this is the strongest of them{made}, {how}"


def report(result: dict, online: dict | None, new: list[dict]) -> list[str]:
    sources, use, ranker = result["sources"], result["use"], result["ranker"]
    lines = ["Your machines"] + [_machine_line(m) for m in result["machines"]]
    ready = [m for m in result["machines"] if m.get("reachable")]
    together = f"  Together: room for {round(sum(max(m.room_gb, 0) for m in ready), 1)} GB of model in memory"
    if ready and all(m.room_now_gb is not None for m in ready):
        together += f", {round(sum(m.room_now_gb for m in ready), 1)} GB of it free right now"
    lines.append(together + ".")
    lines.append("")
    best = result["best"]
    if best:
        lines.append("Best pick" + ("" if use == "general" else f" for {USES[use]}"))
        lines += _row_lines(best, result)
        why = _why_best(best, result)
        if why:
            lines.append(why)
        best_now = result.get("best_now")
        if best_now:
            lines += ["", "Best that fits right now"] + _row_lines(best_now, result)
        elif best.get("fits_now") is False:
            lines += ["", "Nothing on the list fits in memory right now; try again when your machines are quieter."]
        overall = result.get("overall")
        if overall and not (best_now and overall["model"] is best_now["model"]):
            lines += ["", "Strongest overall here, by general score"] + _row_lines(overall, result)
        elif overall:
            lines.append("    (also the strongest overall here, by general score)")
    else:
        lines.append("Nothing on the list runs on these machines"
                     + ("" if use == "general" else f" for {USES[use]}") + ".")
    shown = {id(best), id(result.get("overall"))} | {
        id(r) for r in result["groups"]["fast"] if result.get("best_now") and r["model"] is result["best_now"]["model"]}
    for group, title, limit in (("fast", "Also fits in memory here", 4),
                                ("slow", "Strongest that run slowly from disk (background work)", 2),
                                ("no", "Won't run here", 4)):
        rows = [r for r in result["groups"][group] if id(r) not in shown][:limit]
        if rows:
            lines += ["", title]
            for row in rows:
                lines += _row_lines(row, result) if group != "no" else [
                    f"  {row['model']['name']} ({round(row['quant']['bytes'] / GB, 1)} GB): {row['why']}"]
    lines.append("")
    if new:
        lines.append("Popular on Hugging Face but not on Orchard's list (not ranked, not checked):")
        lines += [f"  huggingface.co/{m['repo']} ({m['downloads']:,} downloads in the last month"
                  + (f", updated {m['modified']})" if m["modified"] else ")") for m in new]
        lines.append("")
    if online is None:
        lines.append(f"Sizes from Orchard's own list ({result['as_of']}); not checked online.")
    else:
        lines.append(f"Sizes checked on Hugging Face just now: {online['checked']} files"
                     + (f", {online['missing']} missing" if online["missing"] else "")
                     + (f"; {online['repos'] - online['reached']} repos unreachable" if online["reached"] < online["repos"] else "")
                     + ".")
    if ranker:
        rest = ("." if result["covered"] == result["listed"] else
                "; the rest come after them" + ("" if use == "general" else ", those made for it first") + ".")
        lines.append(f"Ranked for {USES[use]} by {ranker['name']}: {result['covered']} of the "
                     f"{result['listed']} models have a score{rest}")
    elif use != "general":
        lines.append(f"No {_word(use)} leaderboard on the list yet: ranked by general strength, "
                     "models made for it first.")
    lines.append("Where the scores come from (higher is better unless it says otherwise):")
    for source in sources.values():
        notes = []
        if source.get("higher_is_better") is False:
            notes.append("lower is better")
        if source.get("self_reported"):
            notes.append("the makers' own numbers")
        lines.append(f"  {_word(source.get('task') or 'general')}: {source['name']}, {source.get('url', '')}, "
                     f"read {source.get('read_on', '?')}" + (f" ({'; '.join(notes)})" if notes else ""))
    lines.append("Speeds are rough estimates unless marked measured, from Orchard's sealed records of earlier runs "
                 "on these machines. Room keeps a quarter of each machine's memory for the machine itself, at least "
                 "2 GB (4 GB on a Mac), and less where a run there ran short of memory. Right now leaves 1 GB of "
                 "what's free untouched.")
    return lines


def as_json(result: dict, online: dict | None, new: list[dict]) -> dict:
    ranker = result["ranker"]

    def row(r: dict | None) -> dict | None:
        if r is None:
            return None
        model = r["model"]
        out = {"id": model["id"], "name": model["name"], "repo": model["repo"],
               "owned": bool(r["quant"].get("owned")),
               "quant": r["quant"]["quant"], "files": r["quant"]["files"], "bytes": r["quant"]["bytes"],
               "group": r["group"], "need_gb": r["need_gb"], "strength": score_of(model, result["primary"]),
               "use_score": score_of(model, ranker["id"]) if ranker else None,
               "scores": [dict(s, task=result["sources"].get(s.get("source"), {}).get("task", "general"))
                          for s in model.get("scores") or []],
               "good_at": model.get("good_at") or [], "context_tokens": model.get("context"),
               "params_b": model.get("params_b"), "active_b": model.get("active_b"),
               "licence": model.get("licence"), "gated": model.get("gated", False)}
        if model.get("pictures"):
            out["pictures"] = model["pictures"]
        out["runs"] = r.get("runs") or []
        if r["group"] == "fast":
            out["split"] = [{"machine": m["name"], "gb": share} for m, share in r["split"]]
            out["words_per_second_estimate"] = r["words_per_second"]
            out["fits_now"] = r.get("fits_now")
            out["words_per_second_now_estimate"] = r.get("words_per_second_now")
            out["split_now"] = [{"machine": m["name"], "gb": share} for m, share in r.get("split_now") or []]
            out["short_now"] = [{"machine": m["name"], "share_gb": share, "free_gb": now}
                                for m, share, now in r.get("short_now") or []]
        elif r["group"] == "slow":
            out["home"] = r["home"]["name"]
        else:
            out["why"] = r["why"]
        return out

    def machine(m: Machine) -> dict:
        if not m.get("reachable"):
            return dict(m, room_gb=None)
        return dict(m, room_gb=m.room_gb, full_room_gb=m.full_room_gb, room_now_gb=m.room_now_gb)

    return {"machines": [machine(m) for m in result["machines"]],
            "use": result["use"], "ranked_by": ranker["id"] if ranker else None,
            "scored": result["covered"], "listed": result["listed"],
            "best": row(result["best"]), "overall": row(result.get("overall")),
            "best_now": row(result.get("best_now")),
            "groups": {g: [row(r) for r in rows] for g, rows in result["groups"].items()},
            "sources": list(result["sources"].values()),
            "online": online, "new": new, "primary": result["primary"], "as_of": result["as_of"]}


def mark_owned(models: list[dict]) -> str | None:
    """Mark the quantization `orchard ai` already uses, if it's on the list."""
    try:
        engine = ai.load_settings()["engine"]
    except ai.ConfigError:
        return None
    name = engine["model"].rsplit("/", 1)[-1]
    for model in models:
        for quant in model["quants"]:
            if any(path.rsplit("/", 1)[-1] == name for path in quant["files"]):
                quant["owned"], quant["owned_on"] = True, engine["host"]
                return model["id"]
    return None


def run(offline: bool = False, use: str = "general", only: list[str] | None = None,
        say: Callable[[str], None] = lambda _: None) -> tuple[dict, dict | None, list[dict]]:
    catalogue = builtin_list()
    mark_owned(catalogue["models"])
    online, new = None, []
    say("reading your machines")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        machines_job = pool.submit(read_machines, only)
        if not offline:
            say("checking sizes on Hugging Face")
            online = refresh(catalogue["models"])
            new = new_models({m["repo"] for m in catalogue["models"]})
        machines = machines_job.result()
    return pick(machines, catalogue, use, history(machines, catalogue)), online, new
