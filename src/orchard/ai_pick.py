"""Which of the strongest open models your machines can run, and how.

`orchard ai pick` reads each machine Orchard knows (this one and every worker):
its memory, graphics memory, free disk and how fast it reads memory. It then
goes through a dated list of open models and sorts them into three groups:

- fast: the model fits in memory across your machines, with room to spare
- slow: too big for memory, but built so it can run reading from disk
  (a mixture of experts), for background work
- won't run here, and why

Within each group the strongest come first, by a public leaderboard named with
its date. The split preview says which machine would hold how much. Speeds are
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
import urllib.error
import urllib.parse
import urllib.request
from importlib import resources
from typing import Callable

from . import ai, config

GB = 1e9
HF = "https://huggingface.co"
HF_TIMEOUT = 15
NEW_FROM = ("unsloth", "bartowski", "ggml-org")
NEW_LIMIT = 6
PROBE_TIMEOUT = 90
USER_AGENT = "orchard-ai-pick"

# What a machine keeps for itself: a quarter of its memory, and at least 2 GB.
RESERVE_SHARE = 0.25
RESERVE_MIN_MB = 2048
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


class Machine(dict):
    """One machine's readings, and what it can give a model."""

    @property
    def room_gb(self) -> float:
        """Memory a model can have here: its RAM less what the machine keeps for
        itself, plus most of a graphics card's own memory."""
        ram = self["ram_mb"]
        reserve = max(RESERVE_MIN_MB, ram * RESERVE_SHARE)
        vram = sum(g["memory_mb"] or 0 for g in self["gpus"] if g["kind"] == "nvidia") * VRAM_SHARE
        return round((ram - reserve + vram) / 1024, 1)

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


def refresh(models: list[dict], get_files: Callable[[str], dict[str, int]] = hf_files) -> dict:
    """Check each listed file on Hugging Face and take its current size. A model
    whose files are gone is marked, not dropped. Returns what was checked."""
    repos = sorted({m["repo"] for m in models})
    found: dict[str, dict[str, int] | None] = {}

    def fetch(repo: str) -> None:
        try:
            found[repo] = get_files(repo)
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            found[repo] = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, repos))
    checked = missing = 0
    for model in models:
        files = found.get(model["repo"])
        if files is None:
            continue
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


def plan_memory(need: float, machines: list[Machine]) -> list[tuple[Machine, float]] | None:
    """The fewest machines that hold `need` GB between them, most room first,
    each given a share in proportion to its room. None if they can't."""
    ranked = sorted(machines, key=lambda m: (m.room_gb, m.speed_gbs), reverse=True)
    chosen: list[Machine] = []
    for machine in ranked:
        if machine.room_gb <= 0:
            continue
        chosen.append(machine)
        room = sum(m.room_gb for m in chosen)
        if room >= need:
            return [(m, round(need * m.room_gb / room, 1)) for m in chosen]
    return None


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


def judge(model: dict, quant: dict, machines: list[Machine]) -> dict:
    """Where one model fits: fast (memory), slow (disk) or not here, and why."""
    need = round(need_gb(quant), 1)
    out = {"model": model, "quant": quant, "need_gb": need}
    if quant.get("missing"):
        return out | {"group": "no", "why": "its files are no longer on Hugging Face under that name"}
    if model.get("llama_cpp") is False:
        return out | {"group": "no", "why": "the engine can't run this kind of model yet"}
    if model.get("llama_cpp") is None:
        return out | {"group": "no", "why": "the engine's support for this kind of model isn't confirmed yet"}
    split = plan_memory(need, machines)
    if split:
        return out | {"group": "fast", "split": split, "words_per_second": estimate_words_per_second(model, quant, split)}
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


def strength(model: dict, primary: str) -> float:
    for score in model.get("scores") or []:
        if score.get("source") == primary and isinstance(score.get("score"), (int, float)):
            return float(score["score"])
    return float("-inf")


def pick(machines: list[Machine], catalogue: dict, good_at: str | None = None) -> dict:
    """Every model judged, the strongest first in each group."""
    ready = [m for m in machines if m.get("reachable")]
    primary = catalogue.get("primary")
    rows = []
    for model in catalogue["models"]:
        if good_at and good_at not in (model.get("good_at") or []):
            continue
        # The largest quantization that runs fast; otherwise the first that runs at all.
        verdicts = [judge(model, quant, ready) for quant in model["quants"]]
        order = {"fast": 0, "slow": 1, "no": 2}
        verdicts.sort(key=lambda v: (order[v["group"]], -v["quant"]["bytes"]))
        rows.append(verdicts[0])
    rows.sort(key=lambda r: strength(r["model"], primary), reverse=True)
    groups = {g: [r for r in rows if r["group"] == g] for g in ("fast", "slow", "no")}
    comfortable = [r for r in groups["fast"] if r["words_per_second"] >= COMFORT_WPS]
    full = [r for r in comfortable if not squeezed(r)]
    best = (full or comfortable or groups["fast"] or groups["slow"] or [None])[0]
    return {"machines": machines, "groups": groups, "best": best, "primary": primary,
            "sources": {s["id"]: s for s in catalogue.get("sources", [])}, "as_of": catalogue.get("as_of")}


# Saying it.

def _score_text(model: dict, sources: dict, primary: str | None) -> str:
    for score in model.get("scores") or []:
        if score.get("source") == primary:
            source = sources.get(primary, {})
            return f"{score['score']} on the {source.get('name', primary)} (read {source.get('read_on', '?')})"
    return "not ranked yet"


def _machine_line(m: Machine) -> str:
    if not m.get("reachable"):
        return f"  {m['name']:<10} not reachable: {m.get('why', '')}"
    gpus = "; ".join(g["name"] + (f" {g['memory_mb'] / 1024:.0f} GB" if g["memory_mb"] else "") for g in m["gpus"])
    kind = "Linux in Windows" if m.get("wsl") else {"Darwin": "macOS", "Linux": "Linux"}.get(m["system"], m["system"])
    disk = f"{m['disk_free_gb']} GB free disk" + (f" ({m['disk_note']})" if m.get("disk_note") else "")
    return (f"  {m['name']:<10} {kind} {m['arch']}, {m['ram_mb'] / 1024:.0f} GB memory"
            f"{', ' + gpus if gpus else ''}: room for {m.room_gb} GB of model, {disk}, "
            f"reads memory at {m.speed_gbs} GB/s")


def _row_lines(row: dict, sources: dict, primary: str | None) -> list[str]:
    model, quant = row["model"], row["quant"]
    size = round(quant["bytes"] / GB, 1)
    owned = ", already on your machines" if quant.get("owned") else ""
    bits = bits_per_weight(model, quant)
    squeeze = f", squeezed to {bits} bits a weight: weaker than its score" if squeezed(row) else ""
    lines = [f"  {model['name']} ({quant['quant']}, {size} GB file{owned}{squeeze})",
             f"    strength: {_score_text(model, sources, primary)}"]
    if row["group"] == "fast":
        split = ", ".join(f"{m['name']} {share} GB" for m, share in row["split"])
        lines.append(f"    held in memory: {split}; about {row['words_per_second']} words a second (estimate)")
    elif row["group"] == "slow":
        lines.append(f"    {row['why']}: the engine sits on {row['home']['name']}; expect seconds per word")
    else:
        lines.append(f"    {row['why']}")
    good = ", ".join(model.get("good_at") or [])
    lines.append(f"    good at {good or 'general use'}; licence {model.get('licence') or 'unknown'}"
                 + (", needs a sign-in to download" if model.get("gated") else "")
                 + f"; from huggingface.co/{model['repo']}")
    return lines


def report(result: dict, online: dict | None, new: list[dict]) -> list[str]:
    sources, primary = result["sources"], result["primary"]
    lines = ["Your machines"] + [_machine_line(m) for m in result["machines"]]
    ready = [m for m in result["machines"] if m.get("reachable")]
    lines.append(f"  Together: room for {round(sum(max(m.room_gb, 0) for m in ready), 1)} GB of model in memory.")
    lines.append("")
    best = result["best"]
    if best:
        lines.append("Best pick")
        lines += _row_lines(best, sources, primary)
        if best["group"] == "fast" and best["words_per_second"] >= COMFORT_WPS:
            lines.append(f"    the strongest here that should answer at {COMFORT_WPS:.0f} or more words a second"
                         + ("" if squeezed(best) else f", kept at {FULL_BITS} or more bits a weight"))
    else:
        lines.append("Nothing on the list runs on these machines.")
    for group, title, limit in (("fast", "Also fits in memory here", 4),
                                ("slow", "Strongest that run slowly from disk (background work)", 2),
                                ("no", "Won't run here", 4)):
        rows = [r for r in result["groups"][group] if r is not best][:limit]
        if rows:
            lines += ["", title]
            for row in rows:
                lines += _row_lines(row, sources, primary) if group != "no" else [
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
    source = sources.get(primary, {})
    lines.append(f"Strength from {source.get('name', 'no source')} ({source.get('url', '')}), read {source.get('read_on', '?')}.")
    lines.append("Speeds are rough estimates; a short test run gives the real number. Room keeps a quarter of each "
                 "machine's memory, at least 2 GB, for the machine itself.")
    return lines


def as_json(result: dict, online: dict | None, new: list[dict]) -> dict:
    def row(r: dict | None) -> dict | None:
        if r is None:
            return None
        out = {"id": r["model"]["id"], "name": r["model"]["name"], "repo": r["model"]["repo"],
               "owned": bool(r["quant"].get("owned")),
               "quant": r["quant"]["quant"], "files": r["quant"]["files"], "bytes": r["quant"]["bytes"],
               "group": r["group"], "need_gb": r["need_gb"], "strength": strength(r["model"], result["primary"]),
               "licence": r["model"].get("licence"), "gated": r["model"].get("gated", False)}
        if out["strength"] == float("-inf"):
            out["strength"] = None
        if r["group"] == "fast":
            out["split"] = [{"machine": m["name"], "gb": share} for m, share in r["split"]]
            out["words_per_second_estimate"] = r["words_per_second"]
        elif r["group"] == "slow":
            out["home"] = r["home"]["name"]
        else:
            out["why"] = r["why"]
        return out

    return {"machines": [dict(m, room_gb=m.room_gb if m.get("reachable") else None) for m in result["machines"]],
            "best": row(result["best"]), "groups": {g: [row(r) for r in rows] for g, rows in result["groups"].items()},
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


def run(offline: bool = False, good_at: str | None = None, only: list[str] | None = None,
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
    return pick(machines, catalogue, good_at), online, new
