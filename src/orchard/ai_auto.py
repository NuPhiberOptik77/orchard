"""Name a model, and Orchard lays it out across your machines.

`orchard ai ask --model gemma-4-26b-a4b "..."`, `orchard ai serve --model ...`
and `orchard ai plan gemma-4-26b-a4b` take a model from the picker's list,
find its file on the engine machine, read each machine in the pool (the
engine machine, this machine's helper, each far helper) and work out how many
of the model's layers each one holds:

- the engine machine's graphics card first, as many as fit;
- then helper machines whose memory is at least twice as fast as the engine
  machine's own, fastest first (every word crosses the network to them);
- then the engine machine's own memory;
- then the other helpers, fastest first.

A helper that gets no layers isn't started. Each machine's room is the
picker's: its memory less what it keeps for itself, and less where a run there
ran short of memory. The layout replaces the settings' model, devices, split
and layers for that run; everything else stays as the settings say.
"""

from __future__ import annotations

import concurrent.futures
import copy
import math
import posixpath
import shlex
import sys
from typing import Callable

from . import ai, ai_pick

# A helper goes before the engine machine's own memory only when its memory is this much faster.
HELPER_EDGE = 2.0


def find_model(name: str, catalogue: dict) -> dict:
    """A listed model by its id or name, or the start of one that only it has."""
    key = name.strip().lower()
    models = catalogue["models"]
    exact = [m for m in models if key in (m["id"].lower(), m["name"].lower())]
    if exact:
        return exact[0]
    near = [m for m in models if key and (m["id"].lower().startswith(key) or m["name"].lower().startswith(key))]
    if len(near) == 1:
        return near[0]
    if near:
        raise ai.ConfigError(f"'{name}' could be {', '.join(m['id'] for m in near)}: say which")
    raise ai.ConfigError(f"'{name}' isn't on Orchard's list: `orchard ai pick` shows the models it knows")


def search_folders(cfg: dict) -> list[str]:
    e = cfg["engine"]
    folders = [e.get("models_folder") or "", posixpath.dirname(e["model"])]
    return [f for i, f in enumerate(folders) if f and f not in folders[:i]]


def find_files(cfg: dict, model: dict, remote: Callable | None = None) -> dict[str, str]:
    """Which of the model's files are on the engine machine: quant name to path."""
    remote = remote or ai._remote
    quants = [q for q in model["quants"] if q.get("files")]
    folders = " ".join(ai._rpath(f) for f in search_folders(cfg))
    script = "".join(
        f'echo "{i}=$(find {folders} -maxdepth 3 -name {shlex.quote(posixpath.basename(q["files"][0]))} '
        f'2> /dev/null | head -1)"\n' for i, q in enumerate(quants))
    out = remote(cfg["engine"]["host"], "sh -s", 60, script).stdout
    found = {}
    for line in out.splitlines():
        index, _, path = line.partition("=")
        if index.isdigit() and int(index) < len(quants) and path.strip():
            found[quants[int(index)]["quant"]] = path.strip()
    return found


def read_pool(cfg: dict, read: Callable | None = None) -> list[ai_pick.Machine]:
    """The engine machine, this machine's helper and each far helper, read side
    by side, each marked with its part."""
    read = read or ai_pick.read_machine
    here = "this Mac" if sys.platform == "darwin" else "this machine"
    wanted = [("engine", cfg["engine"]["host"], cfg["engine"]["host"])]
    if ai.has_helper(cfg):
        wanted.append(("helper", here, None))
    wanted += [("far", h["host"], h["host"]) for h in cfg.get("far_helpers", [])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        machines = list(pool.map(lambda w: read(w[1], w[2]), wanted))
    for (part, _, _), m in zip(wanted, machines):
        if not m.get("reachable"):
            raise ai.ConfigError(f"cannot read {m['name']}: {m.get('why') or 'no answer'}")
        m["part"] = part
    return machines


def _devices(machines: list[ai_pick.Machine]) -> list[dict]:
    """Where layers can go, each with its room in GB, in the order they fill."""
    engine = next(m for m in machines if m["part"] == "engine")
    helpers = [m for m in machines if m["part"] != "engine"]
    out = []
    kinds = {g["kind"] for g in engine["gpus"]}
    if "apple" in kinds:
        # One memory for both: the engine's graphics chip takes it all.
        out.append({"machine": engine, "device": "MTL0", "room": engine.room_gb - ai_pick.NEED_EXTRA_GB})
        own = None
    else:
        if engine.vram_gb > 0:
            out.append({"machine": engine, "device": "CUDA0", "room": engine.vram_gb})
        own = {"machine": engine, "device": "CPU", "room": engine.room_gb - engine.vram_gb - ai_pick.NEED_EXTRA_GB}
    fast = sorted([m for m in helpers if own is None or m.speed_gbs >= HELPER_EDGE * engine.speed_gbs],
                  key=lambda m: m.speed_gbs, reverse=True)
    slow = sorted([m for m in helpers if m not in fast], key=lambda m: m.speed_gbs, reverse=True)
    out += [{"machine": m, "device": "RPC", "room": m.room_gb} for m in fast]
    if own:
        out.append(own)
    out += [{"machine": m, "device": "RPC", "room": m.room_gb} for m in slow]
    return out


def lay_out(model: dict, quant: dict, machines: list[ai_pick.Machine]) -> list[dict]:
    """How many layers each device holds. ConfigError if they can't hold them all."""
    layers = model.get("layers")
    if not layers:
        raise ai.ConfigError(f"Orchard doesn't know how many layers {model['name']} has, so it can't lay it out")
    per_layer = quant["bytes"] / ai_pick.GB / layers * ai_pick.NEED_SHARE
    left, out = layers, []
    for dev in _devices(machines):
        take = min(left, max(0, math.floor(dev["room"] / per_layer)))
        out.append({**dev, "layers": take, "gb": round(take * per_layer, 1)})
        left -= take
    if left:
        held = sum(max(0, math.floor(d["room"] / per_layer)) for d in out)
        raise ai.ConfigError(
            f"{model['name']} ({quant['quant']}) has {layers} layers of about {per_layer:.2f} GB each, and these "
            f"machines have room for {held}: `orchard ai pick` shows what fits")
    return out


def apply(cfg: dict, name: str, say: Callable[[str], None] = lambda _: None, *,
          catalogue: dict | None = None, read: Callable | None = None, remote: Callable | None = None,
          announce: bool = True) -> tuple[dict, dict]:
    """Settings for one run of the named model, laid out across the pool, and
    the layout to record."""
    e = cfg["engine"]
    if e.get("overrides"):
        raise ai.ConfigError("--model works out where each layer goes: leave out overrides")
    catalogue = catalogue or ai_pick.builtin_list()
    model = find_model(name, catalogue)
    say(f"finding {model['name']} on {e['host']}")
    found = find_files(cfg, model, remote)
    if not found:
        smallest = min((q for q in model["quants"] if q.get("files")), key=lambda q: q["bytes"])
        raise ai.ConfigError(
            f"{model['name']} isn't on {e['host']} under {' or '.join(search_folders(cfg))}: download "
            f"{posixpath.basename(smallest['files'][0])} or a larger file ({smallest['bytes'] / ai_pick.GB:.1f} GB "
            f"and up) from huggingface.co/{model['repo']}")
    say("reading the machines")
    machines = read_pool(cfg, read)
    past = ai_pick.history(machines, catalogue)
    for m in machines:
        cap = past["caps"].get(m["name"])
        if cap and cap["gb"] < m.full_room_gb:
            m["cap_gb"], m["cap_why"] = cap["gb"], cap["why"]
    # The largest file you have that fits.
    tried = []
    for quant in sorted((q for q in model["quants"] if q["quant"] in found), key=lambda q: -q["bytes"]):
        try:
            layout = lay_out(model, quant, machines)
            break
        except ai.ConfigError as exc:
            tried.append(str(exc))
    else:
        raise ai.ConfigError(tried[0])
    used = [d for d in layout if d["layers"]]
    out = copy.deepcopy(cfg)
    helper_used = any(d["machine"]["part"] == "helper" for d in used)
    if not helper_used:
        out["helper"]["folder"] = ""
    far_used = {d["machine"]["name"] for d in used if d["machine"]["part"] == "far"}
    out["far_helpers"] = [h for h in cfg.get("far_helpers", []) if h["host"] in far_used]
    # RPC numbers follow the engine's --rpc list: this machine's helper, then each far helper kept.
    rpc = (["helper"] if helper_used else []) + [h["host"] for h in out["far_helpers"]]
    offloaded = []
    for d in used:
        if d["device"] == "CPU":
            continue
        if d["device"] == "RPC":
            key = "helper" if d["machine"]["part"] == "helper" else d["machine"]["name"]
            d["device"] = f"RPC{rpc.index(key)}"
        offloaded.append(d)
    offloaded.sort(key=lambda d: (not d["device"].startswith(("CUDA", "MTL")), d["device"]))
    oe = out["engine"]
    oe["model"] = found[quant["quant"]]
    oe["devices"] = ",".join(d["device"] for d in offloaded)
    oe["split"] = ",".join(str(d["layers"]) for d in offloaded)
    oe["layers"] = sum(d["layers"] for d in offloaded)
    record = {"model": model["id"], "name": model["name"], "quant": quant["quant"], "file": oe["model"],
              "layers": model["layers"],
              "parts": [{"machine": d["machine"]["name"], "device": d["device"], "layers": d["layers"], "gb": d["gb"]}
                        for d in used],
              "not_started": [m["name"] for m in machines if m["part"] != "engine"
                              and m["name"] not in {d["machine"]["name"] for d in used}]}
    if announce:
        say("laid out: " + describe(record))
    return out, record


def describe(record: dict) -> str:
    """Each machine once, in the order it fills: its layers, memory, and how many sit on its graphics card."""
    order: list[str] = []
    for p in record["parts"]:
        if p["machine"] not in order:
            order.append(p["machine"])
    parts = []
    for name in order:
        mine = [p for p in record["parts"] if p["machine"] == name]
        gpu = sum(p["layers"] for p in mine if p["device"].startswith(("CUDA", "MTL")))
        layers, gb = sum(p["layers"] for p in mine), round(sum(p["gb"] for p in mine), 1)
        on_card = "" if not gpu else ", all on its graphics chip" if gpu == layers else f", {gpu} on its graphics card"
        parts.append(f"{name} {layers} layers ({gb} GB{on_card})")
    rest = f"; not needed: {', '.join(record['not_started'])}" if record["not_started"] else ""
    return f"{record['name']} ({record['quant']}, {record['layers']} layers): {'; '.join(parts)}{rest}"
