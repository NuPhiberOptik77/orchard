#!/usr/bin/env python3
"""Claude Code PreToolUse hook: route heavy Bash commands through `orchard offload`.

Reads the hook JSON payload on stdin. For unrecognized or ambiguous commands it
stays silent and exits 0 (allow). For a conservative set of clearly CPU-heavy
commands (full pytest suites, torchrun, train-style python scripts) it denies
with an instruction to re-run via the offload wrapper. Any internal error also
fails open, so this hook can never block a session by accident.

stdlib-only, py3.9-compatible (may run under stock /usr/bin/python3).
"""

import json
import re
import sys
from pathlib import Path

HEAVY_PATTERNS = (
    # full-ish pytest runs: bare `pytest`, `python -m pytest`, after separators
    re.compile(r"(?:^|[;&|(]\s*|\bsudo\s+)(?:python[0-9.]*\s+-m\s+)?pytest(?:\s|$)"),
    re.compile(r"\btorchrun\b"),
    # train/finetune/eval/benchmark/sweep scripts run through python
    re.compile(r"\bpython[0-9.]*\s[\s\S]*?\b(?:train|finetune|pretrain|evaluate|eval|benchmark|sweep)\w*\.py\b"),
)

# "already handled": the offload wrapper / escape hatch is in the command
ALLOW_PATTERNS = (
    re.compile(r"\$ORCH\b"),
    re.compile(r"ORCHARD_LOCAL"),
    re.compile(r"\borchard[\"']?\s+offload\b"),
    re.compile(r"\boffload\s+--"),
)

# pytest selectors/options that make a run clearly small
PYTEST_ALLOW_SUBSTRINGS = ("::", "--version", "--help", "--collect-only", "--co", "-k ", "--lf", "--last-failed")


def _deny(reason):
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def main():
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Bash":
        return 0
    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd or any(p.search(cmd) for p in ALLOW_PATTERNS):
        return 0

    hit = False
    for pattern in HEAVY_PATTERNS:
        if pattern.search(cmd):
            hit = True
            break
    if not hit:
        return 0
    if "pytest" in cmd and any(s in cmd for s in PYTEST_ALLOW_SUBSTRINGS):
        return 0

    launcher = Path(__file__).resolve().parent / "orchard"
    _deny(
        "This looks long/CPU-heavy, so it is denied here to keep the laptop responsive. "
        "Re-run it through the spillover wrapper so it uses the Pro when the Pro is free "
        f"(falls back to a local run automatically): `\"{launcher}\" offload -- <your command>`. "
        "If you really want it local, prefix `ORCHARD_LOCAL=1`; for a quick check, run a "
        "specific test (path::name) or pass -k/--lf."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
