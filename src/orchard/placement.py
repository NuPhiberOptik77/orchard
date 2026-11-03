from __future__ import annotations

from dataclasses import dataclass, field

from .job import JobSpec

DISK_HEADROOM_GB = 2.0
DEFAULT_DISK_NEED_GB = 1.0
DEFAULT_MEM_NEED_GB = 1.0
OS_TAG_PREFIX = "os:"
DEFAULT_OS = "macos"
ANY_OS = "any"
PROVEN_OS = "proven"


@dataclass
class Candidate:
    name: str
    kind: str
    caps: dict
    weight: float = 1.0


@dataclass
class Decision:
    candidate: Candidate
    reasons: list[str] = field(default_factory=list)


def worker_os(caps: dict) -> str:
    """The worker's OS from its `os:*` tag. Agents from before OS tags only ran on macOS."""
    oses = sorted(t[len(OS_TAG_PREFIX):] for t in caps.get("tags") or [] if t.startswith(OS_TAG_PREFIX))
    if not oses:
        return DEFAULT_OS
    # Conflicting tags (say a static `os:*` in workers.toml) match no allow-list.
    return ",".join(oses)


def allowed_oses(job: JobSpec) -> set[str]:
    """OSes the job may run on. Default-deny: a job with no `os:*` tag is a macOS job."""
    allowed = {t[len(OS_TAG_PREFIX):] for t in job.tags or [] if t.startswith(OS_TAG_PREFIX)} or {DEFAULT_OS}
    if PROVEN_OS in allowed:
        # Learned portability: macOS plus every OS where a sealed receipt shows
        # this command passing (attached at submit by receipts.attach_proof).
        allowed = (allowed - {PROVEN_OS}) | {DEFAULT_OS} | set(job.requires.get("proven_os") or [])
    return allowed


def problems(job: JobSpec, caps: dict) -> list[str]:
    probs = []
    required = {t for t in job.tags or [] if not t.startswith(OS_TAG_PREFIX)}
    if required:
        missing = sorted(required - set(caps.get("tags") or []))
        if missing:
            probs.append(f"missing tags: {', '.join(missing)}")
    allowed = allowed_oses(job)
    os_name = worker_os(caps)
    if ANY_OS not in allowed and os_name not in allowed:
        hint = f" (add -t {OS_TAG_PREFIX}{os_name} to allow it)" if "," not in os_name else ""
        if OS_TAG_PREFIX + PROVEN_OS in (job.tags or []):
            hint = f" (not proven on {os_name} yet; run orchard prove -- <cmd>)"
        probs.append(f"worker OS is {os_name}; job allows {', '.join(sorted(allowed))}{hint}")
    if not caps.get("platform_ok", True):
        probs.append(f"platform resolve failed ({caps.get('platform_note', 'unknown')})")
    mem_need = float(job.requires.get("mem_gb", DEFAULT_MEM_NEED_GB))
    ram = float(caps.get("ram_gb", 0) or 0)
    if ram < mem_need:
        probs.append(f"needs {mem_need} GB RAM, has {ram} GB")
    disk_need = float(job.requires.get("disk_gb", DEFAULT_DISK_NEED_GB))
    free = float(caps.get("disk_free_gb", 0) or 0)
    if free < disk_need + DISK_HEADROOM_GB:
        probs.append(f"needs {disk_need + DISK_HEADROOM_GB} GB free disk, has {free} GB")
    return probs


def _score(candidate: Candidate) -> float:
    return candidate.weight * float(candidate.caps.get("bench", 1.0) or 1.0)


def choose(job: JobSpec, candidates: list[Candidate], where: str = "auto", background: bool = False) -> Decision:
    if not candidates:
        raise ValueError("no candidates to place the job on")
    local = next((c for c in candidates if c.kind == "local"), candidates[0])

    if where != "auto":
        forced = (local if where in ("local", local.name)
                  else next((c for c in candidates if c.name == where or c.kind == where), None))
        if forced is None:
            raise ValueError(f"unknown worker: {where}")
        issues = problems(job, forced.caps)
        reasons = [f"forced to {forced.name}"]
        reasons += [f"warning: {p}" for p in issues]
        return Decision(candidate=forced, reasons=reasons)

    usable = [c for c in candidates if not problems(job, c.caps)]
    if not background:
        if local in usable:
            return Decision(candidate=local, reasons=["interactive job runs locally"])
        reasons = [f"local ineligible: {p}" for p in problems(job, local.caps)]
        if usable:
            best = max(usable, key=_score)
            reasons.append(f"best eligible worker: {best.name}")
            return Decision(candidate=best, reasons=reasons)
        if job.tags:
            raise ValueError(f"no worker with required tags: {', '.join(job.tags)}")
        return Decision(candidate=local, reasons=reasons + ["no eligible worker; proceeding on local with warnings"])

    remotes = [c for c in usable if c.kind != "local"]
    if remotes:
        best = max(remotes, key=_score)
        reasons = [f"background job offloaded to {best.name}"]
        for c in candidates:
            if c not in usable and c is not local:
                reasons.append(f"skipped {c.name}: {'; '.join(problems(job, c.caps))}")
        return Decision(candidate=best, reasons=reasons)
    if job.tags:
        raise ValueError(f"no worker with required tags: {', '.join(job.tags)}")
    reasons = [f"no eligible worker: {'; '.join(problems(job, local.caps)) or 'worker unavailable'}"]
    reasons.append("background job falls back to local")
    return Decision(candidate=local, reasons=reasons)
