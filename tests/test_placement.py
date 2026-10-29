import pytest

from orchard.job import JobSpec
from orchard.placement import Candidate, choose, problems


def caps(ram=16, disk=50, bench=1000, platform_ok=True, note="resolve ok", tags=None):
    return {
        "ram_gb": ram,
        "disk_free_gb": disk,
        "bench": bench,
        "platform_ok": platform_ok,
        "platform_note": note,
        "tags": tags if tags is not None else [],
    }


def job(**requires):
    return JobSpec(name="j", cmd=["true"], requires=requires)


def test_problems_memory():
    probs = problems(job(mem_gb=4), caps(ram=2))
    assert any("RAM" in p for p in probs)


def test_problems_disk_headroom():
    assert problems(job(disk_gb=6), caps(disk=7)) != []
    assert problems(job(disk_gb=1), caps(disk=7)) == []


def test_problems_platform():
    probs = problems(job(), caps(platform_ok=False, note="no wheel"))
    assert any("resolve" in p for p in probs)


def test_auto_interactive_stays_local():
    candidates = [Candidate("local", "local", caps()), Candidate("pro", "remote", caps(), 0.2)]
    decision = choose(job(), candidates)
    assert decision.candidate.kind == "local"


def test_background_prefers_remote():
    candidates = [Candidate("local", "local", caps()), Candidate("pro", "remote", caps(), 0.2)]
    decision = choose(job(), candidates, background=True)
    assert decision.candidate.name == "pro"


def test_background_falls_back_when_ineligible():
    candidates = [
        Candidate("local", "local", caps()),
        Candidate("pro", "remote", caps(platform_ok=False, note="no wheel"), 0.2),
    ]
    decision = choose(job(), candidates, background=True)
    assert decision.candidate.kind == "local"
    assert any("ineligible" in reason or "no eligible" in reason for reason in decision.reasons)


def test_forced_worker_warns_but_proceeds():
    candidates = [
        Candidate("local", "local", caps()),
        Candidate("pro", "remote", caps(ram=2, platform_ok=False), 0.2),
    ]
    decision = choose(job(mem_gb=4), candidates, where="pro")
    assert decision.candidate.name == "pro"
    assert any("warning" in reason for reason in decision.reasons)


def test_unknown_worker_raises():
    candidates = [Candidate("local", "local", caps())]
    with pytest.raises(ValueError, match="unknown worker"):
        choose(job(), candidates, where="nope")


def tagged_job(*tags):
    return JobSpec(name="b", cmd=["swift"], kind="build", tags=list(tags))


def test_tags_gate():
    assert problems(tagged_job("x86_64"), caps(tags=["arm64"])) == ["missing tags: x86_64"]
    assert problems(tagged_job("x86_64", "swift"), caps(tags=["x86_64", "swift"])) == []
    assert problems(tagged_job("swift"), caps(tags=["x86_64", "swift"])) == []


def test_tagged_job_without_matching_worker_raises():
    candidates = [Candidate("local", "local", caps(tags=["arm64"]))]
    with pytest.raises(ValueError, match="required tags"):
        choose(tagged_job("x86_64"), candidates, background=True)
    with pytest.raises(ValueError, match="required tags"):
        choose(tagged_job("x86_64"), candidates)


def test_tagged_job_picks_matching_remote():
    candidates = [
        Candidate("local", "local", caps(tags=["arm64"])),
        Candidate("pro", "remote", caps(tags=["x86_64"]), 0.2),
    ]
    decision = choose(tagged_job("x86_64"), candidates)
    assert decision.candidate.name == "pro"
    assert any("missing tags" in r for r in decision.reasons)


def test_tag_shared_by_local_stays_local_interactive():
    candidates = [
        Candidate("local", "local", caps(tags=["arm64", "swift"])),
        Candidate("pro", "remote", caps(tags=["x86_64", "swift"]), 0.2),
    ]
    decision = choose(tagged_job("swift"), candidates)
    assert decision.candidate.kind == "local"
