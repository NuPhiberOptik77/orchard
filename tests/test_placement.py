import pytest

from orchard.job import JobSpec
from orchard.placement import Candidate, choose, problems, worker_os


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


def test_worker_os_defaults_to_macos_for_untagged_agents():
    assert worker_os(caps(tags=["x86_64"])) == "macos"
    assert worker_os(caps(tags=["x86_64", "os:linux"])) == "linux"
    assert worker_os(caps(tags=["os:linux", "os:macos"])) == "linux,macos"


def test_os_default_deny_refuses_linux_for_untagged_job():
    linux = caps(tags=["x86_64", "os:linux"])
    assert problems(job(), linux) == ["worker OS is linux; job allows macos (add -t os:linux to allow it)"]
    assert problems(tagged_job("x86_64"), linux) == [
        "worker OS is linux; job allows macos (add -t os:linux to allow it)"
    ]


def test_os_tag_allows_listed_oses():
    linux = caps(tags=["x86_64", "os:linux"])
    mac = caps(tags=["x86_64", "os:macos"])
    assert problems(tagged_job("x86_64", "os:linux"), linux) == []
    assert problems(tagged_job("os:any"), linux) == []
    assert problems(tagged_job("os:any"), mac) == []
    assert problems(tagged_job("os:linux", "os:macos"), linux) == []
    assert problems(tagged_job("os:linux", "os:macos"), mac) == []
    assert "worker OS is macos" in problems(tagged_job("os:linux"), mac)[0]


def test_os_tag_is_not_a_required_tag():
    # os:* tags pick the OS; they never show up as missing capability tags.
    assert problems(tagged_job("os:linux"), caps(tags=["os:linux"])) == []
    assert not any("missing tags" in p for p in problems(tagged_job("os:linux"), caps(tags=["arm64"])))


def test_conflicting_worker_os_tags_match_nothing():
    both = caps(tags=["os:linux", "os:macos"])
    assert problems(job(), both) == ["worker OS is linux,macos; job allows macos"]
    assert problems(tagged_job("os:any"), both) == []


def test_mac_x86_64_job_still_lands_on_pro_beside_linux_worker():
    # A higher-scoring x86_64 Linux worker must never take an untagged Intel Mac job,
    # whether the Pro's agent reports os:macos or predates OS tags.
    for pro_tags in (["x86_64", "swift"], ["x86_64", "swift", "os:macos"]):
        candidates = [
            Candidate("local", "local", caps(tags=["arm64", "swift", "os:macos"])),
            Candidate("pro", "remote", caps(tags=pro_tags), 0.2),
            Candidate("laptop", "remote", caps(tags=["x86_64", "swift", "os:linux"], bench=9000), 1.0),
        ]
        decision = choose(tagged_job("x86_64"), candidates, background=True)
        assert decision.candidate.name == "pro"
        assert any("skipped laptop: worker OS is linux" in r for r in decision.reasons)
        assert choose(tagged_job("x86_64"), candidates).candidate.name == "pro"


def test_untagged_background_job_skips_linux_worker():
    candidates = [
        Candidate("local", "local", caps(tags=["arm64", "os:macos"])),
        Candidate("laptop", "remote", caps(tags=["x86_64", "os:linux"], bench=9000), 1.0),
    ]
    decision = choose(job(), candidates, background=True)
    assert decision.candidate.name == "local"


def test_linux_job_picks_linux_worker_or_raises():
    laptop = Candidate("laptop", "remote", caps(tags=["x86_64", "os:linux"]), 1.0)
    pro = Candidate("pro", "remote", caps(tags=["x86_64", "os:macos"]), 0.2)
    local = Candidate("local", "local", caps(tags=["arm64", "os:macos"]))
    decision = choose(tagged_job("os:linux"), [local, pro, laptop], background=True)
    assert decision.candidate.name == "laptop"
    with pytest.raises(ValueError, match="required tags"):
        choose(tagged_job("os:linux"), [local, pro], background=True)
