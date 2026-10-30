import pytest

from orchard.job import JobSpec, agent_job_payload, capture_env, load_job_file, new_job_id, slugify


def test_load_job_file(tmp_path):
    path = tmp_path / "job.yml"
    path.write_text(
        "name: embed\n"
        "cmd: [python, '-c', 'print(1)']\n"
        "python: '3.11'\n"
        "timeout_s: 60\n"
        "requires: {mem_gb: 2}\n"
    )
    job = load_job_file(path)
    assert job.name == "embed"
    assert job.cmd == ["python", "-c", "print(1)"]
    assert job.python == "3.11"
    assert job.timeout_s == 60
    assert job.requires == {"mem_gb": 2}


def test_load_job_file_accepts_string_cmd(tmp_path):
    path = tmp_path / "job.yml"
    path.write_text("cmd: echo hello\n")
    job = load_job_file(path)
    assert job.cmd == ["echo", "hello"]


def test_load_job_file_kind_and_tags(tmp_path):
    path = tmp_path / "build.yml"
    path.write_text("kind: build\ncmd: [swift, test]\ntags: [x86_64, swift]\n")
    job = load_job_file(path)
    assert job.kind == "build"
    assert job.tags == ["x86_64", "swift"]


def test_load_job_file_defaults_kind_run(tmp_path):
    path = tmp_path / "job.yml"
    path.write_text("cmd: ['true']\n")
    job = load_job_file(path)
    assert job.kind == "run"
    assert job.tags == []


def test_load_job_file_requires_cmd(tmp_path):
    path = tmp_path / "job.yml"
    path.write_text("name: nope\n")
    with pytest.raises(ValueError, match="cmd"):
        load_job_file(path)


def test_load_job_file_env(tmp_path):
    path = tmp_path / "job.yml"
    path.write_text("cmd: ['true']\nenv: [HF_TOKEN, WANDB_API_KEY]\n")
    job = load_job_file(path)
    assert job.env == ["HF_TOKEN", "WANDB_API_KEY"]
    assert job.env_values == {}


def test_load_job_file_env_must_be_list(tmp_path):
    path = tmp_path / "job.yml"
    path.write_text("cmd: ['true']\nenv: HF_TOKEN\n")
    with pytest.raises(ValueError, match="env"):
        load_job_file(path)


def test_capture_env(monkeypatch):
    monkeypatch.setenv("ORCHARD_TEST_TOKEN", "s3cret")
    monkeypatch.delenv("ORCHARD_TEST_MISSING", raising=False)
    values, missing = capture_env(["ORCHARD_TEST_TOKEN", "ORCHARD_TEST_MISSING"])
    assert values == {"ORCHARD_TEST_TOKEN": "s3cret"}
    assert missing == ["ORCHARD_TEST_MISSING"]


def test_agent_job_payload_includes_env():
    job = JobSpec(name="t", cmd=["true"], env=["HF_TOKEN"], env_values={"HF_TOKEN": "x"})
    payload = agent_job_payload(job, "id1", "/j", "/w", None)
    assert payload["env"] == {"HF_TOKEN": "x"}
    assert payload["attempt"] == 1
    assert agent_job_payload(job, "id1", "/j", "/w", None, attempt=3)["attempt"] == 3


def test_slugify():
    assert slugify("Train Model v3!") == "train-model-v3"
    assert slugify("!!!") == "job"


def test_new_job_id():
    job_id = new_job_id("train model")
    assert job_id.startswith("train-model-")
    assert len(job_id.split("-")) == 5
