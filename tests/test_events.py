from orchard import events


def test_append_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "EVENTS_PATH", tmp_path / "state" / "events.jsonl")
    events.append("submitted", job="j1", worker="pro", background=True)
    events.append("held", job="j1", reason="worker busy")
    records = events.load()
    assert [r["event"] for r in records] == ["submitted", "held"]
    assert records[0]["job"] == "j1"
    assert records[0]["background"] is True
    assert records[1]["reason"] == "worker busy"
    assert records[0]["ts"] <= records[1]["ts"]


def test_rotation_keeps_previous_generation(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(events, "EVENTS_PATH", path)
    events.append("first", job="j0")
    monkeypatch.setattr(events, "MAX_BYTES", path.stat().st_size - 1)
    events.append("second", job="j1")
    assert path.with_suffix(".jsonl.1").exists()
    assert not path.with_suffix(".jsonl.2").exists()
    records = events.load()
    assert [r["event"] for r in records] == ["first", "second"]


def test_append_never_raises(tmp_path, monkeypatch):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, not a directory")
    monkeypatch.setattr(events, "EVENTS_PATH", blocker / "events.jsonl")
    events.append("submitted", job="j1")
    assert events.load() == []


def test_load_skips_bad_lines(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    path.write_text('{"ts": 1, "event": "ok"}\nnot json\n{"ts": 2, "event": "ok2"}\n')
    monkeypatch.setattr(events, "EVENTS_PATH", path)
    assert [r["event"] for r in events.load()] == ["ok", "ok2"]
