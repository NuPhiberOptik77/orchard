import pytest

from orchard import events


@pytest.fixture(autouse=True)
def _isolated_events(tmp_path, monkeypatch):
    # Keep test events out of the real ~/.orchard log.
    monkeypatch.setattr(events, "EVENTS_PATH", tmp_path / "events.jsonl")
