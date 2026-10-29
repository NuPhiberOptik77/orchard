import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchard import dispatcher, events  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_events(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(events, "EVENTS_PATH", tmp_path_factory.mktemp("events") / "events.jsonl")
    dispatcher._hold_logged.clear()
