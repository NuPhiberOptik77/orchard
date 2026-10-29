from orchard.env import _filter_requirements


def test_filter_requirements_drops_local_paths():
    lines = [
        "annotated-doc==0.0.5",
        "orchard @ file:///Users/me/My%20Project",
        "-e file:///Users/me/My%20Project/orchard",
        "pytest==9.1.1",
        "-e .",
    ]
    kept, dropped = _filter_requirements(lines)
    assert kept == ["annotated-doc==0.0.5", "pytest==9.1.1"]
    assert dropped == 3


def test_filter_requirements_keeps_plain_pins():
    lines = ["typer==0.27.3", "rich==15.0.0"]
    kept, dropped = _filter_requirements(lines)
    assert kept == lines
    assert dropped == 0
