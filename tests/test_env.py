import pytest

from orchard.env import _filter_requirements, platform_tag


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


def test_platform_tag_knows_the_os():
    assert platform_tag("arm64") == "aarch64-apple-darwin"
    assert platform_tag("x86_64") == "x86_64-apple-darwin"
    assert platform_tag("x86_64", "macos") == "x86_64-apple-darwin"
    assert platform_tag("x86_64", "linux") == "x86_64-unknown-linux-gnu"
    assert platform_tag("aarch64", "linux") == "aarch64-unknown-linux-gnu"
    with pytest.raises(ValueError, match="unsupported platform"):
        platform_tag("x86_64", "freebsd")
    with pytest.raises(ValueError, match="unsupported platform"):
        platform_tag("aarch64", "macos")
