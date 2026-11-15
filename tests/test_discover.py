import json

from typer.testing import CliRunner

from orchard import cli, config, discover

runner = CliRunner()

BROWSE = """Browsing for _ssh._tcp.local.
DATE: ---Sun 15 Nov 2026---
 7:40:32.625  ...STARTING...
Timestamp     A/R    Flags  if Domain               Service Type         Instance Name
 7:40:32.625  Add        3  12 local.               _ssh._tcp.           Old Mac
 7:40:32.626  Add        3  14 local.               _ssh._tcp.           Old Mac
 7:40:32.627  Add        2  12 local.               _ssh._tcp.           Studio
 7:40:33.100  Rmv        0  12 local.               _ssh._tcp.           Gone Box
"""

LOOKUP = """Lookup Studio._ssh._tcp.local.
 7:40:53.832  Studio._ssh._tcp.local. can be reached at studio.local.:22 (interface 12)
"""

AVAHI = """+;wlan0;IPv4;old-box;_ssh._tcp;local
=;wlan0;IPv4;old-box;_ssh._tcp;local;old-box.local;192.0.2.7;22;
=;wlan0;IPv6;old-box;_ssh._tcp;local;old-box.local;fe80::1;22;
=;wlan0;IPv4;nas\\032box;_ssh._tcp;local;nas.local;192.0.2.9;2222;
"""


def test_bonjour_answers_are_read_once_each():
    assert discover.parse_browse(BROWSE) == ["Old Mac", "Studio"]
    assert discover.parse_lookup(LOOKUP) == ("studio.local", 22)
    assert discover.parse_lookup("nothing") is None
    assert discover.parse_avahi(AVAHI) == [{"name": "old-box", "host": "old-box.local", "port": 22},
                                           {"name": "nas box", "host": "nas.local", "port": 2222}]


def test_looking_resolves_each_name(monkeypatch):
    monkeypatch.setattr(discover.shutil, "which", lambda tool: "/usr/bin/dns-sd" if tool == "dns-sd" else None)
    calls = []

    def run(argv, seconds):
        calls.append(argv)
        if argv[1] == "-B":
            return BROWSE
        return f"x can be reached at {argv[2].lower().replace(' ', '-')}.local.:22 (interface 12)"

    assert discover.look(1, run) == [{"name": "Old Mac", "host": "old-mac.local", "port": 22},
                                     {"name": "Studio", "host": "studio.local", "port": 22}]
    assert calls[0] == ["dns-sd", "-B", "_ssh._tcp", "local."]


def test_each_machine_is_this_one_a_worker_one_your_keys_open_or_not(monkeypatch):
    monkeypatch.setattr(config, "load_workers", lambda: [config.Worker(name="old-mac", host="old-mac")])
    found = [{"name": "This", "host": "this-mac.local", "port": 22},
             {"name": "Old Mac", "host": "Old-Mac.local.", "port": 22},
             {"name": "Studio", "host": "studio.local", "port": 22},
             {"name": "Kitchen", "host": "kitchen.local", "port": 2222}]
    tried = []
    result = discover.find(1, look_fn=lambda s: found,
                           hostname=lambda target: "old-mac.local" if target == "old-mac" else target,
                           opens=lambda host, port, user: tried.append((host, port, user)) or host == "studio.local",
                           here=lambda: "this-mac.local", user="me")
    assert [r["status"] for r in result] == ["this machine", "worker", "opens", "closed"]
    assert result[1]["worker"] == "old-mac" and result[2]["suggest"] == "studio"
    assert sorted(tried) == [("kitchen.local", 2222, "me"), ("studio.local", 22, "me")]  # never this one or a worker
    text = "\n".join(discover.report(result, 1))
    assert "Studio: studio.local, your ssh keys open it as me. To add it, put this in" in text
    assert '      name = "studio"\n      host = "studio.local"' in text
    assert "Kitchen: kitchen.local:2222, found, but your ssh keys don't open it as me" in text
    assert "Old Mac: Old-Mac.local., already an Orchard worker (old-mac)" in text
    assert text.endswith("Nothing was changed.")


def test_the_key_check_records_nothing_and_forwards_nothing(monkeypatch):
    seen = []
    monkeypatch.setattr(discover.subprocess, "run", lambda argv, **k: seen.append(argv) or type("R", (), {"returncode": 0})())
    assert discover.keys_open("studio.local", 22, "me")
    argv = seen[0]
    for option in ("BatchMode=yes", "UserKnownHostsFile=/dev/null", "ForwardAgent=no", "ClearAllForwardings=yes"):
        assert option in argv
    assert argv[-2:] == ["me@studio.local", "true"]


def test_orchard_find_prints_and_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(discover, "look", lambda seconds: [{"name": "Studio", "host": "studio.local", "port": 22}])
    monkeypatch.setattr(discover, "keys_open", lambda host, port, user: True)
    monkeypatch.setattr(discover, "_this_host", lambda: "")
    monkeypatch.setattr(discover, "_ssh_hostname", lambda target: target)
    result = runner.invoke(cli.app, ["find", "--seconds", "1"])
    assert result.exit_code == 0 and "your ssh keys open it" in result.stdout
    assert not (tmp_path / "config.toml").exists()
    data = json.loads(runner.invoke(cli.app, ["find", "--json"]).stdout)
    assert data[0]["status"] == "opens"
