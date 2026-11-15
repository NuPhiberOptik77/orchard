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
    found.append({"name": "Cafe", "host": "cafe.local", "port": 22})
    statuses = {"studio.local": "opens", "kitchen.local": "closed", "cafe.local": "unverified"}
    result = discover.find(1, check_keys=True, look_fn=lambda s: found,
                           hostname=lambda target: "old-mac.local" if target == "old-mac" else target,
                           opens=lambda host, port, user: tried.append((host, port, user)) or statuses[host],
                           here=lambda: "this-mac.local", user="me")
    assert [r["status"] for r in result] == ["this machine", "worker", "opens", "closed", "unverified"]
    assert result[1]["worker"] == "old-mac" and result[2]["suggest"] == "studio"
    assert sorted(tried) == [("cafe.local", 22, "me"), ("kitchen.local", 2222, "me"), ("studio.local", 22, "me")]
    text = "\n".join(discover.report(result, 1))
    assert "Studio: studio.local, your ssh keys open it as me. To add it, put this in" in text
    assert '      name = "studio"\n      host = "studio.local"' in text
    assert "Kitchen: kitchen.local:2222, found, but your ssh keys don't open it as me" in text
    assert "Old Mac: Old-Mac.local., already an Orchard worker (old-mac)" in text
    assert "Cafe: cafe.local, its host key isn't one you trust yet, so no key was tried" in text
    assert text.endswith("Nothing was changed.")
    # Without --check-keys no key is tried at all.
    tried.clear()
    plain = discover.find(1, look_fn=lambda s: found, hostname=lambda t: t, here=lambda: "",
                          opens=lambda *a: tried.append(a) or "opens", user="me")
    assert tried == [] and {r["status"] for r in plain} == {"found"}


def test_the_key_check_trusts_only_known_host_keys_and_forwards_nothing(monkeypatch):
    seen, answer = [], {"returncode": 0, "stderr": ""}
    monkeypatch.setattr(discover.subprocess, "run", lambda argv, **k: seen.append(argv) or type("R", (), answer)())
    assert discover.keys_open("studio.local", 22, "me") == "opens"
    argv = seen[0]
    for option in ("BatchMode=yes", "StrictHostKeyChecking=yes", "PasswordAuthentication=no",
                   "KbdInteractiveAuthentication=no", "ForwardAgent=no", "ForwardX11=no", "ClearAllForwardings=yes"):
        assert option in argv
    assert "UserKnownHostsFile=/dev/null" not in argv
    answer.update(returncode=255, stderr="No ED25519 host key is known for studio.local\nHost key verification failed.")
    assert discover.keys_open("studio.local", 22, "me") == "unverified"
    answer.update(stderr="me@studio.local: Permission denied (publickey).")
    assert discover.keys_open("studio.local", 22, "me") == "closed"
    assert argv[-2:] == ["me@studio.local", "true"]


def test_orchard_find_prints_and_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(discover, "look", lambda seconds: [{"name": "Studio", "host": "studio.local", "port": 22}])
    monkeypatch.setattr(discover, "keys_open", lambda host, port, user: "opens")
    monkeypatch.setattr(discover, "_this_host", lambda: "")
    monkeypatch.setattr(discover, "_ssh_hostname", lambda target: target)
    result = runner.invoke(cli.app, ["find", "--seconds", "1", "--check-keys"])
    assert result.exit_code == 0 and "your ssh keys open it" in result.stdout
    assert not (tmp_path / "config.toml").exists()
    data = json.loads(runner.invoke(cli.app, ["find", "--json"]).stdout)
    assert data[0]["status"] == "found"


def test_what_a_machine_announces_is_checked_before_it_is_printed():
    assert discover.parse_lookup("x can be reached at evil;rm -rf ~.local.:22 (interface 12)") is None
    assert discover.parse_lookup("x can be reached at $(id).local.:22") is None
    assert discover.parse_lookup("x can be reached at ok-box.local.:99999") is None
    bad = "=;wlan0;IPv4;n;_ssh._tcp;local;a|b.local;192.0.2.1;22;\n=;wlan0;IPv4;caf\\233\\027[2Jx;_ssh._tcp;local;fine.local;192.0.2.2;22;"
    assert discover.parse_avahi(bad) == [{"name": "caf\u00e9[2Jx", "host": "fine.local", "port": 22}]


def test_add_puts_in_only_machines_your_keys_open_and_undo_takes_it_back(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "workers.toml")
    config.save_workers([config.Worker(name="studio", host="old-box.local")])
    before = (tmp_path / "workers.toml").read_text()
    found = [{"name": "Studio", "host": "studio.local", "port": 22, "status": "opens", "suggest": "studio", "user": "me"},
             {"name": "Cafe", "host": "cafe.local", "port": 22, "status": "unverified", "user": "me"},
             {"name": "Old", "host": "old-box.local", "port": 22, "status": "opens", "suggest": "old-box", "user": "me"}]
    added = discover.add(found)
    assert [(w.name, w.host) for w in added] == [("studio-2", "studio.local")]  # unverified and known skipped
    assert [w.name for w in config.load_workers()] == ["studio", "studio-2"]
    assert discover.undo() == "put the workers file back as it was before orchard find --add"
    assert (tmp_path / "workers.toml").read_text() == before and discover.undo() is None


def test_undo_removes_a_workers_file_that_add_made(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "workers.toml")
    discover.add([{"name": "S", "host": "studio.local", "port": 22, "status": "opens", "suggest": "studio", "user": "me"}])
    assert (tmp_path / "workers.toml").exists()
    assert discover.undo() == "removed the workers file, which orchard find --add had made"
    assert not (tmp_path / "workers.toml").exists()


def test_orchard_find_add_and_undo(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "workers.toml")
    monkeypatch.setattr(discover, "look", lambda seconds: [{"name": "Studio", "host": "studio.local", "port": 22}])
    monkeypatch.setattr(discover, "keys_open", lambda host, port, user: "opens")
    monkeypatch.setattr(discover, "_this_host", lambda: "")
    monkeypatch.setattr(discover, "_ssh_hostname", lambda target: target)
    result = runner.invoke(cli.app, ["find", "--add", "--seconds", "1"])
    assert result.exit_code == 0 and "added worker studio (studio.local)" in result.stdout
    assert "Undo: orchard find --undo" in result.stdout
    result = runner.invoke(cli.app, ["find", "--undo"])
    assert result.exit_code == 0 and not (tmp_path / "workers.toml").exists()
    assert runner.invoke(cli.app, ["find", "--undo"]).exit_code == 1


def test_undo_after_two_adds_goes_back_to_before_the_first(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "workers.toml")
    discover.add([{"name": "A", "host": "a.local", "port": 22, "status": "opens", "suggest": "a", "user": "me"}])
    discover.add([{"name": "B", "host": "b.local", "port": 22, "status": "opens", "suggest": "b", "user": "me"}])
    assert [w.name for w in config.load_workers()] == ["a", "b"]
    assert discover.undo() == "removed the workers file, which orchard find --add had made"
    assert not (tmp_path / "workers.toml").exists() and discover.undo() is None
