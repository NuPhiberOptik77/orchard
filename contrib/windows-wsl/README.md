# A Windows PC as a Linux worker (WSL2)

These files turn a Windows 11 PC into an Orchard worker that runs Linux jobs
inside WSL2. The agent runs in the Linux distro like it does on a Mac: over
SSH, from `~/.orchard`, with no daemon of its own. Windows adds two small
tasks that tell the agent when someone is using the PC.

Linux workers only take jobs that ask for them. A job with no `os:*` tag is a
macOS job, so nothing you already run moves to the PC. Opt a job in with
`-t os:linux`, or `-t os:any` for a job that runs anywhere.

| File | Where it goes |
| --- | --- |
| `wslconfig.txt` | `%UserProfile%\.wslconfig` on Windows |
| `wsl.conf` | `/etc/wsl.conf` in the distro |
| `setup-sshd.sh` | run once in the distro |
| `orchard-host-status.ps1`, `register-task.ps1` | one folder on Windows, for example `%UserProfile%\Orchard` |

## Why Windows needs a helper

Nothing inside WSL can see the Windows keyboard, mouse or charger, and the
Linux disk is a file on C: that can't grow past C:'s free space. So
`orchard-host-status.ps1` writes them to `~/.orchard/host-status.json` in the
distro every 10 seconds:

```json
{"ts":1767225600,"idle_seconds":412.5,"power":"ac","battery_pct":60,"disk_free_gb":210.4,"source":"windows"}
```

The agent fails closed. When the file is missing, unreadable, incomplete or
more than 60 s old (by its mtime, so the clocks don't have to agree), the PC
counts as in use with unknown power. New jobs are held and running ones pause,
and `orchard doctor` and the queue's hold note name the reason:
`host status missing`, `host status stale (75s old)` and so on. Free disk is the smaller of the
distro's own free space and C:'s.

Two tasks run it, and both run as you:

- **The startup task** runs from boot, before anyone signs in. It is the only
  thing that starts the distro or writes the file. WSL's VM belongs to the
  Windows session that first started WSL since the WSL service last started,
  and stops when that session signs out, taking any running job with it.
  Started from boot, it belongs to this task and runs until Windows shuts
  down. If the distro stops, this task starts it again within seconds.
- **The sign-in task** runs in your session, because Windows only reports
  keyboard, mouse and game controller input to the session that owns the
  screen. A service or a SYSTEM task would see a PC that is always idle. It
  writes its numbers to `session-status.json` next to the scripts, and the
  startup task passes them on. A locked screen is fine.

While nobody is signed in, the startup task counts the PC as idle from the
moment it saw that, so jobs start about three minutes after a restart or after
the last person signs out. Those samples say `"source":"windows-signed-out"`.
Windows 11 usually signs you back in by itself after any restart and locks
the screen, while "Use my sign-in info to automatically finish setting up"
is on in Sign-in options. You'll then see `"source":"windows"` instead, which
is fine. The signed-out path only runs with that setting off, and it hasn't
been tested on real hardware yet.
When someone signs in, it marks the PC in use once, then passes on what the
sign-in task reports. If that is missing or more than 30 s old, it writes
nothing and the file goes stale. If it can't tell whether anyone is signed in,
it assumes someone is.

Either way, a restart, such as a Windows update overnight, doesn't take the
worker offline until you next sign in.

## Setup

Windows steps go in PowerShell (`PS C:\...>`), Linux steps in the distro's
terminal (`<linux-user>@<pc>:~$`). Pasting one into the other fails
harmlessly.

1. **Check the PC.** Windows 11 22H2 or newer, and Task Manager > Performance >
   CPU shows "Virtualization: Enabled". Leave room on C:; every job's files
   land on the Linux disk.
2. **Write `.wslconfig`.** Copy `wslconfig.txt` to `%UserProfile%\.wslconfig`
   and set `memory` to what Windows can spare. Mirrored networking puts the
   distro on the PC's own LAN address.
3. **Install the distro** from an admin PowerShell, then restart and create
   the Linux user:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

4. **Write `/etc/wsl.conf`** from `wsl.conf`, with `<linux-user>` replaced.
   It turns on systemd and keeps jobs away from the Windows drives and
   programs. Then run `wsl --shutdown` in PowerShell and reopen the distro.
5. **Start sshd on port 2222, keys only.** Port 22 stays free for Windows'
   own OpenSSH server. Run `sudo -v` on its own line first, so a multi-line
   paste can't land in the password prompt:

   ```sh
   sudo -v
   sudo sh setup-sshd.sh
   ```

   Ubuntu 24.04 starts sshd from `ssh.socket`, which only reads the new port
   after `systemctl daemon-reload`; the script does that and falls back to
   `ssh.service` if the socket still won't listen. Then add the controller's
   public key to `~/.ssh/authorized_keys`.
6. **Open port 2222** from an admin PowerShell. Mirrored mode with
   `firewall=true` filters the distro's traffic through the Hyper-V firewall,
   which needs its own rule. Both rules apply only on Private networks, so
   sshd stays closed on public Wi-Fi, and your home network has to be Private.
   List the networks:

   ```powershell
   Get-NetConnectionProfile | Format-List Name,InterfaceAlias,InterfaceIndex,NetworkCategory,IPv4Connectivity
   ```

   If your home network says `Public`, mark only that one Private, using the
   `InterfaceIndex` the list shows for it, and list it again:

   ```powershell
   Set-NetConnectionProfile -InterfaceIndex <n> -NetworkCategory Private
   Get-NetConnectionProfile -InterfaceIndex <n> | Format-List Name,NetworkCategory
   ```

   Go no further until it says `Private`. Pick the network by its index: the
   alias may not be `Wi-Fi`, and `-InterfaceAlias 'Wi-Fi'` then fails with
   `ObjectNotFound`. Leave a network that isn't your own as it is. Then add
   the rules and check the Hyper-V one. The GUID is WSL's fixed VM creator id.

   ```powershell
   New-NetFirewallHyperVRule -Name OrchardSSH -DisplayName "Orchard SSH (WSL)" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 2222 -Profiles Private
   New-NetFirewallRule -DisplayName "Orchard SSH (WSL)" -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow -Profile Private
   Get-NetFirewallHyperVRule -Name OrchardSSH | Format-List Name,Profiles,Enabled,PortStatuses
   ```

   Profiles should show Private, and Enabled should show True. A Hyper-V rule
   made without `-Profiles` applies everywhere; fix it with
   `Set-NetFirewallHyperVRule -Name OrchardSSH -Profiles Private`.
7. **Keep it awake on the charger**, lid closed included:

   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
   powercfg /setactive SCHEME_CURRENT
   ```

   A laptop that stays plugged in lasts longer with a charge limit, if the
   maker's app has one.
8. **Register the status tasks.** Close other WSL clients first: VS Code
   Remote-WSL, Docker Desktop, and Explorer windows on `\\wsl.localhost`. One
   of them could start WSL before the startup task does, and the VM would be
   its session's. The script only sees `wsl.exe` running in a signed-in
   session, and asks for a restart then; it can't tell whether those other
   clients are open. Put
   both `.ps1` files in one folder and run, from an admin PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\Orchard\register-task.ps1"
   ```

   It restarts the WSL service first, so the startup task is the first to
   start WSL and the VM is its own. `wsl --shutdown` wouldn't do: the VM
   would stay with whichever session started WSL first. That closes open
   distro windows and stops running jobs, which the dispatcher runs again.
   Then it prints `Registered and started 'Orchard WSL
   worker (startup)' and 'Orchard WSL worker'.` Admin is only needed to
   register the startup task, which runs without your password stored; both
   tasks run as you, without admin rights. If your account isn't an admin, the
   admin prompt runs as another one: give the full path to your folder, add
   `-User <pc>\<you>`, and restart Windows afterwards. Run it again after
   updating the scripts: it replaces and restarts both. Pass `-Distro <name>`
   if yours isn't `Ubuntu-24.04`. Before any restart it asks for, check that
   the startup task holds the VM:

   ```powershell
   Start-Sleep 20
   Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" | Format-List ProcessId,SessionId,CommandLine
   ```

   In one entry SessionId must show 0, with `--exec sh -c` in its command
   line: that is the startup task's writer. If none does, restart Windows and
   check again; `Get-ScheduledTaskInfo 'Orchard WSL worker (startup)'` shows
   the task's last result. A minute later `cat ~/.orchard/host-status.json`
   in the distro shows fresh numbers. This only shows that the startup task's
   writer started; the check with nobody signed in at the end of this guide
   is what shows the VM stays up without you.
9. **Clean up old jobs.** Nothing deletes finished job folders on the worker
   by itself, and they fill C: in the end. A cron line in the distro runs the
   agent's cleanup every 6 hours: folders older than 2 days go, then newer
   ones while under 15 GB is free, counting C:. Running and paused jobs are
   kept; one that died with the PC goes once it is 2 days old. The agent
   arrives with the first job, and until then the line does nothing. In the
   distro:

   ```sh
   ( crontab -l 2>/dev/null | grep -v 'orchard-agent.py" gc '; echo '17 */6 * * * test -f "$HOME/.orchard/bin/orchard-agent.py" && /usr/bin/python3 "$HOME/.orchard/bin/orchard-agent.py" gc 2 15 >/dev/null 2>&1' ) | crontab -
   ```

## On the controller

Orchard's worker entry has no port field, so give the PC an SSH alias in
`~/.ssh/config`:

```
Host <alias>
    HostName <pc-name>.local
    Port 2222
    User <linux-user>
    ConnectTimeout 5
```

A DHCP reservation on the router works as well as the `.local` name. Check
`ssh <alias> true` works without a password prompt, then add the worker to
`~/.orchard/workers.toml`:

```toml
[[workers]]
name = "<name>"
host = "<alias>"
```

`orchard workers` should list it with `os:linux` in its tags and a real idle
time, and `orchard doctor` should show no host-status warning. Every machine running the dispatcher needs this version or
newer first: an older one doesn't know `os:*` tags and would send macOS jobs to
the PC.

To check that the worker keeps reporting with nobody signed in, restart the PC
and, before signing in, run on the controller:

```sh
ssh <alias> 'cat ~/.orchard/host-status.json; echo; date +%s'
```

It should show a `ts` no more than about 20 s behind the number on the last
line. The source is `"windows"` if Windows signed you back in by itself (see
above), or `"windows-signed-out"` if it didn't. To try the signed-out path,
sign out from the Start menu and check again. If ssh can't connect after a restart
with nobody signed in, the startup task didn't start WSL: sign in and go back
over step 8. ssh working with a missing file or an old `ts` means the writer
stopped.

## Limits

- **Thermal throttling isn't checked.** Linux in WSL can't read it, so the
  PC's own firmware is the only guard.
- **`max_other_cpu_pct` only sees Linux processes,** not Windows programs.
  User input and power are the real guards here.
- **The RAM the worker reports is the `.wslconfig` cap,** not the PC's total,
  and `requires.mem_gb` is checked against that cap.
- **Someone else on the PC holds every job.** While another account is
  signed in and you aren't, nothing reports. While another account has the
  screen through Switch user, your session reports the PC as in use, and so
  does the Switch user screen itself.
- **Only XInput game controllers count as input.** That covers Xbox pads and
  most PC ones; others, such as a PlayStation pad, read as idle. A pad with
  stick drift keeps the PC in use while it is connected.
- **`wsl --shutdown` stops running jobs,** which the dispatcher runs again.
  The startup task starts the distro again within seconds, and the VM stays
  its own even if you open the distro first.
- **`wsl --terminate` keeps the boot id** and hands pids out again, so the
  agent also compares each job's start time in `/proc` before trusting or
  signalling its pid. A job whose first process has exited while its children
  run on can't be dated that way, and is judged by whether its process group
  is alive.
- **A job that signals its own process group,** for example a shell with
  `trap 'kill 0' EXIT`, signals the agent's `wrap` too, which records it. If
  the agent died as well, that job runs again instead of being collected.
- **Other Linux machines** work as workers too, but they need something that
  writes `host-status.json`, or every job stays held. On a headless server
  nobody uses, a cron job or systemd timer that writes `"power": "ac"`, a large
  `idle_seconds` and the real free disk once a minute is enough.
