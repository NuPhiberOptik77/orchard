# A Windows PC as a Linux worker (WSL2)

These files turn a Windows 11 PC into an Orchard worker that runs Linux jobs
inside WSL2. The agent runs in the Linux distro like it does on a Mac: over
SSH, from `~/.orchard`, with no daemon of its own. Windows adds one sign-in
task that tells the agent when someone is using the PC.

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
{"ts": 1767225600, "idle_seconds": 412.5, "power": "ac", "battery_pct": 60, "disk_free_gb": 210.4, "source": "windows"}
```

The agent fails closed. When the file is missing, unreadable, incomplete or
more than 60 s old (by its mtime, so the clocks don't have to agree), the PC
counts as in use with unknown power. New jobs are held and running ones pause,
and `orchard doctor` and the queue's hold note name the reason:
`host status missing`, `host status stale (75s old)` and so on. Free disk is the smaller of the
distro's own free space and C:'s.

The task must run in your signed-in session. Windows only reports input idle
time to the session that owns the keyboard, so a service or a SYSTEM task
would see a PC that is always idle. A locked screen is fine. After a reboot,
nothing runs until you sign in, and queued jobs fall back to the controller
after the usual 15 minutes.

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
   which needs its own rule. The GUID is WSL's fixed VM creator id.

   ```powershell
   New-NetFirewallHyperVRule -Name OrchardSSH -DisplayName "Orchard SSH (WSL)" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 2222
   New-NetFirewallRule -DisplayName "Orchard SSH (WSL)" -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow -Profile Private
   Get-NetConnectionProfile | Set-NetConnectionProfile -NetworkCategory Private
   ```

   The last line marks your current network Private. Skip it if you'd rather
   do that in Settings, or if this isn't your own network.
7. **Keep it awake on the charger**, lid closed included:

   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
   powercfg /setactive SCHEME_CURRENT
   ```

   A laptop that stays plugged in lasts longer with a charge limit, if the
   maker's app has one.
8. **Register the status task.** Put both `.ps1` files in one folder and run,
   from a normal PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\Orchard\register-task.ps1"
   ```

   It prints `Registered and started 'Orchard WSL worker'.` If Windows says
   "Access is denied", run it from an admin PowerShell; the task still runs
   as you. Pass `-Distro <name>` if yours isn't `Ubuntu-24.04`. The task also
   keeps the distro running, and a minute later
   `cat ~/.orchard/host-status.json` in the distro shows fresh numbers.

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

## Limits

- **Thermal throttling isn't checked.** Linux in WSL can't read it, so the
  PC's own firmware is the only guard.
- **`max_other_cpu_pct` only sees Linux processes,** not Windows programs.
  User input and power are the real guards here.
- **The RAM the worker reports is the `.wslconfig` cap,** not the PC's total,
  and `requires.mem_gb` is checked against that cap.
- **Other Linux machines** work as workers too, but they need something that
  writes `host-status.json`, or every job stays held. On a headless server
  nobody uses, a cron job or systemd timer that writes `"power": "ac"`, a large
  `idle_seconds` and the real free disk once a minute is enough.
