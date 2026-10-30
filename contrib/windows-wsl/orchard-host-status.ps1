<#
.SYNOPSIS
Tells the Orchard agent inside WSL when this PC is in use, on battery or short
of disk, and keeps the WSL distro running.

.DESCRIPTION
Linux inside WSL can't see the Windows keyboard, mouse or charger, so this
script writes them to ~/.orchard/host-status.json inside the distro every few
seconds. The agent fails closed: when that file is missing, unreadable or more
than 60 s old, it holds new jobs and pauses running ones.

register-task.ps1 starts it from two tasks. The startup task runs it with
-Startup from boot, and only that copy starts WSL or writes the file. WSL's VM
belongs to the Windows session that first started WSL since the WSL service
last started, and stops when that session signs out. Started from boot, it
runs until Windows shuts down, and wsl --shutdown doesn't change whose it is;
this copy starts the distro again on its next pass. While nobody is signed in it
counts the PC idle from the moment it saw that. When someone signs in, it
marks the PC in use once, then passes on what the sign-in task reports.

The sign-in task runs it in your session, the only place Windows reports
keyboard, mouse and controller input: run as a service or as SYSTEM, it would
see no input at all and the PC would always look idle. That copy writes its
samples to -SessionFile, next to the scripts by default.
#>
param(
    [string]$Distro = 'Ubuntu-24.04',
    [int]$IntervalSeconds = 10,
    [string]$Drive = 'C',
    [string]$SessionFile = '',
    [switch]$Startup
)

$ErrorActionPreference = 'Stop'
if (-not $SessionFile) { $SessionFile = Join-Path $PSScriptRoot 'session-status.json' }

Add-Type -TypeDefinition @"
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

public static class OrchardHost {
    [StructLayout(LayoutKind.Sequential)]
    struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }

    [StructLayout(LayoutKind.Sequential)]
    struct SYSTEM_POWER_STATUS {
        public byte ACLineStatus;
        public byte BatteryFlag;
        public byte BatteryLifePercent;
        public byte SystemStatusFlag;
        public int BatteryLifeTime;
        public int BatteryFullLifeTime;
    }

    [DllImport("user32.dll")]
    static extern bool GetLastInputInfo(ref LASTINPUTINFO info);

    [DllImport("kernel32.dll")]
    static extern bool GetSystemPowerStatus(out SYSTEM_POWER_STATUS status);

    [DllImport("kernel32.dll")]
    static extern uint WTSGetActiveConsoleSessionId();

    [StructLayout(LayoutKind.Sequential)]
    struct XINPUT_STATE {
        public uint dwPacketNumber;
        public ushort wButtons;
        public byte bLeftTrigger;
        public byte bRightTrigger;
        public short sThumbLX;
        public short sThumbLY;
        public short sThumbRX;
        public short sThumbRY;
    }

    [DllImport("xinput1_4.dll")]
    static extern uint XInputGetState(uint index, out XINPUT_STATE state);

    static uint[] packets = new uint[4];
    static bool[] seen = new bool[4];
    static Stopwatch sinceInput;

    // Seconds since the last keyboard or mouse input in this session; -1 if unknown.
    public static double IdleSeconds() {
        LASTINPUTINFO info = new LASTINPUTINFO();
        info.cbSize = (uint)Marshal.SizeOf(info);
        if (!GetLastInputInfo(ref info)) return -1;
        // Both are 32-bit millisecond tick counts; unsigned subtraction survives the 49.7-day wrap.
        return unchecked((uint)Environment.TickCount - info.dwTime) / 1000.0;
    }

    // Seconds since a game controller last changed state; -1 if none has.
    // GetLastInputInfo doesn't see controllers. A controller's packet number
    // changes whenever its state does, and plugging one in counts too.
    public static double ControllerIdleSeconds() {
        try {
            for (uint i = 0; i < 4; i++) {
                XINPUT_STATE s;
                if (XInputGetState(i, out s) != 0) { seen[i] = false; continue; }
                if (!seen[i] || s.dwPacketNumber != packets[i]) {
                    seen[i] = true;
                    packets[i] = s.dwPacketNumber;
                    if (sinceInput == null) sinceInput = Stopwatch.StartNew(); else sinceInput.Restart();
                }
            }
        } catch (Exception) {
            return -1;
        }
        return sinceInput == null ? -1 : sinceInput.Elapsed.TotalSeconds;
    }

    // False while another session has the screen, as after Switch user.
    public static bool InConsoleSession() {
        return (uint)Process.GetCurrentProcess().SessionId == WTSGetActiveConsoleSessionId();
    }

    public static string Power() {
        SYSTEM_POWER_STATUS s;
        if (!GetSystemPowerStatus(out s)) return "unknown";
        if (s.ACLineStatus == 1) return "ac";
        if (s.ACLineStatus == 0) return "battery";
        return "unknown";
    }

    public static int? BatteryPercent() {
        SYSTEM_POWER_STATUS s;
        if (!GetSystemPowerStatus(out s) || s.BatteryLifePercent > 100) return null;
        return s.BatteryLifePercent;
    }
}
"@

# Prefer the WSL app's own wsl.exe; the one in System32 only forwards to it.
$wsl = Join-Path $env:ProgramFiles 'WSL\wsl.exe'
if (-not (Test-Path $wsl)) { $wsl = 'wsl.exe' }

# Runs in the distro. Each status line on stdin replaces the file with a
# rename, so the agent never reads half a file, and anything that isn't a JSON
# object is skipped. It has no double quotes, so it crosses the Windows
# command line intact; IFS= and set -f stand in for quoting $line.
$writerScript = @'
set -f; IFS=; mkdir -p ~/.orchard && cd ~/.orchard || exit 1; while read -r line; do case $line in '{'*) printf '%s\n' $line > host-status.json.$$ && mv -f host-status.json.$$ host-status.json;; esac; done
'@

function Start-Writer {
    # The writer runs until this script exits, which also keeps the distro up:
    # WSL shuts a distro down once nothing runs in it.
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $wsl
    $info.Arguments = "-d $Distro --cd ~ --exec sh -c `"$writerScript`""
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $p = [System.Diagnostics.Process]::Start($info)
    $p.StandardInput.NewLine = "`n"
    $p.StandardInput.AutoFlush = $true
    # Takes any byte-order mark .NET writes ahead of the first line.
    $p.StandardInput.WriteLine()
    $p
}

function Test-SignedIn {
    # sihost runs for every signed-in user, screen locked or not, and explorer
    # is the desktop; neither runs at the sign-in screen. Any error counts as
    # signed in, so a wrong answer holds jobs instead of running them over
    # someone.
    try {
        foreach ($name in 'sihost', 'explorer') {
            foreach ($p in [System.Diagnostics.Process]::GetProcessesByName($name)) {
                if ($p.SessionId -ne 0) { return $true }
            }
        }
        $false
    } catch {
        $true
    }
}

function Write-SessionFile([string]$Json) {
    # Write then swap, so the startup task never reads half a file.
    $tmp = "$SessionFile.tmp"
    [System.IO.File]::WriteAllText($tmp, $Json)
    if (Test-Path -LiteralPath $SessionFile) {
        [System.IO.File]::Replace($tmp, $SessionFile, [NullString]::Value)
    } else {
        [System.IO.File]::Move($tmp, $SessionFile)
    }
}

function Read-SessionStatus {
    # The sign-in task's last sample, or $null when it is missing, malformed
    # or a few intervals old: that task stopped, or the sample is from before
    # a restart. Its own timestamp is checked, not the file's.
    try {
        $text = [System.IO.File]::ReadAllText($SessionFile).Trim()
        $age = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - ($text | ConvertFrom-Json).ts
        if ($text.StartsWith('{') -and $text -notmatch "`n" -and
            $age -ge -5 -and $age -le 3 * $IntervalSeconds) {
            return $text
        }
    } catch {
    }
    $null
}

function Get-Status([double]$Idle, [string]$Source) {
    [ordered]@{
        ts           = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        idle_seconds = [math]::Round($Idle, 1)
        power        = [OrchardHost]::Power()
        battery_pct  = [OrchardHost]::BatteryPercent()
        disk_free_gb = [math]::Round((New-Object System.IO.DriveInfo $Drive).AvailableFreeSpace / 1e9, 2)
        source       = $Source
    } | ConvertTo-Json -Compress
}

$writer = $null
$lastStart = $null
$signedOutFor = $null
while ($true) {
    try {
        if (-not $Startup) {
            $idle = [OrchardHost]::IdleSeconds()
            $pad = [OrchardHost]::ControllerIdleSeconds()
            if ($pad -ge 0 -and $pad -lt $idle) { $idle = $pad }
            # Unknown counts as in use, and so does someone else having the screen.
            if ($idle -lt 0 -or -not [OrchardHost]::InConsoleSession()) { $idle = 0 }
            Write-SessionFile (Get-Status $idle 'windows')
        } else {
            # Restart a writer that died, say after wsl --shutdown, on the next
            # pass: the file goes stale after 60 s and jobs pause. The pause
            # between passes keeps a failing wsl.exe from spinning.
            if (($null -eq $writer -or $writer.HasExited) -and
                ($null -eq $lastStart -or $lastStart.Elapsed.TotalSeconds -ge 5)) {
                $lastStart = [System.Diagnostics.Stopwatch]::StartNew()
                $writer = Start-Writer
            }
            $json = $null
            if (-not (Test-SignedIn)) {
                # A stopwatch, not the clock, which can jump when Windows sets it after boot.
                if ($null -eq $signedOutFor) { $signedOutFor = [System.Diagnostics.Stopwatch]::StartNew() }
                $json = Get-Status $signedOutFor.Elapsed.TotalSeconds 'windows-signed-out'
            } elseif ($null -ne $signedOutFor) {
                # Someone just signed in. Mark the PC in use, so no job starts
                # before their sign-in task reports.
                $signedOutFor = $null
                $json = Get-Status 0 'windows-signed-out'
            } else {
                # Nothing fresh to pass on, from you or because someone else
                # signed in: write nothing, and the file goes stale.
                $json = Read-SessionStatus
            }
            if ($json) { $writer.StandardInput.WriteLine($json) }
        }
    } catch {
        # A skipped write is safe: the agent reads a stale file as "in use".
    }
    Start-Sleep -Seconds $IntervalSeconds
}
