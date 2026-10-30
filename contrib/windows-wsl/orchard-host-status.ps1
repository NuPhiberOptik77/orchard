<#
.SYNOPSIS
Tells the Orchard agent inside WSL when this PC is in use, on battery or short
of disk, and keeps the WSL distro running.

.DESCRIPTION
Linux inside WSL can't see the Windows keyboard, mouse or charger, so this
script writes them to ~/.orchard/host-status.json inside the distro every few
seconds. The agent fails closed: when that file is missing, unreadable or more
than 60 s old, it holds new jobs and pauses running ones.

Start it from the sign-in task that register-task.ps1 creates. It must run in
your signed-in session: run as a service or as SYSTEM, it sees no keyboard or
mouse at all and the PC would always look idle.
#>
param(
    [string]$Distro = 'Ubuntu-24.04',
    [int]$IntervalSeconds = 10,
    [string]$Drive = 'C'
)

$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
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

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern bool MoveFileEx(string from, string to, int flags);

    // Seconds since the last keyboard or mouse input in this session; -1 if unknown.
    public static double IdleSeconds() {
        LASTINPUTINFO info = new LASTINPUTINFO();
        info.cbSize = (uint)Marshal.SizeOf(info);
        if (!GetLastInputInfo(ref info)) return -1;
        // Both are 32-bit millisecond tick counts; unsigned subtraction survives the 49.7-day wrap.
        return unchecked((uint)Environment.TickCount - info.dwTime) / 1000.0;
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

    // Replaces the target in one step, so the agent never reads half a file.
    public static void Replace(string from, string to) {
        if (!MoveFileEx(from, to, 1)) throw new Win32Exception();
    }
}
"@

function Start-KeepAlive {
    # WSL shuts a distro down once nothing runs in it; this never exits.
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = 'wsl.exe'
    $info.Arguments = "-d $Distro --cd ~ --exec sleep infinity"
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    [System.Diagnostics.Process]::Start($info)
}

function Get-StatusPath {
    $linuxHome = (& wsl.exe -d $Distro --cd '~' --exec sh -c 'echo $HOME' | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxHome.StartsWith('/')) {
        throw "could not read the Linux home folder in $Distro"
    }
    $dir = "\\wsl.localhost\$Distro" + ($linuxHome -replace '/', '\') + '\.orchard'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Join-Path $dir 'host-status.json'
}

$keepAlive = $null
$statusPath = $null
while ($true) {
    try {
        if ($null -eq $keepAlive -or $keepAlive.HasExited) { $keepAlive = Start-KeepAlive }
        if (-not $statusPath) { $statusPath = Get-StatusPath }
        $status = [ordered]@{
            ts           = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
            idle_seconds = [math]::Round([OrchardHost]::IdleSeconds(), 1)
            power        = [OrchardHost]::Power()
            battery_pct  = [OrchardHost]::BatteryPercent()
            disk_free_gb = [math]::Round((New-Object System.IO.DriveInfo $Drive).AvailableFreeSpace / 1e9, 2)
            source       = 'windows'
        }
        $json = $status | ConvertTo-Json -Compress
        $tmp = "$statusPath.tmp"
        [System.IO.File]::WriteAllText($tmp, $json)
        try {
            [OrchardHost]::Replace($tmp, $statusPath)
        } catch {
            # Some shares refuse a replacing rename. A torn read then only
            # fails closed for one sample.
            [System.IO.File]::WriteAllText($statusPath, $json)
        }
    } catch {
        # A skipped write is safe: the agent reads a stale file as "in use".
        # Look the path up again in case the distro was reinstalled.
        $statusPath = $null
    }
    Start-Sleep -Seconds $IntervalSeconds
}
