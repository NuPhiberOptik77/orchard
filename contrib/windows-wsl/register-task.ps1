<#
.SYNOPSIS
Starts orchard-host-status.ps1 at every sign-in, with no window.

.DESCRIPTION
Run it once from a normal PowerShell window, in the folder that holds both
scripts:

    powershell -ExecutionPolicy Bypass -File .\register-task.ps1

The task runs as you, in your signed-in session. It has to: Windows only
reports keyboard and mouse idle time to the session that owns them. If
Windows answers "Access is denied", run the same command from an admin
terminal; the task still runs as you, without admin rights.
#>
param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$TaskName = 'Orchard WSL worker'
)

$ErrorActionPreference = 'Stop'

$script = Join-Path $PSScriptRoot 'orchard-host-status.ps1'
if (-not (Test-Path $script)) {
    throw "orchard-host-status.ps1 must be in the same folder as this script"
}

$user = "$env:USERDOMAIN\$env:USERNAME"
# conhost --headless runs PowerShell with no window at all, even when Windows
# Terminal is the default terminal and would ignore -WindowStyle Hidden.
$action = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument (
    "--headless powershell.exe -NoProfile -ExecutionPolicy Bypass " +
    "-File `"$script`" -Distro $Distro")
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
# It has to keep reporting on battery: that is what pauses jobs when the charger is pulled.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Registered and started '$TaskName'."
