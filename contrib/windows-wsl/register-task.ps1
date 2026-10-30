<#
.SYNOPSIS
Starts orchard-host-status.ps1 at startup and at every sign-in, with no window.

.DESCRIPTION
Run it once from an admin PowerShell, in the folder that holds both scripts:

    powershell -ExecutionPolicy Bypass -File .\register-task.ps1

It registers two tasks, and both run as you without admin rights. The startup
task runs from boot without storing your password. It starts the WSL distro
and writes the status file, whether you are signed in or not. The sign-in task
runs in your session, because Windows only reports keyboard, mouse and
controller input to the session that owns them, and hands those numbers to the
startup task. Windows only lets an admin register a task like the startup one.

Running it again replaces both tasks and restarts them. It restarts the WSL
service first, so the startup task is the first to start WSL again. WSL's VM
belongs to the Windows session that first started WSL since the service last
started, and stops when that session signs out; wsl --shutdown leaves it with
that session. That closes open distro windows and stops running jobs, which
the dispatcher runs again.

If your admin prompt runs as another account, pass -User with your own, and
restart Windows afterwards.
#>
param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$TaskName = 'Orchard WSL worker',
    [string]$User = "$env:USERDOMAIN\$env:USERNAME"
)

$ErrorActionPreference = 'Stop'

$script = Join-Path $PSScriptRoot 'orchard-host-status.ps1'
if (-not (Test-Path $script)) {
    throw "orchard-host-status.ps1 must be in the same folder as this script"
}

# Check before stopping anything, so a normal prompt can't leave no writer.
$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this from an admin PowerShell: only an admin can register the startup task.'
}

# An admin prompt opened with another account's password runs as that
# account, and the tasks would report someone else's input.
$self = "$env:USERDOMAIN\$env:USERNAME"
$console = (Get-CimInstance Win32_ComputerSystem).UserName
if (-not $PSBoundParameters.ContainsKey('User') -and $console -and $console -ne $User) {
    throw "This prompt runs as $User but $console is signed in. Run it again with -User '$console'."
}

$wsl = Join-Path $env:ProgramFiles 'WSL\wsl.exe'
if (-not (Test-Path $wsl)) { $wsl = 'wsl.exe' }

$startupName = "$TaskName (startup)"
$psArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -Distro $Distro"
# It has to keep reporting on battery: that is what pauses jobs when the charger is pulled.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

# A running copy would keep the old script, so stop both first.
foreach ($name in $startupName, $TaskName) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $name
    }
}
# Stopping a task can leave its PowerShell running, and that copy would keep
# writing the status file the old way.
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object CommandLine -like '*orchard-host-status.ps1*' |
    Invoke-CimMethod -MethodName Terminate -ErrorAction SilentlyContinue | Out-Null

# The VM stays with whichever session first started WSL since the WSL service
# last started, even across wsl --shutdown. Restarting the service hands it to
# the startup task, which starts WSL first below.
Write-Host 'Restarting WSL, so the startup task starts it. Open distro windows close,'
Write-Host 'and running jobs stop until the dispatcher runs them again.'
$restart = $false
try {
    Stop-Service -Name WSLService -Force
} catch {
    # An older WSL with no service of its own: only a restart hands the VM over.
    Write-Warning "Couldn't stop the WSL service: $($_.Exception.Message)"
    $restart = $true
    if ($User -eq $self) { & $wsl --shutdown }
}

function Register-Worker($Name, $Action, $Trigger, $Principal) {
    Register-ScheduledTask -TaskName $Name -Action $Action -Trigger $Trigger `
        -Principal $Principal -Settings $settings -Force | Out-Null
    Start-ScheduledTask -TaskName $Name
}

# The startup task goes first, so the distro it starts is the one you use.
# S4U runs it as you with no password stored. It has no desktop, so nothing
# can show on screen.
Register-Worker $startupName `
    (New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "$psArgs -Startup") `
    (New-ScheduledTaskTrigger -AtStartup) `
    (New-ScheduledTaskPrincipal -UserId $User -LogonType S4U -RunLevel Limited)

# conhost --headless runs PowerShell with no window at all, even when Windows
# Terminal is the default terminal and would ignore -WindowStyle Hidden.
Register-Worker $TaskName `
    (New-ScheduledTaskAction -Execute 'conhost.exe' -Argument "--headless powershell.exe $psArgs") `
    (New-ScheduledTaskTrigger -AtLogOn -User $User) `
    (New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited)

Start-Sleep -Seconds 5
$state = (Get-ScheduledTask -TaskName $startupName).State
if ($state -ne 'Running') {
    $result = (Get-ScheduledTaskInfo -TaskName $startupName).LastTaskResult
    Write-Warning "'$startupName' is $state (last result $result), so nothing writes the status file."
}
Write-Host "Registered and started '$startupName' and '$TaskName'."
if ($restart -or $User -ne $self) {
    Write-Host "Restart Windows, so the startup task starts $User's distro."
}
