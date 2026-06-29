#requires -Version 5.1

<#
Start Windows IsaacLab for the SONIC deploy bridge.

Ubuntu side should run deploy/proxy/input with:

    scripts/launch_sonic_local_isaaclab_closed_loop.py --replace --no-isaaclab --windows-ip <WINDOWS_IP>

Windows side should run this script from PowerShell. By default, IsaacLabRoot is
the current directory, so the simplest usage is:

    cd D:\path\to\IsaacLab
    powershell -ExecutionPolicy Bypass -File D:\path\to\GR00T-WholeBodyControl\scripts\start_windows_isaaclab_sonic.ps1 -UbuntuIp <UBUNTU_IP>
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$UbuntuIp,

    [string]$WindowsIp = "",

    [string]$IsaacLabRoot = "",

    [string]$Task = "Isaac-SonicSolo-Locomanipulation-G1-v0",

    [string]$Device = "cpu",

    [int]$DebugPort = 5557,

    [int]$StatePort = 5560,

    [string]$DeployTopic = "g1_debug",

    [string]$StateTopic = "sonic_state",

    [ValidateSet(0, 1)]
    [int]$PhysicsMode = 1,

    [ValidateSet(0, 1)]
    [int]$VisualServoMode = 0,

    [ValidateSet(0, 1)]
    [int]$SelfCollisions = 0,

    [ValidateSet(0, 1)]
    [int]$StabilizeRoot = 1,

    [double]$TargetRateLimit = 0.04,

    [switch]$Headless,

    [switch]$EnablePinocchio,

    [string[]]$KitArg = @()
)

$ErrorActionPreference = "Stop"

function Get-DefaultWindowsIpv4 {
    $candidate = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.IPAddress -notlike "127.*" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.SkipAsSource -eq $false
        } |
        Sort-Object -Property InterfaceMetric |
        Select-Object -First 1

    if ($null -eq $candidate) {
        return ""
    }
    return $candidate.IPAddress
}

if ([string]::IsNullOrWhiteSpace($IsaacLabRoot)) {
    $IsaacLabRoot = (Get-Location).Path
}

if ([string]::IsNullOrWhiteSpace($WindowsIp)) {
    $WindowsIp = Get-DefaultWindowsIpv4
}

if ([string]::IsNullOrWhiteSpace($WindowsIp)) {
    throw "Could not auto-detect Windows IPv4. Pass -WindowsIp explicitly."
}

$IsaacLabBat = Join-Path $IsaacLabRoot "isaaclab.bat"
if (-not (Test-Path $IsaacLabBat)) {
    throw "isaaclab.bat not found under IsaacLabRoot: $IsaacLabRoot"
}

$env:ISAACLAB_MACHINE_A_IP = $UbuntuIp
$env:ISAACLAB_MACHINE_B_IP = $WindowsIp
$env:ISAACLAB_LOCAL_MACHINE_IP = $WindowsIp
$env:ISAACLAB_TRACKING_HUB_IP = $UbuntuIp

$env:SONIC_DEPLOY_TRANSPORT = "zmq"
$env:SONIC_DEPLOY_ENDPOINT = "tcp://${UbuntuIp}:${DebugPort}"
$env:SONIC_DEPLOY_TOPIC = $DeployTopic
$env:SONIC_DEPLOY_TARGET_FIELD = "last_action"
$env:SONIC_DEPLOY_REFERENCE_TARGET_FIELD = "body_q_target"

$env:SONIC_PUBLISH_STATE_ZMQ = "1"
$env:SONIC_STATE_ZMQ_BIND = "tcp://*:${StatePort}"
$env:SONIC_STATE_ZMQ_TOPIC = $StateTopic

$env:SONIC_G1_PHYSICS_MODE = "$PhysicsMode"
$env:SONIC_G1_VISUAL_SERVO_MODE = "$VisualServoMode"
$env:SONIC_G1_SELF_COLLISIONS = "$SelfCollisions"
$env:SONIC_DEPLOY_STABILIZE_ROOT = "$StabilizeRoot"
$env:SONIC_DEPLOY_TARGET_RATE_LIMIT = "$TargetRateLimit"

$kitArgs = @(
    "--/app/vsync=false",
    "--/app/runLoops/main/rateLimitEnabled=false"
) + $KitArg

$isaacArgs = @(
    "-p",
    "scripts\environments\teleoperation\teleop_se3_agent.py",
    "--task",
    $Task,
    "--device",
    $Device,
    "--kit_args",
    ($kitArgs -join " ")
)

if ($Headless) {
    $isaacArgs += "--headless"
}

if ($EnablePinocchio) {
    $isaacArgs += "--enable_pinocchio"
}

Write-Host "[sonic-windows-isaaclab] IsaacLabRoot: $IsaacLabRoot"
Write-Host "[sonic-windows-isaaclab] UbuntuIp: $UbuntuIp"
Write-Host "[sonic-windows-isaaclab] WindowsIp: $WindowsIp"
Write-Host "[sonic-windows-isaaclab] Deploy endpoint: $($env:SONIC_DEPLOY_ENDPOINT)"
Write-Host "[sonic-windows-isaaclab] State bind: $($env:SONIC_STATE_ZMQ_BIND), topic: $StateTopic"
Write-Host "[sonic-windows-isaaclab] Task: $Task, device: $Device"
Write-Host "[sonic-windows-isaaclab] Starting isaaclab.bat"

Set-Location $IsaacLabRoot
& $IsaacLabBat @isaacArgs
exit $LASTEXITCODE
