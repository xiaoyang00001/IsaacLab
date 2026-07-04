#requires -Version 5.1

<#
.SYNOPSIS
Report TCP/UDP endpoints owned by IsaacLab-related processes on Windows.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\tools\check_isaaclab_ports.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\tools\check_isaaclab_ports.ps1 -Port 5557,5560

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\tools\check_isaaclab_ports.ps1 -Pattern isaaclab,sonic,python -AllConnections
#>

[CmdletBinding()]
param(
    # Case-insensitive substrings matched against process name, executable path,
    # and command line. When omitted, common IsaacLab/Isaac Sim/Sonic patterns
    # plus this checkout path are used.
    [string[]]$Pattern = @(),

    # Match either local or remote ports. Useful for ZMQ pairs such as 5557/5560.
    [ValidateRange(1, 65535)]
    [int[]]$Port = @(),

    # Only show listening TCP sockets and bound UDP endpoints.
    [switch]$ListenOnly,

    # Do not restrict rows to matched IsaacLab processes. Useful when checking
    # who owns a specific port.
    [switch]$AllConnections,

    # Emit machine-readable output.
    [switch]$Json,

    # Print the full list of processes matched by -Pattern. By default the
    # process list is only printed when no matching endpoint is found.
    [switch]$ShowProcesses
)

$ErrorActionPreference = "Stop"

function Test-ContainsAny {
    param(
        [AllowNull()]
        [string]$Text,

        [string[]]$Needles
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $false
    }

    foreach ($needle in $Needles) {
        if ([string]::IsNullOrWhiteSpace($needle)) {
            continue
        }

        if ($Text.IndexOf($needle, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $true
        }
    }

    return $false
}

function Get-ShortText {
    param(
        [AllowNull()]
        [string]$Text,

        [int]$MaxLength = 160
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $singleLine = ($Text -replace "\s+", " ").Trim()
    if ($singleLine.Length -le $MaxLength) {
        return $singleLine
    }

    return $singleLine.Substring(0, $MaxLength - 3) + "..."
}

function Test-PortMatch {
    param(
        [int]$LocalPort,

        [AllowNull()]
        [object]$RemotePort,

        [int[]]$WantedPorts
    )

    if ($WantedPorts.Count -eq 0) {
        return $true
    }

    return $WantedPorts -contains $LocalPort -or $WantedPorts -contains $RemotePort
}

function New-PortRow {
    param(
        [string]$Protocol,
        [int]$OwnerPid,
        [string]$LocalAddress,
        [int]$LocalPort,
        [AllowNull()]
        [string]$RemoteAddress,
        [AllowNull()]
        [object]$RemotePort,
        [string]$State,
        [hashtable]$ProcessMap,
        [hashtable]$MatchedProcessMap
    )

    $process = $ProcessMap[$OwnerPid]
    $processName = ""
    $path = ""
    $commandLine = ""

    if ($null -ne $process) {
        $processName = $process.Name
        $path = $process.ExecutablePath
        $commandLine = $process.CommandLine
    }

    $remote = ""
    if (
        -not [string]::IsNullOrWhiteSpace($RemoteAddress) -and
        $null -ne $RemotePort -and
        [int]$RemotePort -ne 0
    ) {
        $remote = "${RemoteAddress}:${RemotePort}"
    }

    [pscustomobject]@{
        PID = $OwnerPid
        Process = $processName
        Protocol = $Protocol
        Local = "${LocalAddress}:${LocalPort}"
        Remote = $remote
        State = $State
        LocalPort = $LocalPort
        RemotePort = $RemotePort
        IsaacLabMatch = $MatchedProcessMap.ContainsKey($OwnerPid)
        Path = $path
        Command = Get-ShortText -Text $commandLine
    }
}

if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) {
    throw "Get-NetTCPConnection is unavailable. Run this script on Windows with the NetTCPIP module."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = ""
try {
    $repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path
} catch {
    $repoRoot = ""
}

$defaultPatterns = @(
    "isaaclab",
    "isaacsim",
    "isaac-sim",
    "_isaac_sim",
    "omni.kit",
    "kit.exe",
    "sonic"
)

if (-not [string]::IsNullOrWhiteSpace($repoRoot)) {
    $defaultPatterns += $repoRoot
}

$activePatterns = $Pattern
if ($activePatterns.Count -eq 0) {
    $activePatterns = $defaultPatterns
}

$currentProcessId = [System.Diagnostics.Process]::GetCurrentProcess().Id
$allProcesses = @(Get-CimInstance Win32_Process)
$processMap = @{}
foreach ($process in $allProcesses) {
    $processMap[[int]$process.ProcessId] = $process
}

$matchedProcesses = @(
    $allProcesses | Where-Object {
        $candidateProcessId = [int]$_.ProcessId
        if ($candidateProcessId -eq $currentProcessId) {
            return $false
        }

        $haystack = @($_.Name, $_.ExecutablePath, $_.CommandLine) -join "`n"
        Test-ContainsAny -Text $haystack -Needles $activePatterns
    } | Sort-Object ProcessId
)

$matchedProcessMap = @{}
foreach ($process in $matchedProcesses) {
    $matchedProcessMap[[int]$process.ProcessId] = $process
}

$rows = New-Object System.Collections.Generic.List[object]

$tcpConnections = @(Get-NetTCPConnection -ErrorAction SilentlyContinue)
foreach ($connection in $tcpConnections) {
    $ownerPid = [int]$connection.OwningProcess
    if (-not $AllConnections -and -not $matchedProcessMap.ContainsKey($ownerPid)) {
        continue
    }

    if ($ListenOnly -and [string]$connection.State -ne "Listen") {
        continue
    }

    if (-not (Test-PortMatch -LocalPort $connection.LocalPort -RemotePort $connection.RemotePort -WantedPorts $Port)) {
        continue
    }

    $rows.Add((New-PortRow `
        -Protocol "TCP" `
        -OwnerPid $ownerPid `
        -LocalAddress $connection.LocalAddress `
        -LocalPort $connection.LocalPort `
        -RemoteAddress $connection.RemoteAddress `
        -RemotePort $connection.RemotePort `
        -State ([string]$connection.State) `
        -ProcessMap $processMap `
        -MatchedProcessMap $matchedProcessMap))
}

$udpCommand = Get-Command Get-NetUDPEndpoint -ErrorAction SilentlyContinue
if ($null -ne $udpCommand) {
    $udpEndpoints = @(Get-NetUDPEndpoint -ErrorAction SilentlyContinue)
    foreach ($endpoint in $udpEndpoints) {
        $ownerPid = [int]$endpoint.OwningProcess
        if (-not $AllConnections -and -not $matchedProcessMap.ContainsKey($ownerPid)) {
            continue
        }

        if (-not (Test-PortMatch -LocalPort $endpoint.LocalPort -RemotePort $null -WantedPorts $Port)) {
            continue
        }

        $rows.Add((New-PortRow `
            -Protocol "UDP" `
            -OwnerPid $ownerPid `
            -LocalAddress $endpoint.LocalAddress `
            -LocalPort $endpoint.LocalPort `
            -RemoteAddress $null `
            -RemotePort $null `
            -State "Bound" `
            -ProcessMap $processMap `
            -MatchedProcessMap $matchedProcessMap))
    }
}

$sortedRows = @($rows | Sort-Object Protocol, LocalPort, RemotePort, PID)

if ($Json) {
    [pscustomobject]@{
        Patterns = $activePatterns
        Ports = $Port
        ListenOnly = [bool]$ListenOnly
        AllConnections = [bool]$AllConnections
        ShowProcesses = [bool]$ShowProcesses
        MatchedProcesses = @(
            $matchedProcesses | Select-Object `
                ProcessId,
                Name,
                ExecutablePath,
                @{Name = "Command"; Expression = { Get-ShortText -Text $_.CommandLine -MaxLength 240 }}
        )
        Connections = $sortedRows
    } | ConvertTo-Json -Depth 5
    exit 0
}

Write-Host "[isaaclab-ports] Patterns: $($activePatterns -join ', ')"
if ($Port.Count -gt 0) {
    Write-Host "[isaaclab-ports] Port filter: $($Port -join ', ')"
}
Write-Host "[isaaclab-ports] Matched processes: $($matchedProcesses.Count)"

if (($ShowProcesses -or $sortedRows.Count -eq 0) -and $matchedProcesses.Count -gt 0) {
    $matchedProcesses |
        Select-Object `
            ProcessId,
            Name,
            @{Name = "Path"; Expression = { Get-ShortText -Text $_.ExecutablePath -MaxLength 100 }},
            @{Name = "Command"; Expression = { Get-ShortText -Text $_.CommandLine -MaxLength 140 }} |
        Format-Table -AutoSize -Wrap
}

if ($sortedRows.Count -eq 0) {
    Write-Host "[isaaclab-ports] No matching TCP/UDP endpoints found."
    if (-not $AllConnections -and $Port.Count -gt 0) {
        Write-Host "[isaaclab-ports] Add -AllConnections to see non-IsaacLab owners of the requested ports."
    }
    exit 0
}

$sortedRows |
    Select-Object PID, Process, Protocol, Local, Remote, State, IsaacLabMatch, Command |
    Format-Table -AutoSize -Wrap
