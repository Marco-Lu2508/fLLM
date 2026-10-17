<#
.SYNOPSIS
    Installs, uninstalls, starts, stops, or inspects the fLLM Windows service.
.DESCRIPTION
    Manages the fLLM inference server as a Windows background service using sc.exe / PowerShell Service cmdlets.
#>
[CmdletBinding()]
param(
    [ValidateSet("Install", "Uninstall", "Start", "Stop", "Status")]
    [string]$Action = "Install",

    [string]$ServiceName = "fLLMServer",
    [string]$DisplayName = "fLLM Inference Server",
    [string]$BinaryPath = "$env:ProgramFiles\fllm\fllm-rs.exe",
    [string]$ConfigPath = "$env:ProgramData\fllm\fllm.toml",
    [string]$ExtraArgs = ""
)

$ErrorActionPreference = "Stop"

function Ensure-Admin {
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Administrator privileges are required to manage the $ServiceName Windows service."
    }
}

switch ($Action) {
    "Install" {
        Ensure-Admin
        if (-not (Test-Path -LiteralPath $BinaryPath)) {
            throw "fllm-rs binary not found at '$BinaryPath'. Pass -BinaryPath to specify its location."
        }
        $configDir = Split-Path -Parent $ConfigPath
        if ($configDir -and -not (Test-Path -LiteralPath $configDir)) {
            New-Item -ItemType Directory -Path $configDir -Force | Out-Null
        }
        $binCmd = "`"$BinaryPath`" serve --config `"$ConfigPath`""
        if ($ExtraArgs) {
            $binCmd = "$binCmd $ExtraArgs"
        }
        if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
            throw "Service '$ServiceName' is already installed. Run with -Action Uninstall first."
        }
        New-Service -Name $ServiceName `
                    -DisplayName $DisplayName `
                    -Description "fLLM local and network LLM inference HTTP server" `
                    -BinaryPathName $binCmd `
                    -StartupType Automatic | Out-Null
        & sc.exe failure $ServiceName reset= 86400 actions= restart/5000/restart/5000/restart/5000 | Out-Null
        Write-Output "Installed service '$ServiceName' -> $binCmd"
    }
    "Uninstall" {
        Ensure-Admin
        $svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        if (-not $svc) {
            Write-Output "Service '$ServiceName' is not installed."
            return
        }
        if ($svc.Status -ne "Stopped") {
            Stop-Service -Name $ServiceName -Force
        }
        & sc.exe delete $ServiceName | Out-Null
        Write-Output "Uninstalled service '$ServiceName'."
    }
    "Start" {
        Ensure-Admin
        Start-Service -Name $ServiceName
        Write-Output "Started service '$ServiceName'."
    }
    "Stop" {
        Ensure-Admin
        Stop-Service -Name $ServiceName -Force
        Write-Output "Stopped service '$ServiceName'."
    }
    "Status" {
        $svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        if (-not $svc) {
            Write-Output "Service '$ServiceName' is not installed."
        } else {
            $svc | Format-List Name, DisplayName, Status, StartType
        }
    }
}
