<#
Collects what is needed to judge fLLM on a Windows machine: devices, CPU and GPU self tests and, with -Model, a short speed
measurement per device. Everything goes to fllm-report.txt (next to this script); send that file back.

    .\amd-check.ps1                                  # devices and self tests
    .\amd-check.ps1 -Model C:\models\model.gguf      # plus decode and prompt speed on cpu and vulkan

Run it from the folder that contains fllm-rs.exe (the unpacked zip). Nothing is installed and nothing leaves the machine.
#>
param(
    [string]$Model = "",
    [int]$Port = 8199
)
$ErrorActionPreference = "Continue"
$exe = Join-Path $PSScriptRoot "fllm-rs.exe"
if (-not (Test-Path $exe)) { $exe = Join-Path (Get-Location) "fllm-rs.exe" }
if (-not (Test-Path $exe)) { Write-Host "fllm-rs.exe not found: run this script from the folder of the unpacked zip."; exit 1 }
$report = Join-Path (Split-Path $exe) "fllm-report.txt"
"fLLM check $(Get-Date -Format s)" | Out-File $report -Encoding utf8

function Section($title, [scriptblock]$body) {
    "`n===== $title" | Out-File $report -Append -Encoding utf8
    Write-Host "== $title"
    try { & $body 2>&1 | Out-String -Width 220 | Out-File $report -Append -Encoding utf8 }
    catch { "ERROR: $_" | Out-File $report -Append -Encoding utf8 }
}

Section "system" {
    Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List
    Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | Format-List
    Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | Format-List
    [System.Environment]::OSVersion.VersionString
}
Section "version" { & $exe --version }
Section "devices" { & $exe devices }
Section "selftest cpu" { & $exe selftest-cpu }
Section "selftest gpu vulkan" { & $exe selftest-gpu --device vulkan }
Section "selftest gpu dx12" { & $exe selftest-gpu --device dx12 }

if ($Model -ne "") {
    foreach ($device in @("cpu", "vulkan")) {
        Section "speed on $device" {
            $log = Join-Path (Split-Path $exe) "fllm-server-$device.log"
            $p = Start-Process -FilePath $exe -ArgumentList @("serve", "--model", $Model, "--device", $device, "--port", $Port, "--model-name", "m", "--ctx", "4096") `
                -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
            $ready = $false
            for ($i = 0; $i -lt 240; $i++) {
                Start-Sleep -Seconds 1
                try { Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2 | Out-Null; $ready = $true; break } catch { }
                if ($p.HasExited) { break }
            }
            if ($ready) {
                & $exe bench --url "http://127.0.0.1:$Port" --model m --scenarios prefill,decode-single --prompt-lens 256 --gen-len 64 --input-len 64 --repeats 2 --warmup 1 --timeout 600 --label $device
            } else {
                "server on $device did not start; log tail:"
                Get-Content "$log.err" -Tail 25
                Get-Content $log -Tail 25
            }
            if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
            Start-Sleep -Seconds 3
        }
    }
}
Write-Host "Done. Send this file: $report"
