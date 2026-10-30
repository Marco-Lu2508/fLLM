<#
Collects what is needed to judge fLLM on a Windows machine: devices, CPU and GPU self tests and, with -Model, a short speed
measurement per device. Everything goes to fllm-report.txt (next to this script); send that file back.

    .\windows-check.ps1                              # devices and self tests
    .\windows-check.ps1 -Model C:\models\model.gguf  # plus decode and prompt speed on cpu and every GPU device found
    .\windows-check.ps1 -Model m.gguf -Devices cpu,cuda:0   # only these devices

Run it from the folder that contains fllm-rs.exe (the unpacked zip). Nothing is installed and nothing leaves the machine.
#>
param(
    [string]$Model = "",
    [int]$Port = 8199,
    [string[]]$Devices = @()
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
Section "nvidia-smi (NVIDIA driver, if present)" { if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi } else { "not found" } }
$inventory = & $exe devices 2>&1 | Out-String
Section "devices" { $inventory }
Section "amd-npu status" { & $exe amd-npu status }
Section "selftest cpu" { & $exe selftest-cpu }
# every native GPU backend that the inventory lists gets the GPU self test; the speed test below uses the same list
$gpu = @()
foreach ($m in [regex]::Matches($inventory, '(?m)Select:\s+--device ((cuda|hip|vulkan|metal)(:\d+)?)\s*$')) { $gpu += $m.Groups[1].Value }
$gpu = $gpu | Select-Object -Unique
if ($gpu.Count -eq 0) { $gpu = @("vulkan") }
foreach ($d in $gpu) { Section "selftest gpu $d" { & $exe selftest-gpu --device $d } }
Section "selftest gpu dx12 (wgpu, experimental)" { & $exe selftest-gpu --device dx12 }
$Devices = @($Devices | ForEach-Object { $_ -split ',' } | Where-Object { $_ })   # -File passes a,b as one string
if ($Devices.Count -eq 0) { $Devices = @("cpu") + $gpu }

if ($Model -ne "") {
    foreach ($device in $Devices) {
        Section "speed on $device" {
            $log = Join-Path (Split-Path $exe) ("fllm-server-" + ($device -replace "[^A-Za-z0-9]", "_") + ".log")
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
Write-Host "(it holds the hardware names, driver versions and speeds from above; no files of yours and no prompts)"
