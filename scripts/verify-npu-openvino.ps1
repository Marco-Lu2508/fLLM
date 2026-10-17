<#
Verification run of the Intel NPU backend over OpenVINO (cargo feature `npu-openvino`) on a Core Ultra machine with an NPU (Windows or Linux).
See docs/backends-verify.md, section Intel NPU. Needs the NPU driver and OpenVINO 2025 or newer with the NPU plugin; the C library is found
through FLLM_OPENVINO_C_LIB, FLLM_OPENVINO_DIR, INTEL_OPENVINO_DIR, PATH or LD_LIBRARY_PATH.

  .\scripts\verify-npu-openvino.ps1 -Model C:\models\llama-q4_0.gguf
  $env:V_NPU_PROBES = 'xdna hexagon ane'; .\scripts\verify-npu-openvino.ps1 -Skip text,bench    # also the probes of the XDNA, Hexagon and ANE research notes

Expected: selftest-gpu reports the same 7 failed q8 attention checks as the CPU backend itself, nothing
else; the script therefore exits 1 on a healthy run. Only prompt-sized quantized_linear calls go to the NPU: the text case `npu-forced`
lowers the thresholds so that short prompts reach it too. Hybrid backend: dense models only.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

$forced = 'FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1'
$extra = @("npu-forced||$forced")
# the host hook (the `host_engine` function of compute.rs) is optional; without it FLLM_NPU_HOST=1 does nothing
if (Select-String -Path (Join-Path $here '../rust/engine/src/compute.rs') -Pattern 'host_engine' -Quiet -ErrorAction SilentlyContinue) { $extra += "cpu-npu-host|cpu|FLLM_NPU_HOST=1 $forced" }

$config = @{
    Backend = 'npu-openvino'; Features = 'npu-openvino'; Device = 'npu'; RefEnv = @('FLLM_NPU=0')
    SelftestEnv = @('FLLM_NPU_MIN_BATCH=1', 'FLLM_NPU_MIN_ELEMS=1'); ExtraCases = $extra
    Pre = {
        $python = Get-Command python3, python -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($python) { [void](Invoke-Step 'openvino-devices' $python.Source @('-c', 'import openvino as ov; print(ov.Core().available_devices)')) }
        else { Skip-Step 'openvino-devices' 'python not found' }
        # compares a 192x512 Q4_0 projection on the NPU with the host result (threshold 2 %)
        [void](Invoke-Step 'npu-hardware-selfcheck' 'cargo' @('test', '--release', '-p', 'fllm-engine', '--features', 'npu-openvino', '--lib', 'npu_hardware_selfcheck', '--', '--ignored', '--nocapture'))
        foreach ($probe in @($env:V_NPU_PROBES -split '\s+' | Where-Object { $_ })) {
            $feature = switch ($probe) { 'xdna' { 'npu-xdna' } 'hexagon' { 'npu-qnn' } 'ane' { 'npu-ane' } default { $null } }
            if (-not $feature) { Skip-Step "probe-$probe" 'unknown probe (xdna, hexagon, ane)'; continue }
            $base = @('run', '--release', '-p', 'fllm-engine', '--example', 'npu_probe', '--features', $feature, '--')
            [void](Invoke-Step "probe-$probe-list" 'cargo' ($base + @('list')))
            [void](Invoke-Step "probe-$probe-1x4096x4096" 'cargo' ($base + @($probe, '1', '4096', '4096')))
            [void](Invoke-Step "probe-$probe-128x4096x4096" 'cargo' ($base + @($probe, '128', '4096', '4096')))
        }
    }
}
Invoke-Verify -Config $config -Arguments $Rest
