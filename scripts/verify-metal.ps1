<#
Verification run of the native Metal backend (cargo feature `metal`) on an Apple-silicon Mac, with PowerShell 7 (pwsh). The bash version
scripts/verify-metal.sh is the primary one; see docs/backends-verify.md, section Mac.

  pwsh scripts/verify-metal.ps1 -ModelUri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf

Order: metal_core_smoke, the quantized kernels with all libraries compiled at start, the ops kernels, devices,
selftest-gpu, text check against the CPU, benchmark. On a failure the self test and the text are repeated once per switch.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

$device = 'metal:0'
if (Select-String -Path (Join-Path $here '../rust/engine/src/devices.rs') -Pattern 'metal-native' -Quiet -ErrorAction SilentlyContinue) { $device = 'metal-native:0' }
else { Write-Host "[verify-metal] note: this tree has no 'metal-native' device name ; a fallback to wgpu would not fail the run, read the server logs" }

$config = @{
    Backend = 'metal'; Features = 'metal'; Device = $device; ExactEnv = @('FLLM_METAL_EXACT=1'); NativeGuard = 'wgpu'
    DiagEnvs = @('FLLM_METAL_SERIAL=1', 'FLLM_METAL_FAST_MATH=0', 'FLLM_METAL_EXACT=1', 'FLLM_METAL_NO_NOCOPY=1', 'FLLM_METAL_SINGLE_LIB=1', 'FLLM_METAL_MSL=3.0', 'FLLM_METAL_MSL=3.1')
    ExtraCases = @('no-nocopy||FLLM_METAL_NO_NOCOPY=1')
    Pre = {
        [void](Invoke-Step 'metal-core-smoke' 'cargo' @('test', '--release', '-p', 'fllm-engine', '--features', 'metal', 'metal_core_smoke', '--', '--nocapture'))
        [void](Invoke-Step 'metal-quant' 'cargo' @('test', '--release', '-p', 'fllm-engine', '--features', 'metal', 'metal_quant_kernels', '--', '--nocapture') @('FLLM_METAL_PREWARM=1', 'FLLM_METAL_DEBUG=1'))
        [void](Invoke-Step 'metal-ops' 'cargo' @('test', '--release', '-p', 'fllm-engine', '--features', 'metal', 'metal_ops_kernels', '--', '--nocapture'))
    }
}
Invoke-Verify -Config $config -Arguments $Rest
