<#
Verification run of the AMD ROCm/HIP backend (cargo feature `hip`) on a machine with a ROCm or HIP SDK GPU (RDNA3, RDNA4, CDNA).
See docs/backends-verify.md, section AMD. Needs the HIP SDK (Windows: HIP_PATH) or ROCm (Linux: ROCM_PATH, 6.2 or newer).

  .\scripts\verify-hip.ps1 -ModelUri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
  $env:FLLM_HIP_ARCHS = 'gfx1201'; .\scripts\verify-hip.ps1 -Model C:\models\llama-q4_k_m.gguf

scripts/hip-verify.sh (bash) does the AMD kernel family on/off comparison; here the common phases run: self test, text check
with the switches FLLM_HIP_NO_MMA, FLLM_HIP_NO_SKINNY, FLLM_HIP_NO_Q4Z, FLLM_CUDA_NO_ATTN4 as diagnostics, benchmark.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

$archs = $env:FLLM_HIP_ARCHS
if (-not $archs) { $archs = 'gfx1100;gfx90a' }
$buildEnv = @('FLLM_HIP=1', "FLLM_HIP_ARCHS=$archs")
if ($env:ROCM_PATH) { $buildEnv += "ROCM_PATH=$($env:ROCM_PATH)" } elseif (-not $env:HIP_PATH -and -not (Test-Path 'C:\Program Files\AMD\ROCm')) { $buildEnv += 'ROCM_PATH=/opt/rocm' }

$config = @{
    Backend = 'hip'; Features = 'hip'; Device = 'hip:0'; BuildEnv = $buildEnv
    ExactEnv = @('FLLM_CUDA_EXACT=1')
    DiagEnvs = @('FLLM_HIP_NO_MMA=1', 'FLLM_HIP_NO_SKINNY=1', 'FLLM_HIP_NO_Q4Z=1', 'FLLM_CUDA_NO_ATTN4=1')
}
Invoke-Verify -Config $config -Arguments $Rest
