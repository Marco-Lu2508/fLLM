<#
Verification run of the Intel SYCL backend (cargo feature `sycl`) on a machine with an Intel GPU (Arc, Xe, Level Zero driver).
See docs/backends-verify.md, section Intel. Needs oneAPI DPC++ (icx; run setvars.bat first on Windows, setvars.sh on Linux).
The Windows branch of that build.rs function (icx, /std:c++17, llvm-lib) has never been run.

  .\scripts\verify-sycl.ps1 -Model C:\models\llama-q4_0.gguf
  $env:ONEAPI_DEVICE_SELECTOR = 'level_zero:0'; .\scripts\verify-sycl.ps1 -Skip build -Bin .\target\release\fllm-rs.exe

The self test (batch <= 5) reaches only the exact kernels; the tiled and XMX paths are checked by the unit test
`linear_matches_cpu_for_every_batch_path` in the pre phase.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

if (-not (Select-String -Path (Join-Path $here '../rust/engine/build.rs') -Pattern 'fn sycl_kernels' -Quiet -ErrorAction SilentlyContinue)) {
    Write-Host '[verify-sycl] warning: build.rs has no sycl_kernels(): the feature builds a stub. Apply rust/engine/sycl/build_rs.patch (git apply) first.'
}

$config = @{
    Backend = 'sycl'; Features = 'sycl'; Device = 'sycl:0'
    ExactEnv = @('FLLM_SYCL_EXACT=1'); DiagEnvs = @('FLLM_SYCL_EXACT=1')
    Pre = {
        [void](Invoke-Step 'sycl-linear-vs-cpu' 'cargo' @('test', '--release', '-p', 'fllm-engine', '--features', 'sycl', '--lib', 'linear_matches_cpu', '--', '--nocapture'))
    }
}
Invoke-Verify -Config $config -Arguments $Rest
