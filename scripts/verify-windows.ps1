<#
Verification run on native Windows (MSVC toolchain): build with scripts\build-windows.ps1, device list as JSON, selftest-gpu on the best
device (`auto`), greedy text against the CPU, benchmark. See docs/backends-verify.md, section Windows.

  .\scripts\verify-windows.ps1 -ModelUri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
  .\scripts\verify-windows.ps1 -Device cuda:0 -BuildArgs -Cuda,-CudaArchs,89          # NVIDIA, one architecture
  .\scripts\verify-windows.ps1 -Device hip:0  -BuildArgs -Hip                          # AMD HIP SDK
  .\scripts\verify-windows.ps1 -Device vulkan-native:0                                 # force the native Vulkan backend
  .\scripts\verify-windows.ps1 -Skip build -Bin D:\fllm\fllm-rs.exe                    # an unpacked release asset

-BuildArgs are passed to scripts\build-windows.ps1 (-Cuda, -CudaArchs "89;120", -Hip, -HipArchs, -Arm64, -NoVulkan, -DryRun ...). The binary
is expected at target\release\fllm-rs.exe (pass -Bin for -Arm64 builds, whose output sits under target\aarch64-pc-windows-msvc\release).
Nothing here has run on Windows with these toolchains: a build error should be reported with the full command line from the log.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

$config = @{
    Backend = 'windows'; Features = 'rust-gpu,vulkan'; Device = 'auto'; NativeGuard = 'wgpu'
    Phases = @('build', 'devices', 'pre', 'selftest', 'text', 'bench')
    BuildCommand = {
        $builder = Join-Path $script:V.Root 'scripts\build-windows.ps1'
        if (Test-Path $builder) {
            $shell = (Get-Process -Id $PID).Path
            $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $builder) + @($script:V.BuildArgs | Where-Object { $_ })
            [void](Invoke-Step 'build' $shell $arguments)
        } else {
            [void](Invoke-Step 'build' 'cargo' @('build', '--release', '-p', 'fllm-engine', '--features', $script:V.Features, '--bin', 'fllm-rs'))
        }
    }
    Pre = {
        if (-not $script:V.DryRun -and -not (Test-Path $script:V.Bin)) { Skip-Step 'devices-json' "no binary at $($script:V.Bin)"; return }
        # capabilities per device (FP16, BF16, Int8-Dot, matrix cores, memory) as JSON
        [void](Invoke-Step 'devices-json' $script:V.Bin @('devices', '--json'))
    }
}
Invoke-Verify -Config $config -Arguments $Rest
