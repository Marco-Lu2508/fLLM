<#
Verification run of the native Vulkan backend (cargo feature `vulkan`) on a machine with a Vulkan GPU: AMD, NVIDIA, Intel; Windows,
Linux or macOS with pwsh. See docs/backends-verify.md. Needs the Vulkan SDK (glslc) at build time; on Windows run it from a
"x64 Native Tools" prompt or let scripts\build-windows.ps1 prepare the environment first.

  .\scripts\verify-vulkan.ps1 -ModelUri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf

Every run uses FLLM_VULKAN_VALIDATE=1. The self test runs twice: FLLM_VULKAN_EXACT=1 (scalar shaders only) and then the fast variants.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

$device = 'vulkan:0'
if (Select-String -Path (Join-Path $here '../rust/engine/src/devices.rs') -Pattern 'vulkan-native' -Quiet -ErrorAction SilentlyContinue) { $device = 'vulkan-native:0' }
else { Write-Host "[verify-vulkan] note: this tree has no 'vulkan-native' device name; a fallback to wgpu would not fail the run, read the server logs" }

$config = @{
    Backend = 'vulkan'; Features = 'vulkan'; Device = $device
    CommonEnv = @('FLLM_VULKAN_VALIDATE=1'); ExactEnv = @('FLLM_VULKAN_EXACT=1')
    DiagEnvs = @('FLLM_VULKAN_NO_COOP=1', 'FLLM_VULKAN_NO_IDOT=1'); SelftestTwice = $true; NativeGuard = 'wgpu'
    Pre = {
        $checker = Join-Path $script:V.Root 'rust/engine/shaders/vulkan/check_all.sh'
        $bash = Get-Command bash -ErrorAction SilentlyContinue
        if ((Test-Path $checker) -and $bash) { [void](Invoke-Step 'shader-check' $bash.Source @($checker)) }
        else { Skip-Step 'shader-check' 'check_all.sh needs bash, glslc and spirv-val (Git Bash or WSL)' }
        [void](Invoke-Step 'vk-smoke' 'cargo' @('test', '--release', '-p', 'fllm-engine', '--features', 'vulkan', '--lib', 'backend::vulkan::instance::smoke', '--', '--ignored', '--nocapture', '--test-threads=1') @('FLLM_VULKAN_VALIDATE=1'))
        [void](Invoke-Step 'vk-hw-tests' 'cargo' @('test', '--release', '-p', 'fllm-engine', '--features', 'vulkan', '--lib', 'backend::vulkan::backend::hw_tests', '--', '--ignored', '--nocapture', '--test-threads=1') @('FLLM_VULKAN_VALIDATE=1'))
    }
}
Invoke-Verify -Config $config -Arguments $Rest
