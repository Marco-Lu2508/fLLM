<#
Verification run of the ARM CPU kernels (NEON, dotprod, i8mm, SVE) on an AArch64 machine with pwsh (macOS, Linux, Windows on ARM). The
bash version scripts/verify-arm.sh is the primary one; see docs/backends-verify.md, section ARM. There is no GPU here: the "device" is the
CPU, the reference is the lowest tier (FLLM_CPU_ISA=neon), and every tier the CPU supports is compared with it.

  pwsh scripts/verify-arm.ps1 -Model /models/llama-q4_k_m.gguf
  $env:V_ARM_QUICK = '1'; pwsh scripts/verify-arm.ps1 -Skip bench

check-arm-cpu.sh (needs bash) builds, runs the CPU unit tests and selftest-cpu for every tier. aarch64-pc-windows-msvc was
not built. The SVE tier is opt-in (FLLM_ARM_SVE=1) and needs vectors of 256 bit or more.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

$config = @{
    Backend = 'arm'; Features = ''; Device = 'cpu'; RefEnv = @('FLLM_CPU_ISA=neon'); Phases = @('build', 'devices', 'pre', 'text', 'bench')
    Pre = {
        $script = Join-Path $script:V.Root 'scripts/check-arm-cpu.sh'
        $bash = Get-Command bash -ErrorAction SilentlyContinue
        if (-not (Test-Path $script)) { Skip-Step 'check-arm-cpu' 'scripts/check-arm-cpu.sh is not in this tree'; return }
        if (-not $bash) { Skip-Step 'check-arm-cpu' 'bash not found'; return }
        $arguments = @($script)
        if ($env:V_ARM_QUICK -eq '1') { $arguments += '--quick' }
        [void](Invoke-Step 'check-arm-cpu' $bash.Source $arguments)
    }
    CaseHook = {
        if (-not (Test-Path $script:V.Bin)) { return }
        $line = (& $script:V.Bin selftest-cpu 2>&1 | Select-String -Pattern '^supported tiers: (.*)$' | Select-Object -First 1)
        if (-not $line) { return }
        foreach ($tier in ($line.Matches[0].Groups[1].Value -split '\s+' | Where-Object { $_ })) {
            switch ($tier) {
                { $_ -in 'scalar', 'neon' } { }
                'neon-sve' { "tier-$tier|cpu|FLLM_CPU_ISA=$tier FLLM_ARM_SVE=1" }
                default { "tier-$tier|cpu|FLLM_CPU_ISA=$tier" }
            }
        }
    }
}
Invoke-Verify -Config $config -Arguments $Rest
