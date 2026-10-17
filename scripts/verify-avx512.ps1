<#
Verification run of the x86 extension kernels (cargo feature `avx512`: AVX-512 VNNI/BF16 and AMX) on an x86-64 machine with AVX-512 (Zen 4/5,
Ice Lake or newer) and, for the AMX paths, a Sapphire Rapids or newer Xeon with Linux 5.16 or newer (AMX is Linux only). The bash version
scripts/verify-avx512.sh is the primary one; see docs/backends-verify.md, section x86. The "device" is the CPU, the reference is the AVX2 tier
(FLLM_CPU_ISA=avx2), and every tier the CPU supports is compared with it.

  .\scripts\verify-avx512.ps1 -Model C:\models\llama-q4_k_m.gguf

check_x86_ext.sh (needs bash) builds with --features avx512, runs the unit tests, selftest-cpu for every tier and for the
policy overrides.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

$config = @{
    Backend = 'avx512'; Features = 'avx512'; Device = 'cpu'; RefEnv = @('FLLM_CPU_ISA=avx2'); Phases = @('build', 'devices', 'pre', 'text', 'bench')
    ExtraCases = @('wide-float||FLLM_X86_WIDE_FLOAT=1', 'no-wide-float||FLLM_X86_WIDE_FLOAT=0', 'no-bf16||FLLM_X86_BF16=0', 'vnni-gemm-all||FLLM_VNNI_GEMM=all', 'amx-from-1-token||FLLM_AMX_MIN_BATCH=1')
    Pre = {
        $script = Join-Path $script:V.Root 'scripts/check_x86_ext.sh'
        $bash = Get-Command bash -ErrorAction SilentlyContinue
        if (-not (Test-Path $script)) { Skip-Step 'check-x86-ext' 'scripts/check_x86_ext.sh is not in this tree'; return }
        if (-not $bash) { Skip-Step 'check-x86-ext' 'bash not found (Git Bash or WSL)'; return }
        [void](Invoke-Step 'check-x86-ext' $bash.Source @($script))
    }
    CaseHook = {
        if (-not (Test-Path $script:V.Bin)) { return }
        $line = (& $script:V.Bin selftest-cpu 2>&1 | Select-String -Pattern '^supported tiers: (.*)$' | Select-Object -First 1)
        if (-not $line) { return }
        foreach ($tier in ($line.Matches[0].Groups[1].Value -split '\s+' | Where-Object { $_ })) {
            if ($tier -notin 'scalar', 'avx2') { "tier-$tier|cpu|FLLM_CPU_ISA=$tier" }
        }
    }
}
Invoke-Verify -Config $config -Arguments $Rest
