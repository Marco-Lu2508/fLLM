<#
Verification run of the Google TPU backend (cargo feature `tpu-pjrt`) on a TPU VM. See docs/backends-verify.md, section TPU. There is no
Windows build of libtpu: this script exists for completeness (pwsh on a Linux TPU VM); on Colab use scripts/verify-tpu.sh.
Needs the pip package `libtpu` (or FLLM_TPU_LIBRARY) at run time, not at build time. A TPU can be opened by one process only.

  pwsh scripts/verify-tpu.ps1 -ModelUri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf

When scripts/colab/tpu_check.py exists, it does build, device list, self test and the greedy comparison TPU against CPU; this
script copies its files to the output directory and adds the timed requests, fllm-rs bench and the JSON report.
Level of this script: L0.
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'verify-common.ps1')

if (-not $env:FLLM_TPU_LIBRARY) {
    $python = Get-Command python3, python -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($python) {
        $lib = (& $python.Source -c 'import libtpu; print(libtpu.get_library_path())' 2>$null)
        if ($lib -and (Test-Path $lib)) { $env:FLLM_TPU_LIBRARY = "$lib".Trim() }
    }
}
$phases = @('build', 'devices', 'pre', 'selftest', 'text', 'bench')
if (Test-Path (Join-Path $here 'colab/tpu_check.py')) { $phases = @('pre', 'bench') }

$config = @{
    Backend = 'tpu'; Features = 'tpu-pjrt'; Device = 'tpu'; CommonEnv = @('FLLM_TPU_EXACT=1'); Phases = $phases
    Pre = {
        $checker = Join-Path $script:V.Root 'scripts/colab/tpu_check.py'
        $python = Get-Command python3, python -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not (Test-Path $checker) -or -not $python) { Skip-Step 'tpu-check' 'scripts/colab/tpu_check.py or python not found'; return }
        $arguments = @($checker, '--report', (Join-Path $script:V.Out 'tpu_check_report.json'), '--port', "$($script:V.Port)", '--tokens', "$($script:V.Tokens)", '--prompt', $script:V.Prompt)
        if ($script:V.BinGiven) { $arguments += @('--skip', 'build', '--bin', $script:V.Bin) }
        if ($script:V.Model) { $arguments += @('--model', $script:V.Model) } elseif ($script:V.ModelUri) { $arguments += @('--model-uri', $script:V.ModelUri) }
        [void](Invoke-Step 'tpu-check' $python.Source $arguments)
        if (-not $script:V.DryRun) {
            Get-ChildItem -Path (Join-Path $script:V.Root 'scripts/colab') -Include 'tpu_selftest.json', 'tpu_check_*.log' -ErrorAction SilentlyContinue | Copy-Item -Destination $script:V.Out
        }
    }
}
Invoke-Verify -Config $config -Arguments $Rest
