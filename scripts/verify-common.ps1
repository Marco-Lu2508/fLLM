<#
Shared driver of scripts\verify-<backend>.ps1 (see docs/backends-verify.md). PowerShell 5.1 and 7; also runs under pwsh on macOS and Linux.

A wrapper builds a configuration hashtable and calls Invoke-Verify. Phases (all on by default, -Skip build,bench leaves some out):
  build     cargo build --release -p fllm-engine --features <Features> --bin fllm-rs   (or the wrapper's BuildCommand)
  devices   fllm-rs devices
  pre       backend specific steps (the wrapper's Pre script block)
  selftest  fllm-rs selftest-gpu --device <Device> --json ...   (exact math first and fast math second when SelftestTwice is set)
  text      greedy text (temperature 0) of the reference (cpu) against the device: default math, exact math, extra cases;
            when the default differs, one more case per switch in DiagEnvs
  bench     timed chat requests and `fllm-rs bench` against a server on the device
Options of every wrapper (also with a leading --): -Model FILE, -ModelUri URI, -Bin PATH, -Device NAME, -Out DIR, -Port N, -Tokens N,
-Prompt TEXT, -Skip a,b, -DryRun, -BuildArgs a,b (extra arguments of the wrapper's BuildCommand).
Output: -Out (default verify-out\<backend>) holds one log per step and report.json with device, commit, command, exit code, duration
and tokens/s. The exit status is 1 when a step failed; a text that differs from the CPU is reported, not counted as a failure.

Level of this script: L0. Parsed with the PowerShell parser, never run.
#>

$script:Records = New-Object System.Collections.ArrayList
$script:Failed = 0
$script:TokS = $null
$script:Server = $null
$script:ServerLog = $null
$script:LastSame = ''
$script:V = $null

function Write-Log([string]$Message) {
    Write-Host "[verify-$($script:V.Backend)] $Message"
}

function Test-WindowsHost { return ($env:OS -eq 'Windows_NT') }

function ConvertTo-ArgString([string[]]$Items) {
    $parts = foreach ($item in $Items) {
        if ($item -eq '') { '""' }
        elseif ($item -match '[\s"]') { '"' + $item.Replace('"', '\"') + '"' }
        else { $item }
    }
    return ($parts -join ' ')
}

function Use-Env([string[]]$Pairs) {
    $saved = @{}
    foreach ($pair in $Pairs) {
        if (-not $pair) { continue }
        $kv = $pair -split '=', 2
        $saved[$kv[0]] = [Environment]::GetEnvironmentVariable($kv[0])
        [Environment]::SetEnvironmentVariable($kv[0], $kv[1])
    }
    return $saved
}

function Restore-Env($Saved) {
    foreach ($key in @($Saved.Keys)) { [Environment]::SetEnvironmentVariable($key, $Saved[$key]) }
}

function Add-Record([string]$Name, [string]$Command, [int]$Exit, [double]$Seconds, [string]$Log, [string]$Note, $Extra) {
    $record = [ordered]@{ name = $Name; command = $Command; exit_code = $Exit; seconds = [math]::Round($Seconds, 1); log = $Log; note = $Note }
    if ($Extra) { foreach ($key in $Extra.Keys) { $record[$key] = $Extra[$key] } }
    [void]$script:Records.Add($record)
}

function Skip-Step([string]$Name, [string]$Reason) {
    Write-Log "skip ${Name}: $Reason"
    Add-Record $Name '' 0 0 '' "skipped: $Reason" @{ skipped = $true }
}

# Runs a program, waits, merges stdout and stderr into $Log, returns the exit code (-1 when it cannot be started).
function Invoke-Native([string]$File, [string[]]$Arguments, [string[]]$Env, [string]$Cwd, [string]$Log) {
    $out = "$Log.out"
    $err = "$Log.err"
    $rc = -1
    $saved = Use-Env $Env
    try {
        $params = @{ FilePath = $File; WorkingDirectory = $Cwd; RedirectStandardOutput = $out; RedirectStandardError = $err; PassThru = $true; NoNewWindow = $true }
        if ($Arguments -and $Arguments.Count -gt 0) { $params.ArgumentList = ConvertTo-ArgString $Arguments }
        $process = Start-Process @params
        $null = $process.Handle
        $process.WaitForExit()
        $rc = [int]$process.ExitCode
    } catch {
        Set-Content -Path $Log -Value "cannot start ${File}: $_"
        return -1
    } finally {
        Restore-Env $saved
    }
    Get-Content -Path $out, $err -ErrorAction SilentlyContinue | Set-Content -Path $Log
    Remove-Item -Path $out, $err -ErrorAction SilentlyContinue
    return $rc
}

# Invoke-Step NAME FILE ARGUMENTS ENV: run in the repository root, log to <Out>\NAME.log, record exit code and duration.
function Invoke-Step([string]$Name, [string]$File, [string[]]$Arguments = @(), [string[]]$Env = @()) {
    $log = Join-Path $script:V.Out "$Name.log"
    $command = (@($Env) + @($File) + @($Arguments) | Where-Object { $_ }) -join ' '
    Write-Log "step ${Name}: $command"
    if ($script:V.DryRun) {
        Add-Record $Name $command 0 0 $log 'dry run' $null
        return 0
    }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $rc = Invoke-Native $File $Arguments $Env $script:V.Root $log
    $timer.Stop()
    Add-Record $Name $command $rc $timer.Elapsed.TotalSeconds $log '' $null
    if ($rc -ne 0) {
        $script:Failed = 1
        Write-Log "  FAILED (exit $rc), see $log"
        Get-Content -Path $log -Tail 5 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" }
    }
    return $rc
}

function Test-Phase([string]$Name) { return ($script:V.Phases -contains $Name) }

function ConvertFrom-VerifyArguments([string[]]$Items) {
    $o = @{
        Model = $env:MODEL; ModelUri = $env:MODEL_URI; Bin = $env:BIN; Device = $env:DEVICE; Out = $env:OUT
        Port = 18080; Tokens = 64; Prompt = 'Write one sentence about the sea.'; Skip = @(); DryRun = $false; BuildArgs = @()
    }
    if ($env:PORT) { $o.Port = [int]$env:PORT }
    if ($env:TOKENS) { $o.Tokens = [int]$env:TOKENS }
    if ($env:PROMPT) { $o.Prompt = $env:PROMPT }
    $i = 0
    while ($Items -and $i -lt $Items.Count) {
        $key = $Items[$i] -replace '^-+', ''
        switch ($key) {
            'model' { $o.Model = $Items[++$i] }
            { $_ -in 'modeluri', 'model-uri' } { $o.ModelUri = $Items[++$i] }
            'bin' { $o.Bin = $Items[++$i] }
            'device' { $o.Device = $Items[++$i] }
            'out' { $o.Out = $Items[++$i] }
            'port' { $o.Port = [int]$Items[++$i] }
            'tokens' { $o.Tokens = [int]$Items[++$i] }
            'prompt' { $o.Prompt = $Items[++$i] }
            'skip' { $o.Skip = ($Items[++$i] -split ',') }
            { $_ -in 'buildargs', 'build-args' } { $o.BuildArgs = ($Items[++$i] -split ',') }
            { $_ -in 'dryrun', 'dry-run' } { $o.DryRun = $true }
            { $_ -in 'help', 'h', '?' } { $o.Help = $true }
            default { throw "unknown option: $($Items[$i])" }
        }
        $i++
    }
    return $o
}

function Stop-Server {
    if ($script:Server) {
        try { if (-not $script:Server.HasExited) { $script:Server.Kill() } } catch { }
        try { [void]$script:Server.WaitForExit(10000) } catch { }
        $script:Server = $null
    }
    if ($script:ServerLog) {
        Get-Content -Path "$($script:ServerLog).out", "$($script:ServerLog).err" -ErrorAction SilentlyContinue | Set-Content -Path $script:ServerLog
        Remove-Item -Path "$($script:ServerLog).out", "$($script:ServerLog).err" -ErrorAction SilentlyContinue
        $script:ServerLog = $null
    }
}

# Starts `fllm-rs serve` in the background and waits until /v1/models answers; returns $true or $false.
function Start-Server([string]$Name, [string]$Device, [string[]]$Env = @()) {
    $log = Join-Path $script:V.Out "server-$Name.log"
    $script:ServerLog = $log
    $serveArgs = ConvertTo-ArgString @('serve', '--model', $script:V.Model, '--device', $Device, '--host', '127.0.0.1', '--port', "$($script:V.Port)", '--ctx', '2048', '--max-seqs', '2')
    $saved = Use-Env $Env
    try {
        $script:Server = Start-Process -FilePath $script:V.Bin -ArgumentList $serveArgs -WorkingDirectory $script:V.Root -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err" -PassThru -NoNewWindow
        $null = $script:Server.Handle
    } catch {
        Set-Content -Path $log -Value "cannot start $($script:V.Bin): $_"
        $script:Server = $null
        return $false
    } finally {
        Restore-Env $saved
    }
    $deadline = (Get-Date).AddSeconds($script:V.ReadyTimeout)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$($script:V.Port)/v1/models" -TimeoutSec 5
            if ($response.StatusCode -eq 200) { return $true }
        } catch { }
        if ($script:Server.HasExited) { break }
        Start-Sleep -Seconds 2
    }
    Stop-Server
    return $false
}

# Greedy chat request(s) against the running server; the first text is written to text-<Name>.txt, the last request is timed.
function Invoke-Ask([string]$Name, [int]$Repeats) {
    $url = "http://127.0.0.1:$($script:V.Port)/v1/chat/completions"
    $texts = @()
    $last = $null
    for ($i = 0; $i -lt $Repeats; $i++) {
        $body = @{ model = 'm'; messages = @(@{ role = 'user'; content = $script:V.Prompt }); temperature = 0; max_tokens = [int]$script:V.Tokens } | ConvertTo-Json -Depth 5 -Compress
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri $url -Method Post -ContentType 'application/json' -Body $body -TimeoutSec $script:V.RequestTimeout
        $timer.Stop()
        $texts += [string]$response.choices[0].message.content
        $count = $null
        if ($response.usage) { $count = $response.usage.completion_tokens }
        $seconds = $timer.Elapsed.TotalSeconds
        $rate = $null
        if ($count -and $seconds -gt 0) { $rate = [math]::Round($count / $seconds, 2) }
        $last = [ordered]@{ seconds = [math]::Round($seconds, 3); completion_tokens = $count; tok_s = $rate }
    }
    [IO.File]::WriteAllText((Join-Path $script:V.Out "text-$Name.txt"), $texts[0], (New-Object Text.UTF8Encoding($false)))
    $last['repeat_equal'] = -not ($texts | Where-Object { $_ -cne $texts[0] })
    $last['requests'] = $Repeats
    return $last
}

function Compare-Text([string]$PathA, [string]$PathB) {
    $a = [IO.File]::ReadAllText($PathA)
    $b = [IO.File]::ReadAllText($PathB)
    if ($a -ceq $b) { return 'same' }
    $limit = [Math]::Min($a.Length, $b.Length)
    $n = 0
    while ($n -lt $limit -and $a[$n] -ceq $b[$n]) { $n++ }
    return "first difference at character $n (lengths $($a.Length) / $($b.Length))"
}

function ConvertTo-EnvList([string]$Text) {
    return @($Text -split '\s+' | Where-Object { $_ })
}

# One server, two requests (the first warms up kernels and pipelines); the text is compared with the reference text.
function Invoke-TextCase([string]$Name, [string]$Device, [string[]]$Env = @()) {
    $command = "serve --device $Device $($Env -join ' ')"
    $serverLog = Join-Path $script:V.Out "server-$Name.log"
    Write-Log "text case ${Name}: --device $Device $($Env -join ' ')"
    $script:LastSame = ''
    if ($script:V.DryRun) {
        Add-Record "text-$Name" $command 0 0 '' 'dry run' $null
        return
    }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    if (-not (Start-Server $Name $Device $Env)) {
        $script:Failed = 1
        $script:LastSame = 'error'
        Add-Record "text-$Name" $command 1 $timer.Elapsed.TotalSeconds $serverLog 'server did not start' $null
        return
    }
    try {
        $request = Invoke-Ask $Name 2
    } catch {
        Stop-Server
        $script:Failed = 1
        $script:LastSame = 'error'
        Add-Record "text-$Name" $command 1 $timer.Elapsed.TotalSeconds $serverLog "request failed: $_" $null
        return
    }
    Stop-Server
    if ($Name -eq 'ref') { $verdict = 'reference' } else { $verdict = Compare-Text (Join-Path $script:V.Out 'text-ref.txt') (Join-Path $script:V.Out "text-$Name.txt") }
    $script:LastSame = $verdict
    $extra = [ordered]@{ text_vs_cpu = $verdict; request = $request }
    # the native Vulkan and Metal backends can fall back to wgpu (Decoder::set_device); a log that mentions wgpu may mean that
    if ($Name -ne 'ref' -and $script:V.NativeGuard -and (Select-String -Path $serverLog -Pattern $script:V.NativeGuard -Quiet -ErrorAction SilentlyContinue)) {
        $extra["log_mentions_$($script:V.NativeGuard)"] = $true
    }
    Add-Record "text-$Name" $command 0 $timer.Elapsed.TotalSeconds $serverLog '' $extra
    if ($Device -eq $script:V.Device -and $Name -ne 'ref' -and $request.tok_s) { $script:TokS = $request.tok_s }
    Write-Log "  ${Name}: $verdict"
}

function Invoke-Pull {
    if ($script:V.Model -or -not $script:V.ModelUri) { return }
    if ($script:V.DryRun) { Add-Record 'pull' "$($script:V.Bin) pull $($script:V.ModelUri)" 0 0 '' 'dry run' $null; return }
    $rc = Invoke-Step 'pull' $script:V.Bin @('pull', $script:V.ModelUri)
    if ($rc -eq 0) {
        $path = (Get-Content -Path (Join-Path $script:V.Out 'pull.log') | Where-Object { $_ } | Select-Object -Last 1)
        if ($path -and (Test-Path -Path $path -PathType Leaf)) { $script:V.Model = $path } else { Write-Log "pull printed '$path', which is not a file" }
    }
}

function Invoke-BuildPhase {
    if ($script:V.BinGiven) { Skip-Step 'build' '-Bin given'; return }
    if ($script:V.BuildCommand) { & $script:V.BuildCommand; return }
    $arguments = @('build', '--release', '-p', 'fllm-engine')
    if ($script:V.Features) { $arguments += @('--features', $script:V.Features) }
    $arguments += @('--bin', 'fllm-rs')
    [void](Invoke-Step 'build' 'cargo' $arguments $script:V.BuildEnv)
}

function Invoke-DevicesPhase {
    if (-not $script:V.DryRun -and -not (Test-Path -Path $script:V.Bin)) { Skip-Step 'devices' "no binary at $($script:V.Bin)"; return }
    [void](Invoke-Step 'devices' $script:V.Bin @('devices') $script:V.CommonEnv)
    if (-not $script:V.DryRun) { Get-Content -Path (Join-Path $script:V.Out 'devices.log') -TotalCount 60 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ } }
}

function Invoke-SelftestPhase {
    if (-not $script:V.DryRun -and -not (Test-Path -Path $script:V.Bin)) { Skip-Step 'selftest' "no binary at $($script:V.Bin)"; return }
    $device = $script:V.Device
    $base = @($script:V.CommonEnv) + @($script:V.SelftestEnv)
    if ($script:V.ExactEnv.Count -gt 0 -and $script:V.SelftestTwice) {
        [void](Invoke-Step 'selftest-exact' $script:V.Bin @('selftest-gpu', '--device', $device, '--json', (Join-Path $script:V.Out 'selftest-exact.json')) ($base + $script:V.ExactEnv))
        [void](Invoke-Step 'selftest-fast' $script:V.Bin @('selftest-gpu', '--device', $device, '--json', (Join-Path $script:V.Out 'selftest-fast.json')) $base)
        return
    }
    $rc = Invoke-Step 'selftest' $script:V.Bin @('selftest-gpu', '--device', $device, '--json', (Join-Path $script:V.Out 'selftest.json')) $base
    if ($rc -ne 0 -and -not $script:V.DryRun) {
        # find the switch that makes the failure go away; these runs do not change the verdict above
        foreach ($switch in (@($script:V.DiagEnvs) + @($script:V.ExactEnv))) {
            if (-not $switch) { continue }
            $name = 'selftest-with-' + ($switch -replace '[^A-Za-z0-9_=.-]', '_')
            $before = $script:Failed
            [void](Invoke-Step $name $script:V.Bin @('selftest-gpu', '--device', $device, '--json', (Join-Path $script:V.Out "$name.json")) ($base + @($switch)))
            $script:Failed = $before
        }
    }
}

function Invoke-TextPhase {
    if (-not $script:V.Model -and -not $script:V.DryRun) { Skip-Step 'text' 'no model (-Model or -ModelUri)'; return }
    Invoke-TextCase 'ref' 'cpu' $script:V.RefEnv
    if ($script:LastSame -eq 'error') { Skip-Step 'text-cases' 'reference run failed'; return }
    Invoke-TextCase 'default' $script:V.Device $script:V.CommonEnv
    $defaultResult = $script:LastSame
    if ($script:V.ExactEnv.Count -gt 0) { Invoke-TextCase 'exact' $script:V.Device (@($script:V.CommonEnv) + $script:V.ExactEnv) }
    # extra cases: "name|device|ENV=VALUE ENV=VALUE" (ExtraCases, then the output of CaseHook)
    $cases = @($script:V.ExtraCases)
    if ($script:V.CaseHook) { $cases += @(& $script:V.CaseHook) }
    foreach ($case in $cases) {
        if (-not $case) { continue }
        $fields = $case -split '\|', 3
        $device = $fields[1]
        if (-not $device) { $device = $script:V.Device }
        $envText = ''
        if ($fields.Count -gt 2) { $envText = $fields[2] }
        Invoke-TextCase $fields[0] $device (ConvertTo-EnvList $envText)
    }
    if ($script:V.DiagEnvs.Count -gt 0 -and $defaultResult -and $defaultResult -ne 'same') {
        Write-Log "default text differs from the CPU: one case per switch ($($script:V.DiagEnvs -join ' '))"
        foreach ($switch in $script:V.DiagEnvs) {
            Invoke-TextCase ('diag-' + ($switch -replace '[^A-Za-z0-9_=.-]', '_')) $script:V.Device (@($script:V.CommonEnv) + @($switch))
        }
    }
}

function Invoke-BenchPhase {
    if (-not $script:V.Model -and -not $script:V.DryRun) { Skip-Step 'bench' 'no model (-Model or -ModelUri)'; return }
    if ($script:V.DryRun) {
        Add-Record 'bench' "$($script:V.Bin) bench --url http://127.0.0.1:$($script:V.Port) --scenarios prefill,decode-single --json bench.json" 0 0 '' 'dry run' $null
        return
    }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    if (-not (Start-Server 'bench' $script:V.Device $script:V.CommonEnv)) {
        $script:Failed = 1
        Add-Record 'bench' "serve --device $($script:V.Device)" 1 $timer.Elapsed.TotalSeconds (Join-Path $script:V.Out 'server-bench.log') 'server did not start' $null
        return
    }
    try {
        $request = Invoke-Ask 'bench-timed' 3
        Add-Record 'bench-timed-request' 'chat completion, 3 requests, last one timed (includes the prompt)' 0 0 '' '' @{ request = $request }
        if ($request.tok_s) { $script:TokS = $request.tok_s }
    } catch {
        $script:Failed = 1
        Add-Record 'bench-timed-request' 'chat completion' 1 0 '' "request failed: $_" $null
    }
    $benchArgs = @('bench', '--url', "http://127.0.0.1:$($script:V.Port)", '--scenarios', 'prefill,decode-single', '--prompt-lens', '512', '--gen-len', '128', '--warmup', '1', '--repeats', '2', '--label', "$($script:V.Backend) $($script:V.Device)", '--json', (Join-Path $script:V.Out 'bench.json'))
    [void](Invoke-Step 'bench' $script:V.Bin $benchArgs)
    Stop-Server
}

function Write-Report {
    $commit = 'unknown'
    $dirty = $false
    try {
        $head = (& git -C $script:V.Root rev-parse --short HEAD 2>$null)
        if ($head) { $commit = "$head".Trim() }
        $dirty = [bool](& git -C $script:V.Root status --porcelain 2>$null)
    } catch { }
    $duration = [int]$script:Total.Elapsed.TotalSeconds
    $report = [ordered]@{
        backend = $script:V.Backend
        device = $script:V.Device
        commit = $commit
        tree_dirty = $dirty
        command = $script:V.Command
        features = $script:V.Features
        model = $script:V.Model
        host = "$([Environment]::OSVersion.VersionString) $env:PROCESSOR_ARCHITECTURE"
        date_utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        exit_code = $script:Failed
        duration_seconds = $duration
        tok_s = $script:TokS
        tok_s_note = 'chat completion of the device server, includes the prompt; not a benchmark (see bench.json for fllm-rs bench)'
        script_level = 'L0 (parsed only before its first run)'
        steps = @($script:Records)
    }
    $path = Join-Path $script:V.Out 'report.json'
    $report | ConvertTo-Json -Depth 10 | Set-Content -Path $path -Encoding UTF8
    Write-Log "report: $path (exit code $($script:Failed), $duration s)"
    if ($script:V.DryRun) { Write-Log 'dry run: nothing was executed' }
    if ($script:Failed -ne 0) { Write-Log 'at least one step failed: the matrix cell stays at L0.' } else { Write-Log 'no step failed.' }
    Write-Log "Levels are assigned by hand from the report: docs/backends-verify.md, section 'Auswertung'."
}

# Config keys: Backend, Features, Device, BuildEnv, CommonEnv, SelftestEnv, ExactEnv, DiagEnvs, RefEnv, ExtraCases (all string[]),
# SelftestTwice (bool), Phases (string[]), NativeGuard (string), Pre / CaseHook / BuildCommand (script blocks).
function Invoke-Verify([hashtable]$Config, [string[]]$Arguments) {
    $script:Total = [Diagnostics.Stopwatch]::StartNew()
    $parsed = ConvertFrom-VerifyArguments $Arguments
    if ($parsed.Help) {
        Write-Host "usage: verify-$($Config.Backend).ps1 [-Model FILE.gguf] [-ModelUri URI] [-Bin PATH] [-Device NAME] [-Out DIR] [-Port N] [-Tokens N] [-Prompt TEXT] [-Skip a,b] [-DryRun]"
        Write-Host 'phases for -Skip: build devices pre selftest text bench; see the header of scripts\verify-common.ps1'
        exit 0
    }
    $root = Split-Path -Parent $PSScriptRoot
    $v = @{
        Backend = $Config.Backend; Features = [string]$Config.Features; Device = $Config.Device
        BuildEnv = @($Config.BuildEnv | Where-Object { $_ }); CommonEnv = @($Config.CommonEnv | Where-Object { $_ })
        SelftestEnv = @($Config.SelftestEnv | Where-Object { $_ }); ExactEnv = @($Config.ExactEnv | Where-Object { $_ })
        DiagEnvs = @($Config.DiagEnvs | Where-Object { $_ }); RefEnv = @($Config.RefEnv | Where-Object { $_ })
        ExtraCases = @($Config.ExtraCases | Where-Object { $_ })
        SelftestTwice = [bool]$Config.SelftestTwice; NativeGuard = $Config.NativeGuard
        Pre = $Config.Pre; CaseHook = $Config.CaseHook; BuildCommand = $Config.BuildCommand
        Phases = @('build', 'devices', 'pre', 'selftest', 'text', 'bench')
        Root = $root; ReadyTimeout = 900; RequestTimeout = 900
    }
    if ($Config.Phases) { $v.Phases = @($Config.Phases | Where-Object { $_ }) }
    $skipList = @($parsed.Skip | Where-Object { $_ })
    $v.Phases = @($v.Phases | Where-Object { $skipList -notcontains $_ })
    if ($parsed.Device) { $v.Device = $parsed.Device }
    foreach ($key in 'Model', 'ModelUri', 'Port', 'Tokens', 'Prompt', 'DryRun', 'BuildArgs') { $v[$key] = $parsed[$key] }
    $v.BinGiven = [bool]$parsed.Bin
    if ($parsed.Bin) { $v.Bin = $parsed.Bin } elseif (Test-WindowsHost) { $v.Bin = Join-Path $root 'target\release\fllm-rs.exe' } else { $v.Bin = Join-Path $root 'target/release/fllm-rs' }
    if ($parsed.Out) { $v.Out = $parsed.Out } else { $v.Out = Join-Path (Join-Path $root 'verify-out') $Config.Backend }
    $v.Command = "$($MyInvocation.PSCommandPath) $($Arguments -join ' ')"
    $script:V = $v
    New-Item -ItemType Directory -Force -Path $v.Out | Out-Null
    Write-Log "root $root, device $($v.Device), features '$($v.Features)', output $($v.Out)"
    try {
        if (Test-Phase 'build') { Invoke-BuildPhase }
        if (Test-Phase 'devices') { Invoke-DevicesPhase }
        if (Test-Phase 'pre') {
            if ($v.Pre) { & $v.Pre } else { Skip-Step 'pre' 'this backend has no extra steps' }
        }
        if (Test-Phase 'selftest') { Invoke-SelftestPhase }
        if ((Test-Phase 'text') -or (Test-Phase 'bench')) { Invoke-Pull }
        if (Test-Phase 'text') { Invoke-TextPhase }
        if (Test-Phase 'bench') { Invoke-BenchPhase }
    } finally {
        Stop-Server
    }
    Write-Report
    exit $script:Failed
}
