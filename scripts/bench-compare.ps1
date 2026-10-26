<#
.SYNOPSIS
  Compares fLLM with llama.cpp and Ollama on this computer with one GGUF model and one fixed prompt.

.DESCRIPTION
  Every engine gets the same chat prompt of about 500 tokens and writes about 100 tokens (greedy). The script prints the speed of reading
  the prompt and of writing the answer, in tokens per second, for the processor and for the graphics chip of each engine that is present.

  Nothing is downloaded: point the parameters at programs you already have.
    -Fllm       fllm-rs.exe (default: the one of the desktop app, or the one on the PATH)
    -LlamaDir   a folder with llama-bench.exe (a llama.cpp release for Windows, the CPU and the Vulkan builds are separate folders)
    -OllamaExe  ollama.exe (a standalone build is enough; the script starts its own server on a private port and a private model folder)

.EXAMPLE
  .\scripts\bench-compare.ps1 -Model C:\models\Qwen3.5-2B-Q4_K_M.gguf -LlamaCpu C:\bench\llama-cpu -LlamaVulkan C:\bench\llama-vulkan -OllamaExe C:\bench\ollama\ollama.exe
#>
param(
  [Parameter(Mandatory)][string]$Model,
  [string]$Fllm = '',
  [string]$LlamaCpu = '',
  [string]$LlamaVulkan = '',
  [string]$OllamaExe = '',
  [int]$PromptRepeats = 48,
  [int]$Tokens = 128,
  [int[]]$Threads = @()
)
$ErrorActionPreference = 'Stop'
if (-not $Fllm) {
  $candidates = @("$env:LOCALAPPDATA\fllm\fllm-rs.exe", (Get-Command fllm-rs -ErrorAction SilentlyContinue).Source) | Where-Object { $_ -and (Test-Path $_) }
  $Fllm = $candidates | Select-Object -First 1
}
$text = 'The quick brown fox jumps over the lazy dog. ' * $PromptRepeats
$rows = New-Object System.Collections.Generic.List[object]

function Wait-Http($url, $seconds = 120) {
  for ($i = 0; $i -lt $seconds / 2; $i++) { Start-Sleep 2; try { if ((Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { return $true } } catch { } }
  return $false
}

function Bench-Fllm($device, $extra, $label) {
  $port = Get-Random -Minimum 48000 -Maximum 49000
  $p = Start-Process $Fllm -ArgumentList (@('serve', '--model', "`"$Model`"", '--device', $device, '--host', '127.0.0.1', '--port', "$port", '--ctx', '4096', '--model-name', 'q') + $extra) -PassThru -WindowStyle Hidden
  try {
    if (-not (Wait-Http "http://127.0.0.1:$port/health")) { Write-Warning "$label did not start"; return }
    $body = @{ model = 'q'; messages = @(@{ role = 'user'; content = "Repeat the last word of this text and then count to twenty. $text" }); max_tokens = $Tokens; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 6
    $null = Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
    $r = Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
    $rows.Add([pscustomobject]@{ Engine = $label; Prompt = [math]::Round($r.timings.prompt_per_second); Answer = [math]::Round($r.timings.predicted_per_second, 1) })
  } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
}

function Bench-LlamaBench($dir, $label) {
  $exe = Join-Path $dir 'llama-bench.exe'
  if (-not (Test-Path $exe)) { Write-Warning "no llama-bench.exe in $dir"; return }
  $out = (& $exe -m $Model -p 512 -n $Tokens -r 2 2>$null) -join "`n"
  $pp = [regex]::Match($out, 'pp512\s*\|\s*([\d.,]+)')
  $tg = [regex]::Match($out, 'tg\d+\s*\|\s*([\d.,]+)')
  if (-not $pp.Success -or -not $tg.Success) { Write-Warning "could not read the result of $label"; return }
  $num = { param($m) [double]::Parse($m.Groups[1].Value.Replace(',', '.'), [Globalization.CultureInfo]::InvariantCulture) }
  $rows.Add([pscustomobject]@{ Engine = $label; Prompt = [math]::Round((& $num $pp)); Answer = [math]::Round((& $num $tg), 1) })
}

function Bench-Ollama($igpu, $label) {
  $work = Join-Path $env:TEMP 'bench-ollama'
  New-Item -ItemType Directory -Force $work | Out-Null
  $env:OLLAMA_MODELS = Join-Path $work 'models'; $env:OLLAMA_HOST = '127.0.0.1:11555'
  if ($igpu) { $env:OLLAMA_VULKAN = '1'; $env:OLLAMA_IGPU_ENABLE = '1' } else { Remove-Item Env:OLLAMA_VULKAN, Env:OLLAMA_IGPU_ENABLE -ErrorAction SilentlyContinue }
  $p = Start-Process $OllamaExe -ArgumentList 'serve' -PassThru -WindowStyle Hidden
  try {
    if (-not (Wait-Http 'http://127.0.0.1:11555/api/version')) { Write-Warning "$label did not start"; return }
    "FROM $Model" | Set-Content (Join-Path $work 'Modelfile') -Encoding ascii
    & $OllamaExe create bench -f (Join-Path $work 'Modelfile') *> $null
    $chat = { param($t) @{ model = 'bench'; stream = $false; think = $false; messages = @(@{ role = 'user'; content = $t }); options = @{ num_predict = $Tokens; temperature = 0; num_ctx = 4096 } } | ConvertTo-Json -Depth 6 }
    $null = Invoke-RestMethod 'http://127.0.0.1:11555/api/chat' -Method Post -Body (& $chat "warm up $text") -ContentType 'application/json' -TimeoutSec 600
    # a prompt that differs from the first word on, so that Ollama cannot reuse its cache of the previous request
    $r = Invoke-RestMethod 'http://127.0.0.1:11555/api/chat' -Method Post -Body (& $chat "Run $(Get-Random): Repeat the last word of this text and then count to twenty. $text") -ContentType 'application/json' -TimeoutSec 600
    $rows.Add([pscustomobject]@{ Engine = $label; Prompt = [math]::Round($r.prompt_eval_count / ($r.prompt_eval_duration / 1e9)); Answer = [math]::Round($r.eval_count / ($r.eval_duration / 1e9), 1) })
  } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force }
}

if ($Fllm) {
  Bench-Fllm 'cpu' @() 'fLLM, processor (default threads)'
  foreach ($t in $Threads) { Bench-Fllm 'cpu' @('--threads', "$t") "fLLM, processor ($t threads)" }
  Bench-Fllm 'vulkan:0' @() 'fLLM, graphics (Vulkan)'
}
if ($LlamaCpu) { Bench-LlamaBench $LlamaCpu 'llama.cpp, processor' }
if ($LlamaVulkan) { Bench-LlamaBench $LlamaVulkan 'llama.cpp, graphics (Vulkan)' }
if ($OllamaExe) { Bench-Ollama $false 'Ollama, processor'; Bench-Ollama $true 'Ollama, graphics (Vulkan)' }

"`nModel: $Model"
"Prompt: about 505 tokens, answer: up to $Tokens tokens, greedy; speeds in tokens per second"
$rows | Format-Table -AutoSize
