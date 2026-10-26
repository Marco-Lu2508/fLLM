<#
.SYNOPSIS
  Compares fLLM, llama.cpp (llama-server) and Ollama under load: prompts of different lengths and several users at the same time.

.DESCRIPTION
  Every engine is started as an HTTP server with room for the largest scenario and is asked through the same OpenAI style
  /v1/chat/completions route with streaming. For each prompt length and each number of simultaneous users the script starts all
  requests at once (every prompt is different, so no engine can reuse a cache) and prints
    total      generated tokens per second over the whole run (what the server delivers to all users together)
    first      average time until the first token arrives, in seconds
    each       generated tokens per second one user sees after its first token
  Nothing is downloaded: point the parameters at programs you already have (see bench-compare.ps1 for the layout).

.EXAMPLE
  .\scripts\bench-load.ps1 -Model C:\models\Qwen3.5-2B-Q4_K_M.gguf -LlamaCpu C:\bench\llama-cpu -LlamaVulkan C:\bench\llama-vulkan -OllamaExe C:\bench\ollama\ollama.exe
#>
param(
  [Parameter(Mandatory)][string]$Model,
  [string]$Fllm = '',
  [string]$LlamaCpu = '',
  [string]$LlamaVulkan = '',
  [string]$OllamaExe = '',
  [int[]]$PromptSizes = @(100, 500, 2000),
  [int[]]$Users = @(1, 4, 8),
  [int]$Tokens = 64,
  [ValidateSet('cpu', 'gpu')][string[]]$Targets = @('gpu', 'cpu'),
  [ValidateSet('fllm', 'llama', 'ollama')][string[]]$Engines = @('fllm', 'llama', 'ollama')
)
$ErrorActionPreference = 'Stop'
$maxUsers = ($Users | Measure-Object -Maximum).Maximum
$slot = (($PromptSizes | Measure-Object -Maximum).Maximum) + $Tokens + 400   # tokens one user may need
$total = $slot * $maxUsers
if (-not $Fllm) {
  $Fllm = @("$env:LOCALAPPDATA\fllm\fllm-rs.exe", (Get-Command fllm-rs -ErrorAction SilentlyContinue).Source) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

Add-Type -ReferencedAssemblies @(if ($PSVersionTable.PSEdition -eq 'Core') { 'System.Net.Http', 'System.Net.Primitives', 'System.Collections', 'System.Linq', 'System.Console' } else { 'System.Net.Http' }) -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

public static class LoadClient {
  public class Result { public double First; public double Total; public int Chunks; public string Error; }

  static async Task<Result> One(HttpClient http, string url, string body) {
    var r = new Result(); var sw = Stopwatch.StartNew();
    try {
      using (var req = new HttpRequestMessage(HttpMethod.Post, url) { Content = new StringContent(body, Encoding.UTF8, "application/json") })
      using (var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead)) {
        if (!resp.IsSuccessStatusCode) { var msg = await resp.Content.ReadAsStringAsync(); r.Error = "HTTP error " + resp.ReasonPhrase + " " + (msg.Length > 160 ? msg.Substring(0, 160) : msg); return r; }
        using (var rd = new StreamReader(await resp.Content.ReadAsStreamAsync())) {
          string line;
          while ((line = await rd.ReadLineAsync()) != null) {
            if (!line.StartsWith("data:")) continue;
            var d = line.Substring(5).Trim();
            if (d == "[DONE]") break;
            // a chunk that carries text (answer or reasoning) is about one token
            if (d.Contains("\"content\":\"") || d.Contains("\"reasoning_content\":\"") || d.Contains("\"reasoning\":\"")) {
              if (r.Chunks == 0) r.First = sw.Elapsed.TotalSeconds;
              r.Chunks++;
            }
          }
        }
      }
    } catch (Exception e) { r.Error = e.Message; }
    r.Total = sw.Elapsed.TotalSeconds;
    return r;
  }

  public static Result[] Run(string url, string[] bodies, out double wall) {
    var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
    var sw = Stopwatch.StartNew();
    var tasks = new List<Task<Result>>();
    foreach (var b in bodies) tasks.Add(Task.Run(() => One(http, url, b)));
    Task.WaitAll(tasks.ToArray());
    wall = sw.Elapsed.TotalSeconds;
    var res = new Result[tasks.Count];
    for (int i = 0; i < res.Length; i++) res[i] = tasks[i].Result;
    return res;
  }
}
'@

function Wait-Http($url, $seconds = 180) {
  for ($i = 0; $i -lt $seconds / 2; $i++) { Start-Sleep 2; try { if ((Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { return $true } } catch { } }
  return $false
}

$sentence = 'The quick brown fox jumps over the lazy dog. '   # about 10 tokens
function New-Body($model, $size, $extra) {
  $text = $sentence * [math]::Max(1, [int]($size / 10))
  $o = @{ model = $model; stream = $true; max_tokens = $Tokens; temperature = 0
    messages = @(@{ role = 'user'; content = "Run $(Get-Random): count the words of this text and then count to twenty. $text" }) }
  foreach ($k in $extra.Keys) { $o[$k] = $extra[$k] }
  $o | ConvertTo-Json -Depth 6 -Compress
}

$rows = New-Object System.Collections.Generic.List[object]
function Measure-Engine($label, $url, $model, $extra) {
  $null = [LoadClient]::Run($url, @((New-Body $model 50 $extra)), [ref]([double]0))   # warm up
  foreach ($size in $PromptSizes) {
    foreach ($n in $Users) {
      $wall = [double]0
      $res = [LoadClient]::Run($url, @(1..$n | ForEach-Object { New-Body $model $size $extra }), [ref]$wall)
      $bad = @($res | Where-Object { $_.Error })
      if ($bad.Count -gt 0) {
        $rows.Add([pscustomobject]@{ Engine = $label; Prompt = $size; Users = $n; 'total tok/s' = 'error'; 'first (s)' = ''; 'each tok/s' = $bad[0].Error }); continue
      }
      $made = ($res | Measure-Object Chunks -Sum).Sum
      $first = ($res | Measure-Object First -Average).Average
      $each = ($res | ForEach-Object { if ($_.Chunks -gt 1 -and $_.Total -gt $_.First) { ($_.Chunks - 1) / ($_.Total - $_.First) } else { 0 } } | Measure-Object -Average).Average
      $rows.Add([pscustomobject]@{ Engine = $label; Prompt = $size; Users = $n; 'total tok/s' = [math]::Round($made / $wall, 1); 'first (s)' = [math]::Round($first, 2); 'each tok/s' = [math]::Round($each, 1) })
      Write-Host ("{0,-28} prompt {1,5} users {2}: {3} tok/s total, first token {4} s" -f $label, $size, $n, $rows[-1].'total tok/s', $rows[-1].'first (s)')
    }
  }
}

function Test-Fllm($device, $label) {
  $port = Get-Random -Minimum 48000 -Maximum 49000
  $p = Start-Process $Fllm -ArgumentList @('serve', '--model', "`"$Model`"", '--device', $device, '--host', '127.0.0.1', '--port', "$port", '--ctx', "$total", '--max-seqs', "$maxUsers", '--model-name', 'm') -PassThru -WindowStyle Hidden
  try {
    if (-not (Wait-Http "http://127.0.0.1:$port/health")) { Write-Warning "$label did not start"; return }
    Measure-Engine $label "http://127.0.0.1:$port/v1/chat/completions" 'm' @{ chat_template_kwargs = @{ enable_thinking = $false } }
  } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
}

function Test-Llama($dir, $gpu, $label) {
  $exe = Join-Path $dir 'llama-server.exe'
  if (-not (Test-Path $exe)) { Write-Warning "no llama-server.exe in $dir"; return }
  $port = Get-Random -Minimum 49000 -Maximum 50000
  $p = Start-Process $exe -ArgumentList @('-m', "`"$Model`"", '-c', "$total", '-np', "$maxUsers", '-ngl', $(if ($gpu) { '99' } else { '0' }), '--host', '127.0.0.1', '--port', "$port", '--alias', 'm', '--no-webui') -PassThru -WindowStyle Hidden
  try {
    if (-not (Wait-Http "http://127.0.0.1:$port/health")) { Write-Warning "$label did not start"; return }
    Measure-Engine $label "http://127.0.0.1:$port/v1/chat/completions" 'm' @{ chat_template_kwargs = @{ enable_thinking = $false } }
  } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
}

function Test-Ollama($igpu, $label) {
  $work = Join-Path $env:TEMP 'bench-ollama'
  New-Item -ItemType Directory -Force $work | Out-Null
  $env:OLLAMA_MODELS = Join-Path $work 'models'; $env:OLLAMA_HOST = '127.0.0.1:11556'
  $env:OLLAMA_NUM_PARALLEL = "$maxUsers"; $env:OLLAMA_CONTEXT_LENGTH = "$slot"; $env:OLLAMA_MAX_QUEUE = '512'
  if ($igpu) { $env:OLLAMA_VULKAN = '1'; $env:OLLAMA_IGPU_ENABLE = '1' } else { Remove-Item Env:OLLAMA_VULKAN, Env:OLLAMA_IGPU_ENABLE -ErrorAction SilentlyContinue }
  $p = Start-Process $OllamaExe -ArgumentList 'serve' -PassThru -WindowStyle Hidden
  try {
    if (-not (Wait-Http 'http://127.0.0.1:11556/api/version')) { Write-Warning "$label did not start"; return }
    "FROM $Model" | Set-Content (Join-Path $work 'Modelfile') -Encoding ascii
    $ErrorActionPreference = 'Continue'   # ollama writes its progress to the error stream
    & $OllamaExe create bench -f (Join-Path $work 'Modelfile') 2>&1 | Out-Null
    $ErrorActionPreference = 'Stop'
    Measure-Engine $label 'http://127.0.0.1:11556/v1/chat/completions' 'bench' @{ reasoning_effort = 'none' }
  } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force }
}

foreach ($t in $Targets) {
  $gpu = $t -eq 'gpu'
  if ($Fllm -and $Engines -contains 'fllm') { Test-Fllm $(if ($gpu) { 'vulkan:0' } else { 'cpu' }) "fLLM, $t" }
  $dir = if ($gpu) { $LlamaVulkan } else { $LlamaCpu }
  if ($dir -and $Engines -contains 'llama') { Test-Llama $dir $gpu "llama.cpp, $t" }
  if ($OllamaExe -and $Engines -contains 'ollama') { Test-Ollama $gpu "Ollama, $t" }
}

"`nModel: $Model"
"Answer: $Tokens tokens (greedy); every request has its own prompt; sizes are tokens of the prompt; speeds in tokens per second"
$rows | Format-Table -AutoSize
