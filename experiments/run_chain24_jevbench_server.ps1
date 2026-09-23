# JevBench public tiers through the real runtime (tez serve -> llama-server 12B), 8k context for the hard tier.
# Stops the 4k server on :8091 (by exact executable path), starts an 8k one (with --swa-full if it fits, otherwise
# without: sliding-window caching changes cache reuse only, not answers), runs release/jevbench/run_public.py against
# tez serve on :8787, then restores the 4k server with embeddings.
$ErrorActionPreference = "Continue"
$exe = "C:\temp\llamacpp\llama-server.exe"
$wd = "C:\temp\llamacpp"
$gemma = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
$log = "$env:TEMP\tez_chain24"
New-Item -ItemType Directory -Force $log | Out-Null
Set-Location "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"

function Stop-Llama {
  Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object { $_.ExecutablePath -eq $exe -and $_.CommandLine -match "--port 8091" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  Start-Sleep -Seconds 4
}

function Start-Llama($ctx, $extra, $tag) {
  Start-Process -FilePath $exe -WorkingDirectory $wd -ArgumentList "-m $gemma -ngl 99 -c $ctx -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui $extra" -WindowStyle Hidden -RedirectStandardError "$log\server_$tag.err" -RedirectStandardOutput "$log\server_$tag.out"
  for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try { $h = (Invoke-RestMethod -Uri "http://127.0.0.1:8091/health" -TimeoutSec 3).status } catch { $h = $null }
    if ($h -eq "ok") { return $true }
    if (-not (Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object { $_.CommandLine -match "--port 8091" })) { return $false }
  }
  return $false
}

Stop-Llama
$mode = "8k-swafull"
if (-not (Start-Llama 8192 "--swa-full" "8k_swafull")) {
  "8k with --swa-full did not start" | Out-File "$log\note.txt"
  Stop-Llama
  $mode = "8k"
  if (-not (Start-Llama 8192 "" "8k")) { "8k did not start" | Out-File "$log\note.txt" -Append; $mode = "failed" }
}
"server mode: $mode" | Out-File "$log\mode.txt"
if ($mode -ne "failed") {
  python release/jevbench/run_public.py --endpoint http://127.0.0.1:8787/v1/systemone --out results/jevbench_tez_server.json *> "$log\run_public.log"
}
Stop-Llama
Start-Llama 4096 "--swa-full --embeddings --pooling last" "4k_restored" | Out-Null
"done" | Out-File "$log\done.txt"
