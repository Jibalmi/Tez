# JevBench hard tier through tez serve at 8k context WITHOUT --swa-full (the full sliding-window cache spills past
# 16 GB on this laptop; without it the answers are identical, only cross-request cache reuse is lost).
# Logs the server's dedicated/shared GPU memory before running, then restores the 4k server with embeddings.
$ErrorActionPreference = "Continue"
$exe = "C:\temp\llamacpp\llama-server.exe"
$wd = "C:\temp\llamacpp"
$gemma = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
$log = "$env:TEMP\tez_chain25"
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
  }
  return $false
}

function Log-GpuMemory($tag) {
  $p = (Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object { $_.CommandLine -match "--port 8091" }).ProcessId
  $lines = (Get-Counter '\GPU Process Memory(*)\Shared Usage','\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples | Where-Object { $_.InstanceName -match "pid_$p" } | ForEach-Object { '{0} {1:N0} MB' -f $_.Path.Split('\')[-1], ($_.CookedValue/1MB) }
  ("$tag pid $p" + [Environment]::NewLine + ($lines -join [Environment]::NewLine)) | Out-File "$log\gpu_$tag.txt"
}

Stop-Llama
if (Start-Llama 8192 "" "8k") {
  Log-GpuMemory "8k_start"
  python release/jevbench/run_public.py --endpoint http://127.0.0.1:8787/v1/systemone --tiers hard --warmup 3 --out results/jevbench_tez_server_hard.json *> "$log\run_hard.log"
  Log-GpuMemory "8k_end"
} else {
  "8k server did not start" | Out-File "$log\note.txt"
}
Stop-Llama
Start-Llama 4096 "--swa-full --embeddings --pooling last" "4k_restored" | Out-Null
"done" | Out-File "$log\done.txt"
