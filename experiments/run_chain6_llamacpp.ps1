# Chain 6 (after chain 5): thinking budget on the JevBench hard tier at 8k, then the in-process
# hidden-state probe (GPU must be free), then restore the 2k server.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:TEZ_BACKEND = "llamacpp"; $env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$log = "$env:TEMP\abl6"; New-Item -ItemType Directory -Force $log | Out-Null
$exe = "C:\temp\llamacpp\llama-server.exe"; $wd = "C:\temp\llamacpp"
$gemma = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
function Start-Server($model, $ctx, $extra) {
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 4
  Start-Process -FilePath $exe -WorkingDirectory $wd -ArgumentList "-m $model -ngl 99 -c $ctx -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui $extra" -WindowStyle Hidden -RedirectStandardError "$log\server_$ctx.err" -RedirectStandardOutput "$log\server_$ctx.out"
  foreach ($i in 1..90) { Start-Sleep 2; try { if ((Invoke-RestMethod http://127.0.0.1:8091/health -TimeoutSec 3).status -eq 'ok') { return $true } } catch {} }
  return $false
}
while (-not (Test-Path "$env:TEMP\abl5\done.txt")) { Start-Sleep 60 }
Start-Server $gemma 8192 "--swa-full" | Out-Null
if (-not (Test-Path "results/jevbench_hard_think64_gemma4-12b-q8_0.summary.json")) { python experiments/think_td.py --jev-tier hard --budget 64 --limit 111 --out results/jevbench_hard_think64_gemma4-12b-q8_0.jsonl *> "$log\think_hard64.log" }
# hidden-state probe needs the whole GPU
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
if (-not (Test-Path "results/hidden_probe_qwen35-4b.json")) { python experiments/hidden_probe.py --model Qwen/Qwen3.5-4B --layers -1,-4,-8,-12 --out results/hidden_probe_qwen35-4b.json *> "$log\probe.log" }
Start-Server $gemma 2048 "--swa-full" | Out-Null
"CHAIN6DONE $(Get-Date)" | Out-File "$log\done.txt"
