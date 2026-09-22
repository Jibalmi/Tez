# Chain 5: after chain 4, Gemma 4 12B Q8 at 8k context for the long prompts: full few-shot (2,000
# decisions, 4 examples in the cached prefix) and all three JevBench public tiers.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:TEZ_BACKEND = "llamacpp"; $env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$log = "$env:TEMP\abl5"; New-Item -ItemType Directory -Force $log | Out-Null
$exe = "C:\temp\llamacpp\llama-server.exe"; $wd = "C:\temp\llamacpp"
$gemma = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
function Start-Server($model, $ctx, $extra) {
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 4
  Start-Process -FilePath $exe -WorkingDirectory $wd -ArgumentList "-m $model -ngl 99 -c $ctx -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui $extra" -WindowStyle Hidden -RedirectStandardError "$log\server_$ctx.err" -RedirectStandardOutput "$log\server_$ctx.out"
  foreach ($i in 1..90) { Start-Sleep 2; try { if ((Invoke-RestMethod http://127.0.0.1:8091/health -TimeoutSec 3).status -eq 'ok') { return $true } } catch {} }
  return $false
}
while (-not (Test-Path "$env:TEMP\abl4\done.txt")) { Start-Sleep 60 }
$ctx = 8192
if (-not (Start-Server $gemma 8192 "--swa-full")) { $ctx = 6144; if (-not (Start-Server $gemma 6144 "--swa-full")) { $ctx = 4096; Start-Server $gemma 4096 "--swa-full" | Out-Null } }
"gemma ctx $ctx" | Out-File "$log\ctx.txt"
if (-not (Test-Path "results/h2h/rows_typed_decisions_tez-fewshot4-all.summary.json")) { python experiments/fewshot_td.py --k 4 --out results/h2h/rows_typed_decisions_tez-fewshot4-all.jsonl *> "$log\fewshot_all.log" }
if (-not (Test-Path "results/jevbench_tez_gemma4-12b-q8_0.json")) { python experiments/bench_jevbench.py --tiers original,easy,hard --out results/jevbench_tez_gemma4-12b-q8_0.json *> "$log\jev12b.log" }
Start-Server $gemma 2048 "--swa-full" | Out-Null
"CHAIN5DONE $(Get-Date)" | Out-File "$log\done.txt"
