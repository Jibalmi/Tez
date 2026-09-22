# Chain 4 on llama-server (C:\temp\llamacpp): thinking budgets, verbalizer ablation, full few-shot,
# JevBench (short tiers), then the Qwen3.5-9B Q8 backbone, then Gemma at 8k context for the hard tier.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:TEZ_BACKEND = "llamacpp"; $env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$log = "$env:TEMP\abl4"; New-Item -ItemType Directory -Force $log | Out-Null
$exe = "C:\temp\llamacpp\llama-server.exe"; $wd = "C:\temp\llamacpp"
$gemma = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
$qwen = "$root/tools/models/Qwen3.5-9B-Q8_0.gguf"
function Start-Server($model, $ctx, $extra) {
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 4
  Start-Process -FilePath $exe -WorkingDirectory $wd -ArgumentList "-m $model -ngl 99 -c $ctx -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui $extra" -WindowStyle Hidden -RedirectStandardError "$log\server_$ctx.err" -RedirectStandardOutput "$log\server_$ctx.out"
  foreach ($i in 1..90) { Start-Sleep 2; try { if ((Invoke-RestMethod http://127.0.0.1:8091/health -TimeoutSec 3).status -eq 'ok') { return $true } } catch {} }
  return $false
}
# --- Gemma 4 12B Q8
try { $h = (Invoke-RestMethod http://127.0.0.1:8091/health -TimeoutSec 3).status } catch { $h = "down" }
if ($h -ne "ok") { if (-not (Start-Server $gemma 2048 "--swa-full")) { "gemma start failed" | Out-File "$log\gemma_failed.txt"; exit 1 } }
if (-not (Test-Path "results/h2h/rows_typed_decisions_tez-think32.summary.json")) { python experiments/think_td.py --budget 32 --limit 300 --out results/h2h/rows_typed_decisions_tez-think32.jsonl *> "$log\think32.log" }
if (-not (Test-Path "results/h2h/rows_typed_decisions_tez-think128.summary.json")) { python experiments/think_td.py --budget 128 --limit 150 --out results/h2h/rows_typed_decisions_tez-think128.jsonl *> "$log\think128.log" }
$env:TEZ_SYMBOLS = "123456789"; $env:TEZ_SYMBOL_WORD = "number"
if (-not (Test-Path "results/authored144_gemma4-12b-q8_0_digits.manifest.json")) { python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_gemma4-12b-q8_0_digits.jsonl --tag digits *> "$log\digits.log" }
$env:TEZ_SYMBOLS = "abcdefghijklmnop"; $env:TEZ_SYMBOL_WORD = "lowercase letter"
if (-not (Test-Path "results/authored144_gemma4-12b-q8_0_lower.manifest.json")) { python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_gemma4-12b-q8_0_lower.jsonl --tag lower *> "$log\lower.log" }
Remove-Item Env:TEZ_SYMBOLS; Remove-Item Env:TEZ_SYMBOL_WORD
if (-not (Test-Path "results/h2h/rows_typed_decisions_tez-fewshot4-all.summary.json")) { python experiments/fewshot_td.py --k 4 --out results/h2h/rows_typed_decisions_tez-fewshot4-all.jsonl *> "$log\fewshot_all.log" }
if (-not (Test-Path "results/jevbench_tez_gemma4-12b-q8_0_short.json")) { python experiments/bench_jevbench.py --tiers original,easy --out results/jevbench_tez_gemma4-12b-q8_0_short.json *> "$log\jev12b_short.log" }
# --- Qwen3.5-9B Q8 backbone
if (Start-Server $qwen 4096 "") {
  $env:TEZ_TEMPLATE = "qwen3"
  python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_qwen35-9b-q8.jsonl --template qwen3 --tag qwen35-9b-q8 *> "$log\q9_auth.log"
  python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_rev_qwen35-9b-q8.jsonl --template qwen3 --reverse --tag qwen35-9b-q8 *> "$log\q9_rev.log"
  python experiments/run_direct.py --data data/semif/perturbations108.jsonl --out results/perturb108_qwen35-9b-q8.jsonl --template qwen3 --tag qwen35-9b-q8 *> "$log\q9_pert.log"
  python experiments/bench_h2h.py --model tez --tasks typed_decisions,massive:en,massive:ja,massive:ar,massive:km,banking77,sst5,boolq,prompt_injections --out results/h2h_qwen35-9b --flip-n 100 *> "$log\q9_h2h.log"
  python experiments/bench_jevbench.py --tiers original,easy --out results/jevbench_tez_qwen35-9b-q8_short.json *> "$log\jev9b_short.log"
  Remove-Item Env:TEZ_TEMPLATE
} else { "qwen server failed" | Out-File "$log\qwen_failed.txt" }
# --- Gemma at 8k for the JevBench hard tier (long states), then restore 2k
if (Start-Server $gemma 8192 "--swa-full") {
  python experiments/bench_jevbench.py --tiers hard --out results/jevbench_tez_gemma4-12b-q8_0_hard.json *> "$log\jev12b_hard.log"
} else { "gemma 8k failed" | Out-File "$log\gemma8k_failed.txt"; Start-Server $gemma 4096 "--swa-full" | Out-Null; python experiments/bench_jevbench.py --tiers hard --out results/jevbench_tez_gemma4-12b-q8_0_hard.json *> "$log\jev12b_hard4k.log" }
Start-Server $gemma 2048 "--swa-full" | Out-Null
"CHAIN4DONE $(Get-Date)" | Out-File "$log\done.txt"
