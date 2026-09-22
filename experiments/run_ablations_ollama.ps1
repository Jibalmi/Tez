# Accuracy-only ablations through the Ollama fallback backend (llama-server binary quarantined by Avast).
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:TEZ_BACKEND = "ollama"; $env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$log = "$env:TEMP\abl"; New-Item -ItemType Directory -Force $log | Out-Null
python experiments/fewshot_td.py --k 4 --limit 600 --out results/h2h/rows_typed_decisions_tez-fewshot4.jsonl *> "$log\fewshot4.log"
python experiments/think_td.py  --budget 32 --limit 300 --out results/h2h/rows_typed_decisions_tez-think32.jsonl *> "$log\think32.log"
python experiments/bench_h2h.py --model tez --tasks typed_decisions --out results/h2h_ollama --flip-n 0 *> "$log\zeroshot_ollama.log"
"ABLDONE $(Get-Date)" | Out-File "$log\done.txt"
