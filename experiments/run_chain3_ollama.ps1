# Chain 3 (Ollama backend): JevBench public tiers with the 12B, then the Qwen3.5-9B Q8 backbone on
# SemIf authored144 (+reversal), typed-decisions and JevBench. Waits for chain 2 and for the Qwen model.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:TEZ_BACKEND = "ollama"; $env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$log = "$env:TEMP\abl3"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl2\done.txt")) { Start-Sleep 60 }
while (-not ((ollama list) -match "qwen35-9b-q8")) { Start-Sleep 60 }

# 12B on JevBench public tiers
$env:TEZ_OLLAMA_MODEL = "gemma4:12b-it-q8_0"; $env:TEZ_TEMPLATE = "gemma4"
python experiments/bench_jevbench.py --out results/jevbench_tez_gemma4-12b-q8_0.json *> "$log\jev12b.log"

# Qwen3.5-9B Q8 as backbone
$env:TEZ_OLLAMA_MODEL = "qwen35-9b-q8"; $env:TEZ_TEMPLATE = "qwen3"
python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_qwen35-9b-q8_ollama.jsonl --template qwen3 --tag qwen35-9b-q8 *> "$log\q9_auth.log"
python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_rev_qwen35-9b-q8_ollama.jsonl --template qwen3 --reverse --tag qwen35-9b-q8 *> "$log\q9_rev.log"
python experiments/run_direct.py --data data/semif/perturbations108.jsonl --out results/perturb108_qwen35-9b-q8_ollama.jsonl --template qwen3 --tag qwen35-9b-q8 *> "$log\q9_pert.log"
python experiments/bench_h2h.py --model tez --tasks typed_decisions,massive:en,massive:ja,massive:ar,banking77,sst5 --out results/h2h_qwen35-9b --flip-n 100 *> "$log\q9_h2h.log"
python experiments/bench_jevbench.py --out results/jevbench_tez_qwen35-9b-q8.json *> "$log\jev9b.log"
"CHAIN3DONE $(Get-Date)" | Out-File "$log\done.txt"
