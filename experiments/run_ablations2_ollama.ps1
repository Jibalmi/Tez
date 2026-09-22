# Second ablation chain on the Ollama fallback backend: thinking budget, verbalizer symbols, backend agreement.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:TEZ_BACKEND = "ollama"; $env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$log = "$env:TEMP\abl2"; New-Item -ItemType Directory -Force $log | Out-Null
python experiments/think_td.py --budget 32 --limit 300 --out results/h2h/rows_typed_decisions_tez-think32.jsonl *> "$log\think32.log"
$env:TEZ_SYMBOLS = "123456789"; $env:TEZ_SYMBOL_WORD = "number"
python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_gemma4-12b-q8_0_ollama_digits.jsonl --tag digits *> "$log\digits.log"
$env:TEZ_SYMBOLS = "abcdefghijklmnop"; $env:TEZ_SYMBOL_WORD = "lowercase letter"
python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_gemma4-12b-q8_0_ollama_lower.jsonl --tag lower *> "$log\lower.log"
Remove-Item Env:TEZ_SYMBOLS; Remove-Item Env:TEZ_SYMBOL_WORD
python experiments/run_direct.py --data data/semif/authored144.jsonl --out results/authored144_gemma4-12b-q8_0_ollama.jsonl --tag ollama *> "$log\letters.log"
python experiments/think_td.py --budget 128 --limit 150 --out results/h2h/rows_typed_decisions_tez-think128.jsonl *> "$log\think128.log"
"ABL2DONE $(Get-Date)" | Out-File "$log\done.txt"
