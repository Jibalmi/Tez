# Full head-to-head chain, detached from any tool timeout.
#   Tez on all tasks (llama-server must be up) -> stop server -> Laya checkpoints on GPU -> restart server
$ErrorActionPreference = "Continue"
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"
$log  = "$env:TEMP\h2h"
New-Item -ItemType Directory -Force $log | Out-Null
Set-Location $root
$env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

python experiments/bench_h2h.py --model tez --tasks all --out results/h2h *> "$log\tez.log"

Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 4
foreach ($m in @("laya-en", "laya-ml", "laya-td")) {
  python experiments/bench_h2h.py --model $m --tasks all --out results/h2h *> "$log\$m.log"
}

Start-Process -FilePath "C:\Users\migue\llamacpp-bin\llama-server.exe" -WorkingDirectory "C:\Users\migue\llamacpp-bin" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 2048 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full" -WindowStyle Hidden -RedirectStandardError "$env:TEMP\llama_q8_swafull.err" -RedirectStandardOutput "$env:TEMP\llama_q8_swafull.out"
"H2HDONE $(Get-Date)" | Out-File "$log\done.txt"
