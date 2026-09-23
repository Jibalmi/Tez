# Chain 22 (GPU, llama.cpp): clean latency recheck of the pruned GGUFs, then the head-to-head suite on the 24-block cut. Queued behind chain 21.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:TEZ_CACHE_PROMPT = "0"
$log = "$env:TEMP\abl22"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl21\done.txt")) { Start-Sleep 20 }
Start-Sleep 20
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
cmd /c "python experiments\latency_recheck.py --out results\pruned_latency_qwen35-4b.json > $log\latency.txt 2>&1"
$env:TEZ_TEMPLATE = "qwen3"
$exe = "C:\temp\llamacpp\llama-server.exe"
if (-not (Test-Path "results\h2h_q4b_L24\summary.json")) {
  $p = Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m $root\tools\models\Qwen3.5-4B-Q8_0-L24.gguf -ngl 99 -c 8192 -b 512 --port 8092 --host 127.0.0.1 -np 1 --no-webui" -WindowStyle Hidden -PassThru -RedirectStandardError "$log\server_L24.err" -RedirectStandardOutput "$log\server_L24.out"
  Start-Sleep 25
  cmd /c "python experiments\bench_h2h.py --model tez --tasks all --server http://127.0.0.1:8092 --out results\h2h_q4b_L24 > $log\h2h_L24.txt 2>&1"
  Stop-Process -Id $p.Id -Force; Start-Sleep 5
}
Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
"CHAIN22DONE $(Get-Date)" | Out-File "$log\done.txt"
