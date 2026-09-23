# Chain 23 (GPU, llama.cpp): where a decision's latency goes (n_probs 0/20/200 vs /embedding) on the cut 4B, full 4B and 12B. Queued behind chain 22.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"
$log = "$env:TEMP\abl23"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl22\done.txt")) { Start-Sleep 20 }
Start-Sleep 20
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
cmd /c "python experiments\latency_breakdown.py --out results\latency_breakdown.json > $log\latency.txt 2>&1"
$exe = "C:\temp\llamacpp\llama-server.exe"
Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
"CHAIN23DONE $(Get-Date)" | Out-File "$log\done.txt"
