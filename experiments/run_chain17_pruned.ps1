# Chain 17 (GPU, llama.cpp): depth-pruned Qwen3.5-4B GGUFs (20/24/29/32 blocks) - letters (= logit lens) and served-embedding probes, with latency.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"
$log = "$env:TEMP\abl17"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl16\done.txt")) { Start-Sleep 20 }
Start-Sleep 20
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
cmd /c "python experiments\pruned_server_bench.py --cuts L20:tools/models/Qwen3.5-4B-Q8_0-L20.gguf,L24:tools/models/Qwen3.5-4B-Q8_0-L24.gguf,L29:tools/models/Qwen3.5-4B-Q8_0-L29.gguf,L32:tools/models/Qwen3.5-4B-Q8_0.gguf --out results\pruned_server_qwen35-4b.json > $log\pruned.txt 2>&1"
$exe = "C:\temp\llamacpp\llama-server.exe"
Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
"CHAIN17DONE $(Get-Date)" | Out-File "$log\done.txt"
