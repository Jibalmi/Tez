# Chain 15 (GPU, in-process Qwen3.5-4B): many-decisions-per-pass layouts, then cross-lingual probe transfer.
# Waits for chain 14 (12B teacher labels on the server), stops the server, restores it at the end.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:USE_TF = "0"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$log = "$env:TEMP\abl15"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl14\done.txt")) { Start-Sleep 20 }
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
if (-not (Test-Path "results\probe_multiq_qwen35-4b.json")) { cmd /c "python experiments\probe_multiq.py --out results\probe_multiq_qwen35-4b.json > $log\multiq.txt 2>&1" }
$env:HF_DATASETS_OFFLINE = "0"; $env:HF_HUB_OFFLINE = "0"; $env:CURL_CA_BUNDLE = ""; $env:REQUESTS_CA_BUNDLE = ""
if (-not (Test-Path "results\probe_crosslingual_qwen35-4b.json")) { cmd /c "python experiments\probe_crosslingual.py --out results\probe_crosslingual_qwen35-4b.json > $log\xling.txt 2>&1" }
$exe = "C:\temp\llamacpp\llama-server.exe"
Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
"CHAIN15DONE $(Get-Date)" | Out-File "$log\done.txt"
