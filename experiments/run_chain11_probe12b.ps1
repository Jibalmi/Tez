# Chain 11: layer sweep of the hidden-state probe on Gemma 4 12B (NF4 in-process, first 40 of 48 layers), queued behind chain 9b and the weight download.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"
$log = "$env:TEMP\abl11"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not ((Test-Path "$env:TEMP\abl9b\done.txt") -and (Test-Path "$log\dl_done.txt"))) { Start-Sleep 30 }
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
if (-not (Test-Path "results\hidden_probe_sweep_gemma4-12b-nf4.json")) { cmd /c "python experiments\hidden_probe_sweep.py --model google/gemma-4-12B-it --tag gemma4-12b-nf4 --load-4bit --truncate 40 --out results\hidden_probe_sweep_gemma4-12b-nf4.json > $log\sweep12b.txt 2>&1" }
$exe = "C:\temp\llamacpp\llama-server.exe"
Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
"CHAIN11DONE $(Get-Date)" | Out-File "$log\done.txt"
