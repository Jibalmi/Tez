# Chain 8: full-layer hidden-state cache + probe sweep on Qwen3.5-4B (GPU must be free), then restore the server.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$log = "$env:TEMP\abl8"; New-Item -ItemType Directory -Force $log | Out-Null
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
cmd /c "python experiments\hidden_probe_sweep.py --model Qwen/Qwen3.5-4B --tag qwen35-4b --out results\hidden_probe_sweep_qwen35-4b.json > $log\sweep.txt 2>&1"
$exe = "C:\temp\llamacpp\llama-server.exe"
Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
"CHAIN8DONE $(Get-Date)" | Out-File "$log\done.txt"
