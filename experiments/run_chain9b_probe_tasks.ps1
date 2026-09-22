# Chain 9b: task probes (fixed for >26 options), queued behind chain 10; restores the server afterwards.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$log = "$env:TEMP\abl9b"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl10\done.txt")) { Start-Sleep 30 }
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
cmd /c "python experiments\hidden_probe_tasks.py --model Qwen/Qwen3.5-4B --tag qwen35-4b --tasks banking77,emotion,sst5,massive:en,xnli:en,boolq --layers 18,22,26,-1 --n-train 2000 > $log\tasks.txt 2>&1"
$exe = "C:\temp\llamacpp\llama-server.exe"
Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
"CHAIN9BDONE $(Get-Date)" | Out-File "$log\done.txt"
