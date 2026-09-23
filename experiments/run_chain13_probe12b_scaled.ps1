# Chain 13: refit the cached 12B NF4 features with per-layer standardisation (CPU only), 4B for parity; restore the server first.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:USE_TF = "0"; $env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"
$log = "$env:TEMP\abl13"; New-Item -ItemType Directory -Force $log | Out-Null
$exe = "C:\temp\llamacpp\llama-server.exe"
if (-not (Get-Process llama-server -ErrorAction SilentlyContinue)) {
  Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
}
if (-not (Test-Path "results\hidden_probe_sweep_gemma4-12b-nf4_std.json")) { cmd /c "python experiments\hidden_probe_sweep.py --model google/gemma-4-12B-it --tag gemma4-12b-nf4 --offline --truncate 40 --standardize --out results\hidden_probe_sweep_gemma4-12b-nf4_std.json > $log\sweep12b_std.txt 2>&1" }
if (-not (Test-Path "results\hidden_probe_sweep_qwen35-4b_std.json")) { cmd /c "python experiments\hidden_probe_sweep.py --model Qwen/Qwen3.5-4B --tag qwen35-4b --offline --standardize --out results\hidden_probe_sweep_qwen35-4b_std.json > $log\sweep4b_std.txt 2>&1" }
"CHAIN13DONE $(Get-Date)" | Out-File "$log\done.txt"
