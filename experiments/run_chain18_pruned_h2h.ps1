# Chain 18 (GPU, llama.cpp): full head-to-head suite, zero-shot letters, on Qwen3.5-4B cut at 29 blocks vs uncut (does removing the top layers help decisions in general?).
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:PYTHONIOENCODING = "utf-8"; $env:TEZ_TEMPLATE = "qwen3"; $env:HF_DATASETS_OFFLINE = "1"; $env:HF_HUB_OFFLINE = "1"
$log = "$env:TEMP\abl18"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl17\done.txt")) { Start-Sleep 20 }
Start-Sleep 20
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
$exe = "C:\temp\llamacpp\llama-server.exe"
foreach ($cut in @(@("L29", "tools/models/Qwen3.5-4B-Q8_0-L29.gguf"), @("L32", "tools/models/Qwen3.5-4B-Q8_0.gguf"))) {
  $name = $cut[0]; $path = "$root\" + $cut[1]
  if (Test-Path "results\h2h_q4b_$name\summary.json") { continue }
  $p = Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m $path -ngl 99 -c 8192 -b 512 --port 8092 --host 127.0.0.1 -np 1 --no-webui" -WindowStyle Hidden -PassThru -RedirectStandardError "$log\server_$name.err" -RedirectStandardOutput "$log\server_$name.out"
  Start-Sleep 25
  cmd /c "python experiments\bench_h2h.py --model tez --tasks all --server http://127.0.0.1:8092 --out results\h2h_q4b_$name > $log\h2h_$name.txt 2>&1"
  Stop-Process -Id $p.Id -Force; Start-Sleep 5
}
Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last" -WindowStyle Hidden -RedirectStandardError "$log\server.err" -RedirectStandardOutput "$log\server.out"
"CHAIN18DONE $(Get-Date)" | Out-File "$log\done.txt"
