# Chain 17b (GPU, llama.cpp): pruned Qwen3.5-4B GGUFs, served-embedding probes and letters, then the full head-to-head suite on 29 vs 32 blocks.
# Prompt caching OFF: llama.cpp b11100 crashes on partial prefix reuse with qwen35 (hybrid recurrent) GGUFs. Writes abl18\done.txt for chain 19.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HUB_OFFLINE = "1"; $env:HF_DATASETS_OFFLINE = "1"; $env:TEZ_CACHE_PROMPT = "0"
$log = "$env:TEMP\abl17b"; New-Item -ItemType Directory -Force $log | Out-Null
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 5
cmd /c "python experiments\pruned_server_bench.py --cuts L20:tools/models/Qwen3.5-4B-Q8_0-L20.gguf,L24:tools/models/Qwen3.5-4B-Q8_0-L24.gguf,L29:tools/models/Qwen3.5-4B-Q8_0-L29.gguf,L32:tools/models/Qwen3.5-4B-Q8_0.gguf --out results\pruned_server_qwen35-4b.json > $log\pruned.txt 2>&1"
$env:TEZ_TEMPLATE = "qwen3"
$exe = "C:\temp\llamacpp\llama-server.exe"
foreach ($cut in @(@("L29", "tools/models/Qwen3.5-4B-Q8_0-L29.gguf"), @("L32", "tools/models/Qwen3.5-4B-Q8_0.gguf"))) {
  $name = $cut[0]; $path = "$root\" + $cut[1]
  if (Test-Path "results\h2h_q4b_$name\summary.json") { continue }
  $p = Start-Process -FilePath $exe -WorkingDirectory "C:\temp\llamacpp" -ArgumentList "-m $path -ngl 99 -c 8192 -b 512 --port 8092 --host 127.0.0.1 -np 1 --no-webui" -WindowStyle Hidden -PassThru -RedirectStandardError "$log\server_$name.err" -RedirectStandardOutput "$log\server_$name.out"
  Start-Sleep 25
  cmd /c "python experiments\bench_h2h.py --model tez --tasks all --server http://127.0.0.1:8092 --out results\h2h_q4b_$name > $log\h2h_$name.txt 2>&1"
  Stop-Process -Id $p.Id -Force; Start-Sleep 5
}
New-Item -ItemType Directory -Force "$env:TEMP\abl18" | Out-Null
"CHAIN17B DONE $(Get-Date)" | Out-File "$env:TEMP\abl18\done.txt"
"CHAIN17B DONE $(Get-Date)" | Out-File "$log\done.txt"
