# Chain 10: retrieval-narrowed Banking77 (waits for chain 9 to finish and restore the llama-server in embeddings mode).
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:USE_TF = "0"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:PYTHONIOENCODING = "utf-8"
$log = "$env:TEMP\abl10"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl9\done.txt")) { Start-Sleep 30 }
Start-Sleep 60
if (-not (Test-Path "results\retrieval_narrow_banking77_minilm.json")) { cmd /c "python experiments\retrieval_narrow.py --embedder minilm --ks 5,10,15,20 --out results\retrieval_narrow_banking77_minilm.json > $log\minilm.txt 2>&1" }
if (-not (Test-Path "results\retrieval_narrow_banking77_minilm_none.json")) { cmd /c "python experiments\retrieval_narrow.py --embedder minilm --ks 10,20 --with-none --out results\retrieval_narrow_banking77_minilm_none.json > $log\minilm_none.txt 2>&1" }
if (-not (Test-Path "results\retrieval_narrow_banking77_server.json")) { cmd /c "python experiments\retrieval_narrow.py --embedder server --ks 10,20 --out results\retrieval_narrow_banking77_server.json > $log\server.txt 2>&1" }
"CHAIN10DONE $(Get-Date)" | Out-File "$log\done.txt"
