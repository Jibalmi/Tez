# Chain 12: option elimination on the head-to-head rows (server must be up; queued behind chain 11).
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$log = "$env:TEMP\abl12"; New-Item -ItemType Directory -Force $log | Out-Null
while (-not (Test-Path "$env:TEMP\abl11\done.txt")) { Start-Sleep 30 }
Start-Sleep 90
if (-not (Test-Path "results\option_elimination_gemma4-12b-q8_0.json")) { cmd /c "python experiments\option_elimination.py --tasks massive:en,emotion,ag_news,typed_decisions --out results\option_elimination_gemma4-12b-q8_0.json > $log\elim.txt 2>&1" }
"CHAIN12DONE $(Get-Date)" | Out-File "$log\done.txt"
