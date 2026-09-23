# Chain 14: 12B zero-shot answers on typed-decisions train+test (pseudo-labels for label-free probes). Needs the server up.
$root = "C:\Users\migue\OneDrive\Desktop\Personal\AILABS\Tez"; Set-Location $root
$env:PYTHONIOENCODING = "utf-8"; $env:HF_DATASETS_OFFLINE = "1"; $env:HF_HUB_OFFLINE = "1"
$log = "$env:TEMP\abl14"; New-Item -ItemType Directory -Force $log | Out-Null
if (-not (Test-Path "results\teacher_labels_gemma4-12b-q8_0.json")) { cmd /c "python experiments\teacher_labels.py --out results\teacher_labels_gemma4-12b-q8_0.npz > $log\teacher.txt 2>&1" }
"CHAIN14DONE $(Get-Date)" | Out-File "$log\done.txt"
