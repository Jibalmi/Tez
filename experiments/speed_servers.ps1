<#
Server control for the speed_* experiments (Windows PowerShell 5.1).

  .\experiments\speed_servers.ps1 stop-prod            stop ONLY the production llama-server (exact command-line match)
  .\experiments\speed_servers.ps1 start-prod           restore the production llama-server with exactly the brief's command
  .\experiments\speed_servers.ps1 start -Port 8091 -Extra "--cache-ram 0"            the production model with extra flags
  .\experiments\speed_servers.ps1 start -Port 8095 -Model <gguf> -Extra "..." -Base "-ngl 99 -c 4096 -b 512 -np 1 --no-webui --embeddings --pooling last"
  .\experiments\speed_servers.ps1 stop -Port 8095      stop the llama-server listening on that port (started by this script)
  .\experiments\speed_servers.ps1 health               /health of :8091 and /healthz of :8787

Every started server is detached (hidden window), logs to %TEMP%\speed_llama_<port>.log, and is waited on until /health is ok.
#>
param(
    [Parameter(Position = 0)][string]$Action = "health",
    [int]$Port = 8091,
    [string]$Model = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795",
    [string]$Base = "-ngl 99 -c 4096 -b 512 -np 1 --no-webui --swa-full --embeddings --pooling last",
    [string]$Extra = ""
)

$Exe = "C:\temp\llamacpp\llama-server.exe"
$ProdCmd = '"C:\temp\llamacpp\llama-server.exe" -m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last'
$ProdArgs = "-m C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795 -ngl 99 -c 4096 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full --embeddings --pooling last"

function Wait-Health([int]$p) {
    for ($i = 0; $i -lt 300; $i++) {
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$p/health" -TimeoutSec 2
            if ($r.status -eq "ok") { return $true }
        } catch { }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Start-Llama([string]$argline, [int]$p) {
    $log = Join-Path $env:TEMP "speed_llama_$p.log"
    $err = Join-Path $env:TEMP "speed_llama_$p.err.log"
    $proc = Start-Process -FilePath $Exe -ArgumentList $argline -WorkingDirectory "C:\temp\llamacpp" -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
    if (Wait-Health $p) { Write-Output "started pid $($proc.Id) on :$p  ($argline)" }
    else { Write-Output "FAILED to start on :$p, see $err"; exit 1 }
}

switch ($Action) {
    "stop-prod" {
        $procs = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object { $_.CommandLine.Trim() -eq $ProdCmd }
        if (-not $procs) { Write-Output "production llama-server not running"; break }
        foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force; Write-Output "stopped production llama-server pid $($p.ProcessId)" }
        Start-Sleep -Seconds 4
    }
    "start-prod" { Start-Llama $ProdArgs 8091 }
    "start" { Start-Llama "-m $Model $Base --port $Port --host 127.0.0.1 $Extra".Trim() $Port }
    "stop" {
        $procs = Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object { $_.CommandLine -match "--port $Port( |$)" }
        foreach ($p in $procs) {
            if ($p.CommandLine.Trim() -eq $ProdCmd) { Write-Output "refusing to stop the production server with 'stop'; use stop-prod"; continue }
            Stop-Process -Id $p.ProcessId -Force; Write-Output "stopped pid $($p.ProcessId) on :$Port"
        }
        Start-Sleep -Seconds 4
    }
    "health" {
        try { $a = Invoke-RestMethod -Uri "http://127.0.0.1:8091/health" -TimeoutSec 5; Write-Output ":8091 /health -> $($a | ConvertTo-Json -Compress)" } catch { Write-Output ":8091 /health -> DOWN ($($_.Exception.Message))" }
        try { $b = Invoke-RestMethod -Uri "http://127.0.0.1:8787/healthz" -TimeoutSec 10; Write-Output ":8787 /healthz -> $($b | ConvertTo-Json -Compress)" } catch { Write-Output ":8787 /healthz -> DOWN ($($_.Exception.Message))" }
        Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | ForEach-Object { Write-Output "llama-server pid $($_.ProcessId): $($_.CommandLine)" }
    }
}
