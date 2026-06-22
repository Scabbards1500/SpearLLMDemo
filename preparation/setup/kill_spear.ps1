# Kill stale SpearSim processes (fixes PID mismatch on reconnect).
# Usage: .\preparation\setup\kill_spear.ps1

$names = @("SpearSim", "SpearSim-Cmd", "UnrealEditor")

foreach ($name in $names) {
  $procs = Get-Process -Name $name -ErrorAction SilentlyContinue
  if ($procs) {
    Write-Host "Stopping $($procs.Count) x $name ..." -ForegroundColor Yellow
    $procs | Stop-Process -Force -ErrorAction SilentlyContinue
  }
}

Start-Sleep -Seconds 1
$left = Get-Process -Name "SpearSim*" -ErrorAction SilentlyContinue
if ($left) {
  Write-Host "Warning: some SpearSim processes may still be running." -ForegroundColor Red
  $left | Format-Table Id, ProcessName -AutoSize
  exit 1
}

Write-Host "No SpearSim processes running. Safe to start agent." -ForegroundColor Green
