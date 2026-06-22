# Check Windows prerequisites for SPEAR / Unreal Engine 5.5 builds.
#
# Usage:
#   .\scripts\check_ue_prerequisites.ps1

$ErrorActionPreference = "Continue"

function Test-RegistryPath($path) {
  return Test-Path "Registry::$path"
}

Write-Host "=== UE 5.5 / SPEAR build prerequisites ===" -ForegroundColor Cyan

$allOk = $true

# 1. Visual Studio C++ tools
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
  Write-Host "[FAIL] vswhere not found — install Visual Studio 2022" -ForegroundColor Red
  $allOk = $false
} else {
  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if ($vsPath) {
    Write-Host "[ OK ] Visual Studio C++ tools: $vsPath" -ForegroundColor Green
  } else {
    Write-Host "[FAIL] VS 2022 C++ workload missing (Desktop development with C++)" -ForegroundColor Red
    $allOk = $false
  }
}

# 2. cl.exe (only if VS dev shell already loaded)
$cl = Get-Command cl -ErrorAction SilentlyContinue
if ($cl) {
  Write-Host "[ OK ] cl.exe: $($cl.Source)" -ForegroundColor Green
} else {
  Write-Host "[WARN] cl.exe not in PATH (setup_spear.ps1 loads VS dev shell automatically)" -ForegroundColor Yellow
}

# 3. .NET Framework SDK (NetFxSDK) — required by UE SwarmInterface module
$netFxKey = "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\NETFXSDK"
$netFxFound = $false
if (Test-Path $netFxKey) {
  Get-ChildItem $netFxKey | ForEach-Object {
    $dir = (Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue).KitsInstallationFolder
    if ($dir -and (Test-Path $dir)) {
      Write-Host "[ OK ] NetFxSDK $($_.PSChildName): $dir" -ForegroundColor Green
      $netFxFound = $true
    }
  }
}
if (-not $netFxFound) {
  $legacy = "${env:ProgramFiles(x86)}\Microsoft SDKs\NETFXSDK"
  if (Test-Path $legacy) {
    Write-Host "[ OK ] NetFxSDK folder: $legacy" -ForegroundColor Green
    $netFxFound = $true
  }
}
if (-not $netFxFound) {
  Write-Host "[FAIL] .NET Framework SDK (NetFxSDK) not found" -ForegroundColor Red
  Write-Host "       UE build error: SwarmInterface / Could not find NetFxSDK install dir" -ForegroundColor Red
  Write-Host ""
  Write-Host "Fix — open Visual Studio Installer -> Modify VS 2022 -> Individual components:" -ForegroundColor Yellow
  Write-Host "  - .NET Framework 4.8 SDK" -ForegroundColor Yellow
  Write-Host "  - (optional) .NET Framework 4.8 targeting pack" -ForegroundColor Yellow
  Write-Host ""
  Write-Host "Or run (admin PowerShell):" -ForegroundColor Yellow
  $installPath = if ($vsPath) { $vsPath } else { "D:\Program Files\Microsoft Visual Studio\2022\Community" }
  Write-Host "  & `"${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vs_installer.exe`" modify ``" -ForegroundColor Gray
  Write-Host "    --installPath `"$installPath`" ``" -ForegroundColor Gray
  Write-Host "    --add Microsoft.Net.Component.4.8.SDK Microsoft.Net.Component.4.8.TargetingPack ``" -ForegroundColor Gray
  Write-Host "    --passive --wait" -ForegroundColor Gray
  $allOk = $false
}

# 4. Unreal Engine 5.5
$ue = "D:\Program Files\Epic Games\UE_5.5"
if (Test-Path $ue) {
  Write-Host "[ OK ] Unreal Engine 5.5: $ue" -ForegroundColor Green
} else {
  Write-Host "[FAIL] UE 5.5 not found at $ue" -ForegroundColor Red
  $allOk = $false
}

# 5. Windows SDK (informational)
$winSdk = "D:\Windows Kits\10"
if (Test-Path $winSdk) {
  Write-Host "[ OK ] Windows SDK: $winSdk" -ForegroundColor Green
} elseif (Test-Path "${env:ProgramFiles(x86)}\Windows Kits\10") {
  Write-Host "[ OK ] Windows SDK: ${env:ProgramFiles(x86)}\Windows Kits\10" -ForegroundColor Green
} else {
  Write-Host "[WARN] Windows 10/11 SDK path not found (may still be OK via VS)" -ForegroundColor Yellow
}

Write-Host ""
if ($allOk) {
  Write-Host "All critical checks passed. Safe to run .\scripts\setup_spear.ps1" -ForegroundColor Green
  exit 0
} else {
  Write-Host "Fix the [FAIL] items above before building SpearSim." -ForegroundColor Red
  exit 1
}
