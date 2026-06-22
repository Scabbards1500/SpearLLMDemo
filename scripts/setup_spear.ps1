# One-time SPEAR build from official github.com/spear-sim/spear
#
# Usage (regular PowerShell is OK — script loads VS dev tools automatically):
#   conda activate spearenv
#   cd d:\python\SpearLLMDemo
#   .\scripts\setup_spear.ps1
#
# Requires: VS 2022 with "Desktop development with C++", UE 5.5, git

$ErrorActionPreference = "Stop"

$SpearRoot = "D:\dev\spear"
$UeDir = "D:\Program Files\Epic Games\UE_5.5"

function Ensure-VsDevTools {
  $cl = Get-Command cl -ErrorAction SilentlyContinue
  if ($cl) {
    Write-Host "VS tools already in PATH: $($cl.Source)" -ForegroundColor Green
    return
  }

  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio 2022 with the C++ desktop workload."
  }

  $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $vsPath) {
    throw "Visual Studio 2022 C++ tools not found. Install workload: 'Desktop development with C++'."
  }

  $devShell = Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1"
  if (-not (Test-Path $devShell)) {
    throw "Launch-VsDevShell.ps1 not found under: $vsPath"
  }

  Write-Host "Loading Visual Studio dev environment..." -ForegroundColor Yellow
  & $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null

  $cl = Get-Command cl -ErrorAction SilentlyContinue
  if (-not $cl) {
    throw "Failed to load cl.exe. Try opening 'Developer PowerShell for VS 2022' manually and rerun."
  }
  Write-Host "VS tools loaded: $($cl.Source)" -ForegroundColor Green
}

function Invoke-SpearStep {
  param(
    [string]$Label,
    [scriptblock]$Command
  )
  Write-Host "`n$Label" -ForegroundColor Yellow
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed (exit $LASTEXITCODE): $Label"
  }
}

Write-Host "=== SPEAR setup ===" -ForegroundColor Cyan
Write-Host "SPEAR root: $SpearRoot"
Write-Host "UE 5.5:     $UeDir"

Ensure-VsDevTools

if (-not (Test-Path $SpearRoot)) {
  throw "SPEAR repo not found at $SpearRoot. Clone: git clone https://github.com/spear-sim/spear $SpearRoot --recurse-submodules"
}
if (-not (Test-Path $UeDir)) {
  throw "Unreal Engine 5.5 not found at $UeDir"
}

Set-Location $SpearRoot

Invoke-SpearStep "[1/6] Submodules..." {
  git submodule update --init --recursive
}

Invoke-SpearStep "[2/6] Terminal check (Windows)..." {
  python tools/check_terminal_windows.py
}

Invoke-SpearStep "[3/6] Third-party C++ libs..." {
  python tools/build_third_party_libs.py
}

Invoke-SpearStep "[4/6] spear_ext extension..." {
  python tools/install_python_extension.py
}

Invoke-SpearStep "[5/6] Copy engine content..." {
  python tools/copy_engine_content.py --unreal-engine-dir $UeDir
}

Write-Host "`n[pre-check] UE build prerequisites..." -ForegroundColor Yellow
$checkScript = Join-Path $PSScriptRoot "check_ue_prerequisites.ps1"
& $checkScript
if ($LASTEXITCODE -ne 0) {
  throw "UE prerequisites missing. Fix items above, then rerun setup (steps 1-5 can be skipped if already done)."
}

Invoke-SpearStep "[6/6] Build SpearSim (long — 30–90+ min)..." {
  python tools/run_uat.py --unreal-engine-dir $UeDir -build -cook -stage -package -archive -pak
}

$Exe = Join-Path $SpearRoot "cpp\unreal_projects\SpearSim\Standalone-Development\Windows\SpearSim.exe"
Write-Host "`nDone. Expected executable:" -ForegroundColor Green
Write-Host $Exe
if (Test-Path $Exe) {
  Write-Host "SPEAR_BUILD: PASS" -ForegroundColor Green
} else {
  throw "SpearSim.exe not found — check build output above."
}
