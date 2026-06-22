# Install the official SPEAR Python client (NOT PyPI `pip install spear`).
#
# Usage:
#   conda activate spearenv
#   .\preparation\setup\ensure_spear_python.ps1

$ErrorActionPreference = "Stop"

$SpearPython = "D:\dev\spear\python"

if (-not (Test-Path $SpearPython)) {
  throw "SPEAR repo missing at D:\dev\spear. Clone with --recurse-submodules first."
}

Write-Host "Removing wrong PyPI package 'spear' if present..." -ForegroundColor Yellow
pip uninstall spear -y *>$null

Write-Host "Installing official spear-sim (editable) from $SpearPython ..." -ForegroundColor Cyan
pip install -e $SpearPython

python -c @"
import spear
print('spear version:', getattr(spear, '__version__', '?'))
print('spear_ext available:', spear.__can_import_spear_ext__)
if not spear.__can_import_spear_ext__:
    print('NOTE: spear_ext=False until setup_spear.ps1 step [4/6] succeeds.')
"@
