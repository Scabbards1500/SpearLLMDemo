# Preparation — environment setup, smoke tests, dev demos

Not part of the assignment deliverable. Use while setting up and debugging.

## Setup (one-time)

```powershell
conda activate spearenv
cd d:\python\SpearLLMDemo

# Install official SPEAR Python package (not PyPI `spear`)
.\preparation\setup\ensure_spear_python.ps1

# Build SpearSim (30-90+ min, first time only)
.\preparation\setup\setup_spear.ps1

# Check UE / VS prerequisites
.\preparation\setup\check_ue_prerequisites.ps1
```

## Smoke tests

```powershell
python preparation/tests/test_ue55.py
python preparation/tests/test_claude.py
```

## Dev demos (no LLM)

```powershell
python preparation/demos/demo_hardcoded.py
python preparation/demos/demo_toggle_view.py
python preparation/demos/explore_scene.py --scene debug
```

## Utilities

```powershell
# Kill leftover SpearSim before reconnecting
.\preparation\setup\kill_spear.ps1
```

## Assignment entry point

```powershell
python -m src.main
```
