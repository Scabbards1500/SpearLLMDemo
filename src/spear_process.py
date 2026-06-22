"""Helpers for stale SpearSim process cleanup on Windows."""

from __future__ import annotations

import subprocess
import sys
import time


def kill_stale_spear_processes() -> None:
    """Force-stop leftover SpearSim instances before launching a new one."""
    if sys.platform != "win32":
        return
    for exe in ("SpearSim.exe", "SpearSim-Cmd.exe"):
        subprocess.run(
            ["taskkill", "/F", "/IM", exe],
            capture_output=True,
            text=True,
        )
    time.sleep(0.8)
