"""Smoke test: locate Unreal Engine 5.5 installation on Windows."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CANDIDATE_ROOTS = [
    Path(r"C:\Program Files\Epic Games"),
    Path(r"D:\Program Files\Epic Games"),
    Path(r"D:\Epic Games"),
    Path(r"E:\Epic Games"),
    Path(r"C:\UE"),
    Path(r"D:\UE"),
    Path(os.path.expandvars(r"%LOCALAPPDATA%\EpicGamesLauncher\Data")) / "InstalledPlugins",
]

REQUIRED_RELATIVE = [
    Path("Engine/Binaries/Win64/UnrealEditor.exe"),
    Path("Engine/Build/Build.version"),
]


def read_engine_version(engine_root: Path) -> dict | None:
    version_file = engine_root / "Engine/Build/Build.version"
    if not version_file.is_file():
        return None
    return json.loads(version_file.read_text(encoding="utf-8"))


def find_ue_installs() -> list[tuple[Path, dict]]:
    found: list[tuple[Path, dict]] = []
    seen: set[str] = set()

    def consider(path: Path) -> None:
        path = path.resolve()
        key = str(path).lower()
        if key in seen:
            return
        editor = path / REQUIRED_RELATIVE[0]
        version = read_engine_version(path)
        if editor.is_file() and version is not None:
            seen.add(key)
            found.append((path, version))

    for root in CANDIDATE_ROOTS:
        if not root.exists():
            continue
        if root.name.startswith("UE_"):
            consider(root)
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("UE_"):
                consider(child)

    # Launcher manifest (common on Windows)
    manifest = (
        Path(os.path.expandvars(r"%PROGRAMDATA%"))
        / "Epic"
        / "UnrealEngineLauncher"
        / "LauncherInstalled.dat"
    )
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for item in data.get("InstallationList", []):
                install = Path(item.get("InstallLocation", ""))
                if install.is_dir():
                    consider(install)
        except json.JSONDecodeError:
            pass

    return found


def main() -> None:
    installs = find_ue_installs()
    if not installs:
        print("UE_TEST: FAIL — no Unreal Engine installation found")
        sys.exit(1)

    print(f"Found {len(installs)} Unreal Engine install(s):")
    ue55_ok = False
    for root, version in installs:
        major = version.get("MajorVersion")
        minor = version.get("MinorVersion")
        label = f"{major}.{minor}"
        editor = root / REQUIRED_RELATIVE[0]
        print(f"  - {root}")
        print(f"    version: {label} (patch {version.get('PatchVersion')})")
        print(f"    editor:  {editor}")
        if major == 5 and minor == 5:
            ue55_ok = True

    if ue55_ok:
        print("UE_TEST: PASS — UE 5.5 detected")
    else:
        print("UE_TEST: FAIL — Unreal Editor found, but no UE 5.5 install")
        sys.exit(1)


if __name__ == "__main__":
    main()
