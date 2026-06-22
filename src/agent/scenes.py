"""Scene presets: level paths and spawn locations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.utils.config import PROJECT_ROOT

DEBUG_LEVEL = "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000.debug_0000"

# Bottom-right corner of the debug house (cm). Tune in scenes.yaml if needed.
DEBUG_HOUSE_SPAWN = {"X": 180.0, "Y": 420.0, "Z": 110.0}


@dataclass
class ScenePreset:
    name: str
    level: str
    spawn: dict[str, float]
    description: str = ""


def load_scene_preset(name: str, config_path: Path | None = None) -> ScenePreset:
    path = config_path or PROJECT_ROOT / "scenes.yaml"
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if name in data:
            entry = data[name]
            spawn = entry.get("spawn", DEBUG_HOUSE_SPAWN)
            return ScenePreset(
                name=name,
                level=entry.get("level", DEBUG_LEVEL),
                spawn={k: float(spawn[k]) for k in ("X", "Y", "Z")},
                description=entry.get("description", ""),
            )

    if name in ("debug", "debug_house"):
        return ScenePreset(
            name="debug_house",
            level=DEBUG_LEVEL,
            spawn=dict(DEBUG_HOUSE_SPAWN),
            description="SPEAR debug house; sphere starts bottom-right.",
        )
    raise KeyError(f"Unknown scene preset: {name}")
