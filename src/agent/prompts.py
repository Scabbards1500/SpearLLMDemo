"""Load navigation goal prompts from prompts.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.utils.config import PROJECT_ROOT


@dataclass
class GoalPrompt:
    prompt_id: str
    label: str
    text: str


def _resolve_prompts_path(config_path: Path | None) -> Path:
    if config_path and config_path.is_file():
        return config_path
    user = PROJECT_ROOT / "prompts.yaml"
    if user.is_file():
        return user
    return PROJECT_ROOT / "prompts.yaml.example"


def load_prompts(config_path: Path | None = None) -> list[GoalPrompt]:
    path = _resolve_prompts_path(config_path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    prompts = [
        GoalPrompt(
            prompt_id=str(p["id"]),
            label=str(p["label"]),
            text=str(p["text"]).strip(),
        )
        for p in data.get("prompts", [])
    ]
    if not prompts:
        prompts.append(
            GoalPrompt(
                prompt_id="default",
                label="Default",
                text="Navigate to the nearest table in the scene.",
            )
        )
    return prompts


def default_prompt_index(config_path: Path | None = None) -> int:
    path = _resolve_prompts_path(config_path)
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return int(data.get("default_index", 0))
