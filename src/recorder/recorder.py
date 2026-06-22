"""In-loop frame-perfect recorder: frames/ + manifest.jsonl."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.agent.env import Observation, WheelAction
from src.agent.prompts import GoalPrompt
from src.agent.scenes import ScenePreset


def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


class FrameRecorder:
    """Write one manifest row + RGB image per simulation frame (zero drops)."""

    def __init__(
        self,
        episode_dir: Path,
        *,
        target_fps: float,
        scene: ScenePreset,
        control_cadence: int,
        llm_model: str,
        enable_plan: bool = True,
    ) -> None:
        self.episode_dir = Path(episode_dir)
        self.frames_dir = self.episode_dir / "frames"
        self.manifest_path = self.episode_dir / "manifest.jsonl"
        self.meta_path = self.episode_dir / "episode_meta.json"
        self.target_fps = target_fps
        self.dt = 1.0 / target_fps
        self.scene = scene
        self.control_cadence = control_cadence
        self.llm_model = llm_model
        self.enable_plan = enable_plan

        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self.manifest_path.open("w", encoding="utf-8")
        self._records_written = 0
        self._expected_frame_idx: int | None = None

    def write_episode_meta(self, goal: GoalPrompt) -> None:
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scene": self.scene.name,
            "level": self.scene.level,
            "spawn": self.scene.spawn,
            "goal": {
                "id": goal.prompt_id,
                "label": goal.label,
                "text": goal.text,
            },
            "target_fps": self.target_fps,
            "control_cadence": self.control_cadence,
            "llm_model": self.llm_model,
            "plan_enabled": self.enable_plan,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def record_frame(
        self,
        obs: Observation,
        action_in_effect: WheelAction,
        action_id: int,
    ) -> None:
        frame_idx = obs.frame_index
        if self._expected_frame_idx is None:
            self._expected_frame_idx = frame_idx
        elif frame_idx != self._expected_frame_idx:
            raise RuntimeError(
                f"Frame gap detected: expected {self._expected_frame_idx}, got {frame_idx}"
            )

        rel_image = f"frames/{frame_idx}.png"
        image_path = self.episode_dir / rel_image
        bgr = _rgb_to_bgr(obs.rgb)
        if not cv2.imwrite(str(image_path), bgr):
            raise RuntimeError(f"Failed to write frame image: {image_path}")

        loc = obs.location
        rot = obs.rotation
        record: dict[str, Any] = {
            "frame_idx": frame_idx,
            "t": round(frame_idx * self.dt, 4),
            "action": {"left": action_in_effect.left, "right": action_in_effect.right},
            "location": [loc["X"], loc["Y"], loc["Z"]],
            "rotation": [rot["Pitch"], rot["Yaw"], rot["Roll"]],
            "image": rel_image,
            "action_id": action_id,
        }
        self._manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._manifest.flush()
        self._records_written += 1
        self._expected_frame_idx = frame_idx + 1

    @property
    def records_written(self) -> int:
        return self._records_written

    def close(self) -> None:
        self._manifest.close()
