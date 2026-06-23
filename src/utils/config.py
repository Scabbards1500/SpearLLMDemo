"""Runtime configuration loaded from environment and defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    """Agent, recorder, and LLM settings."""

    project_root: Path = PROJECT_ROOT
    user_config: Path = PROJECT_ROOT / "user_config.yaml"
    scenes_config: Path = PROJECT_ROOT / "scenes.yaml"
    prompts_config: Path = PROJECT_ROOT / "prompts.yaml"
    recordings_dir: Path = PROJECT_ROOT / "recordings"
    episode_name: str = os.getenv("EPISODE_NAME", "episode_001")
    scene_name: str = os.getenv("SCENE_NAME", "debug_house")

    target_fps: float = float(os.getenv("TARGET_FPS", "30"))
    control_cadence: int = int(os.getenv("CONTROL_CADENCE", "10"))
    max_frames: int = int(os.getenv("MAX_FRAMES", "900"))

    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm_model: str = os.getenv("LLM_MODEL", "")

    def __post_init__(self) -> None:
        if not self.llm_model:
            self.llm_model = (
                "qwen3.6-plus" if self.llm_provider == "qwen" else "claude-opus-4-6"
            )

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "qwen":
            return self.dashscope_api_key
        return self.anthropic_api_key

    show_opencv: bool = os.getenv("SHOW_OPENCV", "1") not in ("0", "false", "False")
    overhead_camera: bool = os.getenv("OVERHEAD_CAMERA", "1") not in ("0", "false", "False")
    enable_recording: bool = os.getenv("RECORDING", "1") not in ("0", "false", "False")
    plan: bool = os.getenv("PLAN", "1") not in ("0", "false", "False")

    @property
    def episode_dir(self) -> Path:
        return self.recordings_dir / self.episode_name

    @property
    def dt(self) -> float:
        return 1.0 / self.target_fps
