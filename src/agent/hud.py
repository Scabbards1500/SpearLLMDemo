"""OpenCV HUD overlays for the agent preview window."""

from __future__ import annotations

import cv2

from src.agent.prompts import GoalPrompt
from src.agent.env import WheelAction

VIEW_AGENT = "agent"
VIEW_OVERHEAD = "overhead"


def draw_agent_hud(
    image,
    goal: GoalPrompt,
    action: WheelAction,
    frame_index: int,
    view_mode: str,
    llm_busy: bool,
    *,
    arrived: bool = False,
) -> None:
    if arrived:
        status = "ARRIVED"
    elif llm_busy:
        status = "LLM thinking..."
    else:
        status = "LLM idle"
    view = "AGENT" if view_mode == VIEW_AGENT else "OVERHEAD"
    lines = [
        f"[{view}] {goal.label}",
        f"frame {frame_index}  {status}",
        f"L={action.left:.2f} R={action.right:.2f}",
        "1/2/3=prompt  Enter=view  Q=quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (12, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255) if i == 0 else ((0, 200, 255) if arrived and i == 1 else (0, 255, 0)),
            2,
            cv2.LINE_AA,
        )
