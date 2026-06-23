"""OpenCV HUD overlays for the agent preview window."""

from __future__ import annotations

import cv2

from src.agent.prompts import GoalPrompt
from src.agent.env import WheelAction

VIEW_AGENT = "agent"
VIEW_OVERHEAD = "overhead"


def normalize_nav_phase(phase: str | None, *, arrived: bool = False) -> str:
    """Map plan phases to HUD labels: explore | approach | arrived."""
    if arrived:
        return "arrived"
    p = (phase or "explore").lower()
    if p == "arrived":
        return "arrived"
    if p in ("approach", "fine_tune"):
        return "approach"
    return "explore"


def _nav_status_line(
    nav_phase: str,
    *,
    arrived: bool,
    landed: bool,
    llm_busy: bool,
    frames_since_llm: int,
    control_cadence: int,
) -> str:
    if arrived:
        return nav_phase
    detail = ""
    if not landed:
        detail = "waiting landing"
    elif llm_busy:
        detail = "LLM calling..."
    else:
        until_next = max(0, control_cadence - frames_since_llm)
        if until_next > 0:
            detail = f"hold (next LLM in {until_next} fr)"
        else:
            detail = "hold"
    return f"{nav_phase} | {detail}"


def draw_agent_hud(
    image,
    goal: GoalPrompt,
    action: WheelAction,
    frame_index: int,
    llm_busy: bool,
    *,
    arrived: bool = False,
    landed: bool = True,
    nav_phase: str | None = None,
    frames_since_llm: int = 0,
    control_cadence: int = 10,
) -> None:
    phase = normalize_nav_phase(nav_phase, arrived=arrived)
    status = _nav_status_line(
        phase,
        arrived=arrived,
        landed=landed,
        llm_busy=llm_busy,
        frames_since_llm=frames_since_llm,
        control_cadence=control_cadence,
    )
    lines = [
        f"[AGENT] {goal.label}",
        f"frame {frame_index}  {status}",
        f"L={action.left:.2f} R={action.right:.2f}",
        "1/2/3=prompt  Q=quit",
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


def draw_overhead_hud(
    image,
    goal: GoalPrompt,
    action: WheelAction,
    frame_index: int,
    location: dict[str, float],
    *,
    arrived: bool = False,
    landed: bool = True,
    llm_busy: bool = False,
    nav_phase: str | None = None,
    frames_since_llm: int = 0,
    control_cadence: int = 10,
) -> None:
    phase = normalize_nav_phase(nav_phase, arrived=arrived)
    status = _nav_status_line(
        phase,
        arrived=arrived,
        landed=landed,
        llm_busy=llm_busy,
        frames_since_llm=frames_since_llm,
        control_cadence=control_cadence,
    )
    lines = [
        f"[OVERHEAD] {goal.label}",
        f"frame {frame_index}  {status}",
        f"X={location['X']:.0f} Y={location['Y']:.0f}  L={action.left:.2f} R={action.right:.2f}",
        "1/2/3=prompt  Q=quit",
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
