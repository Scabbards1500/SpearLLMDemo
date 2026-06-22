"""2D overlays for agent pose on OpenCV views."""

from __future__ import annotations

import math

import cv2
import numpy as np

# BGR
AGENT_ARROW_COLOR = (0, 0, 255)


def draw_agent_arrow(
    image: np.ndarray,
    yaw_deg: float,
    *,
    center: tuple[int, int] | None = None,
    length: int = 55,
    color: tuple[int, int, int] = AGENT_ARROW_COLOR,
    thickness: int = 3,
    label: str = "agent",
) -> np.ndarray:
    """Draw a red heading arrow on a top-down map image (agent at center).

    Overhead camera follows the agent, so the marker is drawn at the image center.
    Yaw follows Unreal convention (0 deg ~ +X world, increases toward +Y).
    """
    out = image.copy()
    h, w = out.shape[:2]
    cx, cy = center if center is not None else (w // 2, h // 2)

    rad = math.radians(float(yaw_deg))
    tip_x = int(cx + length * math.cos(rad))
    tip_y = int(cy + length * math.sin(rad))

    cv2.circle(out, (cx, cy), 8, color, -1, lineType=cv2.LINE_AA)
    cv2.arrowedLine(
        out,
        (cx, cy),
        (tip_x, tip_y),
        color,
        thickness,
        tipLength=0.35,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        out,
        label,
        (cx + 12, cy - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def draw_agent_compass(
    image: np.ndarray,
    yaw_deg: float,
    *,
    size: int = 64,
    margin: int = 12,
    color: tuple[int, int, int] = AGENT_ARROW_COLOR,
) -> np.ndarray:
    """Small heading compass in the corner of the egocentric view."""
    out = image.copy()
    h, _w = out.shape[:2]
    cx = margin + size // 2
    cy = h - margin - size // 2
    return draw_agent_arrow(
        out,
        yaw_deg,
        center=(cx, cy),
        length=size // 2 - 6,
        color=color,
        thickness=2,
        label="",
    )
