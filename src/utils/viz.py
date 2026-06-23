"""2D overlays for agent pose on OpenCV views."""

from __future__ import annotations

import math

import cv2
import numpy as np

# BGR
AGENT_ARROW_COLOR = (0, 0, 255)


def _yaw_to_image_delta(yaw_deg: float, length: float) -> tuple[float, float]:
    """Map Unreal ground yaw to OpenCV pixel offset for a world-aligned top-down view.

    Unreal: yaw=0° faces world +X; overhead camera uses pitch=-90°, yaw=0° (fixed north-up).
    On the capture, world +X ≈ image up, world +Y ≈ image right; OpenCV y grows downward.
    """
    rad = math.radians(float(yaw_deg))
    return length * math.sin(rad), -length * math.cos(rad)


def draw_agent_arrow(
    image: np.ndarray,
    yaw_deg: float,
    *,
    center: tuple[int, int] | None = None,
    length: int = 55,
    color: tuple[int, int, int] = AGENT_ARROW_COLOR,
    thickness: int = 3,
    label: str = "heading",
) -> np.ndarray:
    """Draw a red heading arrow (Unreal yaw on the ground plane).

    Used on the overhead follow-cam: agent stays at image center; arrow shows
    which way the robot will drive (+X local / differential-drive forward).
    """
    out = image.copy()
    h, w = out.shape[:2]
    cx, cy = center if center is not None else (w // 2, h // 2)

    dx, dy = _yaw_to_image_delta(yaw_deg, length)
    tip_x = int(cx + dx)
    tip_y = int(cy + dy)

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
    """Mini top-down heading dial in the egocentric window (world yaw, not image depth)."""
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
