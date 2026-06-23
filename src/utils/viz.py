"""2D overlays for agent pose on OpenCV views."""

from __future__ import annotations

import math

import cv2
import numpy as np

# BGR
HEADING_ARROW_COLOR = (0, 0, 255)
MOTION_ARROW_COLOR = (0, 255, 0)
INTENT_ARROW_COLOR = (255, 0, 0)
AGENT_ARROW_COLOR = HEADING_ARROW_COLOR  # backward compat


def motion_yaw_from_displacement(
    dx: float,
    dy: float,
    *,
    min_dist_cm: float = 0.4,
) -> float | None:
    """World XY displacement (cm) -> motion yaw (degrees), same convention as Unreal yaw."""
    if math.hypot(dx, dy) < min_dist_cm:
        return None
    return math.degrees(math.atan2(dy, dx))


def intent_yaw_from_wheels(
    heading_yaw_deg: float,
    left: float,
    right: float,
    *,
    min_forward: float = 0.05,
    min_turn: float = 0.05,
    max_bias_deg: float = 50.0,
    pivot_bias_deg: float = 35.0,
) -> float | None:
    """Wheel command -> intended drive direction (heading + differential turn bias)."""
    forward = (left + right) * 0.5
    turn = right - left
    if forward < min_forward and abs(turn) < min_turn:
        return None
    if forward < min_forward:
        bias = pivot_bias_deg if turn > 0 else -pivot_bias_deg
        return heading_yaw_deg + bias
    bias = math.degrees(math.atan2(turn * 0.5, forward))
    bias = max(-max_bias_deg, min(max_bias_deg, bias))
    return heading_yaw_deg + bias


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
    color: tuple[int, int, int] = HEADING_ARROW_COLOR,
    thickness: int = 3,
    label: str = "heading",
    label_offset: tuple[int, int] = (12, -12),
) -> np.ndarray:
    """Draw a heading or motion arrow (Unreal yaw on the ground plane).

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
    if label:
        lx = cx + label_offset[0]
        ly = cy + label_offset[1]
        cv2.putText(
            out,
            label,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def draw_pose_arrows(
    image: np.ndarray,
    heading_yaw_deg: float,
    motion_yaw_deg: float | None,
    *,
    intent_yaw_deg: float | None = None,
    center: tuple[int, int] | None = None,
    length: int = 55,
) -> np.ndarray:
    """Red = heading; green = ground motion; blue = wheel intent (approach phase)."""
    out = image
    if motion_yaw_deg is not None:
        out = draw_agent_arrow(
            out,
            motion_yaw_deg,
            center=center,
            length=max(length - 8, 24),
            color=MOTION_ARROW_COLOR,
            thickness=2,
            label="motion",
            label_offset=(12, 20),
        )
    if intent_yaw_deg is not None:
        out = draw_agent_arrow(
            out,
            intent_yaw_deg,
            center=center,
            length=max(length - 4, 28),
            color=INTENT_ARROW_COLOR,
            thickness=2,
            label="intent",
            label_offset=(-52, 8),
        )
    return draw_agent_arrow(
        out,
        heading_yaw_deg,
        center=center,
        length=length,
        color=HEADING_ARROW_COLOR,
        thickness=3,
        label="heading",
        label_offset=(12, -12),
    )


def draw_agent_compass(
    image: np.ndarray,
    yaw_deg: float,
    motion_yaw_deg: float | None = None,
    *,
    intent_yaw_deg: float | None = None,
    size: int = 64,
    margin: int = 12,
    color: tuple[int, int, int] = HEADING_ARROW_COLOR,
) -> np.ndarray:
    """Mini top-down dial: red heading, green motion, optional blue wheel intent."""
    out = image.copy()
    h, _w = out.shape[:2]
    cx = margin + size // 2
    cy = h - margin - size // 2
    center = (cx, cy)
    if motion_yaw_deg is not None:
        out = draw_agent_arrow(
            out,
            motion_yaw_deg,
            center=center,
            length=size // 2 - 10,
            color=MOTION_ARROW_COLOR,
            thickness=2,
            label="",
        )
    if intent_yaw_deg is not None:
        out = draw_agent_arrow(
            out,
            intent_yaw_deg,
            center=center,
            length=size // 2 - 8,
            color=INTENT_ARROW_COLOR,
            thickness=2,
            label="",
        )
    return draw_agent_arrow(
        out,
        yaw_deg,
        center=center,
        length=size // 2 - 6,
        color=color,
        thickness=2,
        label="",
    )
