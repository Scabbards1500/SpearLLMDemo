"""Hardcoded wheel actions + dual visualization (Unreal window + OpenCV agent view).

Run (after SpearSim is built and user_config.yaml exists):
    conda activate spearenv
    cd d:\\python\\SpearLLMDemo
    python scripts/demo_hardcoded.py

You should see:
  1. SpearSim game window (3D scene)
  2. OpenCV window "agent_view" (egocentric RGB from onboard camera)

Press ESC or Q in the OpenCV window to quit early.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.spear_env import SpearEnv, WheelAction  # noqa: E402


def hardcoded_action(frame_index: int) -> WheelAction:
    """Simple schedule: forward -> turn right -> forward -> turn left -> stop."""
    if frame_index < 90:
        return WheelAction(left=0.8, right=0.8)
    if frame_index < 120:
        return WheelAction(left=0.25, right=0.95)
    if frame_index < 210:
        return WheelAction(left=0.8, right=0.8)
    if frame_index < 240:
        return WheelAction(left=0.95, right=0.25)
    return WheelAction(left=0.0, right=0.0)


def draw_hud(obs_rgb: np.ndarray, action: WheelAction, frame_index: int) -> np.ndarray:
    """Overlay pose + action on the agent camera image for debugging."""
    hud = obs_rgb.copy()
    lines = [
        f"frame {frame_index}",
        f"left={action.left:.2f} right={action.right:.2f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            hud,
            text,
            (12, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return hud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardcoded SPEAR agent visualization demo")
    parser.add_argument("--frames", type=int, default=300, help="Number of simulation frames")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "user_config.yaml",
        help="Path to SPEAR user_config.yaml",
    )
    parser.add_argument(
        "--no-opencv",
        action="store_true",
        help="Skip OpenCV window (only Unreal game window)",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=1,
        help="cv2.waitKey delay (ms); increase to slow playback",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = SpearEnv(user_config=args.config)
    print("Connecting to SpearSim (game window should appear)...")
    env.connect()

    try:
        for i in range(args.frames):
            action = hardcoded_action(i)
            obs = env.step(action)

            if not args.no_opencv:
                hud = draw_hud(obs.rgb, obs.action, obs.frame_index)
                cv2.imshow("agent_view", hud)
                key = cv2.waitKey(args.delay_ms) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("Stopped early by user.")
                    break

            if i % 30 == 0:
                loc = obs.location
                rot = obs.rotation
                print(
                    f"frame {obs.frame_index:4d}  "
                    f"pos=({loc['X']:.1f}, {loc['Y']:.1f}, {loc['Z']:.1f})  "
                    f"yaw={rot['Yaw']:.1f}  "
                    f"L={obs.action.left:.2f} R={obs.action.right:.2f}",
                )
    finally:
        env.close()
        if not args.no_opencv:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
