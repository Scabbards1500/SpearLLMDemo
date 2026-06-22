"""Toggle agent / overhead view with Enter in the OpenCV window.

Run:
    conda activate spearenv
    cd d:\\python\\SpearLLMDemo
    python scripts/demo_toggle_view.py

Controls (focus the OpenCV window):
    Enter  - toggle agent camera <-> overhead bird's-eye
    Q/ESC  - quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_hardcoded import draw_hud, hardcoded_action  # noqa: E402
from src.spear_env import SpearEnv  # noqa: E402
from src.viz import draw_agent_arrow  # noqa: E402

VIEW_AGENT = "agent"
VIEW_OVERHEAD = "overhead"


def draw_view_hud(
    image,
    view_mode: str,
    frame_index: int,
    location: dict[str, float],
) -> None:
    mode_label = "AGENT VIEW" if view_mode == VIEW_AGENT else "OVERHEAD (follow agent)"
    lines = [
        mode_label,
        f"frame {frame_index}",
        f"pos X={location['X']:.0f} Y={location['Y']:.0f} Z={location['Z']:.0f}",
        "Enter=toggle  Q/Esc=quit",
    ]
    for i, text in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (0, 255, 0)
        cv2.putText(
            image,
            text,
            (12, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPEAR demo with Enter-to-toggle camera view")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "user_config.yaml")
    p.add_argument("--delay-ms", type=int, default=1)
    p.add_argument(
        "--overhead-height",
        type=float,
        default=2800.0,
        help="Camera height above agent in cm (2800 = 28m)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    view_mode = VIEW_AGENT

    env = SpearEnv(
        user_config=args.config,
        overhead_camera=True,
        overhead_height=args.overhead_height,
    )
    print("Connecting to SpearSim...")
    print("OpenCV controls: Enter = toggle view, Q/Esc = quit")
    env.connect()

    window = "spear_view"
    try:
        for i in range(args.frames):
            obs = env.step(hardcoded_action(i))

            if view_mode == VIEW_AGENT:
                display = draw_hud(obs.rgb, obs.action, obs.frame_index)
            else:
                base = obs.overhead_rgb.copy() if obs.overhead_rgb is not None else obs.rgb.copy()
                display = draw_agent_arrow(base, obs.rotation["Yaw"])

            draw_view_hud(display, view_mode, obs.frame_index, obs.location)
            cv2.imshow(window, display)

            key = cv2.waitKey(args.delay_ms)
            if key == -1:
                pass
            elif key in (13, 10):  # Enter / numpad Enter
                view_mode = VIEW_OVERHEAD if view_mode == VIEW_AGENT else VIEW_AGENT
                print(f"Switched to {view_mode} view")
            elif (key & 0xFF) in (27, ord("q"), ord("Q")):
                print("Stopped by user.")
                break

            if i % 30 == 0:
                loc = obs.location
                print(
                    f"[{view_mode:8s}] frame {obs.frame_index:4d}  "
                    f"pos=({loc['X']:.1f}, {loc['Y']:.1f}, {loc['Z']:.1f})",
                )
    finally:
        env.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
