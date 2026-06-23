"""Toggle agent / overhead view with Enter (dev demo, no LLM)."""

from __future__ import annotations

import argparse

import cv2

from preparation.bootstrap import ensure_project_root
from preparation.demos.demo_hardcoded import hardcoded_action

ensure_project_root()

from src.agent.env import SpearEnv
from src.agent.hud import VIEW_AGENT, VIEW_OVERHEAD
from src.utils.config import Config
from src.utils.viz import draw_agent_arrow


def draw_view_hud(image, view_mode: str, frame_index: int, location: dict) -> None:
    label = "AGENT VIEW" if view_mode == VIEW_AGENT else "OVERHEAD"
    lines = [
        label,
        f"frame {frame_index}",
        f"pos X={location['X']:.0f} Y={location['Y']:.0f} Z={location['Z']:.0f}",
        "Enter=toggle  Q/Esc=quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            image, text, (12, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255) if i == 0 else (0, 255, 0), 2, cv2.LINE_AA,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--delay-ms", type=int, default=1)
    p.add_argument("--overhead-height", type=float, default=900.0)
    args = p.parse_args()

    cfg = Config()
    view_mode = VIEW_AGENT
    env = SpearEnv(
        user_config=cfg.user_config,
        overhead_camera=True,
        overhead_height=args.overhead_height,
    )
    print("Connecting... Enter=toggle view, Q=quit")
    env.connect()

    try:
        for i in range(args.frames):
            obs = env.step(hardcoded_action(i))
            if view_mode == VIEW_AGENT:
                display = obs.rgb.copy()
            else:
                base = obs.overhead_rgb if obs.overhead_rgb is not None else obs.rgb
                display = draw_agent_arrow(base.copy(), obs.rotation["Yaw"])
            draw_view_hud(display, view_mode, obs.frame_index, obs.location)
            cv2.imshow("spear_view", display)

            key = cv2.waitKey(args.delay_ms)
            if key in (13, 10):
                view_mode = VIEW_OVERHEAD if view_mode == VIEW_AGENT else VIEW_AGENT
            elif key != -1 and (key & 0xFF) in (27, ord("q"), ord("Q")):
                break
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
