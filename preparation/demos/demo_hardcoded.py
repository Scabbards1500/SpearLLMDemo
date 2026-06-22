"""Hardcoded wheel actions + OpenCV preview (no LLM)."""

from __future__ import annotations

import argparse

import cv2

from preparation.bootstrap import ensure_project_root

ensure_project_root()

from src.agent.env import SpearEnv, WheelAction
from src.utils.config import Config


def hardcoded_action(frame_index: int) -> WheelAction:
    if frame_index < 90:
        return WheelAction(left=0.8, right=0.8)
    if frame_index < 120:
        return WheelAction(left=0.25, right=0.95)
    if frame_index < 210:
        return WheelAction(left=0.8, right=0.8)
    if frame_index < 240:
        return WheelAction(left=0.95, right=0.25)
    return WheelAction(left=0.0, right=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Hardcoded SPEAR agent visualization demo")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--delay-ms", type=int, default=1)
    p.add_argument("--no-opencv", action="store_true")
    args = p.parse_args()

    cfg = Config()
    env = SpearEnv(user_config=cfg.user_config)
    print("Connecting to SpearSim...")
    env.connect()

    try:
        for i in range(args.frames):
            obs = env.step(hardcoded_action(i))
            if not args.no_opencv:
                cv2.imshow("agent_view", obs.rgb)
                if cv2.waitKey(args.delay_ms) & 0xFF in (27, ord("q"), ord("Q")):
                    break
            if i % 30 == 0:
                loc, rot = obs.location, obs.rotation
                print(
                    f"frame {obs.frame_index:4d} pos=({loc['X']:.1f},{loc['Y']:.1f},{loc['Z']:.1f}) "
                    f"yaw={rot['Yaw']:.1f}"
                )
    finally:
        env.close()
        if not args.no_opencv:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
