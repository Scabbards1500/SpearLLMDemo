"""Entry point: LLM-driven sphere agent with prompt switching."""

from __future__ import annotations

import argparse
import sys

import cv2

from src.config import Config
from src.llm_controller import AsyncLLMController, GoalPrompt, default_prompt_index, load_prompts
from src.scenes import load_scene_preset
from src.spear_env import SpearEnv, WheelAction
from src.spear_process import kill_stale_spear_processes
from src.viz import draw_agent_arrow, draw_agent_compass

VIEW_AGENT = "agent"
VIEW_OVERHEAD = "overhead"


def draw_hud(
    image,
    goal: GoalPrompt,
    action: WheelAction,
    frame_index: int,
    view_mode: str,
    llm_busy: bool,
) -> None:
    status = "LLM thinking..." if llm_busy else "LLM idle"
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
            (0, 255, 255) if i == 0 else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM sphere agent (prompt-driven navigation)")
    p.add_argument("--scene", default=None, help="Scene preset name (default: debug_house)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--cadence", type=int, default=None, help="LLM control cadence in frames")
    p.add_argument("--no-opencv", action="store_true")
    p.add_argument("--no-overhead", action="store_true")
    p.add_argument("--delay-ms", type=int, default=1)
    p.add_argument(
        "--no-kill-stale",
        action="store_true",
        help="Do not kill leftover SpearSim.exe before connect",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    if not cfg.anthropic_api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    scene_name = args.scene or cfg.scene_name
    scene = load_scene_preset(scene_name, cfg.scenes_config if cfg.scenes_config.is_file() else None)
    prompts = load_prompts(cfg.prompts_config if cfg.prompts_config.is_file() else None)
    prompt_index = default_prompt_index(
        cfg.prompts_config if cfg.prompts_config.is_file() else None
    )
    prompt_index = min(prompt_index, len(prompts) - 1)

    max_frames = args.max_frames or cfg.max_frames
    cadence = args.cadence or cfg.control_cadence
    show_cv = cfg.show_opencv and not args.no_opencv
    use_overhead = cfg.overhead_camera and not args.no_overhead

    print(f"Scene: {scene.name} -> {scene.level}")
    print(f"Spawn: X={scene.spawn['X']} Y={scene.spawn['Y']} Z={scene.spawn['Z']}")
    print("Prompts:")
    for i, pr in enumerate(prompts):
        mark = "*" if i == prompt_index else " "
        print(f"  {mark} [{i + 1}] {pr.label}: {pr.text[:60]}...")
    print(f"LLM every {cadence} frames | model={cfg.llm_model}")

    env = SpearEnv(
        user_config=cfg.user_config,
        scene_level=scene.level,
        spawn_location=scene.spawn,
        overhead_camera=use_overhead,
    )
    llm = AsyncLLMController(api_key=cfg.anthropic_api_key, model=cfg.llm_model)

    action_in_effect = WheelAction(left=0.0, right=0.0)
    view_mode = VIEW_AGENT
    goal_changed = True
    last_llm_frame = -cadence

    print("Connecting to SpearSim...")
    if not args.no_kill_stale:
        print("Stopping any leftover SpearSim processes...")
        kill_stale_spear_processes()
    try:
        env.connect()
    except AssertionError as exc:
        print(
            "\nERROR: Could not connect to SpearSim (PID mismatch or stale process).\n"
            "  1. Close all SpearSim game windows\n"
            "  2. Run: .\\scripts\\kill_spear.ps1\n"
            "  3. Retry: python -m src.main\n"
        )
        raise SystemExit(1) from exc
    print("Running. Focus OpenCV window for keys 1/2/3/Enter/Q.")

    try:
        for frame in range(max_frames):
            obs = env.step(action_in_effect)

            # Switch prompt with keys 1/2/3
            if show_cv:
                key = cv2.waitKey(args.delay_ms)
                if key != -1:
                    k = key & 0xFF
                    if k in (ord("1"), ord("2"), ord("3")):
                        idx = k - ord("1")
                        if idx < len(prompts):
                            prompt_index = idx
                            goal_changed = True
                            print(f"Prompt -> [{idx + 1}] {prompts[prompt_index].label}")
                    elif key in (13, 10):
                        view_mode = VIEW_OVERHEAD if view_mode == VIEW_AGENT else VIEW_AGENT
                    elif k in (27, ord("q"), ord("Q")):
                        print("Stopped by user.")
                        break

            goal = prompts[prompt_index]

            # Trigger LLM on cadence or after prompt change
            need_llm = goal_changed or (frame - last_llm_frame >= cadence)
            if need_llm and not llm.busy:
                llm.request(obs, goal)
                last_llm_frame = frame
                goal_changed = False
                print(f"frame {obs.frame_index}: LLM request ({goal.label})")

            new_action, err = llm.poll()
            if err is not None:
                print(f"LLM error: {err}")
            if new_action is not None:
                action_in_effect = new_action
                env.set_action(action_in_effect)
                print(
                    f"frame {obs.frame_index}: new action "
                    f"L={action_in_effect.left:.2f} R={action_in_effect.right:.2f}"
                )

            if show_cv:
                if view_mode == VIEW_AGENT:
                    display = draw_agent_compass(obs.rgb.copy(), obs.rotation["Yaw"])
                else:
                    display = draw_agent_arrow(
                        (obs.overhead_rgb or obs.rgb).copy(),
                        obs.rotation["Yaw"],
                    )
                draw_hud(display, goal, action_in_effect, obs.frame_index, view_mode, llm.busy)
                cv2.imshow("llm_agent", display)

            if frame % 30 == 0:
                loc = obs.location
                print(
                    f"frame {obs.frame_index:4d} pos=({loc['X']:.0f},{loc['Y']:.0f},{loc['Z']:.0f}) "
                    f"L={action_in_effect.left:.2f} R={action_in_effect.right:.2f}"
                )
    finally:
        env.close()
        if show_cv:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
