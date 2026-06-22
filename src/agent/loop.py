"""LLM agent frame loop: sync SPEAR stepping + async LLM cadence + in-loop recording."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from src.agent.env import SpearEnv, WheelAction
from src.agent.hud import VIEW_AGENT, VIEW_OVERHEAD, draw_agent_hud
from src.agent.llm import AsyncLLMController
from src.agent.prompts import GoalPrompt, default_prompt_index, load_prompts
from src.agent.scenes import ScenePreset, load_scene_preset
from src.recorder.recorder import FrameRecorder
from src.utils.config import Config
from src.utils.process import kill_stale_spear_processes
from src.utils.viz import draw_agent_arrow, draw_agent_compass


@dataclass
class AgentRunOptions:
    scene_name: str
    max_frames: int
    control_cadence: int
    show_opencv: bool
    use_overhead: bool
    delay_ms: int
    kill_stale: bool
    enable_recording: bool


def build_run_options(cfg: Config, args) -> AgentRunOptions:
    return AgentRunOptions(
        scene_name=args.scene or cfg.scene_name,
        max_frames=args.max_frames or cfg.max_frames,
        control_cadence=args.cadence or cfg.control_cadence,
        show_opencv=cfg.show_opencv and not args.no_opencv,
        use_overhead=cfg.overhead_camera and not args.no_overhead,
        delay_ms=args.delay_ms,
        kill_stale=not args.no_kill_stale,
        enable_recording=cfg.enable_recording and not args.no_record,
    )


def load_run_context(cfg: Config, opts: AgentRunOptions) -> tuple[ScenePreset, list[GoalPrompt], int]:
    scenes_path = cfg.scenes_config if cfg.scenes_config.is_file() else None
    prompts_path = cfg.prompts_config if cfg.prompts_config.is_file() else None
    scene = load_scene_preset(opts.scene_name, scenes_path)
    prompts = load_prompts(prompts_path)
    idx = min(default_prompt_index(prompts_path), len(prompts) - 1)
    return scene, prompts, idx


def connect_env(cfg: Config, scene: ScenePreset, use_overhead: bool, kill_stale: bool) -> SpearEnv:
    if kill_stale:
        print("Stopping any leftover SpearSim processes...")
        kill_stale_spear_processes()
    env = SpearEnv(
        user_config=cfg.user_config,
        scene_level=scene.level,
        spawn_location=scene.spawn,
        overhead_camera=use_overhead,
    )
    print("Connecting to SpearSim...")
    try:
        env.connect()
    except AssertionError as exc:
        raise RuntimeError(
            "Could not connect to SpearSim (PID mismatch or stale process).\n"
            "  1. Close all SpearSim game windows\n"
            "  2. Run: .\\preparation\\setup\\kill_spear.ps1\n"
            "  3. Retry: python -m src.main"
        ) from exc
    return env


def run_agent_loop(
    cfg: Config,
    opts: AgentRunOptions,
    env: SpearEnv,
    llm: AsyncLLMController,
    prompts: list[GoalPrompt],
    prompt_index: int,
    recorder: FrameRecorder | None = None,
) -> None:
    action_in_effect = WheelAction(left=0.0, right=0.0)
    action_id = 0
    llm_decision_count = 0
    view_mode = VIEW_AGENT
    goal_changed = True
    last_llm_frame = -opts.control_cadence

    print("Running. Focus OpenCV window for keys 1/2/3/Enter/Q.")
    try:
        for frame in range(opts.max_frames):
            obs = env.step(action_in_effect)
            goal = prompts[prompt_index]

            if recorder is not None:
                recorder.record_frame(obs, action_in_effect, action_id)

            if opts.show_opencv:
                key = cv2.waitKey(opts.delay_ms)
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

            if (goal_changed or frame - last_llm_frame >= opts.control_cadence) and not llm.busy:
                llm.request(obs, goal)
                last_llm_frame = frame
                goal_changed = False
                print(f"frame {obs.frame_index}: LLM request ({goal.label})")

            new_action, err = llm.poll()
            if err is not None:
                print(f"LLM error: {err}")
            if new_action is not None:
                if llm_decision_count > 0:
                    action_id += 1
                llm_decision_count += 1
                action_in_effect = new_action
                env.set_action(action_in_effect)
                print(
                    f"frame {obs.frame_index}: new action "
                    f"L={action_in_effect.left:.2f} R={action_in_effect.right:.2f}"
                )

            if opts.show_opencv:
                if view_mode == VIEW_AGENT:
                    display = draw_agent_compass(obs.rgb.copy(), obs.rotation["Yaw"])
                else:
                    display = draw_agent_arrow(
                        (obs.overhead_rgb or obs.rgb).copy(),
                        obs.rotation["Yaw"],
                    )
                draw_agent_hud(
                    display, goal, action_in_effect, obs.frame_index, view_mode, llm.busy
                )
                cv2.imshow("llm_agent", display)

            if frame % 30 == 0:
                loc = obs.location
                print(
                    f"frame {obs.frame_index:4d} pos=({loc['X']:.0f},{loc['Y']:.0f},{loc['Z']:.0f}) "
                    f"L={action_in_effect.left:.2f} R={action_in_effect.right:.2f}"
                )
    finally:
        env.close()
        if recorder is not None:
            recorder.close()
            print(f"Recorded {recorder.records_written} frames -> {recorder.episode_dir}")
        if opts.show_opencv:
            cv2.destroyAllWindows()
        print("Done.")
