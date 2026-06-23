"""LLM agent frame loop: sync SPEAR stepping + async LLM cadence + in-loop recording."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from src.agent.cognition import CognitionStore, LandDetector, StuckDetector
from src.agent.env import SpearEnv, WheelAction
from src.agent.hud import draw_agent_hud, draw_overhead_hud, normalize_nav_phase
from src.agent.llm import AsyncLLMController
from src.agent.prompts import GoalPrompt, default_prompt_index, load_prompts
from src.agent.scenes import ScenePreset, load_scene_preset
from src.recorder.recorder import FrameRecorder
from src.utils.config import Config
from src.utils.process import kill_stale_spear_processes
from src.utils.viz import (
    draw_agent_compass,
    draw_pose_arrows,
    intent_yaw_from_wheels,
    motion_yaw_from_displacement,
)


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
    enable_plan: bool


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
        enable_plan=cfg.plan and not args.no_plan,
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
        overhead_height=cfg.overhead_height,
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
    cognition: CognitionStore | None = None,
) -> None:
    action_in_effect = WheelAction(left=0.0, right=0.0)
    action_id = 0
    llm_decision_count = 0
    goal_changed = False
    last_llm_frame = 0
    first_llm_sent = False
    land_detector = LandDetector(already_landed=env.spawn_settled)
    stuck_detector = StuckDetector() if cognition is not None else None
    pending_stuck_hint: str | None = None
    arrived = False
    windows_placed = False
    last_location: dict[str, float] | None = None

    print("Running. OpenCV: agent + overhead windows. Keys 1/2/3/Q (focus either window).")
    if cognition is not None:
        print("Plan + memory enabled (memory.json / plan.json in episode dir).")
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
                            arrived = False
                            first_llm_sent = False
                            land_detector.reset()
                            if cognition is not None:
                                cognition.reset_goal(prompts[prompt_index], obs.frame_index)
                            print(f"Prompt -> [{idx + 1}] {prompts[prompt_index].label}")
                    elif k in (27, ord("q"), ord("Q")):
                        print("Stopped by user.")
                        break

            if not arrived and stuck_detector is not None:
                stuck_hint = stuck_detector.update(obs, action_in_effect)
                if stuck_hint:
                    pending_stuck_hint = stuck_hint
                    dead_id = cognition.register_code_dead_end(obs, stuck_hint)
                    print(f"frame {obs.frame_index}: stuck -> registered {dead_id}")

            if not arrived:
                decision, err = llm.poll()
                if err is not None:
                    print(f"LLM error: {err}")
                if decision is not None:
                    if llm_decision_count > 0:
                        action_id += 1
                    llm_decision_count += 1
                    action_in_effect = decision.action
                    if decision.arrived:
                        arrived = True
                        action_in_effect = WheelAction(left=0.0, right=0.0)
                    env.set_action(action_in_effect)
                    if cognition is not None:
                        cognition.apply_decision(
                            action_id=action_id,
                            frame_idx=obs.frame_index,
                            obs=obs,
                            action=action_in_effect,
                            memory_update=decision.memory_update,
                            plan_patch=decision.plan,
                            arrived=decision.arrived,
                        )
                    if decision.arrived:
                        print(f"frame {obs.frame_index}: arrived")
                    elif cognition is not None:
                        phase = cognition.plan.get("phase", "?")
                        print(
                            f"frame {obs.frame_index}: new action "
                            f"L={action_in_effect.left:.2f} R={action_in_effect.right:.2f} "
                            f"phase={phase}"
                        )
                    else:
                        print(
                            f"frame {obs.frame_index}: new action "
                            f"L={action_in_effect.left:.2f} R={action_in_effect.right:.2f}"
                        )
                    pending_stuck_hint = None

                just_landed = land_detector.update(obs)
                need_llm = False
                if goal_changed:
                    need_llm = land_detector.landed
                elif just_landed or (land_detector.landed and not first_llm_sent):
                    need_llm = True
                elif land_detector.landed and frame - last_llm_frame >= opts.control_cadence:
                    need_llm = True

                if need_llm and not llm.busy:
                    cognition_context = None
                    if cognition is not None:
                        cognition_context = cognition.context_for_llm(pending_stuck_hint)
                    llm.request(obs, goal, cognition_context=cognition_context)
                    last_llm_frame = frame
                    goal_changed = False
                    first_llm_sent = True
                    if just_landed:
                        print(f"frame {obs.frame_index}: landed -> LLM request ({goal.label})")
                    else:
                        print(f"frame {obs.frame_index}: LLM request ({goal.label})")

            if opts.show_opencv:
                motion_yaw: float | None = None
                if last_location is not None:
                    motion_yaw = motion_yaw_from_displacement(
                        obs.location["X"] - last_location["X"],
                        obs.location["Y"] - last_location["Y"],
                    )
                last_location = dict(obs.location)

                raw_phase = cognition.plan.get("phase") if cognition is not None else None
                nav_phase = normalize_nav_phase(raw_phase, arrived=arrived)
                intent_yaw: float | None = None
                if nav_phase == "approach":
                    intent_yaw = intent_yaw_from_wheels(
                        obs.rotation["Yaw"],
                        action_in_effect.left,
                        action_in_effect.right,
                    )

                agent_display = draw_agent_compass(
                    obs.rgb.copy(),
                    obs.rotation["Yaw"],
                    motion_yaw,
                    intent_yaw_deg=intent_yaw,
                )
                draw_agent_hud(
                    agent_display,
                    goal,
                    action_in_effect,
                    obs.frame_index,
                    llm.busy,
                    arrived=arrived,
                    landed=land_detector.landed,
                    nav_phase=raw_phase,
                    frames_since_llm=frame - last_llm_frame,
                    control_cadence=opts.control_cadence,
                )
                cv2.imshow("llm_agent", agent_display)

                if opts.use_overhead and obs.overhead_rgb is not None:
                    overhead_display = draw_pose_arrows(
                        obs.overhead_rgb.copy(),
                        obs.rotation["Yaw"],
                        motion_yaw,
                        intent_yaw_deg=intent_yaw,
                    )
                    draw_overhead_hud(
                        overhead_display,
                        goal,
                        action_in_effect,
                        obs.frame_index,
                        obs.location,
                        arrived=arrived,
                        landed=land_detector.landed,
                        llm_busy=llm.busy,
                        nav_phase=raw_phase,
                        frames_since_llm=frame - last_llm_frame,
                        control_cadence=opts.control_cadence,
                    )
                    cv2.imshow("llm_overhead", overhead_display)
                    if not windows_placed:
                        cv2.moveWindow("llm_agent", 40, 40)
                        cv2.moveWindow("llm_overhead", 700, 40)
                        windows_placed = True

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
