"""Entry point: LLM-driven SPEAR agent."""

from __future__ import annotations

import argparse
import sys

from src.agent.loop import (
    build_run_options,
    connect_env,
    load_run_context,
    run_agent_loop,
)
from src.agent.llm import AsyncLLMController
from src.recorder.recorder import FrameRecorder
from src.utils.config import Config


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
    p.add_argument(
        "--no-record",
        action="store_true",
        help="Disable in-loop frame recording (Part B)",
    )
    p.add_argument("--episode", default=None, help="Recording episode folder name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    if not cfg.anthropic_api_key:
        print("ERROR: Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    opts = build_run_options(cfg, args)
    scene, prompts, prompt_index = load_run_context(cfg, opts)

    print(f"Scene: {scene.name} -> {scene.level}")
    print(f"Spawn: X={scene.spawn['X']} Y={scene.spawn['Y']} Z={scene.spawn['Z']}")
    print("Prompts:")
    for i, pr in enumerate(prompts):
        mark = "*" if i == prompt_index else " "
        print(f"  {mark} [{i + 1}] {pr.label}: {pr.text[:60]}...")
    print(f"LLM every {opts.control_cadence} frames | model={cfg.llm_model}")

    recorder: FrameRecorder | None = None
    if opts.enable_recording:
        episode_name = args.episode or cfg.episode_name
        episode_dir = cfg.recordings_dir / episode_name
        recorder = FrameRecorder(
            episode_dir,
            target_fps=cfg.target_fps,
            scene=scene,
            control_cadence=opts.control_cadence,
            llm_model=cfg.llm_model,
        )
        recorder.write_episode_meta(prompts[prompt_index])
        print(f"Recording enabled -> {episode_dir}")

    env = connect_env(cfg, scene, opts.use_overhead, opts.kill_stale)
    llm = AsyncLLMController(api_key=cfg.anthropic_api_key, model=cfg.llm_model)
    run_agent_loop(cfg, opts, env, llm, prompts, prompt_index, recorder=recorder)


if __name__ == "__main__":
    main()
