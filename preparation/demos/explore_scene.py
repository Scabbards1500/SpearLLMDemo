"""Explore a SPEAR scene: spawn pose, nav-mesh goals, optional markers.

Usage:
    python preparation/demos/explore_scene.py --scene debug
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import spear

from preparation.bootstrap import PROJECT_ROOT, ensure_project_root

ensure_project_root()

from src.agent.scenes import DEBUG_LEVEL

SCENES = {
    "apartment": "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000.apartment_0000",
    "debug": DEBUG_LEVEL,
    "debug1": "/Game/SPEAR/Scenes/debug_0001/Maps/debug_0001.debug_0001",
}


def dist_xy(a: dict, b: dict) -> float:
    dx = a["X"] - b["X"]
    dy = a["Y"] - b["Y"]
    return math.hypot(dx, dy)


def open_level(instance, game, gameplay_statics, level_path: str):
    spear.log(f"Opening level: {level_path}")
    with instance.begin_frame():
        pass
    with instance.end_frame():
        game.invalidate()
        gameplay_statics.OpenLevel(LevelName=level_path, bAbsolute=True, Options="")
    return instance.get_game(
        wait=True,
        wait_max_time_seconds=30.0,
        wait_sleep_time_seconds=1.0,
        warm_up=True,
        warm_up_time_seconds=3.0,
        warm_up_num_frames=2,
    )


def find_player_start(game):
    """Return (location dict, rotation dict) or None."""
    try:
        ps = game.unreal_service.find_actor_by_name(
            uclass="APlayerStart",
            actor_name="Settings/PlayerStart",
        )
        if ps is not None:
            loc = ps.K2_GetActorLocation()
            rot = ps.K2_GetActorRotation()
            return dict(loc), dict(rot)
    except Exception:
        pass
    return None


def sample_nav_points(game, num_points: int = 20) -> list[np.ndarray]:
    sp_nav = game.get_unreal_object(uclass="USpNavigationSystemV1")
    nav_sys = sp_nav.GetNavigationSystem()
    old_rebuild = nav_sys.bSupportRebuilding.get()
    nav_sys.bSupportRebuilding = True
    sp_nav.Build(NavigationSystem=nav_sys)
    sp_nav.AddNavigationBuildLock(NavigationSystem=nav_sys, Flags="Custom")
    nav_sys.bSupportRebuilding = old_rebuild

    nav_data = sp_nav.GetNavDataForAgentName(NavigationSystem=nav_sys, AgentName="Default")
    points = game.navigation_service.get_random_points(navigation_data=nav_data, num_points=num_points)
    return list(points)


def spawn_marker(game, bp_axes_uclass, point: dict, scale: float = 2.0):
    game.unreal_service.spawn_actor(
        uclass=bp_axes_uclass,
        location={"X": point["X"], "Y": point["Y"], "Z": point["Z"] + 30.0},
        rotation={"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explore SPEAR scene layout and goal candidates")
    p.add_argument("--scene", choices=list(SCENES.keys()), default="apartment")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "user_config.yaml")
    p.add_argument("--num-goals", type=int, default=15, help="Nav-mesh points to sample")
    p.add_argument(
        "--spawn-markers",
        action="store_true",
        help="Spawn axis gizmos at PlayerStart + top goal points (visible in game window)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    level = SCENES[args.scene]

    config = spear.get_config(user_config_files=[str(args.config.resolve())])
    spear.configure_system(config=config)
    instance = spear.Instance(config=config)
    game = instance.get_game()

    with instance.begin_frame():
        gameplay_statics = game.get_unreal_object(uclass="UGameplayStatics")
    with instance.end_frame():
        pass

    game = open_level(instance, game, gameplay_statics, level)
    with instance.begin_frame():
        gameplay_statics = game.get_unreal_object(uclass="UGameplayStatics")
    with instance.end_frame():
        pass

    print("\n=== SPEAR scene explorer ===")
    print(f"Scene key:  {args.scene}")
    print(f"Level path: {level}")

    spawn_loc, spawn_rot = find_player_start(game)
    if spawn_loc:
        print("\n--- PlayerStart (recommended agent spawn) ---")
        print(f"  location: X={spawn_loc['X']:.1f}  Y={spawn_loc['Y']:.1f}  Z={spawn_loc['Z']:.1f}  (cm)")
        print(f"  rotation: Yaw={spawn_rot['Yaw']:.1f} deg")
    else:
        print("\n[WARN] PlayerStart not found; demo uses hardcoded spawn from control_simple_agent.")
        spawn_loc = {"X": -10.0, "Y": 280.0, "Z": 150.0}

    print("\n--- Coordinate cheat sheet (Unreal) ---")
    print("  Units: centimeters.  Z = up.  Yaw = heading (0=+X, 90=+Y).")
    print("  Agent 'forward' follows its local +X after you apply rotation.")

    try:
        raw_points = sample_nav_points(game, num_points=args.num_goals)
        candidates = []
        for pt in raw_points:
            p = {"X": float(pt[0]), "Y": float(pt[1]), "Z": float(pt[2])}
            candidates.append((dist_xy(spawn_loc, p), p))

        candidates.sort(key=lambda x: x[0], reverse=True)
        print(f"\n--- Nav-mesh goal candidates ({len(candidates)} sampled) ---")
        print("  Pick one as GOAL in README / config (distance from PlayerStart):")
        for i, (d, p) in enumerate(candidates[:8]):
            print(f"  [{i}] dist={d/100:.1f}m  X={p['X']:.1f}  Y={p['Y']:.1f}  Z={p['Z']:.1f}")

        if candidates:
            _, best = candidates[0]
            print("\n--- Suggested episode goal (farthest sampled point) ---")
            print(f"  GOAL_X={best['X']:.1f}")
            print(f"  GOAL_Y={best['Y']:.1f}")
            print(f"  GOAL_Z={best['Z']:.1f}")
            print("  Success: agent within ~100cm of goal (tune in config).")

        if args.spawn_markers:
            with instance.begin_frame():
                bp_axes = game.unreal_service.load_class(
                    uclass="AActor", name="/SpContent/Blueprints/BP_Axes.BP_Axes_C"
                )
                spawn_marker(game, bp_axes, spawn_loc, scale=3.0)
                for _, p in candidates[:3]:
                    spawn_marker(game, bp_axes, p, scale=2.0)
            with instance.end_frame():
                pass
            instance.step(num_frames=5)
            print("\n[Markers spawned] Large axes = PlayerStart; smaller = top 3 goals.")
            print("Look at the SpearSim game window, then close it or Ctrl+C.")

    except Exception as exc:
        print(f"\n[WARN] Nav-mesh sampling failed: {exc}")
        print("  You can still set a manual GOAL by driving the agent and reading pose from demo output.")

    print("\nNext: python preparation/demos/demo_hardcoded.py")
    print("      python -m src.main  (LLM agent)\n")


if __name__ == "__main__":
    main()
