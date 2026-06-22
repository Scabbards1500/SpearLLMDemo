"""SPEAR environment wrapper: synchronous stepping, observations, wheel actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spear

# Sphere agent uses force + yaw; map differential-drive (left/right) to these.
BASE_FORCE = 1000.0
BASE_YAW_DEG = 2.5

AGENT_BP = "/SpContent/Blueprints/BP_SphereAgent.BP_SphereAgent_C"
OVERHEAD_CAMERA_BP = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
DEFAULT_SPAWN = {"X": -10.0, "Y": 280.0, "Z": 150.0}
DEFAULT_SCENE = "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000.debug_0000"
CAMERA_COMPONENT = "DefaultSceneRoot.final_tone_curve_hdr_"
SPHERE_COMPONENT_PATH = "DefaultSceneRoot.sphere_"


def _as_xyz_dict(spear_vector: Any) -> dict[str, float]:
    """Normalize SPEAR FVector (keys may be x/y/z or X/Y/Z)."""
    raw = spear_vector if isinstance(spear_vector, dict) else dict(spear_vector)
    lowered = {str(k).lower(): float(v) for k, v in raw.items()}
    if {"x", "y", "z"}.issubset(lowered):
        return {"X": lowered["x"], "Y": lowered["y"], "Z": lowered["z"]}
    arr = spear.math.to_numpy_array_from_spear_vector(spear_vector=raw)
    return {"X": float(arr[0]), "Y": float(arr[1]), "Z": float(arr[2])}


def _as_rot_dict(spear_rotator: Any) -> dict[str, float]:
    """Normalize SPEAR FRotator (keys may vary in case)."""
    raw = spear_rotator if isinstance(spear_rotator, dict) else dict(spear_rotator)
    lowered = {str(k).lower(): float(v) for k, v in raw.items()}
    if {"pitch", "yaw", "roll"}.issubset(lowered):
        return {
            "Pitch": lowered["pitch"],
            "Yaw": lowered["yaw"],
            "Roll": lowered["roll"],
        }
    arr = spear.math.to_numpy_array_from_spear_rotator(spear_rotator=raw)
    return {"Pitch": float(arr[0]), "Yaw": float(arr[1]), "Roll": float(arr[2])}


@dataclass
class WheelAction:
    """Differential-drive command in [0, 1] per wheel (assignment-compatible)."""

    left: float = 0.0
    right: float = 0.0

    def clamp(self) -> WheelAction:
        return WheelAction(
            left=float(np.clip(self.left, 0.0, 1.0)),
            right=float(np.clip(self.right, 0.0, 1.0)),
        )


@dataclass
class Observation:
    frame_index: int
    rgb: np.ndarray
    location: dict[str, float]
    rotation: dict[str, float]
    action: WheelAction
    overhead_rgb: np.ndarray | None = None


class SpearEnv:
    """Minimal sync SPEAR env: spawn sphere agent, step, read egocentric RGB + pose."""

    def __init__(
        self,
        user_config: Path,
        scene_level: str = DEFAULT_SCENE,
        spawn_location: dict[str, float] | None = None,
        warm_up_frames: int = 2,
        overhead_camera: bool = False,
        overhead_height: float = 2800.0,
        debug_arrow: bool = True,
    ) -> None:
        self.user_config = Path(user_config)
        self.scene_level = scene_level
        self.spawn_location = spawn_location or dict(DEFAULT_SPAWN)
        self.warm_up_frames = warm_up_frames
        self.overhead_camera = overhead_camera
        self.overhead_height = overhead_height
        self.debug_arrow = debug_arrow

        self._config: Any = None
        self._instance: spear.Instance | None = None
        self._game: Any = None
        self._gameplay_statics: Any = None
        self._agent: Any = None
        self._root: Any = None
        self._sphere: Any = None
        self._camera: Any = None
        self._overhead_actor: Any = None
        self._overhead_root: Any = None
        self._overhead_camera: Any = None
        self._kismet: Any = None
        self._frame_index = 0
        self._action_in_effect = WheelAction()

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def action_in_effect(self) -> WheelAction:
        return self._action_in_effect

    def set_action(self, action: WheelAction) -> None:
        self._action_in_effect = action.clamp()

    @staticmethod
    def _normalize_level_path(level: str) -> str:
        """Compare map paths despite .MapName suffix differences."""
        level = level.strip().strip('"')
        if "." in level.rsplit("/", 1)[-1]:
            level = level.rsplit(".", 1)[0]
        return level.lower()

    def _should_open_level(self) -> bool:
        if not self.scene_level:
            return False
        try:
            init = self._config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE
            if init.OVERRIDE_GAME_DEFAULT_MAP and init.GAME_DEFAULT_MAP:
                if self._normalize_level_path(self.scene_level) == self._normalize_level_path(
                    init.GAME_DEFAULT_MAP
                ):
                    spear.log("Scene already default at launch, skipping OpenLevel:", self.scene_level)
                    return False
        except Exception:
            pass
        return True

    def connect(self) -> None:
        if not self.user_config.is_file():
            raise FileNotFoundError(
                f"Missing {self.user_config}. Copy user_config.yaml.example and set GAME_EXECUTABLE."
            )

        self._config = spear.get_config(user_config_files=[str(self.user_config.resolve())])
        spear.configure_system(config=self._config)
        self._instance = spear.Instance(config=self._config)
        self._game = self._instance.get_game()

        with self._instance.begin_frame():
            self._gameplay_statics = self._game.get_unreal_object(uclass="UGameplayStatics")
        with self._instance.end_frame():
            pass

        if self.scene_level and self._should_open_level():
            self._open_level(self.scene_level)

        self._spawn_agent()
        if self.overhead_camera:
            self._spawn_overhead_camera()
        self._instance.step(num_frames=self.warm_up_frames)

    def _open_level(self, level_name: str) -> None:
        assert self._instance is not None and self._game is not None
        spear.log(f"Opening level: {level_name}")
        with self._instance.begin_frame():
            pass
        with self._instance.end_frame():
            self._game.invalidate()
            self._gameplay_statics.OpenLevel(LevelName=level_name, bAbsolute=True, Options="")

        self._game = self._instance.get_game(
            wait=True,
            wait_max_time_seconds=30.0,
            wait_sleep_time_seconds=1.0,
            warm_up=True,
            warm_up_time_seconds=5.0,
            warm_up_num_frames=1,
        )
        with self._instance.begin_frame():
            self._gameplay_statics = self._game.get_unreal_object(uclass="UGameplayStatics")
        with self._instance.end_frame():
            pass

    def _spawn_agent(self) -> None:
        assert self._instance is not None and self._game is not None
        with self._instance.begin_frame():
            bp_class = self._game.unreal_service.load_class(uclass="AActor", name=AGENT_BP)
            self._agent = self._game.unreal_service.spawn_actor(
                uclass=bp_class,
                location=self.spawn_location,
            )
            self._root = self._game.unreal_service.get_component_by_name(
                actor=self._agent,
                component_name="DefaultSceneRoot",
                uclass="USceneComponent",
            )
            self._camera = self._game.unreal_service.get_component_by_name(
                actor=self._agent,
                component_name=CAMERA_COMPONENT,
                uclass="USpSceneCaptureComponent2D",
            )
            self._sphere = self._game.unreal_service.get_component_by_path(
                actor=self._agent,
                component_path=SPHERE_COMPONENT_PATH,
                uclass="USceneComponent",
            )
            self._camera.Initialize()
            self._camera.initialize_sp_funcs()
            if self.debug_arrow:
                self._kismet = self._game.get_unreal_object(uclass="UKismetSystemLibrary")
            self._game.unreal_service.execute_console_command("stat fps")

        with self._instance.end_frame():
            pass

    def _spawn_overhead_camera(self) -> None:
        assert self._instance is not None and self._game is not None
        with self._instance.begin_frame():
            bp_class = self._game.unreal_service.load_class(uclass="AActor", name=OVERHEAD_CAMERA_BP)
            self._overhead_actor = self._game.unreal_service.spawn_actor(uclass=bp_class)
            self._overhead_root = self._game.unreal_service.get_component_by_name(
                actor=self._overhead_actor,
                component_name="DefaultSceneRoot",
                uclass="USceneComponent",
            )
            self._overhead_camera = self._game.unreal_service.get_component_by_name(
                actor=self._overhead_actor,
                component_name=CAMERA_COMPONENT,
                uclass="USpSceneCaptureComponent2D",
            )
            viewport_desc = self._game.rendering_service.get_current_viewport_desc()
            self._game.rendering_service.align_camera_with_viewport(
                camera_sensor=self._overhead_actor,
                camera_components=self._overhead_camera,
                viewport_desc=viewport_desc,
                widths=1280,
                heights=720,
            )
            self._overhead_camera.Initialize()
            self._overhead_camera.initialize_sp_funcs()
            self._update_overhead_camera(self.spawn_location)

        with self._instance.end_frame():
            pass

    def _update_overhead_camera(self, agent_location: dict[str, float]) -> None:
        if self._overhead_actor is None:
            return
        self._overhead_actor.K2_SetActorLocationAndRotation(
            bSweep=False,
            bTeleport=True,
            NewLocation={
                "X": agent_location["X"],
                "Y": agent_location["Y"],
                "Z": agent_location["Z"] + self.overhead_height,
            },
            NewRotation={"Pitch": -90.0, "Yaw": 0.0, "Roll": 0.0},
        )

    def _draw_debug_arrow(self, agent_loc: dict[str, float]) -> None:
        if not self.debug_arrow or self._kismet is None:
            return
        rot_mat = spear.math.to_numpy_matrix_from_spear_rotator(
            spear_rotator=self._root.K2_GetComponentRotation(),
            as_matrix=True,
        )
        forward = rot_mat * np.matrix([250.0, 0.0, 0.0]).T
        start = np.array([agent_loc["X"], agent_loc["Y"], agent_loc["Z"]], dtype=np.float64)
        end = start + np.asarray(forward).reshape(3)
        z_floor = agent_loc["Z"] - 40.0
        start[2] = z_floor
        end[2] = z_floor
        self._kismet.DrawDebugArrow(
            LineStart=spear.math.to_spear_vector_from_numpy_array(numpy_array=start),
            LineEnd=spear.math.to_spear_vector_from_numpy_array(numpy_array=end),
            ArrowSize=90.0,
            LineColor={"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0},
            Duration=0.05,
            Thickness=6.0,
        )

    def step(self, action: WheelAction | None = None) -> Observation:
        """Advance one simulation frame; apply action, return RGB + pose."""
        if action is not None:
            self.set_action(action)
        assert self._instance is not None and self._game is not None

        act = self._action_in_effect
        forward = (act.left + act.right) * 0.5
        turn = (act.right - act.left) * 0.5

        with self._instance.begin_frame():
            self._gameplay_statics.SetGamePaused(bPaused=False)
            agent_loc = _as_xyz_dict(self._root.K2_GetComponentLocation())
            if self._overhead_camera is not None:
                self._update_overhead_camera(agent_loc)
            self._draw_debug_arrow(agent_loc)
            if turn != 0.0:
                self._root.K2_AddRelativeRotation(
                    DeltaRotation={"Pitch": 0.0, "Yaw": turn * BASE_YAW_DEG * 2.0, "Roll": 0.0},
                )
            if forward > 0.0:
                rotator = self._root.K2_GetComponentRotation()
                rot_mat = spear.math.to_numpy_matrix_from_spear_rotator(
                    spear_rotator=rotator,
                    as_matrix=True,
                )
                force_local = np.matrix([forward * BASE_FORCE, 0.0, 0.0]).T
                force_world = rot_mat * force_local
                self._sphere.AddForce(
                    Force=spear.math.to_spear_vector_from_numpy_array(numpy_array=force_world),
                )

        with self._instance.end_frame():
            bundle = self._camera.read_pixels()
            rgb = bundle["arrays"]["data"]
            overhead_rgb = None
            if self._overhead_camera is not None:
                oh_bundle = self._overhead_camera.read_pixels()
                overhead_rgb = oh_bundle["arrays"]["data"]
            location = _as_xyz_dict(self._root.K2_GetComponentLocation())
            rotation = _as_rot_dict(self._root.K2_GetComponentRotation())
            self._gameplay_statics.SetGamePaused(bPaused=True)

        obs = Observation(
            frame_index=self._frame_index,
            rgb=rgb,
            location=location,
            rotation=rotation,
            action=act,
            overhead_rgb=overhead_rgb,
        )
        self._frame_index += 1
        return obs

    def close(self) -> None:
        if self._instance is None or self._game is None or self._agent is None:
            return
        try:
            with self._instance.begin_frame():
                pass
            with self._instance.end_frame():
                if self._camera is not None:
                    self._camera.terminate_sp_funcs()
                    self._camera.Terminate()
                if self._overhead_camera is not None:
                    self._overhead_camera.terminate_sp_funcs()
                    self._overhead_camera.Terminate()
                if self._overhead_actor is not None:
                    self._game.unreal_service.destroy_actor(actor=self._overhead_actor)
                self._game.unreal_service.destroy_actor(actor=self._agent)
        except Exception as exc:
            spear.log(f"close() cleanup skipped (engine may be in error state): {exc}")
        self._agent = None
        self._overhead_actor = None
