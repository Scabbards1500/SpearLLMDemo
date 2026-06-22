"""Episodic memory + plan state for LLM navigation (optional, plan=True)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.agent.env import Observation, WheelAction
from src.agent.prompts import GoalPrompt


def _dist_xy(a: dict[str, float], b: dict[str, float]) -> float:
    dx = a["X"] - b["X"]
    dy = a["Y"] - b["Y"]
    return math.hypot(dx, dy)


def _initial_memory(episode: str, goal: GoalPrompt) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "episode": episode,
        "updated_at_frame": 0,
        "updated_at_action_id": 0,
        "summary": "Episode started.",
        "places": [],
        "dead_ends": [],
        "decision_log": [],
        "constraints": [],
    }


def _initial_plan(goal: GoalPrompt) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "goal_prompt_id": goal.prompt_id,
        "goal_text": goal.text.strip(),
        "updated_at_frame": 0,
        "updated_at_action_id": 0,
        "phase": "explore",
        "phase_reason": "Episode start; target not yet visible.",
        "target": {
            "type": "explore_frontier",
            "description": "Map the room and search for the nearest table.",
            "estimated_location": None,
            "confidence": 0.3,
        },
        "subgoals": [
            {"id": "sg1", "status": "active", "text": "Leave the starting area safely"},
            {"id": "sg2", "status": "pending", "text": "Find the nearest table in view"},
            {"id": "sg3", "status": "pending", "text": "Approach the nearest table and stop nearby"},
        ],
        "active_subgoal_id": "sg1",
        "next_action_intent": None,
        "avoid": [],
        "replan_triggers": ["stuck_for_30_frames", "goal_visible", "prompt_changed"],
    }


class LandDetector:
    """Detect when the spawn drop is finished (Z stops changing)."""

    def __init__(
        self,
        *,
        already_landed: bool = False,
        stable_frames: int = 4,
        dz_threshold: float = 1.5,
    ) -> None:
        self.landed = already_landed
        self.stable_frames = stable_frames
        self.dz_threshold = dz_threshold
        self._stable = stable_frames if already_landed else 0
        self._last_z: float | None = None

    def reset(self) -> None:
        self.landed = False
        self._stable = 0
        self._last_z = None

    def update(self, obs: Observation) -> bool:
        """Return True on the first frame landing is detected."""
        if self.landed:
            return False
        z = obs.location["Z"]
        if self._last_z is not None and abs(z - self._last_z) < self.dz_threshold:
            self._stable += 1
        else:
            self._stable = 0
        self._last_z = z
        if self._stable >= self.stable_frames:
            self.landed = True
            return True
        return False


class StuckDetector:
    """Detect low displacement despite forward command."""

    def __init__(
        self,
        window: int = 30,
        min_dist_cm: float = 8.0,
        min_forward: float = 0.15,
    ) -> None:
        self.window = window
        self.min_dist_cm = min_dist_cm
        self.min_forward = min_forward
        self._origin: dict[str, float] | None = None
        self._frames = 0

    def reset(self) -> None:
        self._origin = None
        self._frames = 0

    def update(self, obs: Observation, action: WheelAction) -> str | None:
        forward = (action.left + action.right) * 0.5
        if forward < self.min_forward:
            self.reset()
            return None

        loc = obs.location
        if self._origin is None:
            self._origin = dict(loc)
            self._frames = 1
            return None

        self._frames += 1
        if self._frames < self.window:
            return None

        if _dist_xy(self._origin, loc) < self.min_dist_cm:
            self.reset()
            return (
                f"Robot barely moved in the last {self.window} frames while driving. "
                "Treat current heading as a dead end; turn away and explore elsewhere."
            )

        self._origin = dict(loc)
        self._frames = 0
        return None


class CognitionStore:
    """Maintains memory.json + plan.json snapshots per LLM decision."""

    def __init__(
        self,
        episode: str,
        goal: GoalPrompt,
        episode_dir: Path | None = None,
    ) -> None:
        self.episode = episode
        self.episode_dir = Path(episode_dir) if episode_dir else None
        self.memory = _initial_memory(episode, goal)
        self.plan = _initial_plan(goal)
        self._dead_end_seq = 0

    def reset_goal(self, goal: GoalPrompt, frame_idx: int) -> None:
        self.plan = _initial_plan(goal)
        self.plan["updated_at_frame"] = frame_idx
        self.plan["phase_reason"] = "Navigation prompt changed."
        self.memory["constraints"].append(f"New goal: {goal.label}")
        self.save()

    def context_for_llm(self, stuck_hint: str | None = None) -> str:
        avoid_ids = self.plan.get("avoid") or []
        dead_labels = [
            de.get("dead_end_id", "?")
            for de in self.memory.get("dead_ends", [])
            if de.get("dead_end_id") in avoid_ids
        ]
        lines = [
            "Current memory summary:",
            self.memory.get("summary", ""),
            f"Plan phase: {self.plan.get('phase')} — {self.plan.get('phase_reason', '')}",
            f"Active subgoal: {self.plan.get('active_subgoal_id')}",
            f"Avoid dead ends: {', '.join(dead_labels) if dead_labels else 'none yet'}",
        ]
        constraints = self.memory.get("constraints") or []
        if constraints:
            lines.append("Constraints: " + "; ".join(constraints[-4:]))
        if stuck_hint:
            lines.append(f"STUCK ALERT: {stuck_hint}")
        return "\n".join(lines)

    def register_code_dead_end(self, obs: Observation, reason: str) -> str:
        self._dead_end_seq += 1
        dead_id = f"de_{self._dead_end_seq:03d}"
        loc = obs.location
        rot = obs.rotation
        entry = {
            "dead_end_id": dead_id,
            "location": [loc["X"], loc["Y"], loc["Z"]],
            "yaw_deg": rot["Yaw"],
            "discovered_frame": obs.frame_index,
            "reason": reason,
            "avoid_until_frame": None,
        }
        self.memory.setdefault("dead_ends", []).append(entry)
        avoid = self.plan.setdefault("avoid", [])
        if dead_id not in avoid:
            avoid.append(dead_id)
        return dead_id

    def apply_decision(
        self,
        *,
        action_id: int,
        frame_idx: int,
        obs: Observation,
        action: WheelAction,
        memory_update: dict[str, Any] | None,
        plan_patch: dict[str, Any] | None,
        arrived: bool = False,
    ) -> None:
        loc = obs.location
        rot = obs.rotation

        log_entry: dict[str, Any] = {
            "action_id": action_id,
            "frame_idx": frame_idx,
            "location": [loc["X"], loc["Y"], loc["Z"]],
            "yaw_deg": rot["Yaw"],
            "action": {"left": action.left, "right": action.right},
        }
        if arrived:
            log_entry["arrived"] = True
        if memory_update:
            if memory_update.get("observation"):
                log_entry["observation"] = memory_update["observation"]
            if memory_update.get("outcome"):
                log_entry["outcome"] = memory_update["outcome"]
        self.memory.setdefault("decision_log", []).append(log_entry)

        if memory_update:
            if memory_update.get("summary"):
                self.memory["summary"] = str(memory_update["summary"])
            for place in memory_update.get("add_places") or []:
                self.memory.setdefault("places", []).append(place)
            for dead in memory_update.get("add_dead_ends") or []:
                self.memory.setdefault("dead_ends", []).append(dead)
                dead_id = dead.get("dead_end_id")
                if dead_id:
                    avoid = self.plan.setdefault("avoid", [])
                    if dead_id not in avoid:
                        avoid.append(dead_id)
            for c in memory_update.get("constraints") or []:
                constraints = self.memory.setdefault("constraints", [])
                if c not in constraints:
                    constraints.append(str(c))

        if arrived:
            if plan_patch is None:
                plan_patch = {}
            plan_patch = {
                **plan_patch,
                "phase": "arrived",
                "phase_reason": plan_patch.get(
                    "phase_reason", "Navigation goal reached."
                ),
            }
            if not (memory_update and memory_update.get("summary")):
                summary = str(self.memory.get("summary", "")).strip()
                self.memory["summary"] = (
                    f"{summary} Arrived at goal.".strip() if summary else "Arrived at goal."
                )

        if plan_patch:
            for key, value in plan_patch.items():
                if key in ("schema_version", "goal_prompt_id", "goal_text"):
                    continue
                self.plan[key] = value

        self.memory["updated_at_frame"] = frame_idx
        self.memory["updated_at_action_id"] = action_id
        self.plan["updated_at_frame"] = frame_idx
        self.plan["updated_at_action_id"] = action_id
        self.save()

    def save(self) -> None:
        if self.episode_dir is None:
            return
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "memory.json").write_text(
            json.dumps(self.memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.episode_dir / "plan.json").write_text(
            json.dumps(self.plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
