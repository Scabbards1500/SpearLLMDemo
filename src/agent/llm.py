"""LLM vision controller: RGB + pose + goal prompt -> wheel action JSON."""

from __future__ import annotations

import base64
import json
import re
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.agent.env import Observation, WheelAction
from src.agent.prompts import GoalPrompt

JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

SYSTEM_PROMPT = """You control a differential-drive sphere robot in an Unreal Engine indoor simulation.

Each step you receive:
- An egocentric RGB camera image (robot forward = image forward / deeper into scene)
- Robot world pose (X, Y, Z in cm; Yaw in degrees)

Output ONLY one JSON object, no markdown:
{"left": <float 0..1>, "right": <float 0..1>, "arrived": <bool optional>}

When you are close enough to the navigation goal, set "arrived": true and "left": 0, "right": 0.

Wheel semantics:
- left == right > 0 : drive forward (higher = faster)
- left > right      : turn right while moving
- left < right      : turn left while moving
- both near 0       : only when you must stop briefly

If the goal object is not visible, explore: move forward slowly (e.g. left=0.35, right=0.35)
while turning (e.g. left=0.25, right=0.55) to scan the room. Never stay at 0,0 for long.
Use walls and furniture in the image to avoid collisions. Prefer steady forward motion over stopping."""

SYSTEM_PROMPT_PLAN = """You control a differential-drive sphere robot in an Unreal Engine indoor simulation.

You receive an RGB image, pose, navigation goal, current memory summary, and current plan.
Return ONLY one JSON object (no markdown) with this shape:

{
  "action": {"left": <0..1>, "right": <0..1>},
  "arrived": <bool>,
  "plan": {
    "phase": "explore|approach|arrived",
    "phase_reason": "<short>",
    "target": {
      "type": "explore_frontier|goal_object|waypoint",
      "description": "<what you search for or move toward, egocentric — no world coordinates>"
    },
    "subgoals": [{"id":"sg1","status":"pending|active|done","text":"..."}],
    "active_subgoal_id": "sg1",
    "next_action_intent": {
      "left": <0..1>, "right": <0..1>,
      "rationale": "<why this action>"
    },
    "avoid": ["de_001"]
  },
  "memory_update": {
    "summary": "<1-3 sentence running summary>",
    "observation": "<what you see this step>",
    "outcome": "<expected result of this action>",
    "add_places": [],
    "add_dead_ends": [],
    "constraints": ["<optional rule>"]
  }
}

Rules:
- When close enough to the goal, set arrived=true, action left=0 right=0, and plan.phase="arrived".
- Never drive into a listed dead end or narrow corner with no exit.
- If the goal is not visible, phase=explore and keep moving (do not stay at 0,0).
- When the goal is visible and you are steering toward it, phase=approach.
- Do not output world X/Y/Z for the target; steer via action and next_action_intent instead.
- add_dead_ends only when the view shows a true dead end; include dead_end_id like de_003.
- action.left/right are the wheels to execute now; match next_action_intent when possible."""


@dataclass
class LLMDecision:
    action: WheelAction
    arrived: bool = False
    plan: dict[str, Any] | None = None
    memory_update: dict[str, Any] | None = None


def _encode_rgb_jpeg(rgb: np.ndarray, max_side: int = 768, quality: int = 85) -> tuple[str, str]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode observation image")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in LLM response: {text[:200]}")
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError(f"Unbalanced JSON in LLM response: {text[:200]}")


def _parse_action_data(data: dict[str, Any]) -> WheelAction:
    if "action" in data and isinstance(data["action"], dict):
        act = data["action"]
    else:
        act = data
    return WheelAction(left=float(act["left"]), right=float(act["right"])).clamp()


def _parse_arrived(data: dict[str, Any], plan: dict[str, Any] | None) -> bool:
    raw = data.get("arrived")
    if raw is True or (isinstance(raw, str) and raw.lower() in ("true", "1", "yes")):
        return True
    if plan and str(plan.get("phase", "")).lower() == "arrived":
        return True
    return False


def _parse_decision(text: str, *, plan_mode: bool) -> LLMDecision:
    if not plan_mode:
        match = JSON_RE.search(text)
        if not match:
            data = _extract_json_object(text)
        else:
            data = json.loads(match.group())
        plan = None
    else:
        data = _extract_json_object(text)
        plan = data.get("plan") if isinstance(data.get("plan"), dict) else None

    arrived = _parse_arrived(data, plan)
    action = WheelAction(left=0.0, right=0.0) if arrived else _parse_action_data(data)
    if plan_mode:
        return LLMDecision(
            action=action,
            arrived=arrived,
            plan=plan,
            memory_update=data.get("memory_update") if isinstance(data.get("memory_update"), dict) else None,
        )
    return LLMDecision(action=action, arrived=arrived)


def _build_user_text(
    obs: Observation,
    goal: GoalPrompt,
    *,
    last_action: WheelAction | None = None,
    cognition_context: str | None = None,
) -> str:
    loc = obs.location
    rot = obs.rotation
    explore_hint = ""
    if (
        last_action is not None
        and last_action.left < 0.05
        and last_action.right < 0.05
    ):
        explore_hint = (
            "\nYou were stopped. If the goal is NOT yet reached, explore: "
            "drive forward slowly or turn to search. "
            "If you ARE at the goal, set arrived=true with left=0 right=0.\n"
        )
    cognition_block = ""
    if cognition_context:
        cognition_block = f"\n{cognition_context.strip()}\n"
    return (
        f"Navigation goal: {goal.text.strip()}\n"
        f"Pose: X={loc['X']:.1f} Y={loc['Y']:.1f} Z={loc['Z']:.1f} "
        f"Yaw={rot['Yaw']:.1f}\n"
        f"Frame: {obs.frame_index}\n"
        f"{explore_hint}"
        f"{cognition_block}"
        "Return JSON only."
    )


class _AnthropicBackend:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        plan_mode: bool = False,
        base_url: str | None = None,
    ) -> None:
        import anthropic

        self.model = model
        self.plan_mode = plan_mode
        client_kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**client_kwargs)

    def decide(
        self,
        obs: Observation,
        goal: GoalPrompt,
        *,
        last_action: WheelAction | None = None,
        cognition_context: str | None = None,
    ) -> LLMDecision:
        b64, media_type = _encode_rgb_jpeg(obs.rgb)
        user_text = _build_user_text(
            obs, goal, last_action=last_action, cognition_context=cognition_context
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=768 if self.plan_mode else 128,
            system=SYSTEM_PROMPT_PLAN if self.plan_mode else SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )
        block = msg.content[0]
        text = block.text if hasattr(block, "text") else str(block)
        return _parse_decision(text, plan_mode=self.plan_mode)


class _QwenBackend:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        plan_mode: bool = False,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.plan_mode = plan_mode
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def decide(
        self,
        obs: Observation,
        goal: GoalPrompt,
        *,
        last_action: WheelAction | None = None,
        cognition_context: str | None = None,
    ) -> LLMDecision:
        b64, media_type = _encode_rgb_jpeg(obs.rgb)
        user_text = _build_user_text(
            obs, goal, last_action=last_action, cognition_context=cognition_context
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=768 if self.plan_mode else 128,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_PLAN if self.plan_mode else SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        )
        text = resp.choices[0].message.content or ""
        return _parse_decision(text, plan_mode=self.plan_mode)


class LLMController:
    """Synchronous vision controller (Anthropic or Qwen via DashScope)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        provider: str = "anthropic",
        plan_mode: bool = False,
        base_url: str | None = None,
    ) -> None:
        provider = provider.strip().lower()
        if provider == "qwen":
            self._backend: _AnthropicBackend | _QwenBackend = _QwenBackend(
                api_key,
                model,
                plan_mode=plan_mode,
                base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif provider == "anthropic":
            self._backend = _AnthropicBackend(
                api_key,
                model,
                plan_mode=plan_mode,
                base_url=base_url,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def decide(
        self,
        obs: Observation,
        goal: GoalPrompt,
        *,
        last_action: WheelAction | None = None,
        cognition_context: str | None = None,
    ) -> LLMDecision:
        return self._backend.decide(
            obs,
            goal,
            last_action=last_action,
            cognition_context=cognition_context,
        )


class AsyncLLMController:
    """Non-blocking wrapper: frame loop never waits on the API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        provider: str = "anthropic",
        plan_mode: bool = False,
        base_url: str | None = None,
    ) -> None:
        self.plan_mode = plan_mode
        self._sync = LLMController(
            api_key=api_key,
            model=model,
            provider=provider,
            plan_mode=plan_mode,
            base_url=base_url,
        )
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: LLMDecision | None = None
        self._error: Exception | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def request(
        self,
        obs: Observation,
        goal: GoalPrompt,
        *,
        cognition_context: str | None = None,
    ) -> None:
        with self._lock:
            if self._busy:
                return
            if self._result is not None or self._error is not None:
                return
            self._busy = True
            snap = Observation(
                frame_index=obs.frame_index,
                rgb=obs.rgb.copy(),
                location=dict(obs.location),
                rotation=dict(obs.rotation),
                action=obs.action,
            )
            last_action = WheelAction(
                left=obs.action.left,
                right=obs.action.right,
            )

        def worker() -> None:
            try:
                decision = self._sync.decide(
                    snap,
                    goal,
                    last_action=last_action,
                    cognition_context=cognition_context,
                )
                with self._lock:
                    self._result = decision
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._result = None
                    self._error = exc
            finally:
                with self._lock:
                    self._busy = False

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def poll(self) -> tuple[LLMDecision | None, Exception | None]:
        with self._lock:
            if self._result is not None:
                decision, self._result = self._result, None
                err, self._error = self._error, None
                return decision, err
            if self._error is not None:
                err, self._error = self._error, None
                return None, err
            return None, None
