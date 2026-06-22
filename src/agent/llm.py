"""LLM vision controller: RGB + pose + goal prompt -> wheel action JSON."""

from __future__ import annotations

import base64
import json
import re
import threading

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
{"left": <float 0..1>, "right": <float 0..1>}

Wheel semantics:
- left == right > 0 : drive forward (higher = faster)
- left > right      : turn right while moving
- left < right      : turn left while moving
- both near 0       : stop or crawl slowly to adjust

Use vision to follow the user's navigation goal. Avoid walls. Progress toward the goal each decision."""


def _encode_rgb_jpeg(rgb: np.ndarray, max_side: int = 768, quality: int = 85) -> tuple[str, str]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode observation image")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"


def _parse_action(text: str) -> WheelAction:
    match = JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in LLM response: {text[:200]}")
    data = json.loads(match.group())
    return WheelAction(left=float(data["left"]), right=float(data["right"])).clamp()


class LLMController:
    """Synchronous Anthropic vision controller."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def decide(self, obs: Observation, goal: GoalPrompt) -> WheelAction:
        b64, media_type = _encode_rgb_jpeg(obs.rgb)
        loc = obs.location
        rot = obs.rotation
        user_text = (
            f"Navigation goal: {goal.text.strip()}\n"
            f"Pose: X={loc['X']:.1f} Y={loc['Y']:.1f} Z={loc['Z']:.1f} "
            f"Yaw={rot['Yaw']:.1f}\n"
            f"Frame: {obs.frame_index}\n"
            "Return JSON only."
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=128,
            system=SYSTEM_PROMPT,
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
        return _parse_action(text)


class AsyncLLMController:
    """Non-blocking wrapper: frame loop never waits on the API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._sync = LLMController(api_key=api_key, model=model)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: WheelAction | None = None
        self._error: Exception | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def request(self, obs: Observation, goal: GoalPrompt) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True
            self._result = None
            self._error = None
            snap = Observation(
                frame_index=obs.frame_index,
                rgb=obs.rgb.copy(),
                location=dict(obs.location),
                rotation=dict(obs.rotation),
                action=obs.action,
            )

        def worker() -> None:
            try:
                action = self._sync.decide(snap, goal)
                with self._lock:
                    self._result = action
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

    def poll(self) -> tuple[WheelAction | None, Exception | None]:
        with self._lock:
            if self._result is not None:
                action, self._result = self._result, None
                err, self._error = self._error, None
                return action, err
            if self._error is not None:
                err, self._error = self._error, None
                return None, err
            return None, None
