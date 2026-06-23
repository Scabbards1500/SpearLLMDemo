"""Smoke test: ANTHROPIC_API_KEY + Claude connectivity."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

key = os.getenv("ANTHROPIC_API_KEY", "")
base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
model = os.getenv("CLAUDE_TEST_MODEL") or os.getenv("LLM_MODEL", "claude-opus-4-6")
if not model.startswith("claude"):
    model = "claude-opus-4-6"
print(f"API key present: {bool(key) and len(key) > 10}")
print(f"Base URL: {base_url or '(anthropic SDK default)'}")
print(f"Model: {model}")

try:
    import anthropic

    client_kwargs: dict[str, str] = {"api_key": key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**client_kwargs)
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model,
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    latency_s = time.perf_counter() - t0
    print(f"Response model: {msg.model}")
    print(f"Response: {msg.content[0].text.strip()}")
    print(f"Latency: {latency_s * 1000:.0f} ms ({latency_s:.2f} s)")
    print("CLAUDE_TEST: PASS")
except Exception as exc:
    print("CLAUDE_TEST: FAIL")
    print(f"{type(exc).__name__}: {exc}")
    sys.exit(1)
