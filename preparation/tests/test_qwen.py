"""Smoke test: DASHSCOPE_API_KEY + Qwen (OpenAI-compatible) connectivity."""

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

key = os.getenv("DASHSCOPE_API_KEY", "")
model = os.getenv("QWEN_TEST_MODEL") or os.getenv("LLM_MODEL", "qwen3.6-plus")
if not model.startswith("qwen"):
    model = "qwen3.6-plus"
base_url = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
print(f"API key present: {bool(key) and len(key) > 10}")
print(f"Model: {model}")
print(f"Base URL: {base_url}")

try:
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base_url)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    latency_s = time.perf_counter() - t0
    text = (resp.choices[0].message.content or "").strip()
    print(f"Response model: {resp.model}")
    print(f"Response: {text}")
    print(f"Latency: {latency_s * 1000:.0f} ms ({latency_s:.2f} s)")
    print("QWEN_TEST: PASS")
except Exception as exc:
    print("QWEN_TEST: FAIL")
    print(f"{type(exc).__name__}: {exc}")
    sys.exit(1)
