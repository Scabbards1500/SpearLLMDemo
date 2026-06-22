"""Smoke test: ANTHROPIC_API_KEY + Claude connectivity."""

from __future__ import annotations

import os
import sys

from preparation.bootstrap import PROJECT_ROOT, ensure_project_root

ensure_project_root()

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

key = os.getenv("ANTHROPIC_API_KEY", "")
print(f"API key present: {bool(key) and len(key) > 10}")

try:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.getenv("LLM_MODEL", "claude-opus-4-6"),
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    print(f"Model: {msg.model}")
    print(f"Response: {msg.content[0].text.strip()}")
    print("CLAUDE_TEST: PASS")
except Exception as exc:
    print("CLAUDE_TEST: FAIL")
    print(f"{type(exc).__name__}: {exc}")
    sys.exit(1)
