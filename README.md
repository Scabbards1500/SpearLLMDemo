# SPEAR LLM-Controlled Agent + Frame-Perfect Recording

UE-2 deliverable: LLM-driven SPEAR ground agent with in-loop, frame-perfect recording.

## Quick start

```powershell
conda activate spearenv
cd d:\python\SpearLLMDemo
pip install -r requirements.txt
pip install -e D:\dev\spear\python

# Copy configs
copy user_config.yaml.example user_config.yaml
copy .env.example .env          # set ANTHROPIC_API_KEY

# Run agent (records every frame by default)
python -m src.main
```

First-time environment setup and dev tools live under **`preparation/`** (see `preparation/README.md`).

## Configuration

| File | Purpose |
|------|---------|
| `user_config.yaml` | SpearSim.exe path, default map (`debug_0000`) |
| `.env` | API key, LLM provider/model, cadence, frame limits, episode name |
| `scenes.yaml` | Scene level + agent spawn point |
| `prompts.yaml` | LLM navigation goals (switch with keys 1/2/3) |

### Environment variables (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `qwen` |
| `LLM_MODEL` | provider default | e.g. `claude-opus-4-6`, `qwen3.6-plus` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `DASHSCOPE_API_KEY` | — | Required when `LLM_PROVIDER=qwen` |
| `CONTROL_CADENCE` | `10` | LLM request interval in simulation frames |
| `MAX_FRAMES` | `600` | Episode length (one record per frame) |
| `TARGET_FPS` | `30` | Nominal fps for manifest `t` timestamps |
| `EPISODE_NAME` | `episode_001` | Output folder under `recordings/` |
| `RECORDING` | `1` | Set `0` to disable Part B |
| `PLAN` | `1` | Set `0` to disable plan + memory |

API keys are read from the environment only — never hardcoded in source. Copy `.env.example` → `.env` and keep `.env` out of git.

## LLM provider (Part A)

**Default:** Anthropic vision LLM via the official Python SDK (`anthropic`), model `claude-opus-4-6`.

**Alternate:** Qwen via DashScope OpenAI-compatible API (`openai` package), set `LLM_PROVIDER=qwen`.

Each control step sends the LLM:

- Egocentric RGB frame (JPEG, base64)
- World pose (`X`, `Y`, `Z`, `Yaw`)
- Natural-language navigation goal from `prompts.yaml`
- Optional plan/memory context when `PLAN=1`

The LLM returns structured JSON, e.g. `{"left": 0.5, "right": 0.5}` or with plan mode `{"action": {"left": ..., "right": ...}, "arrived": false, ...}`. Actions are differential-drive wheel speeds in `[0, 1]`.

**Control loop:** the simulation frame loop runs every frame (sync SPEAR step + record). The LLM runs on a slower cadence (`CONTROL_CADENCE`) in a background thread (`AsyncLLMController`); the frame loop never blocks on the API. Between LLM decisions the current action is held — one LLM decision spans many recorded frames.

```powershell
# Anthropic (default)
set ANTHROPIC_API_KEY=sk-ant-...
python -m src.main

# Qwen
set LLM_PROVIDER=qwen
set LLM_MODEL=qwen3.6-plus
set DASHSCOPE_API_KEY=sk-...
python -m src.main

# Shorter episode, faster LLM cadence
python -m src.main --max-frames 600 --cadence 5 --episode episode_600
```

Smoke tests: `python preparation/tests/test_claude.py`, `python preparation/tests/test_qwen.py`.

## Agent

Ground agent: SPEAR built-in **`BP_SphereAgent`** (`/SpContent/Blueprints/BP_SphereAgent.BP_SphereAgent_C`) — the standard simple navigation embodiment in the provided SPEAR build (same as `examples/control_simple_agent`). Egocentric camera + physics sphere; actions mapped to forward force and yaw from `left`/`right` wheel commands.

## Project layout

```
src/
  main.py              # CLI entry: python -m src.main
  agent/               # Part A: LLM agent
    loop.py            # Frame loop + async LLM cadence
    env.py             # SPEAR sync env, observations, actions
    llm.py             # Anthropic vision -> left/right JSON
    prompts.py         # Goal prompt loading
    scenes.py          # Scene presets (debug_house, ...)
    hud.py             # OpenCV HUD overlays
  recorder/            # Part B: frame-perfect in-loop recording
    recorder.py        # frames/ + manifest.jsonl per episode
  utils/               # Shared helpers
    config.py          # Settings from .env
    viz.py             # OpenCV arrow/compass overlays
    process.py         # Kill stale SpearSim on reconnect

preparation/         # Setup scripts, smoke tests, dev demos (not submitted)
recordings/          # Output bundles (Part B)
```

## Episode goal

Navigate the scene per natural-language prompt (default: reach the **nearest table**). See `prompts.yaml`.

**Termination:** LLM sets `arrived=true` when at the goal (agent stops, prints `arrived`), or episode ends at `MAX_FRAMES`, or Q/Esc in OpenCV.

## Recording bundle (Part B)

Each run writes to `recordings/<episode_name>/`:

| Path | Content |
|------|---------|
| `frames/0.png` | Egocentric RGB, one file per simulation frame |
| `manifest.jsonl` | One JSON line per frame: `frame_idx`, `t`, `action`, `location`, `rotation`, `image`, `action_id` |
| `episode_meta.json` | Scene, spawn, goal, fps, cadence (optional metadata) |
| `memory.json` | Episodic memory snapshot (when `PLAN=1`) |
| `plan.json` | Current navigation plan (when `PLAN=1`) |

Recording is on by default. Disable with `--no-record` or `RECORDING=0` in `.env`.
Plan + memory are on by default (`PLAN=1`). Disable with `--no-plan` or `PLAN=0`.

## Controls (OpenCV window)

| Key | Action |
|-----|--------|
| 1/2/3 | Switch navigation prompt |
| Enter | Toggle agent / overhead view |
| Q/Esc | Quit |

## External dependencies

- Windows, UE 5.5, VS 2022 C++
- Official [spear-sim/spear](https://github.com/spear-sim/spear) at `D:\dev\spear`
- Compiled `SpearSim.exe` (see `preparation/setup/setup_spear.ps1`)
