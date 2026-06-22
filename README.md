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
| `.env` | API key, cadence, frame limits, episode name |
| `scenes.yaml` | Scene level + agent spawn point |
| `prompts.yaml` | LLM navigation goals (switch with keys 1/2/3) |

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
