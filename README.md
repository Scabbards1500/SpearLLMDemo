# SPEAR LLM-Controlled Agent + Frame-Perfect Recording

UE-2 deliverable: LLM-driven SPEAR ground agent with in-loop, frame-perfect recording.

## Prerequisites

- Windows + Unreal Engine 5.5（本机：`D:\Program Files\Epic Games\UE_5.5`）
- 官方 [spear-sim/spear](https://github.com/spear-sim/spear) 仓库（**无 dashboard**，TA 确认直接用 GitHub）
- Python 3.11（conda env `spearenv`）
- Visual Studio 2022（含 C++ 游戏开发 workload，用于编译 SpearSim）

## 仓库分工

| 路径 | 用途 |
|------|------|
| `D:\dev\spear` | 官方 SPEAR 源码 + 编译 SpearSim |
| `D:\python\SpearLLMDemo` | 本作业代码（LLM agent + 内录） |

## Environment setup

```powershell
# 1. Conda 环境（已完成可跳过）
conda create -n spearenv python=3.11 -y
conda activate spearenv

# 2. 作业项目依赖
cd d:\python\SpearLLMDemo
pip install -r requirements.txt

# 3. SPEAR Python 客户端（从官方仓库 editable 安装）
pip install -e D:\dev\spear\python
```

### 编译 SpearSim（首次，耗时 30–90+ 分钟）

在 **Developer PowerShell for VS 2022** 中运行：

```powershell
conda activate spearenv
cd d:\python\SpearLLMDemo
.\scripts\setup_spear.ps1
```

或按 [getting_started](https://github.com/spear-sim/spear/blob/main/docs/getting_started.md) 逐步执行。成功后：

```
D:\dev\spear\cpp\unreal_projects\SpearSim\Standalone-Development\Windows\SpearSim.exe
```

### 安装 [spear-sim/spear](https://github.com/spear-sim/spear)

**不能**用 `pip install spear`（PyPI 上是 2010 年的无关旧包）。

```powershell
git clone https://github.com/spear-sim/spear D:\dev\spear --recurse-submodules
pip install -e D:\dev\spear\python
```

默认场景可用官方自带的 `apartment_0000` / `debug_0000` 等（见 SPEAR docs）。

## Configuration

1. Copy `user_config.yaml.example` → `user_config.yaml` and set `GAME_EXECUTABLE` to your `SpearSim.exe`.
2. Copy `.env.example` → `.env` and set `ANTHROPIC_API_KEY`.

## Run (once implemented)

```powershell
conda activate spearenv
python -m src.main
```

Recordings land in `recordings/<episode>/` (`frames/` + `manifest.jsonl`).

## Episode goal

_TBD — point-goal navigation to a target world location._

## Project layout

```
src/
  main.py           # frame loop entry
  spear_env.py      # SPEAR sync step + observations
  llm_controller.py # LLM action decisions
  recorder.py       # in-loop manifest + frame images
  config.py         # settings
recordings/         # output bundles
demo/               # demo.mp4 (30–90s screen capture)
```
