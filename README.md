# RimWorld Visual-Behavior Agent

This repository implements a Windows-friendly RimWorld assistant that learns to
interact with the game using computer vision, safe input injection and multiple
learning strategies (Behavior Cloning, DAgger and PPO). The system is designed
for Python 3.11 and targets RimWorld running in windowed mode with configurable
resolution.

## Repository Structure

```
agent/
  window_manager.py        # Window handle discovery and focus checks
  screen_capture.py        # High-speed capture using mss with frame stacking
  perception/             # Template matching, OCR and UI maps
  policy/                 # Behavior-tree primitives and scripted helpers
  overlay.py              # PySide6 HUD with real-time charts and recorder toggle
  demo_recorder.py        # Saves frame stacks + macro actions for imitation
  app.py                  # High level runtime harness
rl/
  envs/rimworld_wrapper.py # Gymnasium-compatible environment wrapper
  envs/tasks.py            # Curriculum-aware task & reward definitions
  algo/                    # BC, PPO, DAgger and replay utilities
  models/                  # CNN backbone + multi-head policy/value heads
  train.py                 # Unified training CLI with config overrides
  eval.py                  # Evaluation/recording helper
scripts/sweep.py           # Hyper-parameter sweep helper
configs/                   # Agent, RL, model and reward configuration templates
data/                      # Checkpoints, captures and demos (gitkept)
```

## Quick Start

1. **Install dependencies** (Python 3.11):
   ```bash
   pip install -e .[dev]
   ```
   Optional packages such as `pywin32` only install on Windows. PySide6 and
   OpenCV are required for the HUD and perception stack.

2. **Collect demonstrations** using the runtime HUD (Ctrl+Alt+D toggle) or by
   calling `DemoRecorder` programmatically. Recordings are stored under
   `data/demos/` as compressed `.npz` bundles containing frame stacks, scalar
   features, macro IDs, parameters and timestamps.

3. **Pre-train with behavior cloning**:
   ```bash
   python -m rl.train --mode bc --config configs/rl.yaml
   ```
   TensorBoard summaries are written to `data/checkpoints/bc-*/`.

4. **Fine-tune with PPO**:
   ```bash
   python -m rl.train --mode ppo --config configs/rl.yaml
   ```
   The default configuration uses a stubbed environment that produces synthetic
   frames so training scripts can be exercised without RimWorld. Disable the
   stub (`env.stub: false`) to operate on the real window; ensure RimWorld is in
   windowed mode and focused to allow safe input injection.

5. **Aggregate demonstrations with DAgger**:
   ```bash
   python -m rl.train --mode dagger --config configs/rl.yaml
   ```
   The trainer queries a scripted expert whenever policy confidence is low (or
   at random according to `dagger.query_prob`) and periodically retrains the BC
   model on the aggregated dataset.

6. **Evaluate checkpoints**:
   ```bash
   python -m rl.eval --config configs/rl.yaml --checkpoint data/checkpoints/ppo-5000.pt --mode ppo
   ```
   Add `--record data/captures/evals` to dump per-episode `.npz` captures.

## Configuration & Overrides

Configuration files are standard YAML documents validated via the lightweight
loader in `agent/config.py`. Values can be overridden from the CLI using dot
notation, for example:

```bash
python -m rl.train --config configs/rl.yaml --mode ppo train.total_steps=20000 ppo.lr=1e-4 action.param_bounds="[-0.8,0.8]"
```

Key files:

- `configs/agent.yaml` – capture resolution, frame stacking and HUD switches for
  the runtime app.
- `configs/rl.yaml` – RL/IL defaults (actions, PPO hyper-parameters, curriculum
  rules, dashboard options). Automatically loads `configs/reward.yaml` if a
  reward table is not inlined.
- `configs/reward.yaml` – per-event shaping coefficients used by
  `RewardTable` (select colonist, open architect, open zone, stockpile, etc.).
- `configs/model.yaml` – placeholder for model-architecture overrides.

## Reward & Mission Tuning

`rl/envs/tasks.py` defines the curriculum used by the `RimWorldEnv`. Each task is
composed of ordered `TaskStep`s and uses `RewardTable` values for shaping:

- `select_colonist`: first colonist selection detection
- `open_architect`: architect menu opening
- `open_zone`: zone submenu activation
- `create_stockpile`: final stockpile placement

Misclicks and timeouts apply negative shaping while a small per-step time
penalty encourages quick completion. Adjust the magnitudes in
`configs/reward.yaml` and reload to experiment with different curricula. The
HUD's line charts (reward, success rate, skill score) provide immediate visual
feedback on tuning changes.

## Real-time Visualization

The PySide6 HUD overlays current task information, event logs and rolling charts
for reward, success rate and the EMA-based `SkillScore`. All metrics are also
mirrored to TensorBoard (`train/*`, `ppo/*`, `bc/*`, `dagger/*` namespaces).

## Safety Features

- Global hotkey placeholder (Ctrl+Alt+S) for manual interruption.
- Input injection is suppressed whenever the RimWorld window is unfocused.
- The training wrapper pauses actions if the window is minimised or occluded
  (via focus checks in `WindowManager`).
- No memory hooking or packet manipulation; only screen capture + input events
  are used.

## Hyper-parameter Sweeps

Use `scripts/sweep.py` to run grid searches with automatic override handling:

```bash
python scripts/sweep.py --mode ppo \
  --sweep ppo.lr 3e-4 1e-4 \
  --sweep ppo.clip_ratio 0.1 0.2
```

Append `--dry-run` to preview commands without executing them. Each combination
is executed via `python -m rl.train` with the provided overrides and isolated
output directories.

## Testing

Two lightweight sanity tests are included:

- `tests/offline_detection.py` – verifies template registration and matching via
  OpenCV using synthetic frames.
- `tests/env_stub.py` – checks reward transitions and observation formatting for
  the Gym wrapper using the synthetic stub environment.

Run the suite with `pytest`.

## License

This project is released under the MIT License. Respect the RimWorld EULA – do
not perform memory inspection, network packet manipulation or other prohibited
modifications. Screen capture and simulated input are the only interaction
mechanisms provided.
