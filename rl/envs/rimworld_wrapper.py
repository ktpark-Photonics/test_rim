"""Gymnasium compatible wrapper for interacting with RimWorld via screen/input."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # pragma: no cover - gymnasium is optional for lightweight testing
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback stubs for tests without gymnasium
    class _DummyEnv:
        observation_space = None
        action_space = None

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            return None, {}

        def step(self, action):
            raise NotImplementedError

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            return np.zeros(self.shape, dtype=self.dtype)

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def sample(self) -> int:
            return 0

    class _Dict(dict):
        def sample(self):
            return {key: space.sample() for key, space in self.items()}

    class _Spaces:
        Box = _Box
        Discrete = _Discrete
        Dict = _Dict

    class _GymModule:
        Env = _DummyEnv

    gym = _GymModule()
    spaces = _Spaces()

from agent.action_executor import ActionExecutor
from agent.overlay import HudDashboard
from agent.screen_capture import CaptureConfig, FrameStacker, ScreenCapture
from agent.window_manager import WindowManager

from .tasks import CurriculumManager, CurriculumRule, RewardTable, Task, TaskResult, build_default_tasks


DEFAULT_ACTIONS = [
    "NOOP",
    "SELECT_COLONIST_0",
    "OPEN_ARCHITECT",
    "OPEN_ZONE",
    "CREATE_STOCKPILE",
    "SET_SPEED_1",
    "SET_SPEED_2",
    "SET_SPEED_3",
    "TOGGLE_PAUSE",
]


@dataclass
class ObservationConfig:
    width: int = 320
    height: int = 180
    frame_stack: int = 4
    grayscale: bool = False
    scalar_features: int = 8


@dataclass
class ActionConfig:
    names: list[str]
    param_dim: int = 4
    param_bounds: tuple[float, float] = (-1.0, 1.0)


class RimWorldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Dict,
        frame_provider=None,
        hud: Optional[HudDashboard] = None,
    ) -> None:
        super().__init__()
        obs_cfg_data = cfg.get("obs", {})
        self.obs_cfg = ObservationConfig(
            width=obs_cfg_data.get("width", 320),
            height=obs_cfg_data.get("height", 180),
            frame_stack=obs_cfg_data.get("frame_stack", 4),
            grayscale=obs_cfg_data.get("grayscale", False),
            scalar_features=obs_cfg_data.get("scalar_features", 8),
        )
        action_names = cfg.get("actions", DEFAULT_ACTIONS)
        self.action_cfg = ActionConfig(
            names=action_names,
            param_dim=cfg.get("action", {}).get("param_dim", 4),
            param_bounds=tuple(cfg.get("action", {}).get("param_bounds", (-1.0, 1.0))),
        )
        reward_cfg = cfg.get("reward", {})
        self.reward_table = RewardTable(
            select_colonist=reward_cfg.get("select_colonist", 0.2),
            open_architect=reward_cfg.get("open_architect", 0.4),
            open_zone=reward_cfg.get("open_zone", 0.4),
            create_stockpile=reward_cfg.get("create_stockpile", 1.0),
            misclick=reward_cfg.get("misclick", 0.1),
            timeout=reward_cfg.get("timeout", 0.5),
            time_penalty=reward_cfg.get("time_penalty", 0.01),
            success_bonus=reward_cfg.get("success_bonus", 1.0),
        )
        self.window_manager = WindowManager()
        capture_config = CaptureConfig(
            width=self.obs_cfg.width,
            height=self.obs_cfg.height,
            frame_stack=self.obs_cfg.frame_stack,
            grayscale=self.obs_cfg.grayscale,
        )
        if frame_provider is None:
            frame_provider = lambda: np.zeros((self.obs_cfg.height, self.obs_cfg.width, 3), dtype=np.uint8)
        self.capture = ScreenCapture(self.window_manager, capture_config, frame_provider=frame_provider)
        self.frame_stacker = FrameStacker(
            capacity=self.obs_cfg.frame_stack,
            frame_shape=(self.obs_cfg.height, self.obs_cfg.width, 1 if self.obs_cfg.grayscale else 3),
        )
        self.executor = ActionExecutor(self.window_manager, enable=cfg.get("enable_input", False))
        self.hud = hud or HudDashboard(enable_gui=cfg.get("dashboard", {}).get("hud_charts", False))

        action_low, action_high = self.action_cfg.param_bounds
        channels = self.obs_cfg.frame_stack * (1 if self.obs_cfg.grayscale else 3)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(channels, self.obs_cfg.height, self.obs_cfg.width), dtype=np.uint8),
                "scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_cfg.scalar_features,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Dict(
            {
                "id": spaces.Discrete(len(self.action_cfg.names)),
                "params": spaces.Box(low=action_low, high=action_high, shape=(self.action_cfg.param_dim,), dtype=np.float32),
            }
        )
        self.action_lookup = {idx: name for idx, name in enumerate(self.action_cfg.names)}
        tasks = build_default_tasks(self.action_lookup, self.reward_table)
        rule_cfg = cfg.get("curriculum", {})
        self.curriculum = CurriculumManager(
            tasks,
            rule=CurriculumRule(
                promotion_threshold=rule_cfg.get("promotion_threshold", 0.8),
                demotion_threshold=rule_cfg.get("demotion_threshold", 0.3),
                window=rule_cfg.get("window", 20),
            ),
        )
        self.current_task: Task = self.curriculum.current_task()
        self.episode_steps = 0
        self.last_action_id = 0
        self.last_reward = 0.0
        self.success_ema = 0.5
        self.reward_ema = 0.0
        self.skill_score = 50.0
        self.hud.set_current_task(self.current_task.name)

    # ------------------------------------------------------------------ Gym API
    def reset(self, *, seed: int | None = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_task = self.curriculum.current_task()
        self.current_task.reset()
        self.hud.set_current_task(self.current_task.name)
        self.frame_stacker.reset()
        self.episode_steps = 0
        self.last_reward = 0.0
        self.last_action_id = 0
        for _ in range(self.obs_cfg.frame_stack):
            frame = self.capture.capture()
            self.frame_stacker.push(frame)
        obs = self._build_observation()
        info = self._build_info([], reset=True)
        return obs, info

    def step(self, action: Dict[str, np.ndarray]):
        action_id = int(action.get("id", 0))
        params = action.get("params")
        if params is None:
            params = np.zeros(self.action_cfg.param_dim, dtype=np.float32)
        params = np.asarray(params, dtype=np.float32)
        action_name = self.action_lookup.get(action_id, "NOOP")
        self.executor.execute_macro(action_name, params.tolist())

        result: TaskResult = self.current_task.step(action_name, params.tolist())
        reward = float(result.reward)
        terminated = result.done and result.success
        truncated = result.done and not result.success
        self.episode_steps += 1
        self.last_action_id = action_id
        self.last_reward = reward
        self._update_skill_score(result.success, reward)
        self._log_events(result)

        frame = self.capture.capture()
        self.frame_stacker.push(frame)
        obs = self._build_observation()
        info = self._build_info(result.events, reset=False, result=result)

        if result.done:
            self.curriculum.record_outcome(self.current_task, result.success)
            self.current_task = self.curriculum.current_task()
            self.current_task.reset()
            self.hud.set_current_task(self.current_task.name)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ helpers
    def _build_observation(self) -> Dict[str, np.ndarray]:
        stack = self.frame_stacker.stack
        if stack.ndim == 4:  # (stack, H, W, C)
            stack = np.transpose(stack, (0, 3, 1, 2))
            stack = stack.reshape(-1, self.obs_cfg.height, self.obs_cfg.width)
        scalars = self._collect_scalars()
        return {"image": stack.astype(np.uint8), "scalars": scalars.astype(np.float32)}

    def _collect_scalars(self) -> np.ndarray:
        task_progress = self.current_task.stage / max(len(self.current_task.steps), 1)
        step_norm = self.episode_steps / max(self.current_task.max_steps, 1)
        success_flag = 1.0 if self.current_task.success else 0.0
        action_norm = self.last_action_id / max(len(self.action_cfg.names) - 1, 1)
        reward_norm = np.tanh(self.last_reward)
        skill_norm = self.skill_score / 100.0
        ema_reward = self.reward_ema
        ema_success = self.success_ema
        features = np.array(
            [
                task_progress,
                step_norm,
                success_flag,
                action_norm,
                reward_norm,
                skill_norm,
                ema_reward,
                ema_success,
            ],
            dtype=np.float32,
        )
        if self.obs_cfg.scalar_features > len(features):
            pad = np.zeros(self.obs_cfg.scalar_features - len(features), dtype=np.float32)
            features = np.concatenate([features, pad])
        return features[: self.obs_cfg.scalar_features]

    def _build_info(self, events, reset: bool, result: TaskResult | None = None) -> Dict:
        info = {
            "task": self.current_task.name,
            "events": [event.__dict__ for event in events],
            "skill_score": self.skill_score,
            "success_ema": self.success_ema,
            "reward_ema": self.reward_ema,
            "stage": self.current_task.stage,
            "total_stages": len(self.current_task.steps),
        }
        if result is not None:
            info.update({"success": result.success, "done": result.done})
        self.hud.plot_scalar("reward", self.last_reward)
        self.hud.plot_scalar("SkillScore", self.skill_score)
        self.hud.plot_scalar("success_rate", self.success_ema)
        if events:
            for event in events:
                self.hud.log_event(f"{event.name}: {event.reward:+.2f} - {event.description}")
        return info

    def _update_skill_score(self, success: bool, reward: float) -> None:
        alpha = 0.1
        self.success_ema = (1 - alpha) * self.success_ema + alpha * (1.0 if success else 0.0)
        self.reward_ema = (1 - alpha) * self.reward_ema + alpha * reward
        ratio = (self.success_ema + 1e-3) / (1 - self.success_ema + 1e-3)
        score = 50 + 10 * math.log(ratio)
        score += 5 * np.tanh(self.reward_ema)
        self.skill_score = float(np.clip(score, 0.0, 100.0))

    def _log_events(self, result: TaskResult) -> None:
        if result.events:
            for event in result.events:
                self.hud.log_event(f"{event.name}: {event.reward:+.2f}")

    def render(self):  # pragma: no cover - not required for automated tests
        pass

    def close(self):  # pragma: no cover - resource cleanup not required in tests
        pass


__all__ = ["RimWorldEnv", "ObservationConfig", "ActionConfig"]
