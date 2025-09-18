"""Dataset Aggregation (DAgger) utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

import numpy as np
from torch.utils.data import DataLoader

from rl.algo.bc import BehaviorCloningLearner, DemoDataset


@dataclass
class DAggerConfig:
    enabled: bool = True
    query_prob: float = 0.1
    update_interval: int = 100
    max_buffer_size: int = 50_000


@dataclass
class DAggerSample:
    image: np.ndarray
    scalars: np.ndarray
    action_id: int
    params: np.ndarray


class DAggerTrainer:
    """Collects additional demonstrations during interaction to refine the policy."""

    def __init__(
        self,
        config: DAggerConfig,
        learner: BehaviorCloningLearner,
        expert_policy: Callable[[dict], dict],
    ) -> None:
        self.config = config
        self.learner = learner
        self.expert_policy = expert_policy
        self.buffer: List[DAggerSample] = []
        self.steps_since_update = 0

    def maybe_query_expert(self, observation: dict, policy_action: dict) -> dict:
        if random.random() < self.config.query_prob:
            expert_action = self.expert_policy(observation)
            self._append_sample(observation, expert_action)
            return expert_action
        self._append_sample(observation, policy_action)
        return policy_action

    def _append_sample(self, observation: dict, action: dict) -> None:
        if not self.config.enabled:
            return
        sample = DAggerSample(
            image=np.array(observation["image"], copy=True),
            scalars=np.array(observation["scalars"], copy=True),
            action_id=int(action.get("id", 0)),
            params=np.array(action.get("params", np.zeros(1, dtype=np.float32)), copy=True),
        )
        self.buffer.append(sample)
        if len(self.buffer) > self.config.max_buffer_size:
            self.buffer = self.buffer[-self.config.max_buffer_size :]
        self.steps_since_update += 1

    def should_update(self) -> bool:
        return self.config.enabled and self.steps_since_update >= self.config.update_interval

    def update_policy(self) -> dict:
        if not self.should_update():
            return {}
        dataset = self._to_dataset()
        loader = DataLoader(dataset, batch_size=self.learner.config.batch_size, shuffle=True)
        history = self.learner.fit(loader)
        self.steps_since_update = 0
        return history

    def _to_dataset(self) -> DemoDataset:
        tmp_dir = Path("data/demos/_dagger")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / "buffer.npz"
        frames = np.stack([sample.image for sample in self.buffer], axis=0)
        scalars = np.stack([sample.scalars for sample in self.buffer], axis=0)
        actions = np.array([sample.action_id for sample in self.buffer], dtype=np.int64)
        params = np.stack([sample.params for sample in self.buffer], axis=0)
        timestamps = np.zeros(len(self.buffer), dtype=np.float64)
        np.savez_compressed(path, frames=frames, scalars=scalars, actions=actions, params=params, timestamps=timestamps)
        return DemoDataset([path])


__all__ = ["DAggerTrainer", "DAggerConfig"]
