"""Replay buffer utilities supporting optional prioritisation."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Tuple

import numpy as np


@dataclass
class Transition:
    image: np.ndarray
    scalars: np.ndarray
    action_id: int
    params: np.ndarray
    reward: float
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, prioritized: bool = False, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.prioritized = prioritized
        self.alpha = alpha
        self.priorities: Deque[float] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        priority = abs(transition.reward) + 1e-5
        if self.prioritized:
            self.priorities.append(priority)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        if not self.prioritized or len(self.buffer) == 0:
            return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        probs = np.array(self.priorities, dtype=np.float64)
        probs = probs ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False, p=probs)
        return [list(self.buffer)[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class EpisodeRecorder:
    def __init__(self, directory: str | Path = "data/captures") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames: List[np.ndarray] = []
        self.actions: List[int] = []
        self.params: List[np.ndarray] = []
        self.rewards: List[float] = []

    def record(self, image: np.ndarray, action_id: int, params: np.ndarray, reward: float) -> None:
        self.frames.append(np.array(image, copy=True))
        self.actions.append(int(action_id))
        self.params.append(np.array(params, copy=True))
        self.rewards.append(float(reward))

    def save(self, name: str) -> Path:
        path = self.directory / f"{name}.npz"
        np.savez_compressed(
            path,
            frames=np.stack(self.frames, axis=0),
            actions=np.array(self.actions, dtype=np.int64),
            params=np.stack(self.params, axis=0),
            rewards=np.array(self.rewards, dtype=np.float32),
        )
        self.clear()
        return path

    def clear(self) -> None:
        self.frames.clear()
        self.actions.clear()
        self.params.clear()
        self.rewards.clear()


__all__ = ["ReplayBuffer", "Transition", "EpisodeRecorder"]
