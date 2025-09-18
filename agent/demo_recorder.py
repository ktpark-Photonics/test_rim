"""Utilities for recording imitation learning demonstrations."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .logger import logger


@dataclass
class DemoStep:
    timestamp: float
    image: np.ndarray
    scalars: np.ndarray
    action_id: int
    action_params: np.ndarray


@dataclass
class DemoSession:
    task_name: str
    meta: Dict[str, str] = field(default_factory=dict)
    steps: List[DemoStep] = field(default_factory=list)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        frames = np.stack([step.image for step in self.steps], axis=0)
        scalars = np.stack([step.scalars for step in self.steps], axis=0)
        actions = np.array([step.action_id for step in self.steps], dtype=np.int64)
        params = np.stack([step.action_params for step in self.steps], axis=0)
        timestamps = np.array([step.timestamp for step in self.steps], dtype=np.float64)
        return {
            "frames": frames,
            "scalars": scalars,
            "actions": actions,
            "params": params,
            "timestamps": timestamps,
        }


class DemoRecorder:
    def __init__(self, output_dir: str | Path = "data/demos") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[DemoSession] = None

    @property
    def is_recording(self) -> bool:
        return self._session is not None

    def start(self, task_name: str, meta: Optional[Dict[str, str]] = None) -> None:
        if self.is_recording:
            logger.warning("DemoRecorder already active; restarting session")
        self._session = DemoSession(task_name=task_name, meta=dict(meta or {}))
        logger.info("Demo recording started for task '{}'.", task_name)

    def stop(self) -> Optional[Path]:
        if not self.is_recording:
            logger.info("DemoRecorder stop called while inactive")
            return None
        session = self._session
        self._session = None
        if not session.steps:
            logger.warning("Skipping empty demonstration for task '{}'.", session.task_name)
            return None
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = self.output_dir / f"demo-{session.task_name}-{timestamp}.npz"
        arrays = session.to_arrays()
        np.savez_compressed(out_path, **arrays)
        meta_path = out_path.with_suffix(".json")
        meta = dict(session.meta)
        meta.update({"task": session.task_name, "num_steps": len(session.steps), "created_at": timestamp})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Demo saved to %s", out_path)
        return out_path

    def toggle(self, task_name: str, meta: Optional[Dict[str, str]] = None) -> Optional[Path]:
        if self.is_recording:
            return self.stop()
        self.start(task_name, meta)
        return None

    def record_step(self, observation: Dict[str, np.ndarray], action: Dict[str, np.ndarray | int], info: Optional[dict] = None) -> None:
        if not self.is_recording:
            return
        session = self._session
        assert session is not None
        image = observation.get("image")
        scalars = observation.get("scalars")
        if image is None or scalars is None:
            raise ValueError("Observation must contain 'image' and 'scalars' entries")
        image = np.asarray(image)
        scalars = np.asarray(scalars)
        action_id = int(action.get("id", 0))
        params = np.asarray(action.get("params", np.zeros(1, dtype=np.float32)))
        if params.ndim == 0:
            params = params.reshape(1)
        timestamp = float(action.get("timestamp", time.time()))
        session.steps.append(DemoStep(timestamp=timestamp, image=image, scalars=scalars, action_id=action_id, action_params=params))
        logger.debug("Recorded demo step %d for task '%s'", len(session.steps), session.task_name)


__all__ = ["DemoRecorder"]
