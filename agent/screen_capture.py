"""Screen capture utilities backed by :mod:`mss` with graceful fallbacks."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import numpy as np

try:  # pragma: no cover - optional dependency during tests
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional on non-Windows
    import mss
except Exception:  # pragma: no cover
    mss = None  # type: ignore

from .window_manager import WindowManager


FrameProvider = Callable[[], np.ndarray]


@dataclass
class CaptureConfig:
    width: int
    height: int
    frame_stack: int = 4
    grayscale: bool = False


class FrameStacker:
    """Maintains a rolling stack of the most recent frames."""

    def __init__(self, capacity: int, frame_shape: tuple[int, int, int]) -> None:
        self.capacity = capacity
        self.buffer: Deque[np.ndarray] = deque(maxlen=capacity)
        self.frame_shape = frame_shape

    def reset(self) -> None:
        self.buffer.clear()

    def push(self, frame: np.ndarray) -> np.ndarray:
        if len(self.buffer) == 0 and self.capacity > 1:
            for _ in range(self.capacity - 1):
                self.buffer.append(np.zeros_like(frame))
        self.buffer.append(frame)
        return self.stack

    @property
    def stack(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros((self.capacity, *self.frame_shape), dtype=np.uint8)
        frames = list(self.buffer)
        if len(frames) < self.capacity:
            pad = [np.zeros_like(frames[0]) for _ in range(self.capacity - len(frames))]
            frames = pad + frames
        return np.stack(frames, axis=0)


class ScreenCapture:
    """Captures frames from the RimWorld window using :mod:`mss`."""

    def __init__(
        self,
        window_manager: WindowManager,
        config: CaptureConfig,
        frame_provider: Optional[FrameProvider] = None,
    ) -> None:
        self.window_manager = window_manager
        self.config = config
        self.frame_provider = frame_provider
        self._mss: Optional[mss.mss] = None if mss is None else mss.mss()
        self._monitor: Optional[dict] = None
        self._last_timestamp: float = 0.0

    def set_window(self, handle: int | None) -> None:
        if handle is None or mss is None:
            self._monitor = None
            return
        try:
            mon = mss.windows.get_window_rect(handle)
        except Exception:  # pragma: no cover - Windows specific
            mon = None
        if mon is None:
            self._monitor = None
        else:
            left, top, right, bottom = mon
            self._monitor = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            }

    def capture(self) -> np.ndarray:
        if self.frame_provider is not None:
            frame = self.frame_provider()
        elif self._mss is not None and self._monitor is not None:
            raw = np.asarray(self._mss.grab(self._monitor))
            frame = raw[:, :, :3]
        else:
            frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)

        if cv2 is not None and frame.shape[:2] != (self.config.height, self.config.width):
            interpolation = cv2.INTER_AREA if frame.shape[0] > self.config.height else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (self.config.width, self.config.height), interpolation=interpolation)
        else:
            frame = np.array(frame, copy=False)
            frame = frame[: self.config.height, : self.config.width, :3]

        if self.config.grayscale:
            if cv2 is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = np.mean(frame, axis=2, keepdims=True).astype(np.uint8)
        return frame

    def capture_with_rate_limit(self, fps: float) -> np.ndarray:
        now = time.perf_counter()
        min_interval = 1.0 / max(fps, 1e-6)
        if self._last_timestamp:
            sleep_time = self._last_timestamp + min_interval - now
            if sleep_time > 0:
                time.sleep(sleep_time)
        frame = self.capture()
        self._last_timestamp = time.perf_counter()
        return frame


__all__ = ["ScreenCapture", "CaptureConfig", "FrameStacker"]
