"""High level orchestrator for the RimWorld agent runtime."""
from __future__ import annotations

import signal
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .action_executor import ActionExecutor
from .config import ConfigError, load_config
from .demo_recorder import DemoRecorder
from .logger import logger, setup_logging
from .overlay import HudDashboard
from .screen_capture import CaptureConfig, FrameStacker, ScreenCapture
from .window_manager import WindowManager


class AgentApp:
    def __init__(self, config_path: str | Path = "configs/agent.yaml") -> None:
        try:
            cfg_bundle = load_config(Path(config_path))
        except ConfigError as exc:  # pragma: no cover - configuration missing
            logger.error("%s", exc)
            raise
        self.cfg = cfg_bundle.data
        setup_logging(self.cfg.get("log_level", "INFO"))
        self.window_manager = WindowManager()
        capture_cfg = self.cfg.get("capture", {})
        self.capture = ScreenCapture(
            self.window_manager,
            CaptureConfig(
                width=capture_cfg.get("width", 320),
                height=capture_cfg.get("height", 180),
                frame_stack=capture_cfg.get("frame_stack", 4),
                grayscale=capture_cfg.get("grayscale", False),
            ),
        )
        self.frame_stacker = FrameStacker(
            self.capture.config.frame_stack,
            (self.capture.config.height, self.capture.config.width, 1 if self.capture.config.grayscale else 3),
        )
        self.executor = ActionExecutor(self.window_manager)
        self.recorder = DemoRecorder()
        hud_cfg = self.cfg.get("hud", {})
        self.hud = HudDashboard(enable_gui=hud_cfg.get("enable_gui", False), recorder=self.recorder)
        self._stop_event = threading.Event()
        self._register_hotkeys()

    # ------------------------------------------------------------------ lifecycle
    def _register_hotkeys(self) -> None:
        logger.info("Global hotkeys are not available in this environment; use CLI controls instead.")

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - long running loop
        logger.info("Agent runtime started. Press Ctrl+C to exit.")
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        self.window_manager.select_window(self.cfg.get("window_keyword", "RimWorld"))
        self.capture.set_window(self.window_manager.handle)
        while not self._stop_event.is_set():
            frame = self.capture.capture_with_rate_limit(self.cfg.get("loop_fps", 10))
            stacked = self.frame_stacker.push(frame)
            obs = {"image": stacked, "scalars": np.zeros(8, dtype=np.float32)}
            action = {"id": 0, "params": np.zeros(4, dtype=np.float32)}
            self.hud.set_status("idle")
            self.hud.record_step(obs, action, info=None)
            time.sleep(0.01)
        logger.info("Agent runtime stopped")


__all__ = ["AgentApp"]
