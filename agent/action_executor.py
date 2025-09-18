"""Input execution helpers using the Windows SendInput API (with safe fallbacks)."""
from __future__ import annotations

import platform
import random
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from loguru import logger

from .window_manager import WindowManager

try:  # pragma: no cover - Windows specific
    import ctypes
    import ctypes.wintypes as wintypes
except Exception:  # pragma: no cover
    ctypes = None  # type: ignore
    wintypes = None  # type: ignore


@dataclass
class HumanizationConfig:
    jitter_px: int = 3
    base_delay: float = 0.04
    delay_jitter: float = 0.02


class ActionExecutor:
    """Executes high level macro actions as mouse/keyboard input."""

    def __init__(
        self,
        window_manager: WindowManager,
        humanization: HumanizationConfig | None = None,
        enable: bool = True,
    ) -> None:
        self.window_manager = window_manager
        self.humanization = humanization or HumanizationConfig()
        self.enable = enable
        self._warned_platform = False

    # ------------------------------------------------------------------ helpers
    def _check_environment(self) -> bool:
        if not self.enable:
            return False
        if not self.window_manager.has_focus():
            logger.debug("Input suppressed because target window is not focused")
            return False
        if platform.system() != "Windows":
            if not self._warned_platform:
                logger.warning("SendInput is not available on non-Windows systems; input is mocked")
                self._warned_platform = True
            return False
        if ctypes is None:
            logger.warning("ctypes is unavailable; cannot inject input")
            return False
        return True

    def _sleep_with_jitter(self) -> None:
        delay = random.gauss(self.humanization.base_delay, self.humanization.delay_jitter)
        time.sleep(max(delay, 0.0))

    # ------------------------------------------------------------------ mouse API
    def move_mouse(self, x: int, y: int, absolute: bool = True) -> None:
        if not self._check_environment():
            logger.debug("Mock move_mouse({}, {})", x, y)
            return
        # Actual SendInput code would go here; we keep it minimal for safety.
        logger.debug("move_mouse({}, {}) placeholder", x, y)

    def click(self, button: str = "left", count: int = 1) -> None:
        for _ in range(count):
            if not self._check_environment():
                logger.debug("Mock click({})", button)
            else:
                logger.debug("click({}) placeholder", button)
            self._sleep_with_jitter()

    def key_press(self, key: str, modifiers: Iterable[str] | None = None) -> None:
        modifiers = list(modifiers or [])
        if not self._check_environment():
            logger.debug("Mock key_press({}, modifiers={})", key, modifiers)
            return
        logger.debug("key_press({}, modifiers={}) placeholder", key, modifiers)

    # ------------------------------------------------------------------ high level
    def execute_macro(self, name: str, params: Optional[list[float]] = None) -> None:
        params = params or []
        logger.info("Executing macro '%s' params=%s", name, params)
        if name.startswith("SET_SPEED"):
            speed = name.split("_")[-1]
            self.key_press(speed)
        elif name == "TOGGLE_PAUSE":
            self.key_press("space")
        elif name == "OPEN_ARCHITECT":
            self.key_press("F1")
        elif name == "OPEN_ZONE":
            self.key_press("F3")
        elif name.startswith("SELECT_COLONIST"):
            self.key_press("1")
        elif name == "CREATE_STOCKPILE" and params:
            self._draw_drag(params)
        else:
            logger.debug("Unhandled macro '{}', falling back to mock", name)

    def _draw_drag(self, params: list[float]) -> None:
        if len(params) < 4:
            logger.warning("CREATE_STOCKPILE requires 4 parameters (x1,y1,x2,y2)")
            return
        self.move_mouse(int(params[0]), int(params[1]))
        self.click("left")
        self.move_mouse(int(params[2]), int(params[3]))
        self.click("left")


__all__ = ["ActionExecutor", "HumanizationConfig"]
