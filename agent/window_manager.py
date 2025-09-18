"""Window management utilities for targeting the RimWorld client window."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

try:  # pragma: no cover - optional Windows dependency
    import win32con
    import win32gui
except Exception:  # pragma: no cover - running on non-Windows platforms
    win32con = None  # type: ignore
    win32gui = None  # type: ignore


@dataclass
class WindowInfo:
    handle: int
    title: str
    rect: tuple[int, int, int, int]


class WindowManager:
    """Finds and validates the RimWorld window.

    The implementation degrades gracefully on non-Windows platforms so the rest
    of the stack can be unit tested without direct OS interaction.
    """

    def __init__(self) -> None:
        self._handle: Optional[int] = None

    @property
    def handle(self) -> Optional[int]:
        return self._handle

    def list_windows(self) -> List[WindowInfo]:
        if win32gui is None:
            return []

        windows: List[WindowInfo] = []

        def _callback(handle: int, _: int) -> None:
            title = win32gui.GetWindowText(handle)
            if title and win32gui.IsWindowVisible(handle):
                rect = win32gui.GetClientRect(handle)
                windows.append(WindowInfo(handle=handle, title=title, rect=rect))

        win32gui.EnumWindows(_callback, 0)
        return windows

    def select_window(self, title_keyword: str | None = None, handle: int | None = None) -> Optional[WindowInfo]:
        if handle is not None:
            self._handle = handle
            logger.info("Window handle selected manually: {}", handle)
            return WindowInfo(handle=handle, title="manual", rect=(0, 0, 0, 0))

        if title_keyword is None:
            title_keyword = "RimWorld"

        for info in self.list_windows():
            if title_keyword.lower() in info.title.lower():
                self._handle = info.handle
                logger.info("Selected window '{}' (handle={})", info.title, info.handle)
                return info

        logger.warning("Window with keyword '{}' not found", title_keyword)
        self._handle = None
        return None

    def has_focus(self) -> bool:
        if win32gui is None:
            return True
        if self._handle is None:
            return False
        return win32gui.GetForegroundWindow() == self._handle

    def bring_to_front(self) -> bool:
        if win32gui is None or win32con is None:
            logger.debug("bring_to_front is a no-op on {}", platform.system())
            return True
        if self._handle is None:
            return False
        try:
            win32gui.ShowWindow(self._handle, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self._handle)
            return True
        except Exception as exc:  # pragma: no cover - OS level failure
            logger.error("Failed to bring window to front: {}", exc)
            return False


__all__ = ["WindowManager", "WindowInfo"]
