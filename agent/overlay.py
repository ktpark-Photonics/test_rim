"""On-screen debugging HUD and lightweight dashboard utilities."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from loguru import logger

from .demo_recorder import DemoRecorder

try:  # pragma: no cover - GUI optional during tests
    from PySide6 import QtCore, QtWidgets
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    QtCore = None  # type: ignore
    QtWidgets = None  # type: ignore
    pg = None  # type: ignore


class HudDashboard:
    """Displays key runtime metrics and recording controls."""

    def __init__(self, enable_gui: bool = False, recorder: Optional[DemoRecorder] = None) -> None:
        self.enable_gui = enable_gui and QtWidgets is not None
        self.recorder = recorder or DemoRecorder()
        self.current_task: str = ""
        self.status_text: str = ""
        self.event_log: Deque[Tuple[float, str]] = deque(maxlen=50)
        self.scalar_history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=600))
        self._thread: Optional[threading.Thread] = None
        self._app: Optional[QtWidgets.QApplication] = None if QtWidgets else None
        self._window: Optional[QtWidgets.QWidget] = None if QtWidgets else None
        self._status_label = None
        self._task_label = None
        self._event_widget = None
        self._plot_widget = None
        self._plots: Dict[str, pg.PlotWidget] = {} if pg else {}
        if self.enable_gui:
            self._start_gui()

    # ------------------------------------------------------------------ GUI layer
    def _start_gui(self) -> None:
        assert QtWidgets is not None and pg is not None
        if QtWidgets.QApplication.instance():
            self._app = QtWidgets.QApplication.instance()
        else:
            self._app = QtWidgets.QApplication([])
        self._window = QtWidgets.QWidget()
        self._window.setWindowTitle("RimWorld Agent HUD")
        layout = QtWidgets.QVBoxLayout(self._window)
        self._status_label = QtWidgets.QLabel("Status: idle")
        layout.addWidget(self._status_label)
        self._task_label = QtWidgets.QLabel("Task: -")
        layout.addWidget(self._task_label)
        self._event_widget = QtWidgets.QListWidget()
        layout.addWidget(self._event_widget)
        self._plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self._plot_widget)
        self._window.resize(400, 600)
        self._window.show()

        def _run():  # pragma: no cover - GUI thread not exercised in tests
            while True:
                self._app.processEvents()
                time.sleep(0.05)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ data API
    def set_status(self, text: str) -> None:
        self.status_text = text
        if self.enable_gui and self._status_label is not None:
            self._status_label.setText(f"Status: {text}")

    def set_current_task(self, name: str) -> None:
        self.current_task = name
        if self.enable_gui and self._task_label is not None:
            self._task_label.setText(f"Task: {name}")

    def log_event(self, text: str) -> None:
        timestamp = time.time()
        logger.info("HUD event: %s", text)
        self.event_log.append((timestamp, text))
        if self.enable_gui and self._event_widget is not None:
            self._event_widget.insertItem(0, text)

    def plot_scalar(self, name: str, value: float) -> None:
        timestamp = time.time()
        self.scalar_history[name].append((timestamp, float(value)))
        if self.enable_gui and self._plot_widget is not None:
            plot = self._plots.get(name)
            if plot is None:
                row = len(self._plots)
                plot = self._plot_widget.addPlot(row=row, col=0, title=name)
                plot.showGrid(x=True, y=True, alpha=0.3)
                self._plots[name] = plot
            xs, ys = zip(*self.scalar_history[name])
            plot.clear()
            plot.plot(xs, ys, pen=pg.mkPen("#7fdbff"))

    def get_recent_rewards(self, window_seconds: float = 60.0) -> List[Tuple[float, float]]:
        now = time.time()
        data = [pair for pair in self.scalar_history.get("reward", []) if now - pair[0] <= window_seconds]
        return data

    # ------------------------------------------------------------------ recording
    def toggle_recording(self, task_name: str, meta: Optional[dict] = None) -> Optional[Path]:
        path = self.recorder.toggle(task_name, meta)
        state = "ON" if self.recorder.is_recording else "OFF"
        self.log_event(f"Recording toggled {state} for task {task_name}")
        return path

    def record_step(self, observation: Dict[str, float], action: Dict[str, float], info: Optional[dict] = None) -> None:
        self.recorder.record_step(observation, action, info)


__all__ = ["HudDashboard"]
