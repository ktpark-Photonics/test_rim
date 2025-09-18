"""Definitions of key RimWorld UI regions using normalized coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class UiRegion:
    name: str
    top_left: Tuple[float, float]
    bottom_right: Tuple[float, float]

    def as_pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = int(self.top_left[0] * width)
        y1 = int(self.top_left[1] * height)
        x2 = int(self.bottom_right[0] * width)
        y2 = int(self.bottom_right[1] * height)
        return x1, y1, x2, y2


DEFAULT_UI_MAP: Dict[str, UiRegion] = {
    "architect_button": UiRegion("architect_button", (0.92, 0.06), (0.98, 0.12)),
    "zone_button": UiRegion("zone_button", (0.92, 0.14), (0.98, 0.21)),
    "speed_controls": UiRegion("speed_controls", (0.45, 0.94), (0.55, 0.99)),
    "colonist_bar": UiRegion("colonist_bar", (0.05, 0.88), (0.95, 0.97)),
    "status_panel": UiRegion("status_panel", (0.70, 0.70), (0.98, 0.96)),
}


__all__ = ["UiRegion", "DEFAULT_UI_MAP"]
