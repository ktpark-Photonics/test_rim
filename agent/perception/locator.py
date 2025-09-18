"""Template matching helpers used to ground UI elements."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

try:  # pragma: no cover - optional heavy dependency during tests
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .template_store import Template, TemplateStore


@dataclass
class TemplateMatch:
    name: str
    score: float
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]
    scale: float

    def to_bbox(self) -> tuple[int, int, int, int]:
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        return x1, y1, x2, y2


class TemplateLocator:
    def __init__(self, store: TemplateStore, nms_threshold: float = 0.3) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for template matching")
        self.store = store
        self.nms_threshold = nms_threshold

    def match(
        self,
        frame: np.ndarray,
        template_names: Sequence[str] | None = None,
        scales: Sequence[float] = (1.0,),
    ) -> List[TemplateMatch]:
        matches: List[TemplateMatch] = []
        search_space: Iterable[Template]
        if template_names is None:
            search_space = list(self.store)
        else:
            search_space = [self.store.get(name) for name in template_names]

        gray_frame = self._prepare_frame(frame)

        for template in search_space:
            template_img = self._prepare_template(template.image)
            for scale in scales:
                scaled = self._scale_template(template_img, scale)
                res = cv2.matchTemplate(gray_frame, scaled, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= template.threshold)
                for x, y in zip(xs, ys):
                    h, w = scaled.shape[:2]
                    score = float(res[y, x])
                    matches.append(
                        TemplateMatch(
                            name=template.name,
                            score=score,
                            top_left=(int(x), int(y)),
                            bottom_right=(int(x + w), int(y + h)),
                            scale=float(scale),
                        )
                    )
        return self._non_maximum_suppression(matches)

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _scale_template(self, template: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-6:
            return template
        h, w = template.shape[:2]
        new_size = (max(int(w * scale), 1), max(int(h * scale), 1))
        return cv2.resize(template, new_size, interpolation=cv2.INTER_LINEAR)

    def _prepare_template(self, template: np.ndarray) -> np.ndarray:
        if template.ndim == 2:
            return template
        return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    def _non_maximum_suppression(self, matches: List[TemplateMatch]) -> List[TemplateMatch]:
        if not matches:
            return []
        matches = sorted(matches, key=lambda m: m.score, reverse=True)
        keep: List[TemplateMatch] = []
        for candidate in matches:
            if all(self._iou(candidate, existing) <= self.nms_threshold for existing in keep):
                keep.append(candidate)
        return keep

    def _iou(self, a: TemplateMatch, b: TemplateMatch) -> float:
        ax1, ay1, ax2, ay2 = a.to_bbox()
        bx1, by1, bx2, by2 = b.to_bbox()
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union


__all__ = ["TemplateMatch", "TemplateLocator"]
