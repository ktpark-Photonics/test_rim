"""OCR helper utilities for HUD text extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency during tests
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore


@dataclass
class OcrResult:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


class OcrEngine:
    def __init__(self, language: str = "eng+kor", psm: int = 6) -> None:
        self.language = language
        self.psm = psm
        if pytesseract is None:
            raise RuntimeError("pytesseract is required for OCR operations")

    def read_text(self, image: np.ndarray) -> List[OcrResult]:
        config = f"--psm {self.psm}"
        data = pytesseract.image_to_data(image, lang=self.language, config=config, output_type=pytesseract.Output.DICT)
        results: List[OcrResult] = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue
            conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            results.append(OcrResult(text=text, confidence=conf, bbox=(x, y, x + w, y + h)))
        return results


__all__ = ["OcrEngine", "OcrResult"]
