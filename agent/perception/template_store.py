"""Template storage and management utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

try:  # pragma: no cover - optional heavy dependency during tests
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass
class Template:
    name: str
    image: np.ndarray
    threshold: float = 0.85
    metadata: dict | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]


class TemplateStore:
    """Keeps track of template images used by the perception stack."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else None
        self._templates: Dict[str, Template] = {}

    def register(self, name: str, path: str | Path, threshold: float = 0.85, metadata: Optional[dict] = None) -> Template:
        file_path = Path(path)
        if not file_path.is_absolute() and self.root is not None:
            file_path = self.root / file_path
        image = self._load_image(file_path)
        template = Template(name=name, image=image, threshold=threshold, metadata=metadata)
        self._templates[name] = template
        return template

    def _load_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        if cv2 is None:
            raise RuntimeError("OpenCV is required to load templates")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to load template: {path}")
        return image

    def get(self, name: str) -> Template:
        return self._templates[name]

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __iter__(self) -> Iterable[Template]:
        return iter(self._templates.values())

    def items(self):  # pragma: no cover - trivial wrapper
        return self._templates.items()


__all__ = ["Template", "TemplateStore"]
