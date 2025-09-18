import numpy as np
import cv2

from agent.perception.template_store import TemplateStore
from agent.perception.locator import TemplateLocator


def test_template_matching(tmp_path):
    template_img = np.zeros((20, 20, 3), dtype=np.uint8)
    template_img[5:15, 5:15] = 255
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[30:40, 40:50] = 255
    template_path = tmp_path / "template.png"
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(template_path), template_img)
    cv2.imwrite(str(frame_path), frame)
    store = TemplateStore(root=tmp_path)
    store.register("square", template_path, threshold=0.8)
    locator = TemplateLocator(store)
    matches = locator.match(cv2.imread(str(frame_path)))
    assert matches, "Template should be detected in synthetic frame"
    match = matches[0]
    assert match.name == "square"
    assert match.score >= 0.8
