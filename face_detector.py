from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    bbox: BBox
    conf: float


class FaceDetector:
    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self._retina_model = self._load_retinaface_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
        device = (device or "auto").lower()
        if device not in {"auto", "cpu", "cuda"}:
            return "cpu"
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_retinaface_model(self):
        try:
            from facexlib.detection import init_detection_model

            model = init_detection_model("retinaface_resnet50", device=self.device)
            model.eval()
            return model
        except Exception as exc:
            raise RuntimeError(
                "RetinaFace dependencies missing. Run: pip install -r requirements.txt"
            ) from exc

    def detect(self, image: np.ndarray, conf_thresh: float = 0.35) -> List[Detection]:
        detections: List[Detection] = []

        with torch.no_grad():
            response = self._retina_model.detect_faces(image, conf_thresh)

        if response is None or len(response) == 0:
            return detections

        for row in response:
            if len(row) < 5:
                continue
            x1, y1, x2, y2, conf = row[:5]
            conf = float(conf)
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection((x1, y1, x2, y2), conf))
        return detections
