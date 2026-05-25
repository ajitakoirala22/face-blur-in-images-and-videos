from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def blur_face(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    face_w = max(1, x2 - x1)
    k = max(15, (face_w // 7) | 1)
    blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=0, sigmaY=0)
    frame[y1:y2, x1:x2] = blurred
    return frame


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
