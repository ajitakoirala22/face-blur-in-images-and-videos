from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from blur_utils import blur_face, enhance_frame
from face_detector import Detection, FaceDetector
from temporal_filter import TemporalFilter


def merge_detections_nms(detections: List[Detection], iou_thresh: float = 0.4) -> List[Detection]:
    if not detections:
        return []
    boxes = []
    scores = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        boxes.append([x1, y1, w, h])
        scores.append(float(det.conf))

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_thresh)
    if indices is None or len(indices) == 0:
        return detections

    keep = []
    flat = np.array(indices).reshape(-1).tolist()
    for idx in flat:
        keep.append(detections[idx])
    return keep


def build_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    codecs = ("avc1", "H264", "mp4v")
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise ValueError("Could not initialize video writer with supported codecs")


def process_image(
    input_path: str,
    output_path: str,
    detector: FaceDetector,
    conf_thresh: float = 0.2,
    enhance: bool = False,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> str:
    if progress_cb:
        progress_cb(5, "Reading image")

    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError("Could not read image")

    if progress_cb:
        progress_cb(35, "Detecting faces")

    detections = detector.detect(frame, conf_thresh=conf_thresh)
    if enhance:
        enhanced = enhance_frame(frame)
        detections += detector.detect(enhanced, conf_thresh=max(0.15, conf_thresh - 0.05))

    detections = merge_detections_nms(detections, iou_thresh=0.4)

    if progress_cb:
        progress_cb(70, "Blurring faces")

    for det in detections:
        frame = blur_face(frame, det.bbox)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame)

    if progress_cb:
        progress_cb(100, "Completed")

    return output_path


def process_video(
    input_path: str,
    output_path: str,
    detector: FaceDetector,
    conf_thresh: float = 0.2,
    enhance: bool = False,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> str:
    if progress_cb:
        progress_cb(3, "Opening video")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = width - (width % 2)
    out_height = height - (height % 2)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = build_video_writer(output_path, fps, out_width, out_height)

    temporal = TemporalFilter(iou_thresh=0.3, min_hits_mid=2, max_misses=3, grace=2)
    high_conf = max(conf_thresh, 0.4)
    mid_conf = min(conf_thresh, 0.2)
    processed_frames = 0

    if progress_cb:
        progress_cb(8, "Processing frames")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[1] != out_width or frame.shape[0] != out_height:
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

        proc_frame = enhance_frame(frame) if enhance else frame
        detections = detector.detect(proc_frame, conf_thresh=mid_conf)
        detections = merge_detections_nms(detections, iou_thresh=0.4)
        accepted_boxes = temporal.update(detections, high_conf=high_conf, mid_conf=mid_conf)

        for bbox in accepted_boxes:
            frame = blur_face(frame, bbox)

        writer.write(frame)
        processed_frames += 1

        if progress_cb and total_frames > 0:
            pct = 8 + int((processed_frames / total_frames) * 88)
            pct = min(96, max(8, pct))
            progress_cb(pct, f"Processing frames ({processed_frames}/{total_frames})")

    cap.release()
    writer.release()

    if progress_cb:
        progress_cb(100, "Completed")

    return output_path
