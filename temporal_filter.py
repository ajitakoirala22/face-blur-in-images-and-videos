from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from face_detector import Detection


BBox = Tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class Track:
    bbox: BBox
    hits: int = 1
    misses: int = 0
    grace_left: int = 0


class TemporalFilter:
    def __init__(self, iou_thresh: float = 0.3, min_hits_mid: int = 2, max_misses: int = 3, grace: int = 2):
        self.iou_thresh = iou_thresh
        self.min_hits_mid = min_hits_mid
        self.max_misses = max_misses
        self.grace = grace
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: List[Detection], high_conf: float, mid_conf: float) -> List[BBox]:
        accepted: List[BBox] = []
        matched_track_ids = set()

        for det in detections:
            best_track_id = None
            best_iou = 0.0
            for tid, track in self.tracks.items():
                score = iou(det.bbox, track.bbox)
                if score > best_iou and score >= self.iou_thresh:
                    best_iou = score
                    best_track_id = tid

            if best_track_id is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(bbox=det.bbox)
            else:
                tid = best_track_id
                tr = self.tracks[tid]
                tr.bbox = det.bbox
                tr.hits += 1
                tr.misses = 0
                tr.grace_left = self.grace

            matched_track_ids.add(tid)
            track = self.tracks[tid]

            if det.conf >= high_conf:
                accepted.append(det.bbox)
            elif det.conf >= mid_conf and track.hits >= self.min_hits_mid:
                accepted.append(det.bbox)

        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                track = self.tracks[tid]
                track.misses += 1
                if track.grace_left > 0:
                    accepted.append(track.bbox)
                    track.grace_left -= 1
                if track.misses > self.max_misses:
                    del self.tracks[tid]

        return accepted
