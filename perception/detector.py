"""YOLO11 detector wrapper — .pt in dev, TensorRT .engine in deployment (§5.1).

Phase 1. Keeps COCO class indexing (person=0, backpack=24, handbag=26,
suitcase=28) so pretrained heads transfer. imgsz defaults to 1280 for small
distant bags on high-res CCTV.
"""

from __future__ import annotations

from common.config import COCO_CLASSES


class Detector:
    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.25):
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.keep_ids = sorted(COCO_CLASSES)
        self._model = None  # lazy-load ultralytics.YOLO

    def __call__(self, frame, watch_rois=None):  # noqa: ARG002
        """Detect. Inside watch_rois, accept candidates down to conf 0.15 (§6.7)."""
        raise NotImplementedError("Phase 1: load ultralytics.YOLO and run inference")
