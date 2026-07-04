"""Dual-rate MOG2 static-object redundancy channel (§5.5). Phase 1/2.

Two background models: B_fast (absorbs a static object in ~5 s) and B_slow
(~3 min). A newly-static candidate is foreground in B_slow AND background in
B_fast, area 300-20000 px^2, persisting > 10 s. Candidates with no overlapping
YOLO bag detection are queued for a slow-path VLM crop check (§5.7 P-roles).

Inside watch-ROIs this runs every frame instead of every 3rd (§6.7).
"""

from __future__ import annotations


class DualBackgroundChannel:
    def __init__(self, fast_lr: float = 0.02, slow_lr: float = 0.0002,
                 min_area: int = 300, max_area: int = 20000, persist_s: float = 10.0):
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.min_area = min_area
        self.max_area = max_area
        self.persist_s = persist_s

    def update(self, frame, rois=None):  # noqa: ARG002
        """Return newly-static blob candidates [{bbox, area, age_s}]."""
        raise NotImplementedError("Phase 1: two MOG2 models + disagreement blobs")
