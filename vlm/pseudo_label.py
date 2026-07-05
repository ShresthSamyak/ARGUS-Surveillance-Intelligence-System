"""LocateAnything teacher — batch pseudo-labeling + temporal filter + hard-example
miner (§5.2, §5.7 P5).

Batch offline: prompt the VLM per frame ("person"/"backpack"/"handbag"/"suitcase"
+ "bag left on the floor"), class-aware NMS (IoU 0.6). Temporal consistency
filter: keep a box only if a matching box (same class, IoU>0.5) appears in >=2 of
3 consecutive sampled frames — kills single-frame hallucinations. Human audit gate:
proceed only if per-class precision > 90%.

v0.2 makes the teacher permanent: every live fast/slow disagreement is auto-cropped
into data/hard_examples/ with provenance — a self-mining curriculum. Retrain gate:
>= 500 new corrected examples AND the new checkpoint beats the old on the frozen
video-held-out val.

Phase 4 (distillation ablation) — interface stubbed now so the queue plumbs through.
"""

from __future__ import annotations

PROMPTS = ("person", "backpack", "handbag", "suitcase", "bag left on the floor")


def pseudo_label_batch(provider, frames):  # noqa: ARG001
    raise NotImplementedError("Phase 4")


def temporal_consistency_filter(per_frame_boxes, iou_thresh: float = 0.5, k: int = 2):  # noqa: ARG001
    raise NotImplementedError("Phase 4")
