"""Re-split detection labels by VIDEO, not by frame (§9.2 — run before training).

The #1 dataset risk: if train/val was split randomly over frames, val is
near-duplicate frames of train and every mAP is fiction. This holds out entire
clips (e.g. one corridor, one gate, one baggage cam) so no frame from a val clip
ever appears in train.

Operates on the labeled dataset once it exists (data/labels/ + data/frames/).
It is a NO-OP today because there are no hand labels yet — the ingestion pipeline
(sample_frames.py -> CVAT labeling) has to produce them first. Kept as the
canonical splitter so it's ready the moment labels land.

    python scripts/resplit_by_video.py --val-cams cam02 cam08 cam11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / "data" / "frames"
LABELS_DIR = ROOT / "data" / "labels"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-cams", nargs="+", required=False,
                    help="cam ids to hold out entirely as validation")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "split")
    ap.parse_args()

    if not LABELS_DIR.exists() or not any(LABELS_DIR.rglob("*.txt")):
        sys.exit(
            "No labels found in data/labels/. This script splits an EXISTING labeled\n"
            "set by video. First: sample_frames.py -> label in CVAT -> export YOLO txt,\n"
            "then re-run with --val-cams to hold out whole clips (§9.2)."
        )
    raise NotImplementedError(
        "Phase 1: write train/val frame lists partitioned by cam id, assert zero "
        "cross-split clip leakage, emit an Ultralytics data.yaml."
    )


if __name__ == "__main__":
    main()
