"""Zero-shot class census — count person/bag instances per camera (§9.2).

Runs stock COCO-pretrained YOLO11-s over the sampled frames and tallies
detections per class per camera. This is NOT ground truth — it is the cheap
pre-labeling reality check the spec demands: "Handbag/suitcase will be scarce;
you cannot fix what you haven't measured." If suitcase comes back near zero,
you learn it before investing days in CVAT, not after.

Doubles as the first look at the zero-shot detector the fine-tune must beat.

    python scripts/class_census.py                 # over data/frames/
    python scripts/class_census.py --conf 0.25 --imgsz 1280
    python scripts/class_census.py --from-video --stride 25   # sample clips directly

Requires the [perception] extra (ultralytics). Weights auto-download on first run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

# Reduce fragmentation on the 8 GB card; set before torch/ultralytics import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import COCO_CLASSES  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
FRAMES_DIR = ROOT / "data" / "frames"
KEEP_IDS = sorted(COCO_CLASSES)  # [0, 24, 26, 28]


def iter_frame_groups(cams: list[str] | None):
    """Yield (cam_id, [image_paths]) for each camera's sampled frames."""
    for cam_dir in sorted(p for p in FRAMES_DIR.iterdir() if p.is_dir()):
        if cams and cam_dir.name not in cams:
            continue
        imgs = sorted(cam_dir.glob("*.jpg"))
        if imgs:
            yield cam_dir.name, imgs


def predict_batched(model, items, conf, imgsz, batch):
    """Yield Results for `items` (paths or frames) in OOM-safe chunks.

    The original code passed a whole camera's image list (up to 209) to predict in
    one call — ultralytics batches a list into a single forward, which OOMs an 8 GB
    card at imgsz 1280. Here we chunk to `batch`; on CUDA OOM we empty the cache and
    halve the batch (down to 1) and retry the same chunk.
    """
    import torch

    i, bs, n = 0, max(1, batch), len(items)
    while i < n:
        chunk = items[i:i + bs]
        try:
            for r in model.predict(source=chunk, conf=conf, imgsz=imgsz,
                                   classes=KEEP_IDS, verbose=False):
                yield r
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [oom] batch too big, retrying at batch={bs}")


def census_frames(model, cams, conf, imgsz, batch) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cam_id, imgs in iter_frame_groups(cams):
        paths = [str(p) for p in imgs]
        for r in predict_batched(model, paths, conf, imgsz, batch):
            counts[cam_id]["_frames"] += 1
            for c in r.boxes.cls.tolist():
                counts[cam_id][COCO_CLASSES[int(c)]] += 1
        print(f"  {cam_id}: scanned {counts[cam_id]['_frames']} frames")
    return counts


def census_video(model, cams, conf, imgsz, stride, batch) -> dict[str, dict[str, int]]:
    import cv2
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    clips = sorted(RAW_DIR.glob("*.mp4"))
    if cams:
        clips = [c for c in clips if c.stem in set(cams)]
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        idx = 0
        frames: list = []
        while True:
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append(frame)
            idx += 1
            if len(frames) >= batch:
                _tally(model, frames, clip.stem, counts, conf, imgsz, batch)
                frames = []
        if frames:
            _tally(model, frames, clip.stem, counts, conf, imgsz, batch)
        cap.release()
        print(f"  {clip.stem}: scanned {counts[clip.stem]['_frames']} frames")
    return counts


def _tally(model, frames, cam_id, counts, conf, imgsz, batch) -> None:
    for r in predict_batched(model, frames, conf, imgsz, batch):
        counts[cam_id]["_frames"] += 1
        for c in r.boxes.cls.tolist():
            counts[cam_id][COCO_CLASSES[int(c)]] += 1


def print_table(counts: dict[str, dict[str, int]]) -> None:
    classes = [COCO_CLASSES[i] for i in KEEP_IDS]
    hdr = f"{'cam_id':<10}{'frames':>8}" + "".join(f"{c:>10}" for c in classes)
    print("\n" + hdr)
    print("-" * len(hdr))
    totals = defaultdict(int)
    for cam_id in sorted(counts):
        row = counts[cam_id]
        line = f"{cam_id:<10}{row['_frames']:>8}"
        for c in classes:
            line += f"{row.get(c, 0):>10}"
            totals[c] += row.get(c, 0)
        totals["_frames"] += row["_frames"]
        print(line)
    print("-" * len(hdr))
    tline = f"{'TOTAL':<10}{totals['_frames']:>8}" + "".join(f"{totals[c]:>10}" for c in classes)
    print(tline)
    scarce = [c for c in classes if c != "person" and totals[c] < 150]
    if scarce:
        print(f"\n⚠  Scarce (< 150 zero-shot instances): {', '.join(scarce)} "
              f"— plan targeted footage or report the limitation (§9.2).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--from-video", action="store_true",
                    help="sample clips directly instead of using data/frames/")
    ap.add_argument("--stride", type=int, default=25, help="frame stride for --from-video")
    ap.add_argument("--batch", type=int, default=8,
                    help="images per forward pass (auto-halves on CUDA OOM; lower if 8GB tight)")
    ap.add_argument("--out", type=Path, default=FRAMES_DIR / "class_census.csv")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed — run: pip install -e \".[perception]\"")

    if not args.from_video and not FRAMES_DIR.exists():
        sys.exit(f"{FRAMES_DIR} missing — run sample_frames.py first, or use --from-video")

    model = YOLO(args.weights)
    if args.from_video:
        counts = census_video(model, args.cams, args.conf, args.imgsz, args.stride, args.batch)
    else:
        counts = census_frames(model, args.cams, args.conf, args.imgsz, args.batch)

    if not counts:
        sys.exit("No frames found to census.")

    print_table(counts)

    classes = [COCO_CLASSES[i] for i in KEEP_IDS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cam_id", "frames", *classes])
        for cam_id in sorted(counts):
            w.writerow([cam_id, counts[cam_id]["_frames"],
                        *(counts[cam_id].get(c, 0) for c in classes)])
    print(f"\nWrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
