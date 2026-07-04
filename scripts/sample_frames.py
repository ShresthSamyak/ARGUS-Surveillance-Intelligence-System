"""Motion-aware frame sampler — produces frames to hand-label (§9.2, §5.2).

Uniform time sampling wastes the annotator's budget on empty corridors. This
picks frames weighted by motion (MOG2 foreground fraction) while enforcing:
  * a per-camera quota, so the busy intersection cam doesn't dominate the set,
  * a minimum temporal gap between kept frames, to avoid near-duplicate labels
    (the same leakage risk that resplit_by_video.py fixes on the split side).

Two passes per clip:
  1. SCAN (fast): grab frames at --probe-stride, retrieve only those, score
     motion on a downscaled grayscale MOG2 foreground mask.
  2. EXTRACT: greedily select highest-motion frames subject to the min-gap
     constraint until the quota is met, then re-seek and write full-res JPEGs.

Output: data/frames/<cam_id>/<cam_id>_<frameidx>.jpg + data/frames/manifest.csv

    python scripts/sample_frames.py --total 2500
    python scripts/sample_frames.py --per-video 200 --cams cam02 cam06
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FRAMES_DIR = ROOT / "data" / "frames"


def scan_motion(path: Path, probe_stride: int, scan_w: int) -> tuple[list[tuple[int, float]], float]:
    """Return [(frame_idx, motion_score)] at probe cadence, plus source fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

    scores: list[tuple[int, float]] = []
    idx = 0
    pbar = tqdm(total=total, desc=f"scan {path.stem}", unit="f", leave=False)
    while True:
        ok = cap.grab()               # cheap: advance without decoding to BGR
        if not ok:
            break
        if idx % probe_stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (scan_w, max(1, int(h * scan_w / w))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            fg = mog.apply(gray)
            score = float((fg > 0).mean())   # fraction of moving pixels
            scores.append((idx, score))
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return scores, fps


def select(scores: list[tuple[int, float]], quota: int, min_gap_frames: int) -> list[int]:
    """Greedy: highest motion first, reject any frame within min_gap of a keeper."""
    kept: list[int] = []
    for fidx, _ in sorted(scores, key=lambda s: s[1], reverse=True):
        if len(kept) >= quota:
            break
        if all(abs(fidx - k) >= min_gap_frames for k in kept):
            kept.append(fidx)
    return sorted(kept)


def extract(path: Path, cam_id: str, frame_idxs: list[int], out_dir: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for fidx in tqdm(frame_idxs, desc=f"write {cam_id}", unit="img", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        fname = f"{cam_id}_{fidx:07d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"cam_id": cam_id, "file": fname, "frame_idx": fidx,
                     "timestamp_s": round(fidx / fps, 2)})
    cap.release()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=2500,
                    help="target total frames across all cams (split evenly per cam)")
    ap.add_argument("--per-video", type=int, default=None,
                    help="override: fixed quota per cam (ignores --total)")
    ap.add_argument("--cams", nargs="*", default=None, help="limit to these cam ids")
    ap.add_argument("--probe-stride", type=int, default=5,
                    help="score every Nth frame in the scan pass")
    ap.add_argument("--min-gap-s", type=float, default=2.0,
                    help="minimum seconds between kept frames (anti-duplicate)")
    ap.add_argument("--scan-width", type=int, default=320, help="downscale width for scoring")
    args = ap.parse_args()

    clips = sorted(RAW_DIR.glob("*.mp4"))
    if args.cams:
        clips = [c for c in clips if c.stem in set(args.cams)]
    if not clips:
        print(f"No clips in {RAW_DIR}. Run organize_footage.py first.")
        return

    quota = args.per_video or max(1, math.ceil(args.total / len(clips)))
    print(f"{len(clips)} clips, quota ~{quota}/clip, min gap {args.min_gap_s}s")

    all_rows: list[dict] = []
    for clip in clips:
        scores, fps = scan_motion(clip, args.probe_stride, args.scan_width)
        min_gap_frames = int(args.min_gap_s * fps)
        picks = select(scores, quota, min_gap_frames)
        rows = extract(clip, clip.stem, picks, FRAMES_DIR / clip.stem)
        all_rows.extend(rows)
        print(f"  {clip.stem}: kept {len(rows)} frames")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = FRAMES_DIR / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cam_id", "file", "frame_idx", "timestamp_s"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nTotal {len(all_rows)} frames -> {FRAMES_DIR.relative_to(ROOT)}/  (manifest.csv)")


if __name__ == "__main__":
    main()
