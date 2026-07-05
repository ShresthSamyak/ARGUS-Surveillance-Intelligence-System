"""Motion-aware frame sampler — produces frames to hand-label (§9.2, §5.2).

Uniform time sampling wastes the annotator's budget on empty corridors. This
picks frames weighted by motion (MOG2 foreground fraction) while enforcing:
  * a per-camera quota, so the busy intersection cam doesn't dominate the set,
  * a minimum temporal gap between kept frames, to avoid near-duplicate labels
    (the same leakage risk that resplit_by_video.py fixes on the split side).

Two passes per clip:
  1. SCAN (fast): grab frames at --probe-stride, retrieve only those, score
     motion on a downscaled grayscale MOG2 foreground mask.
  2. EXTRACT: greedily select the best frames subject to the min-gap constraint
     until the quota is met, then write full-res JPEGs via sequential decode.

--balance-by-class (opt-in, needs the [perception] extra): during the scan pass,
also run zero-shot YOLO (bags only, low conf) on each probed frame and score it by
weighted bag rarity (suitcase 3x > handbag 2x > backpack 1x). Selection then reserves
--bag-frac of each cam's quota for bag-containing frames before filling the rest by
motion. This attacks person-domination directly: a motion-only seed is thousands of
persons and a handful of bags; the labeler's budget should see bags. Slower (a
detector inference per probed frame) and requires ultralytics.

Output: data/frames/<cam_id>/<cam_id>_<frameidx>.jpg + data/frames/manifest.csv
Manifest columns: cam_id, file, frame_idx, timestamp_s, bag_score, source(bag|motion).

    python scripts/sample_frames.py --total 2500
    python scripts/sample_frames.py --per-video 200 --cams cam02 cam06
    python scripts/sample_frames.py --total 2500 --balance-by-class --bag-frac 0.5
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

# Silence FFmpeg's per-frame "Could not find ref with POC" HEVC noise. Must be set
# before cv2 imports / VideoCapture creation. (With sequential decode below there
# are no such errors anyway, but this keeps output clean if a GOP starts mid-file.)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FRAMES_DIR = ROOT / "data" / "frames"

# COCO bag ids weighted by rarity — the seed set should over-sample the rare ones.
BAG_WEIGHTS = {24: 1.0, 26: 2.0, 28: 3.0}   # backpack, handbag, suitcase
BAG_IDS = list(BAG_WEIGHTS)


def make_bag_detector(weights: str, imgsz: int, conf: float):
    """Return score(frame)->float, the weighted-rarity sum of bag detections.

    Bags only (fast), low conf (recall over precision — a labeler corrects false
    positives; a missed bag frame is a lost sampling opportunity). None-safe: the
    caller only builds this when --balance-by-class is set.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("--balance-by-class needs ultralytics: pip install -e \".[perception]\"")
    model = YOLO(weights)

    def score(frame) -> float:
        res = model.predict(frame, imgsz=imgsz, conf=conf, classes=BAG_IDS, verbose=False)
        s = 0.0
        for c in res[0].boxes.cls.tolist():
            s += BAG_WEIGHTS.get(int(c), 0.0)
        return s

    return score


def scan(path: Path, probe_stride: int, scan_w: int, bag_detector=None):
    """Return ([(frame_idx, motion_score, bag_score)], fps) at probe cadence.

    bag_score is 0 unless bag_detector is provided. Sequential decode throughout
    (grab() every frame, retrieve() at probe cadence) — never seeks.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

    scores: list[tuple[int, float, float]] = []
    idx = 0
    desc = f"scan+det {path.stem}" if bag_detector else f"scan {path.stem}"
    pbar = tqdm(total=total, desc=desc, unit="f", leave=False)
    while True:
        ok = cap.grab()               # cheap: advance without decoding to BGR
        if not ok:
            break
        if idx % probe_stride == 0:
            ok, frame = cap.retrieve()
            if ok:   # a single undecodable frame must NOT abort the whole scan
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (scan_w, max(1, int(h * scan_w / w))))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                fg = mog.apply(gray)
                motion = float((fg > 0).mean())   # fraction of moving pixels
                bag = bag_detector(frame) if bag_detector else 0.0
                scores.append((idx, motion, bag))
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return scores, fps


def select(scores, quota: int, min_gap_frames: int, bag_frac: float = 0.0):
    """Choose up to `quota` frame indices; return (sorted_indices, bag_picked_set).

    bag_frac == 0: greedy by motion, min-gap dedup (the original behavior).
    bag_frac > 0:  first fill round(quota*bag_frac) slots from bag-containing frames
    (ranked by weighted rarity), then fill the remainder by motion. If too few bag
    frames exist, motion transparently backfills — so the quota is still met.
    """
    def greedy(ranked: list[int], cap: int, already: list[int]) -> list[int]:
        chosen: list[int] = []
        for fidx in ranked:
            if len(already) + len(chosen) >= cap:
                break
            if all(abs(fidx - k) >= min_gap_frames for k in (*already, *chosen)):
                chosen.append(fidx)
        return chosen

    kept: list[int] = []
    bag_picked: set[int] = set()

    if bag_frac > 0:
        bag_quota = round(quota * bag_frac)
        bag_ranked = [i for i, _m, b in sorted(scores, key=lambda s: s[2], reverse=True) if b > 0]
        picks_bag = greedy(bag_ranked, bag_quota, kept)
        kept.extend(picks_bag)
        bag_picked.update(picks_bag)

    motion_ranked = [i for i, _m, _b in sorted(scores, key=lambda s: s[1], reverse=True)
                     if i not in set(kept)]
    kept.extend(greedy(motion_ranked, quota, kept))
    return sorted(kept), bag_picked


def extract(path: Path, cam_id: str, frame_idxs: list[int], out_dir: Path) -> list[dict]:
    """Write full-res JPEGs at the selected indices via SEQUENTIAL decode.

    Do NOT seek with CAP_PROP_POS_FRAMES: on inter-frame codecs (HEVC/H.264) it
    lands on a frame whose reference frames were never decoded, yielding grey
    "Could not find ref with POC" corruption (~90% of the frame lost). Decoding
    in order keeps the decoder's reference state intact, so every retrieved frame
    is clean. We grab() through the stream and retrieve() only at target indices,
    stopping once the last target is written.
    """
    if not frame_idxs:
        return []
    targets = set(frame_idxs)
    last = max(frame_idxs)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):     # clear stale frames so a re-run leaves no orphans
        old.unlink()
    rows: list[dict] = []
    idx = 0
    pbar = tqdm(total=len(targets), desc=f"write {cam_id}", unit="img", leave=False)
    while idx <= last:
        if not cap.grab():                 # advance decoder (keeps reference state)
            break
        if idx in targets:
            ok, frame = cap.retrieve()     # decode only the frames we keep
            if ok:
                fname = f"{cam_id}_{idx:07d}.jpg"
                cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"cam_id": cam_id, "file": fname, "frame_idx": idx,
                             "timestamp_s": round(idx / fps, 2)})
                pbar.update(1)
        idx += 1
    pbar.close()
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
    # ---- class-balanced sampling (opt-in) ----
    ap.add_argument("--balance-by-class", action="store_true",
                    help="bias selection toward bag-containing frames (needs [perception])")
    ap.add_argument("--bag-frac", type=float, default=0.5,
                    help="fraction of each cam's quota reserved for bag frames")
    ap.add_argument("--weights", default="yolo11s.pt", help="detector for --balance-by-class")
    ap.add_argument("--det-imgsz", type=int, default=960,
                    help="detector input size for bag scoring (>640 helps small bags)")
    ap.add_argument("--det-conf", type=float, default=0.2,
                    help="low conf = recall over precision when hunting bag frames")
    args = ap.parse_args()

    clips = sorted(RAW_DIR.glob("*.mp4"))
    if args.cams:
        clips = [c for c in clips if c.stem in set(args.cams)]
    if not clips:
        print(f"No clips in {RAW_DIR}. Run organize_footage.py first.")
        return

    quota = args.per_video or max(1, math.ceil(args.total / len(clips)))
    bag_frac = args.bag_frac if args.balance_by_class else 0.0
    detector = None
    if args.balance_by_class:
        print(f"balance-by-class ON: reserving {bag_frac:.0%} of quota for bag frames "
              f"(detector {args.weights} @ imgsz {args.det_imgsz}, conf {args.det_conf}) — slower")
        detector = make_bag_detector(args.weights, args.det_imgsz, args.det_conf)
    print(f"{len(clips)} clips, quota ~{quota}/clip, min gap {args.min_gap_s}s")

    all_rows: list[dict] = []
    n_bag_total = 0
    for clip in clips:
        scores, fps = scan(clip, args.probe_stride, args.scan_width, detector)
        bagmap = {i: b for i, _m, b in scores}
        min_gap_frames = int(args.min_gap_s * fps)
        picks, bag_picked = select(scores, quota, min_gap_frames, bag_frac)
        rows = extract(clip, clip.stem, picks, FRAMES_DIR / clip.stem)
        for r in rows:
            idx = r["frame_idx"]
            r["bag_score"] = round(bagmap.get(idx, 0.0), 1)
            r["source"] = "bag" if idx in bag_picked else "motion"
        all_rows.extend(rows)
        n_bag = sum(1 for r in rows if r["source"] == "bag")
        n_bag_total += n_bag
        extra = f" ({n_bag} bag / {len(rows) - n_bag} motion)" if args.balance_by_class else ""
        print(f"  {clip.stem}: kept {len(rows)} frames{extra}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = FRAMES_DIR / "manifest.csv"
    fields = ["cam_id", "file", "frame_idx", "timestamp_s", "bag_score", "source"]
    processed = {c.stem for c in clips}
    # Merge: preserve manifest rows for cams NOT (re)processed this run, so
    # `--cams cam09` updates only cam09 instead of clobbering the whole manifest.
    merged: list[dict] = []
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            merged = [r for r in csv.DictReader(f) if r.get("cam_id") not in processed]
    merged.extend(all_rows)
    merged.sort(key=lambda r: (r["cam_id"], int(r["frame_idx"])))
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fields})
    summary = f" ({n_bag_total} bag-prioritized)" if args.balance_by_class else ""
    print(f"\nWrote {len(all_rows)} rows for {sorted(processed)}; manifest now {len(merged)} "
          f"rows{summary} -> {FRAMES_DIR.relative_to(ROOT)}/manifest.csv")


if __name__ == "__main__":
    main()
