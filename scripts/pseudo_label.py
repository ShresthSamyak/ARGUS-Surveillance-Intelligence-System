"""LocateAnything pseudo-labeling — pre-annotate frames for CVAT correction (§5.2).

Grounds each sampled frame with LocateAnything (open-vocab → catches small/odd bags
a fixed-class detector misses), dedups LA's repeated boxes, and writes YOLO-format
CANDIDATE labels + a provenance manifest for import into CVAT. These are labels a
human verifies/corrects — NOT final ground truth: LA over/under-detects, emits no
confidence, and occasionally hallucinates. That's why the spec gates on a human audit
(>90% per-class precision) before any pseudo-label reaches training.

MUST run in the argus-vlm env (has transformers 4.57.1 + LA deps):
    & "C:\\Users\\HP\\anaconda3\\envs\\argus-vlm\\python.exe" scripts/pseudo_label.py --load-4bit --cams cam06 --limit 20
    & "$vpy" scripts/pseudo_label.py --load-4bit          # full run, ~15-20 h at 4-bit

Output (data/labels/pseudo/):
    <cam>/<frame>.txt   YOLO: `cls cx cy w h` (normalized)   -- one per frame
    classes.txt         person, backpack, handbag, suitcase
    data.yaml           ultralytics dataset stub
    manifest.csv        per-frame: n boxes/class + model/config provenance (§8.8)

Resumable: frames already labeled are skipped unless --overwrite.

--temporal (OFF by default): consistency filter for FUTURE consecutive-frame
expansion — keep a box only if a same-class box (IoU>0.5) recurs in >=2 of 3 adjacent
frames. Our current seed set is motion-SPREAD (non-adjacent), so it doesn't apply here.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRAMES_DIR = ROOT / "data" / "frames"
OUT_DIR = ROOT / "data" / "labels" / "pseudo"

# Our detection classes -> contiguous YOLO ids (fine-tune head is re-init'd for nc=4).
CLASSES = ["person", "backpack", "handbag", "suitcase"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def to_yolo(box: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return cx, cy, bw, bh


def valid_box(box: list[float], w: int, h: int, min_px: float, max_frac: float) -> bool:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < min_px or bh < min_px:
        return False               # degenerate / sub-pixel
    if bw > max_frac * w and bh > max_frac * h:
        return False               # spans almost the whole frame -> LA junk box
    return True


def frames_for(cams, limit) -> list[Path]:
    imgs: list[Path] = []
    for cam_dir in sorted(p for p in FRAMES_DIR.iterdir() if p.is_dir()):
        if cams and cam_dir.name not in cams:
            continue
        cam_imgs = sorted(cam_dir.glob("*.jpg"))
        if limit:
            cam_imgs = cam_imgs[:limit]
        imgs.extend(cam_imgs)
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max frames per cam (trial runs)")
    ap.add_argument("--load-4bit", action="store_true", help="NF4 (fits 8 GB, ~5x faster)")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--prompts", nargs="+", default=CLASSES,
                    help="grounding prompts; must be a subset of the class names")
    ap.add_argument("--min-px", type=float, default=6.0, help="drop boxes smaller than this")
    ap.add_argument("--max-frac", type=float, default=0.95, help="drop near-full-frame boxes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from PIL import Image

    from vlm.providers import LocateAnythingProvider

    unknown = [p for p in args.prompts if p not in CLASS_ID]
    if unknown:
        sys.exit(f"prompts must be class names {CLASSES}; got unknown {unknown}")

    imgs = frames_for(args.cams, args.limit)
    if not imgs:
        sys.exit(f"No frames under {FRAMES_DIR} (run sample_frames.py first).")

    prov = LocateAnythingProvider(load_in_4bit=args.load_4bit, max_side=args.max_side,
                                  max_new_tokens=args.max_new_tokens)
    print(f"loading LA ({'4-bit' if args.load_4bit else 'fp16'})...")
    prov.load()
    revision = getattr(getattr(prov._model, "config", None), "_commit_hash", "") or ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.as_posix()}\nnc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n", encoding="utf-8")

    manifest_rows: list[dict] = []
    t_start = time.time()
    for i, fp in enumerate(imgs, 1):
        cam = fp.parent.name
        out_txt = OUT_DIR / cam / (fp.stem + ".txt")
        if out_txt.exists() and not args.overwrite:
            continue
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        t0 = time.time()
        boxes = prov.ground(img, args.prompts)

        lines, per_class = [], {c: 0 for c in CLASSES}
        for b in boxes:
            if b["label"] not in CLASS_ID or not valid_box(b["box"], w, h, args.min_px, args.max_frac):
                continue
            cx, cy, bw, bh = to_yolo(b["box"], w, h)
            lines.append(f"{CLASS_ID[b['label']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_class[b["label"]] += 1
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        manifest_rows.append({"cam_id": cam, "file": fp.name, "n_boxes": len(lines),
                              **per_class, "secs": round(time.time() - t0, 1)})
        dt = time.time() - t_start
        eta = dt / i * (len(imgs) - i)
        print(f"[{i}/{len(imgs)}] {cam}/{fp.name}: {len(lines)} boxes "
              f"({', '.join(f'{k}={v}' for k, v in per_class.items() if v)}) "
              f"| {manifest_rows[-1]['secs']}s | ETA {eta/60:.0f}m")

    # merge-append manifest (like sample_frames), keep other cams' rows
    man = OUT_DIR / "manifest.csv"
    fields = ["cam_id", "file", "n_boxes", *CLASSES, "secs"]
    processed_files = {(r["cam_id"], r["file"]) for r in manifest_rows}
    prev = []
    if man.exists():
        with open(man, newline="", encoding="utf-8") as f:
            prev = [r for r in csv.DictReader(f)
                    if (r["cam_id"], r["file"]) not in processed_files]
    allrows = prev + manifest_rows
    with open(man, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        for r in allrows:
            wtr.writerow({k: r.get(k, "") for k in fields})

    tot = sum(r["n_boxes"] for r in manifest_rows)
    print(f"\nLabeled {len(manifest_rows)} new frames, {tot} candidate boxes -> "
          f"{OUT_DIR.relative_to(ROOT)}/  (model nvidia/LocateAnything-3B rev {revision[:12] or '?'})")
    print("These are CANDIDATE labels — import into CVAT and correct before training (§5.2 audit gate).")


if __name__ == "__main__":
    main()
