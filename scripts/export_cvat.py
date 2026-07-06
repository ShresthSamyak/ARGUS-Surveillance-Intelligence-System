"""Package pseudo-labels into CVAT-importable YOLO 1.1 zips for correction (§9.3).

The pseudo-labeler emits 2 classes (person, bag). For CVAT correction the operator
must ASSIGN the bag subclass, so we import under a 5-label scheme:
    0 person   1 backpack   2 handbag   3 suitcase   4 bag   (<- placeholder)
Our generic bag (pseudo id 1) maps to id 4 ("bag") so the operator can SEE which
boxes still need subclassing; they reclassify each 4 -> 1/2/3. At training-data prep
we assert zero class-4 boxes remain (scripts/resplit_by_video.py checks this).

One zip PER CAMERA (CVAT tasks are per-camera-manageable). Import flow:
  1. In CVAT: Create Task, add the 5 labels above (same order!), attach the frames
     for that cam (data/frames/<cam>/*.jpg).
  2. Task > Upload annotations > format "YOLO 1.1" > pick <cam>_cvat_yolo11.zip.
  3. Correct boxes + reclassify every "bag" to backpack/handbag/suitcase.
  4. Export as "YOLO 1.1" back into data/labels/corrected/<cam>/ for resplit.

    python scripts/export_cvat.py                 # all pseudo-labeled cams
    python scripts/export_cvat.py --cams cam05     # one cam

Runs in base env (no GPU / no ML deps).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PSEUDO_DIR = ROOT / "data" / "labels" / "pseudo"
FRAMES_DIR = ROOT / "data" / "frames"
OUT_DIR = ROOT / "data" / "labels" / "cvat_import"

# CVAT task label scheme (order defines YOLO ids). "bag" (4) is the placeholder the
# operator reclassifies to a real subclass.
CVAT_NAMES = ["person", "backpack", "handbag", "suitcase", "bag"]
# pseudo id -> CVAT id : person 0->0, bag 1->4
PSEUDO_TO_CVAT = {0: 0, 1: 4}


def build_zip(cam: str) -> tuple[Path, int, int]:
    cam_lbl = PSEUDO_DIR / cam
    txts = sorted(cam_lbl.glob("*.txt"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / f"{cam}_cvat_yolo11.zip"

    n_frames, n_boxes = 0, 0
    obj_names = "\n".join(CVAT_NAMES) + "\n"
    obj_data = (f"classes = {len(CVAT_NAMES)}\n"
                "train = data/train.txt\nnames = data/obj.names\nbackup = backup/\n")
    train_lines, label_files = [], {}

    for txt in txts:
        stem = txt.stem                       # e.g. cam05_0001234
        out_lines = []
        for ln in txt.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            cid = PSEUDO_TO_CVAT.get(int(parts[0]))
            if cid is None:
                continue
            out_lines.append(f"{cid} {' '.join(parts[1:])}")
        label_files[f"obj_train_data/{stem}.txt"] = "\n".join(out_lines) + \
            ("\n" if out_lines else "")
        train_lines.append(f"data/obj_train_data/{stem}.jpg")
        n_frames += 1
        n_boxes += len(out_lines)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", obj_names)
        z.writestr("obj.data", obj_data)
        z.writestr("train.txt", "\n".join(train_lines) + "\n")
        for name, content in label_files.items():
            z.writestr(name, content)
    return zip_path, n_frames, n_boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cams", nargs="*", default=None, help="cams to export (default: all)")
    args = ap.parse_args()

    if not PSEUDO_DIR.exists():
        raise SystemExit(f"{PSEUDO_DIR} missing — run pseudo_label.py first.")
    cams = args.cams or sorted(d.name for d in PSEUDO_DIR.iterdir() if d.is_dir())
    if not cams:
        raise SystemExit("No pseudo-labeled cams found.")

    print(f"CVAT label scheme (add these to the task, in order): {CVAT_NAMES}")
    print(f"Frames per cam are under data/frames/<cam>/ — attach those when creating the task.\n")
    for cam in cams:
        if not (PSEUDO_DIR / cam).is_dir():
            print(f"  {cam}: no pseudo labels, skipped")
            continue
        zp, nf, nb = build_zip(cam)
        print(f"  {cam}: {nf} frames, {nb} boxes -> {zp.relative_to(ROOT)}")
    print(f"\nImport each zip via CVAT > Upload annotations > 'YOLO 1.1'. "
          f"Reclassify every 'bag' box to backpack/handbag/suitcase before exporting.")


if __name__ == "__main__":
    main()
