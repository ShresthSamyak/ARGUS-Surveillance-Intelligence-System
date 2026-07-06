"""LocateAnything pseudo-labeling — pre-annotate frames for CVAT correction (§5.2).

Grounds each sampled frame with LocateAnything (open-vocab → catches small/odd bags
a fixed-class detector misses), dedups LA's repeated boxes, and writes YOLO-format
CANDIDATE labels + a provenance manifest for import into CVAT. These are labels a
human verifies/corrects — NOT final ground truth: LA over/under-detects, emits no
confidence, and occasionally hallucinates. The spec gates on a human audit
(>90% per-class precision) before any pseudo-label reaches training.

LABELING SCHEMES (LA's <box> output carries NO class, so we prompt per class and tag
by prompt — running all 4 bag/person prompts triple-counts the same bag):
  * bag2  (default): prompts {person, bag}. No backpack/handbag/suitcase confusion,
    ~2x faster; the human assigns the bag SUBCLASS in CVAT. Final dataset is still
    4-class after refinement. Plays to LA's strength, avoids its weak subclassing.
  * full4: prompts {person, backpack, handbag, suitcase} — one box per class per
    location; expect cross-class bag dups a human must delete.

MUST run in the argus-vlm env (transformers 4.57.1 + LA deps):
    & "$vpy" scripts/pseudo_label.py --load-4bit --cams cam06 --limit 20   # trial
    & "$vpy" scripts/pseudo_label.py --load-4bit                            # full run

Output (data/labels/pseudo/): <cam>/<frame>.txt (YOLO), classes.txt, data.yaml,
manifest.csv (per-frame counts + model/revision provenance). Resumable (skips done
frames unless --overwrite).
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

# (class_label, LA prompt text). The bag prompt lists synonyms to maximize recall;
# all its boxes collapse to the single "bag" class for later human subclassing.
SCHEMES: dict[str, list[tuple[str, str]]] = {
    "bag2": [("person", "person"),
             ("bag", "backpack, handbag, suitcase, or bag on the floor")],
    "full4": [("person", "person"), ("backpack", "backpack"),
              ("handbag", "handbag"), ("suitcase", "suitcase")],
}

# Dense library-baggage cams: static racks pile up dozens of bags that LA triple-
# counts (~143 noisy boxes/frame) -> cheaper to hand-label from scratch than to clean.
# Design = merge of scheme options 1 (bag2) + 3 (LA on sparse cams only): pseudo-label
# the sparse cams; RESERVE these for manual CVAT annotation. A full run skips them
# unless --include-manual; naming one explicitly in --cams still overrides.
MANUAL_CAMS = {"cam06", "cam07"}


def to_yolo(box: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h


def valid_box(box: list[float], w: int, h: int, min_px: float, max_frac: float) -> bool:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < min_px or bh < min_px:
        return False                       # degenerate / sub-pixel
    if bw > max_frac * w and bh > max_frac * h:
        return False                       # near-full-frame -> LA junk box
    return True


def rebuild_manifest_from_disk(out_dir: Path, class_names: list[str]) -> int:
    """Rebuild manifest.csv by scanning ALL label .txt on disk (interruption-proof).

    An interrupted run leaves label files but never reaches the manifest write, so
    a row-merge approach silently drops those frames. Reconstructing from the .txt
    files (the real deliverable) always yields a complete, correct summary.
    """
    rows = []
    for cam_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        for txt in sorted(cam_dir.glob("*.txt")):
            counts = [0] * len(class_names)
            for ln in txt.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln:
                    cid = int(ln.split()[0])
                    if 0 <= cid < len(class_names):
                        counts[cid] += 1
            rows.append({"cam_id": cam_dir.name, "file": txt.stem + ".jpg",
                         "n_boxes": sum(counts),
                         **{c: counts[i] for i, c in enumerate(class_names)}})
    fields = ["cam_id", "file", "n_boxes", *class_names]
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def frames_for(cams, limit, include_manual) -> list[Path]:
    imgs: list[Path] = []
    for cam_dir in sorted(p for p in FRAMES_DIR.iterdir() if p.is_dir()):
        name = cam_dir.name
        if cams:                                  # explicit list overrides the reserve
            if name not in cams:
                continue
        elif name in MANUAL_CAMS and not include_manual:
            continue                              # reserved for hand-labeling
        cam_imgs = sorted(cam_dir.glob("*.jpg"))
        imgs.extend(cam_imgs[:limit] if limit else cam_imgs)
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheme", choices=list(SCHEMES), default="bag2")
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max frames per cam (trials)")
    ap.add_argument("--load-4bit", action="store_true", help="NF4 (fits 8 GB, ~5x faster)")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--min-px", type=float, default=6.0)
    ap.add_argument("--max-frac", type=float, default=0.95)
    ap.add_argument("--include-manual", action="store_true",
                    help=f"also pseudo-label the reserved dense cams {sorted(MANUAL_CAMS)}")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest-only", action="store_true",
                    help="rebuild manifest.csv from existing labels on disk, then exit (no GPU)")
    args = ap.parse_args()

    scheme = SCHEMES[args.scheme]
    class_names = [lbl for lbl, _ in scheme]
    class_id = {lbl: i for i, lbl in enumerate(class_names)}

    if args.manifest_only:
        n = rebuild_manifest_from_disk(OUT_DIR, class_names)
        print(f"manifest.csv rebuilt from {n} label files on disk.")
        return

    from PIL import Image

    from vlm.providers import LocateAnythingProvider

    imgs = frames_for(args.cams, args.limit, args.include_manual)
    if not imgs:
        sys.exit(f"No frames under {FRAMES_DIR} (run sample_frames.py first).")
    if not args.cams and not args.include_manual:
        print(f"Reserved for MANUAL hand-labeling (skipped): {sorted(MANUAL_CAMS)} "
              "(dense library baggage — draw fresh in CVAT).")

    prov = LocateAnythingProvider(load_in_4bit=args.load_4bit, max_side=args.max_side,
                                  max_new_tokens=args.max_new_tokens)
    print(f"scheme={args.scheme} classes={class_names} | loading LA "
          f"({'4-bit' if args.load_4bit else 'fp16'})...")
    prov.load()
    revision = getattr(getattr(prov._model, "config", None), "_commit_hash", "") or ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.as_posix()}\nnc: {len(class_names)}\nnames: {class_names}\n",
        encoding="utf-8")
    (OUT_DIR.parent / "MANUAL_CAMS.txt").write_text(
        "Hand-label these cams from scratch in CVAT (dense library baggage; NOT "
        "pseudo-labeled — LA triple-counts static racks):\n"
        + "\n".join(sorted(MANUAL_CAMS)) + "\n", encoding="utf-8")

    rows: list[dict] = []
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

        # One ground() call per class so each box gets a definite label; dedup is
        # per-label inside ground(). person and bag legitimately overlap -> we do
        # NOT NMS across classes.
        lines, per_class = [], {c: 0 for c in class_names}
        for lbl, prompt_text in scheme:
            for b in prov.ground(img, [prompt_text]):
                if not valid_box(b["box"], w, h, args.min_px, args.max_frac):
                    continue
                cx, cy, bw, bh = to_yolo(b["box"], w, h)
                lines.append(f"{class_id[lbl]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                per_class[lbl] += 1
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        rows.append({"cam_id": cam, "file": fp.name, "n_boxes": len(lines),
                     **per_class, "secs": round(time.time() - t0, 1)})
        eta = (time.time() - t_start) / i * (len(imgs) - i)
        print(f"[{i}/{len(imgs)}] {cam}/{fp.name}: {len(lines)} boxes "
              f"({', '.join(f'{k}={v}' for k, v in per_class.items() if v)}) "
              f"| {rows[-1]['secs']}s | ETA {eta/60:.0f}m")

    # Rebuild the whole manifest from disk — interruption-proof (see fn docstring).
    total = rebuild_manifest_from_disk(OUT_DIR, class_names)

    tot = sum(r["n_boxes"] for r in rows)
    print(f"\nLabeled {len(rows)} new frames ({tot} boxes) this run; manifest now covers "
          f"{total} frames ({args.scheme}) -> {OUT_DIR.relative_to(ROOT)}/  "
          f"(nvidia/LocateAnything-3B rev {revision[:12] or '?'})")
    print("CANDIDATE labels — import into CVAT, assign bag subclass + correct (§5.2 audit gate).")


if __name__ == "__main__":
    main()
