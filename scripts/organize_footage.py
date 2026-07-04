"""Organize raw CCTV footage into data/raw/ and generate camera configs + manifest.

Run once, first. It:
  1. finds *.mp4 in the repo root (or --src),
  2. parses the CCTV filename convention (name_YYYYmmddHHMMSS_YYYYmmddHHMMSS_id.mp4),
  3. probes resolution / fps / frame count with OpenCV,
  4. MOVES each clip to data/raw/camNN.mp4 (atomic rename on the same volume),
  5. writes configs/cameras/camNN.yaml stubs (homography + zones left for calibration),
  6. writes data/raw/manifest.csv (the only tracked artifact — metadata, no media).

Idempotent-ish: clips already inside data/raw/ are re-probed and re-manifested; the
move step is skipped for anything already there. Nothing is deleted.

    python scripts/organize_footage.py            # move root *.mp4 -> data/raw/
    python scripts/organize_footage.py --dry-run  # print the plan, touch nothing
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CAM_CFG_DIR = ROOT / "configs" / "cameras"

TS_RE = re.compile(r"^\d{14}$")  # YYYYmmddHHMMSS


def parse_name(stem: str) -> tuple[str, str, str]:
    """Return (location_name, start_iso, end_iso) parsed from a CCTV filename stem.

    Convention: <name tokens>_<14-digit start>_<14-digit end>_<trailing id>.
    Falls back gracefully (empty times) when the pattern is absent.
    """
    tokens = stem.split("_")
    ts_idx = [i for i, t in enumerate(tokens) if TS_RE.match(t)]
    if len(ts_idx) >= 2:
        name = "_".join(tokens[: ts_idx[0]]).strip()
        start = _iso(tokens[ts_idx[0]])
        end = _iso(tokens[ts_idx[1]])
        return name or stem, start, end
    return stem, "", ""


def _iso(ts14: str) -> str:
    try:
        return datetime.strptime(ts14, "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        return ""


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_s"] = round(info["frames"] / info["fps"], 1) if info["fps"] else 0.0
    return info


def det_stride_for(fps: float) -> int:
    """Target ~12.5 Hz detections: stride 2 @ 25fps, stride 1 @ 20fps (§4.1)."""
    return 2 if fps >= 24 else 1


def write_camera_yaml(cam_id: str, name: str, uri: str, info: dict, src_name: str,
                      start: str, end: str) -> None:
    cfg = {
        "id": cam_id,
        "name": name,
        "uri": uri,
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "det_stride": det_stride_for(info["fps"]),
        "homography": None,   # TODO scripts/calibrate_homography.py (§4.2)
        "zones": [],          # TODO scripts/define_zones.py (§4.3)
        "thresholds": {},
        "source_filename": src_name,
        "start_time": start,
        "end_time": end,
    }
    out = CAM_CFG_DIR / f"{cam_id}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# {name}  ({src_name})\n")
        f.write("# Auto-generated stub. Calibrate homography + define zones before L2 runs.\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=ROOT, help="dir to scan for *.mp4")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CAM_CFG_DIR.mkdir(parents=True, exist_ok=True)

    # Collect candidates: loose clips in --src, plus anything already in data/raw/.
    loose = sorted(p for p in args.src.glob("*.mp4") if p.parent != RAW_DIR)
    existing = sorted(RAW_DIR.glob("*.mp4"))
    if not loose and not existing:
        print("No .mp4 files found. Nothing to do.")
        return

    rows: list[dict] = []
    cam_n = 0
    # Deterministic cam ids: assign to loose clips first (sorted), then keep existing.
    for src in loose:
        cam_n += 1
        cam_id = f"cam{cam_n:02d}"
        name, start, end = parse_name(src.stem)
        info = probe(src)
        dst = RAW_DIR / f"{cam_id}.mp4"
        print(f"{cam_id}  {info['width']}x{info['height']} {info['fps']}fps "
              f"{info['duration_s']/60:.1f}min  <- {src.name}")
        if not args.dry_run:
            shutil.move(str(src), str(dst))
            write_camera_yaml(cam_id, name, str(dst.relative_to(ROOT).as_posix()),
                              info, src.name, start, end)
        rows.append({"cam_id": cam_id, "name": name, "file": dst.name,
                     "source_filename": src.name, "start_time": start, "end_time": end,
                     **info})

    for src in existing:
        # Already organized (re-run): re-probe, refresh manifest, don't move.
        cam_id = src.stem
        cfg_path = CAM_CFG_DIR / f"{cam_id}.yaml"
        name = ""
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                name = (yaml.safe_load(f) or {}).get("name", "")
        info = probe(src)
        rows.append({"cam_id": cam_id, "name": name, "file": src.name,
                     "source_filename": "", "start_time": "", "end_time": "", **info})

    if args.dry_run:
        print("\n[dry-run] no files moved, no configs written.")
        return

    rows.sort(key=lambda r: r["cam_id"])
    manifest = RAW_DIR / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_min = sum(r["duration_s"] for r in rows) / 60
    print(f"\n{len(rows)} clips, {total_min:.1f} min total.")
    print(f"Manifest: {manifest.relative_to(ROOT)}")
    print(f"Camera stubs: {CAM_CFG_DIR.relative_to(ROOT)}/  "
          f"(homography + zones still TODO)")


if __name__ == "__main__":
    main()
