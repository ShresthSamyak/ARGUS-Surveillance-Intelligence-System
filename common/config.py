"""Pydantic v2 schemas and loaders for ARGUS configuration.

Two config surfaces:
  * per-camera YAML  -> CameraConfig     (configs/cameras/camNN.yaml)
  * global FSM sheet -> FSMConfig         (configs/fsm.yaml, Appendix A)

Keeping these validated and typed is design principle #7 (real-time honesty):
a mis-typed threshold should fail loudly at load, not silently mis-rank a bag.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# COCO class indices we keep (§5.1 — retain COCO indexing so pretrained heads transfer).
COCO_CLASSES: dict[int, str] = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}
BAG_CLASSES: frozenset[str] = frozenset({"backpack", "handbag", "suitcase"})


class ZoneType(str, Enum):
    baggage = "baggage"        # expected stationary bags -> lower base risk, longer timers
    transit = "transit"        # corridors/gates -> bags should not dwell
    restricted = "restricted"  # entry itself is an event
    entry_exit = "entry_exit"  # scene boundary; crossing = person EXITED (hard evidence)
    ignore = "ignore"          # masks: screens, glass reflections, foliage


class Zone(BaseModel):
    name: str
    type: ZoneType
    # Polygon vertices in WORLD (ground-plane, metres) coordinates — see §4.3.
    polygon_world: list[list[float]] = Field(default_factory=list)
    weight: Optional[float] = None  # optional per-zone risk weight override (§6.6)

    @field_validator("polygon_world")
    @classmethod
    def _min_three_points(cls, v: list[list[float]]) -> list[list[float]]:
        if v and len(v) < 3:
            raise ValueError("polygon_world needs >= 3 vertices to form a zone")
        for pt in v:
            if len(pt) != 2:
                raise ValueError(f"polygon vertex must be [x, y], got {pt!r}")
        return v


class CameraConfig(BaseModel):
    id: str                              # "cam01"
    name: str = ""                       # human-readable location
    uri: str                             # file path or rtsp://...
    fps: float = 25.0
    width: int = 0
    height: int = 0
    det_stride: int = 2                  # detect every Nth frame (§4.1)
    # 3x3 pixel->ground-plane homography. None until calibrated (scripts/calibrate_homography.py).
    homography: Optional[list[list[float]]] = None
    zones: list[Zone] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)  # per-cam FSM overrides
    # Provenance / bookkeeping (populated by scripts/organize_footage.py).
    source_filename: str = ""
    start_time: str = ""
    end_time: str = ""

    @property
    def calibrated(self) -> bool:
        return self.homography is not None

    @field_validator("homography")
    @classmethod
    def _check_homography_shape(cls, v):
        if v is not None and (len(v) != 3 or any(len(row) != 3 for row in v)):
            raise ValueError("homography must be a 3x3 matrix")
        return v


class FSMConfig(BaseModel):
    """Global parameter sheet (Appendix A). Single source of truth for L2."""

    ownership: dict = Field(default_factory=dict)
    placement: dict = Field(default_factory=dict)
    fsm: dict = Field(default_factory=dict)
    zones: dict = Field(default_factory=dict)
    risk: dict = Field(default_factory=dict)
    reid: dict = Field(default_factory=dict)
    tracker: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_camera_config(path: str | Path) -> CameraConfig:
    with open(path, "r", encoding="utf-8") as f:
        return CameraConfig.model_validate(yaml.safe_load(f))


def load_all_cameras(camera_dir: str | Path) -> list[CameraConfig]:
    camera_dir = Path(camera_dir)
    return [load_camera_config(p) for p in sorted(camera_dir.glob("*.yaml"))]


def load_fsm_config(path: str | Path) -> FSMConfig:
    with open(path, "r", encoding="utf-8") as f:
        return FSMConfig.model_validate(yaml.safe_load(f))
