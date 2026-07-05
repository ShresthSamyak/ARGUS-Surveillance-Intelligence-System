"""Smoke tests for the config layer — runnable today (no GPU, no video).

Verifies configs/fsm.yaml parses into FSMConfig with all v0.2 sections present,
and that the CameraConfig schema validates + rejects a bad homography.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import CameraConfig, FSMConfig, ZoneType, load_fsm_config

ROOT = Path(__file__).resolve().parents[1]


def test_fsm_yaml_loads_with_v02_sections():
    cfg = load_fsm_config(ROOT / "configs" / "fsm.yaml")
    assert isinstance(cfg, FSMConfig)
    # v0.1 core
    assert cfg.fsm["r1_m"] == 2.0
    assert cfg.risk["alert_tier"] == 75
    # v0.2 additions must be present
    assert cfg.presence["t_unconfirmed_s"] == 20
    assert cfg.watch_roi["accept_conf"] == 0.15
    assert cfg.vlm["provider"] == "locateanything3b"
    assert cfg.approach["k"] == 3


def test_camera_config_minimal():
    cam = CameraConfig(id="cam99", uri="data/raw/cam99.mp4", fps=25)
    assert cam.calibrated is False       # no homography yet
    assert cam.det_stride == 2


def test_camera_config_rejects_bad_homography():
    with pytest.raises(ValueError):
        CameraConfig(id="cam99", uri="x.mp4", homography=[[1, 2], [3, 4]])  # not 3x3


def test_zone_polygon_validation():
    with pytest.raises(ValueError):
        # 2 vertices can't form a zone
        from common.config import Zone
        Zone(name="z", type=ZoneType.baggage, polygon_world=[[0, 0], [1, 1]])
