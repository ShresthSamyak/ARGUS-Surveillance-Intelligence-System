"""Camera worker — the fast-path loop, one process per camera (§4.1, §5).

decode -> detect (stride) -> track (BoT-SORT + OSNet) -> project to world coords
-> publish compact track JSON to Redis channel tracks.{cam_id}.

v0.2: subscribes to watch-ROIs from the reasoner (§6.7) and, inside those ROIs,
lowers the detector acceptance bar and runs the static channel every frame.

Phase 1. Skeleton mirrors the spec pseudocode.
"""

from __future__ import annotations

from common.config import CameraConfig


def run(cam_cfg: CameraConfig) -> None:
    # src     = FrameSource(cam_cfg.uri)              # PyAV; yields (frame, ts)
    # det     = Detector(cam_cfg.model, imgsz=1280)
    # tracker = BoTSORT(reid="osnet_x1_0", gmc=None)  # GMC off: fixed camera
    # static  = DualBackgroundChannel()
    # H       = np.array(cam_cfg.homography)          # pixel -> ground plane
    #
    # for i, (frame, ts) in enumerate(src):
    #     rois  = latest_watch_rois(cam_cfg.id)       # §6.7 expectation feedback
    #     dets  = det(frame, watch_rois=rois) if i % cam_cfg.det_stride == 0 else []
    #     tracks = tracker.update(dets, frame)
    #     for t in tracks:
    #         t.world_xy = project(H, feet_point(t.bbox))
    #     blobs = static.update(frame, rois)
    #     publish(f"tracks.{cam_cfg.id}", pack(ts, tracks, blobs, embeddings_if_refreshed(tracks)))
    raise NotImplementedError("Phase 1")
