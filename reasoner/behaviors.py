"""Behavior rule engine (§7.1). Runs on the world-coord trajectories the entity
layer already maintains — a free ride on the reasoner tick.

Rules: loitering, pacing, erratic path, restricted entry, running, crowd
density, crowd surge, motion-energy spike. v0.2 adds repeated-approach and
coordinated-interest (rules over the event log, not single frames).

Baselines (p95 speed, zone density) are learned online per camera from the
first N hours — "normal" is calibrated per viewpoint, not hardcoded.

Phase 3.
"""

from __future__ import annotations


class BehaviorEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, graph, event_log):
        """Emit BEHAVIOR_FLAG events with score + evidence window."""
        raise NotImplementedError("Phase 3")
