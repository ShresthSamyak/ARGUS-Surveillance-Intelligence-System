"""Cold-start inventory + reconciliation sweeps (§5.7, v0.2).

cold_start_scan: on pipeline boot / camera reconnect, ground bags on the full
frame to seed pre-existing bags the dual-BG channel is blind to (they were
background from frame 1). Seeded bags get placed_before_boot=true, ownership
UNRESOLVED, strict timers. Closes S11's ground truth.

reconcile: every ~90 s per camera, diff VLM grounding against the entity graph.
VLM-only bag -> watch-ROI candidate (fast path must corroborate). Graph-only bag
with a clear ROI -> presence-decay accelerant.

Hallucination guard applies: sweeps never mint entities directly (§5.7).

Phase 2.
"""

from __future__ import annotations


def cold_start_scan(provider, frame, homography):  # noqa: ARG001
    raise NotImplementedError("Phase 2")


def reconcile(provider, frame, graph):  # noqa: ARG001
    raise NotImplementedError("Phase 2")
