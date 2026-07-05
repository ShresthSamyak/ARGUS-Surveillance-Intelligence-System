"""Event enrichment (§5.7 P4). Best-effort grounded descriptions for the operator.

On alert raise, ground the keyframe into a human string for the timeline:
"black backpack, red side pocket, ~48 cm". Off the critical path — the alert
already fired on the fast path.

Phase 3/4.
"""

from __future__ import annotations


def enrich_keyframe(provider, keyframe_path):  # noqa: ARG001
    raise NotImplementedError("Phase 3/4")
