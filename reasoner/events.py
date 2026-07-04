"""Append-only event log — the system of record (§6.2, §8.8).

Every fact is an immutable event; all current state must be re-derivable by
replay. Payloads carry {ts, cam_id, keyframe_path, bbox} plus, in v0.2, the
{model_hash, config_hash, code_version} reproducibility triple (§8.8).

Phase 2. The EventType enum below is the canonical list — extend it here only.
"""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    ENTITY_CREATED = "ENTITY_CREATED"
    ENTITY_EXITED = "ENTITY_EXITED"
    ENTITY_REACQUIRED = "ENTITY_REACQUIRED"
    CARRYING_CONFIRMED = "CARRYING_CONFIRMED"
    PLACED = "PLACED"
    OWNER_LEFT_R1 = "OWNER_LEFT_R1"
    OWNER_LEFT_R2 = "OWNER_LEFT_R2"
    OWNER_OUT_OF_VIEW = "OWNER_OUT_OF_VIEW"
    OWNER_EXITED_SCENE = "OWNER_EXITED_SCENE"
    OWNER_REACQUIRED = "OWNER_REACQUIRED"
    STATE_CHANGED = "STATE_CHANGED"
    PICKED_UP = "PICKED_UP"
    HANDOFF_SUSPECTED = "HANDOFF_SUSPECTED"
    # ---- v0.2 ----
    PRESENCE_CHANGED = "PRESENCE_CHANGED"      # §6.6
    REMOVED_UNSEEN = "REMOVED_UNSEEN"          # §6.6 theft-under-cover
    VLM_VERDICT = "VLM_VERDICT"                # §5.7
    FINGERPRINTED = "FINGERPRINTED"            # §5.7
    # ----
    ZONE_ENTERED = "ZONE_ENTERED"
    ZONE_EXITED = "ZONE_EXITED"
    BEHAVIOR_FLAG = "BEHAVIOR_FLAG"
    ALERT_RAISED = "ALERT_RAISED"
    ALERT_UPDATED = "ALERT_UPDATED"
    ALERT_RESOLVED = "ALERT_RESOLVED"
    CAMERA_STALE = "CAMERA_STALE"              # §8.5 blind camera = security event


class EventWriter:
    """Append-only writer. SQLite for Phase 1-2, Postgres later (§8.2)."""

    def emit(self, type: EventType, **payload) -> int:
        raise NotImplementedError("Phase 2: append row, return seq")

    def replay(self, since_seq: int = 0):
        raise NotImplementedError("Phase 2: yield events for state reconstruction")
