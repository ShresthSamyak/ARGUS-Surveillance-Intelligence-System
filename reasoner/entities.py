"""Persistent entities + track<->entity binding (§6.1).

Tracker tracks are ephemeral (they die on occlusion); entities are persistent —
they own embedding banks, trajectories, relationships, and state, and survive
track death. A new track first tries to bind to an UNSEEN entity via Re-ID
(§6.9); only if no match does it mint a new entity.

Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from common.config import BagSource


class EntityStatus(str, Enum):
    ACTIVE = "ACTIVE"
    UNSEEN = "UNSEEN"
    EXITED = "EXITED"


@dataclass
class PersonEntity:
    id: str
    status: EntityStatus = EntityStatus.ACTIVE
    active_track: int | None = None
    owns: set[str] = field(default_factory=set)
    last_seen: tuple | None = None          # (ts, world_xy, cam_id)
    # bank: EmbeddingBank, trajectory: RingBuffer  (attached at runtime)


@dataclass
class BagEntity:
    id: str
    cls: str                                 # backpack | handbag | suitcase
    position: tuple | None = None            # world_xy, authoritative while stationary
    owner: str | None = None
    ownership_confidence: float = 0.0
    risk: float = 0.0
    source: BagSource = BagSource.detector
    placed_before_boot: bool = False         # v0.2: seeded by cold-start inventory
    fingerprint: dict | None = None          # v0.2: parts + metric size + colors (§5.7)
    # fsm: BagFSM (custody), presence: PresenceFSM  (attached at runtime)


class EntityGraph:
    """In-memory authority for world state; snapshotted every 30 s (§6.11)."""

    def __init__(self):
        self.persons: dict[str, PersonEntity] = {}
        self.bags: dict[str, BagEntity] = {}

    def bind_tracks_to_entities(self, batch):
        raise NotImplementedError("Phase 2: Re-ID rebind on new tracks, else mint")
