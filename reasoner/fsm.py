"""Bag custody FSM + presence FSM + additive risk (§6.5, §6.6, §6.8).

Two ORTHOGONAL state machines per bag — conflating them is how v0.1 silently
mishandled removal-under-occlusion:
  * CustodyFSM  — WHO is responsible: CARRIED->ATTENDED->OWNER_NEAR->OWNER_AWAY
                  ->UNATTENDED->ABANDONED (+ IN_TRANSFER on non-owner pickup).
  * PresenceFSM — is it PHYSICALLY still there: CONFIRMED->OCCLUDED->UNCONFIRMED
                  ->(VLM verify)->REMOVED. REMOVED while custody in
                  {UNATTENDED, ABANDONED} with no PICKED_UP -> REMOVED_UNSEEN
                  theft alert.

Hysteresis: escalation guards must hold dwell_s (3 s); de-escalation is
immediate but risk decays smoothly. Risk is additive and human-readable by
design (§6.8) — "62 = 60 (UNATTENDED) + 8 (2x t1) + (-10) (baggage)".

Phase 2. Pure functions -> tests/test_fsm.py drives scripted (t, x, y) sequences.
"""

from __future__ import annotations

from enum import Enum


class Custody(str, Enum):
    CARRIED = "CARRIED"
    ATTENDED = "ATTENDED"
    OWNER_NEAR = "OWNER_NEAR"
    OWNER_AWAY = "OWNER_AWAY"
    UNATTENDED = "UNATTENDED"
    ABANDONED = "ABANDONED"
    IN_TRANSFER = "IN_TRANSFER"


class Presence(str, Enum):
    CONFIRMED = "CONFIRMED"
    OCCLUDED = "OCCLUDED"
    UNCONFIRMED = "UNCONFIRMED"
    REMOVED = "REMOVED"


class CustodyFSM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = Custody.CARRIED

    def step(self, obs) -> Custody:
        raise NotImplementedError("Phase 2")


class PresenceFSM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = Presence.CONFIRMED

    def step(self, obs) -> Presence:
        raise NotImplementedError("Phase 2")


def compute_risk(bag, cfg: dict) -> tuple[float, list[str]]:
    """Return (risk 0-100, human-readable term breakdown for the dashboard)."""
    raise NotImplementedError("Phase 2")
