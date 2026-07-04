"""Slow-path service process — strict-priority job queue, profile-scheduled (§5.7).

Crash-isolated OS process. Consumes jobs from Redis, returns evidence to the
reasoner via vlm.results. Aging prevents starvation; stale/superseded jobs are
dropped. Profiles (§3.3): P-offline | P-timeslice (default) | P-dedicated.

Hallucination guard (hard rule): a VLM-proposed new object NEVER mints an entity
directly — it becomes a watch-ROI the fast path must corroborate within 2 s.
Absence claims are accepted only from an unoccluded ROI.

Phase 1 stub (queue only) -> Phase 2 (model + P0/P1/P2 live).
"""

from __future__ import annotations

from enum import IntEnum


class Priority(IntEnum):
    P0_REMOVAL_VERIFY = 0    # presence UNCONFIRMED & custody >= UNATTENDED; SLA < 30 s
    P1_FINGERPRINT = 1       # on every PLACED
    P2_COLDSTART_SCAN = 2    # worker boot / camera reconnect
    P3_RECON_SWEEP = 3       # per camera every 90 s, staggered
    P4_ENRICH = 4            # alert keyframes -> grounded descriptions
    P5_MINING = 5            # fast/slow disagreement crops -> hard examples


def main() -> None:
    raise NotImplementedError("Phase 1: strict-priority queue; Phase 2: load model")


if __name__ == "__main__":
    main()
