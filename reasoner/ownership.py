"""Ownership by evidence accumulation (§6.3). The core of the thesis.

Per tick, for each (person i, bag j) within 3 m:
    e_ij = w1*IoU + w2*exp(-d/sigma) + w3*vel_alignment
    E_ij <- lambda*E_ij + e_ij               (decay ~20 s memory)
Assign owner(j)=argmax_i E_ij iff E_best > tau_own AND E_best >= margin*E_second
AND sustained >= confirm_s. Failing the margin -> UNRESOLVED (strict timers).

Bag-birth lookback: a 5 s ring buffer of track states lets a bag born already
stationary get evidence computed retroactively — who was co-located and
decelerating here just before it appeared. Without it, every placed-from-carry
bag starts ownerless.

Phase 2. Pure function of track streams -> unit-test with synthetic trajectories.
"""

from __future__ import annotations


def vel_alignment(v_i, v_j, eps: float = 1e-6):
    """max(0, cos angle) * min(|v_i|,|v_j|) / max(|v_i|,|v_j|, eps). Both must move."""
    raise NotImplementedError("Phase 2")


class OwnershipEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg  # configs/fsm.yaml -> ownership block

    def accumulate(self, graph, ring_buffer):
        raise NotImplementedError("Phase 2: update E_ij matrix, apply assignment rule")
