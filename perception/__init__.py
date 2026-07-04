"""Layer 1 — Perception (per camera). Fast path: decode -> detect -> track -> embed.

GPU-bound; one process per camera. Publishes compact track updates to Redis.
See §5. Implemented in Phase 1.
"""
