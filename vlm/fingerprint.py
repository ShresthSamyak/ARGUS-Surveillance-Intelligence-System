"""Attribute fingerprints — part grounding + metric size + colors (§5.7, v0.2).

On every PLACED event, ground bag parts (zipper/logo/sticker/strap/wheels/handle/
side pocket/bottle) + dominant colors + metric dimensions via homography. Stored
on the bag entity. Used for twin disambiguation, handoff verification, long-gap
bag re-association:

    match(a, b) = 0.40*cos(emb) + 0.30*Jaccard(parts) + 0.20*size_sim_m + 0.10*color_sim

The 0.20 metric-size term is the discriminator embeddings lack: a 62 cm suitcase
and a 45 cm backpack can be twins to OSNet, never to a tape measure.

Phase 2.
"""

from __future__ import annotations

PART_PROMPTS = ("zipper", "logo", "sticker", "strap", "wheels", "handle",
                "side pocket", "bottle")


def match(a: dict, b: dict, weights=(0.40, 0.30, 0.20, 0.10)) -> float:
    """Weighted emb / parts-Jaccard / metric-size / color similarity."""
    raise NotImplementedError("Phase 2")
