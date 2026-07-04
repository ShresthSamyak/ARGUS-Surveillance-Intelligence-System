"""OSNet Re-ID embeddings + EmbeddingBank (§5.4). Phase 1/2.

EmbeddingBank is per-ENTITY (entities outlive tracks): K=10 diverse 512-d
L2-normalized slots. admit() keeps views diverse (cosine < 0.92) rather than
storing 10 near-copies; match() returns max cosine over slots.
"""

from __future__ import annotations


class EmbeddingBank:
    def __init__(self, slots: int = 10, admit_below_cos: float = 0.92, ema_alpha: float = 0.9):
        self.slots = slots
        self.admit_below_cos = admit_below_cos
        self.ema_alpha = ema_alpha
        self._vecs: list = []  # np.ndarray (512,), L2-normalized

    def admit(self, e) -> None:
        raise NotImplementedError("Phase 2: diverse-admit / EMA-update nearest slot")

    def match(self, e) -> float:
        raise NotImplementedError("Phase 2: return max cosine over slots")


class ReID:
    """OSNet-x1.0 embedder. ~1-2 ms / 16 crops; called on new/refresh tracks only."""

    def __init__(self, model: str = "osnet_x1_0"):
        self.model = model
        self._net = None

    def embed(self, crops):  # noqa: ARG002
        raise NotImplementedError("Phase 1: batch OSNet inference -> (N, 512)")
