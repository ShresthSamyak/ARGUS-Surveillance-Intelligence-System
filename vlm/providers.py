"""VLMProvider — the model is a plug, not a pillar (§5.7, §12).

The architecture never depends on one vendor's license. LocateAnything-3B is
the research-licensed primary; Florence-2 (MIT) and Grounding DINO / OWLv2
(Apache-2.0) implement the same contract for commercial deployment. The eval
suite (§9.5) re-runs per provider.

Phase 1: interface + registry. Phase 2: load the primary in P-timeslice.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VLMProvider(Protocol):
    def ground(self, img, prompts: list[str]) -> list[dict]:
        """Return [{box, label, score}] for open-vocabulary prompts."""
        ...


PROVIDERS = ("locateanything3b", "florence2", "grounding_dino", "owlv2")


def load_provider(name: str) -> VLMProvider:
    raise NotImplementedError("Phase 2: instantiate the named provider")
