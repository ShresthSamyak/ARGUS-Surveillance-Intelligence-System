"""VLMProvider — the model is a plug, not a pillar (§5.7, §12).

The architecture never depends on one vendor's license. LocateAnything-3B is the
research-licensed primary; Florence-2 (MIT) and Grounding DINO / OWLv2 (Apache-2.0)
implement the same `ground()` contract for commercial deployment. The eval suite
(§9.5) re-runs per provider.

LocateAnything-3B specifics (nvidia/LocateAnything-3B, model card verified 2026-07-05):
  * loaded via AutoModel(trust_remote_code=True); custom Parallel-Box-Decoding code
  * detection prompt: "Locate all the instances that matches the following
    description: <CATEGORIES>."
  * returns coords in <box><x1><y1><x2><y2></box> normalized to [0, 1000]
  * ~3B params (~6-7 GB bf16); on the 8 GB 4060 we MUST downscale frames and run it
    ALONE on the GPU (P-timeslice, §3.3). No FlashAttention on Windows -> dense SDPA.

NOTE: the exact processor/generate plumbing below follows the documented signature
but the custom remote code owns preprocessing — VERIFY on first load (smoke test)
and adjust `_encode`/`_generate` if the kwargs differ. Prompt template and box
parsing are exact per the card and should not change.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_DETECT_TMPL = "Locate all the instances that matches the following description: {cats}."


@runtime_checkable
class VLMProvider(Protocol):
    def ground(self, img, prompts: list[str]) -> list[dict]:
        """Ground open-vocab prompts in a PIL image.

        Returns [{"box": [x1, y1, x2, y2] in pixels, "label": str, "score": float}].
        """
        ...


PROVIDERS = ("locateanything3b", "florence2", "grounding_dino", "owlv2")


def parse_boxes(answer: str, width: int, height: int, label: str) -> list[dict]:
    """Parse <box>..</box> tokens (coords in [0,1000]) into pixel-space dicts."""
    out = []
    for m in _BOX_RE.finditer(answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        out.append({
            "box": [x1 / 1000 * width, y1 / 1000 * height,
                    x2 / 1000 * width, y2 / 1000 * height],
            "label": label,
            "score": 1.0,   # LocateAnything does not emit per-box confidence
        })
    return out


class LocateAnythingProvider:
    """nvidia/LocateAnything-3B behind the VLMProvider contract.

    max_side downscales the long edge before inference — required to fit the 8 GB
    card and speeds decoding. Grounding is per-prompt (one generate() per category)
    so each returned box carries a definite label; callers batching many prompts
    should mind the added latency (0.5-2 s each).
    """

    def __init__(self, model_id: str = "nvidia/LocateAnything-3B", device: str = "cuda",
                 dtype: str = "bfloat16", max_side: int = 1536,
                 generation_mode: str = "hybrid", max_new_tokens: int = 8192):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_side = max_side
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tok = None
        self._proc = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        td = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(self.dtype, "auto")
        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, dtype=td).to(self.device).eval()
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def _downscale(self, img):
        w, h = img.size
        s = self.max_side / max(w, h)
        if s < 1.0:
            img = img.resize((round(w * s), round(h * s)))
        return img

    def _detect_one(self, img, category: str) -> str:
        """Run one detection prompt; return the raw model answer string.

        LA's custom processor (processing_locateanything.py) wants images as a LIST
        and a chat-formatted prompt with an <image-1> placeholder built by
        py_apply_chat_template. Its model.generate() asserts use_cache=True and
        batch_size==1, and converts a numpy image_grid_hws onto the pixel device.
        """
        import torch

        prompt = _DETECT_TMPL.format(cats=category)
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._proc.py_apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._proc(images=[img], text=text, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        image_grid_hws = inputs["image_grid_hws"]   # numpy; generate() moves it to device
        with torch.inference_mode():
            out = self._model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_hws=image_grid_hws,
                tokenizer=self._tok,
                max_new_tokens=self.max_new_tokens,
                generation_mode=self.generation_mode,
                use_cache=True,
            )
        return self._tok.decode(out[0], skip_special_tokens=False)

    def ground(self, img, prompts: list[str]) -> list[dict]:
        if self._model is None:
            self.load()
        orig_w, orig_h = img.size
        img = self._downscale(img.convert("RGB"))
        results: list[dict] = []
        for category in prompts:
            answer = self._detect_one(img, category)
            # parse against ORIGINAL size (coords are normalized, so scale-invariant)
            results.extend(parse_boxes(answer, orig_w, orig_h, label=category))
        return results


def load_provider(name: str, **kwargs) -> VLMProvider:
    name = name.lower()
    if name in ("locateanything3b", "locateanything", "locate_anything"):
        return LocateAnythingProvider(**kwargs)
    raise NotImplementedError(
        f"provider {name!r} not implemented yet — LocateAnything-3B is the primary; "
        "florence2 / grounding_dino / owlv2 are license-clean swaps for later (§12)."
    )
