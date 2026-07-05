"""LocateAnything-3B smoke test — does it load + ground + fit in 8 GB? (§5.7)

Non-destructive feasibility probe. Loads nvidia/LocateAnything-3B, grounds a few
bag prompts on real sampled frames, and reports boxes + timing + peak VRAM. Prints
full tracebacks so we can adjust the provider if the trust_remote_code API differs
from the documented signature.

First run downloads ~6-7 GB (uses the HF token already in your cache). It does NOT
change any installed packages. If it errors importing `decord`/`lmdb`, install them
(`pip install lmdb decord`) and rerun — those are LA data-loader deps, not inference.

    python scripts/vlm_smoke.py
    python scripts/vlm_smoke.py --max-side 1024   # if VRAM is tight
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pick_frames() -> list[str]:
    """A bag-rich frame (library) + a sparse one (corridor), else any frames."""
    picks = []
    for cam in ("cam06", "cam01", "cam04"):
        g = sorted(glob.glob(str(ROOT / "data" / "frames" / cam / "*.jpg")))
        if g:
            picks.append(g[len(g) // 2])
    if not picks:
        picks = sorted(glob.glob(str(ROOT / "data" / "frames" / "**" / "*.jpg"),
                                 recursive=True))[:2]
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nvidia/LocateAnything-3B")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()

    import torch
    from PIL import Image

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"numpy {__import__('numpy').__version__} | "
          f"transformers {__import__('transformers').__version__}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    frames = pick_frames()
    if not frames:
        sys.exit("No sampled frames found under data/frames/ — run sample_frames.py first.")
    print("test frames:", [Path(f).name for f in frames])

    from vlm.providers import LocateAnythingProvider
    prov = LocateAnythingProvider(model_id=args.model, device=dev, dtype=args.dtype,
                                  max_side=args.max_side, max_new_tokens=args.max_new_tokens)

    print("\n[1/2] loading model (first run downloads ~7 GB)...")
    t0 = time.time()
    try:
        prov.load()
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nLOAD FAILED — paste this whole traceback. If it's ModuleNotFoundError "
              "for decord/lmdb: `pip install lmdb decord` and rerun.")
        sys.exit(1)
    if dev == "cuda":
        torch.cuda.synchronize()
        print(f"  loaded in {time.time()-t0:.1f}s | VRAM alloc "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB")

    prompts = ["backpack", "handbag", "suitcase", "person"]
    print(f"\n[2/2] grounding {prompts} ...")
    for fp in frames:
        img = Image.open(fp).convert("RGB")
        t0 = time.time()
        try:
            res = prov.ground(img, prompts)
            print(f"\n{Path(fp).name} ({img.size[0]}x{img.size[1]}): "
                  f"{len(res)} boxes in {time.time()-t0:.1f}s")
            for r in res[:25]:
                b = r["box"]
                print(f"  {r['label']:10} [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
        except Exception:
            import traceback
            traceback.print_exc()
            print("GROUND FAILED — paste the traceback; the provider's generate/processor "
                  "call likely needs adjusting to LA's custom API.")
            break
        if dev == "cuda":
            print(f"  VRAM peak so far: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if dev == "cuda":
        print(f"\n=== PEAK VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB / "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB ===")


if __name__ == "__main__":
    main()
