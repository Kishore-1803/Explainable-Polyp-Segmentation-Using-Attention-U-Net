"""Measures inference latency / throughput for the replication-study models."""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from variants import build_model          # noqa: E402
from paper_model import PaperAttentionUNet  # noqa: E402


@torch.no_grad()
def bench(model, device, n=100, warm=20):
    model.eval().to(device)
    x = torch.randn(1, 3, 256, 256, device=device)
    for _ in range(warm):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    return dt * 1000, 1.0 / dt


def main():
    out = {}
    for name, dev in [("cuda", "cuda"), ("cpu", "cpu")]:
        if name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(dev)
        for label, model in [("replication", PaperAttentionUNet(base=64)),
                             ("se_ds", build_model("se_ds", base=64))]:
            ms, fps = bench(model, device, n=100 if dev == "cuda" else 20)
            params = sum(p.numel() for p in model.parameters())
            out[f"{label}_{name}"] = {"ms_per_frame": round(ms, 2),
                                      "fps": round(fps, 2), "params": params}
            print(f"{label:14s} {name:5s}  {ms:7.2f} ms/frame   {fps:6.2f} FPS   "
                  f"{params:,} params")
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

    res = ROOT / "results" / "latency_benchmark.json"
    json.dump({"device_name": torch.cuda.get_device_name(0)
               if torch.cuda.is_available() else "cpu", "results": out},
              open(res, "w"), indent=2)
    print(f"\nSaved {res}")


if __name__ == "__main__":
    main()
