"""Per-layer GEMM speed on the real 2B weight shapes: which quantized kernel is fastest on this GPU?

    .venv/bin/python scripts/bench_gemm.py

Kernels: bf16 cuBLAS (reference), int8 weight-only as currently shipped (dequantize, then cuBLAS), GemLite A16W8
with default and autotuned ("max") configs, GemLite A8W8 (dynamic per-token int8 activations, int8 tensor cores),
GemLite A16W4 and bitsandbytes NF4. M = tokens per call (row pass ~64-400, state pass up to 16k).
"""
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "src")
from opendecider.guards import gpu_lock  # noqa: E402
from opendecider.quant import Int8Linear, gemlite_linear  # noqa: E402

SHAPES = [(2048, 6144), (6144, 2048), (2048, 2048)]          # (K, N): qkv/gate/up, down, out/o
MS = [64, 512, 2048, 8192]


def bench(f, n=10):
    for _ in range(3):
        f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t) / n


def main():
    import gemlite
    from gemlite.helper import A8W8_int8_dynamic
    dev = "cuda"
    out = {}
    with gpu_lock("bench_gemm"), torch.no_grad():
        for K, N in SHAPES:
            lin = nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)
            nn.init.normal_(lin.weight, std=0.02)
            kern = {"bf16 cuBLAS": lin, "int8 dequant+cuBLAS (current)": Int8Linear(lin)}
            gemlite.set_autotune(False)
            kern["GemLite A16W8 default"] = gemlite_linear(lin, 8)
            kern["GemLite A16W4 default"] = gemlite_linear(lin, 4)
            try:
                import bitsandbytes as bnb
                l4 = bnb.nn.Linear4bit(K, N, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
                l4.weight = bnb.nn.Params4bit(lin.weight.data.float().cpu(), requires_grad=False, quant_type="nf4")
                kern["bitsandbytes NF4"] = l4.to(dev)
            except Exception as e:                                   # pragma: no cover
                print("bnb:", e)
            X = {M: torch.randn(M, K, device=dev, dtype=torch.bfloat16) for M in MS}   # inputs outside the timing
            res = {}
            for name, m in kern.items():
                res[name] = [bench(lambda m=m, M=M: m(X[M])) for M in MS]
            gemlite.set_autotune("max")
            for name, mk in (("GemLite A16W8 autotuned", lambda: gemlite_linear(lin, 8)),
                             ("GemLite A16W4 autotuned", lambda: gemlite_linear(lin, 4)),
                             ("GemLite A8W8 int8 autotuned",          # from_linear may quantize in place: use a copy
                              lambda: A8W8_int8_dynamic(device=dev, dtype=torch.bfloat16).from_linear(copy.deepcopy(lin)))):
                try:
                    m = mk()
                    res[name] = [bench(lambda m=m, M=M: m(X[M])) for M in MS]
                except Exception as e:
                    print(name, "FAILED:", type(e).__name__, str(e)[:200])
            out[f"{K}x{N}"] = res
            print(f"== K={K} N={N}   ms at M = {MS}", flush=True)
            for name, v in res.items():
                print(f"   {name:32s} " + "  ".join(f"{x:7.3f}" for x in v), flush=True)
    os.makedirs("runs/deploy", exist_ok=True)
    json.dump({"M": MS, "results": out}, open("runs/deploy/gemm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
