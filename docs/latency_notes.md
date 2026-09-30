# Latency notes (Jetson AGX Orin 64GB, MODE_50W, 2026-09-22), parked for M4

Setup:
- Measured on-device, with no network.
- GPU devfreq ramps 306 -> 816 MHz (50W max) within the first call. All numbers below are sustained (816 MHz).
- CPU governor schedutil, max 1.5 GHz.

Time per backbone forward (ms), Qwen3.5-0.8B:

| config | T=16 | T=128 | T=512 | T=1024 | B=16,T=64 |
|---|---|---|---|---|---|
| bf16 eager | 161.7 | 160.6 | 158.1 | 160.8 | 159.4 |
| bf16 CUDA graph | 18.6 | 33.7 | 80.7 | 151.0 | 144.9 |
| int8 weight-only eager | 193.1 | 189.7 | 185.6 | 215.6 | 215.7 |
| int8 weight-only CUDA graph | 74.3 | 87.6 | 138.5 | 209.2 | 203.6 |

## Findings

Kernels:
- GDN runs on fla's Triton `ChunkGatedDeltaRuleFunction` (18 calls per forward). It matches the torch reference
  within 5e-4 and is 9x faster than the reference at T=1024.
- The causal-conv1d CUDA kernel is also active.

Eager mode:
- Eager mode is CPU launch-bound: ~3000 small ops at 40–50 us each on the Orin CPU. This gives a flat ~160 ms.
- The fla chunk op takes 2.1 ms eager vs 0.17 ms inside a CUDA graph at T=128.

CUDA graphs:
- Capture works (fla and causal-conv1d are capturable).
- At T=1024 the cost (151 ms) exceeds the matmul-only estimate (~60 ms at 17 TFLOPs, measured for
  1024x1024x3584).
- Suspected cause: fp32 upcast elementwise passes (RMSNorm, gated norm, decay gates), which are memory-bound.
- Next: try torch.compile per layer. Warm up before graph capture, because capture-time JIT fails.

int8 weight-only:
- Dequantization inside each matmul adds a ~55 ms constant: 0.5B weights are dequantized per call.
- Options: a fused int8 weight-only GEMM (inductor mixed-mm / torchao), W8A8 via torch._int_mm (watch Qwen
  massive activations), or bf16 weights on the Orin (memory is not the constraint: 64 GB).
- bitsandbytes LLM.int8 fails on sm_87 (cuBLASLt status 15). nf4 works but is lossy on 0.8B.

## Plan for M4

1. Bucketed static shapes (B,T) with CUDA graphs for the state encode and the question pass.
2. torch.compile per layer.
3. ONNX/TensorRT. Verify that TRT parses GDN: under torch.export, transformers uses the torch reference path.

System-level option: MAXN power mode plus `sudo jetson_clocks`. This is the operator's decision and is not
changed.
