"""Where a training step's time goes: wall time per phase and the top GPU/CPU ops (torch.profiler).

    .venv/bin/python scripts/profile_train.py --config configs/train_2b_clean.yaml --steps 20 --warmup 5

Runs real training steps (same data sampling, losses and backward as opendecider.train) on a throwaway copy of the
model, no checkpoint written. Reports: backbone tokens per step, s/step, GPU busy fraction, and the ops that dominate.
"""
import argparse
import random
import sys
import time

import torch

sys.path.insert(0, "src")
import opendecider.train as T  # noqa: E402
from opendecider.backbone import Backbone, BackboneConfig, checkpoint_backbone_layers  # noqa: E402
from opendecider.guards import gpu_lock, limit_gpu_memory  # noqa: E402
from opendecider.models import ModelConfig, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_2b_clean.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trace", default="", help="optional chrome trace output path")
    ap.add_argument("--stacks", action="store_true", help="also attribute the top ops to Python call sites")
    a = ap.parse_args()
    cfg = T.load_config(a.config, a.set)
    t = cfg["train"]
    with gpu_lock("profile_train"):
        limit_gpu_memory(t["max_gpu_gb"])
        bcfg = BackboneConfig(**cfg["backbone"])
        if t["backbone_weight_quant"] is not None:
            bcfg.weight_quant = t["backbone_weight_quant"]
        bb = Backbone.load(bcfg)
        if t.get("materialize_weights"):
            from opendecider.quant import materialize_int8_
            print(f"materialized {materialize_int8_(bb.lm, bb.dtype, None if t["materialize_weights"] is True else float(t["materialize_weights"]))} int8 linears", flush=True)
            torch.cuda.empty_cache()
        if t["backbone_checkpointing"]:
            checkpoint_backbone_layers(bb)
        model = build_model(bb, ModelConfig(**cfg["model"])).to(bb.device).train()
        if t.get("init_from"):
            T.warm_start(model, t["init_from"])
        T._WRAP_P.update(p=float(t.get("option_wrap_p") or 0), max_opts=int(t.get("max_train_options") or 0),
                         none_p=float(t.get("none_p") or 0), none_drop_p=float(t.get("none_drop_p") or 0))
        files = [t["data"]] if isinstance(t["data"], str) else list(t["data"])
        pools = [T.load_jsonl(f, 5000) for f in files]
        weights = t.get("data_weights")
        params = model.trainable_parameters()
        opt = torch.optim.AdamW(params, lr=1e-6, fused=True)
        rng = random.Random(0)
        k = max(1, int(t.get("grad_accum") or 1))

        def step():
            opt.zero_grad(set_to_none=True)
            for _ in range(k):
                recs = [rng.choice(pools[i]) for i in rng.choices(range(len(pools)), weights=weights, k=t["states_per_step"])] \
                    if weights else rng.sample([r for p in pools for r in p], t["states_per_step"])
                T.step_loss(model, recs, rng, t, t["noise_std_train"])
            torch.nn.utils.clip_grad_norm_(params, t["grad_clip"])
            opt.step()

        for _ in range(a.warmup):
            step()
        torch.cuda.synchronize()
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=acts, record_shapes=a.stacks, with_stack=a.stacks) as prof:
            t0 = time.perf_counter()
            for _ in range(a.steps):
                step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        ev = prof.key_averages()
        gpu_us = sum(e.self_device_time_total for e in ev if e.device_type == torch.autograd.DeviceType.CUDA) \
            if hasattr(ev[0], "self_device_time_total") else float("nan")
        print(f"\n== {a.steps} steps: {wall / a.steps:.2f} s/step wall; GPU kernel time {gpu_us / 1e6 / a.steps:.2f} s/step "
              f"({100 * gpu_us / 1e6 / wall:.0f}% busy); peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        print("\n== top ops by GPU time")
        print(ev.table(sort_by="self_device_time_total", row_limit=18, max_name_column_width=60))
        print("\n== top ops by CPU time (Python-side overhead shows here)")
        print(ev.table(sort_by="self_cpu_time_total", row_limit=18, max_name_column_width=60))
        if a.stacks:
            print("\n== copy_/mul by call site (top 12)")
            print(prof.key_averages(group_by_stack_n=6).table(sort_by="self_cpu_time_total", row_limit=12,
                                                               max_name_column_width=40, max_src_column_width=110))
        if a.trace:
            prof.export_chrome_trace(a.trace)


if __name__ == "__main__":
    main()
