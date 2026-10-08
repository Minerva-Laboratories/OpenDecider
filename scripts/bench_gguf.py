"""Mercurius-1-4B vs Qwen3.5-4B GGUFs on llama.cpp (the Mercurius fork): memory and latency at equal weight bits.
Per config: one llama-server (-c CTX, all layers on GPU); llama.cpp's own buffer sizes from the load log; the real
footprint as the drop in MemAvailable (Orin: unified memory; unreliable with mmap'd weights, kept for reference); prompt
(prefill) and generation speed for prompts of 1k/4k/15k tokens + 64 greedy tokens. One slot, no host-RAM prompt
cache, no context checkpoints (both would add memory unrelated to the model).
    .venv/bin/python scripts/bench_gguf.py --out runs/gguf_bench.json"""
import argparse, json, os, re, subprocess, threading, time, urllib.request
from transformers import AutoTokenizer

BIN = "/mnt/ext/opendecider/llama-tq/tom/build/bin/llama-server"
G = "/mnt/ext/opendecider/gguf"
CONFIGS = [  # name, file, extra args
    ("mercurius NF4 (TQ 4-bit latent KV)", "mercurius-1-4b-NF4.gguf", ["-fa", "on"]),
    ("qwen NF4 + f16 KV", "qwen3.5-4b-NF4-pure.gguf", ["-fa", "on"]),
    ("qwen NF4 + q8_0 KV", "qwen3.5-4b-NF4-pure.gguf", ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]),
    ("qwen NF4 + turbo4 KV", "qwen3.5-4b-NF4-pure.gguf", ["-fa", "on", "-ctk", "turbo4", "-ctv", "turbo4"]),
    ("mercurius IQ4_NL (TQ 4-bit latent KV)", "mercurius-1-4b-IQ4_NL.gguf", ["-fa", "on"]),
    ("qwen IQ4_NL + f16 KV", "qwen3.5-4b-IQ4_NL-pure.gguf", ["-fa", "on"]),
    ("qwen IQ4_NL + turbo4 KV", "qwen3.5-4b-IQ4_NL-pure.gguf", ["-fa", "on", "-ctk", "turbo4", "-ctv", "turbo4"]),
]
BUF = re.compile(r"CUDA0 (model|KV|RS|compute) buffer size = +([\d.]+) MiB")          # GPU buffers only


def avail_mib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024


def post(port, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--lens", type=int, nargs="+", default=[1000, 4000, 15000])
    ap.add_argument("--text", default="docs/RESULTS.md")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/gguf_bench.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("/mnt/ext/opendecider/models/Qwen__Qwen3.5-4B@851bf6e806ef")
    ids = tok(open(a.text).read() + open("README.md").read(), add_special_tokens=False)["input_ids"]
    assert len(ids) >= max(a.lens) + 1, len(ids)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    port = 18090
    for name, f, extra in CONFIGS:
        if a.only and a.only not in name:
            continue
        base = avail_mib()
        log = open(f"/tmp/bench_gguf_{port}.log", "w")
        p = subprocess.Popen([BIN, "-m", f"{G}/{f}", "-ngl", "99", "-c", str(a.ctx), "-b", "2048", "-ub", "512",
                              "--host", "127.0.0.1", "--port", str(port), "-np", "1", "--cache-ram", "0",
                              "--ctx-checkpoints", "0", "-lv", "4", *extra],
                             stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                try:
                    if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read():
                        break
                except Exception:
                    time.sleep(1)
            time.sleep(2)
            loaded = base - avail_mib()
            bufs = {}
            for m in BUF.finditer(open(log.name).read()):
                bufs[m.group(1)] = round(bufs.get(m.group(1), 0) + float(m.group(2)), 1)
            bufs["total"] = round(sum(bufs.values()), 1)
            r = {"file": f, "file_gib": round(os.path.getsize(f"{G}/{f}") / 2**30, 3), "buffers_mib": bufs,
                 "footprint_loaded_mib": round(loaded)}
            low = [avail_mib()]
            stop = threading.Event()

            def watch():
                while not stop.is_set():
                    low[0] = min(low[0], avail_mib()); time.sleep(0.1)
            th = threading.Thread(target=watch); th.start()
            post(port, {"prompt": ids[:256], "n_predict": 8, "temperature": 0})          # warm-up
            for n in a.lens:
                o = post(port, {"prompt": ids[:n], "n_predict": 64, "temperature": 0, "top_k": 1,
                                "ignore_eos": True, "cache_prompt": False})
                t = o["timings"]
                r[f"prefill_s_{n}"] = round(t["prompt_ms"] / 1000, 3)
                r[f"prefill_tok_s_{n}"] = round(t["prompt_per_second"], 1)
                r[f"gen_tok_s_{n}"] = round(t["predicted_per_second"], 2)
                print(name, n, r[f"prefill_s_{n}"], "s prefill,", r[f"gen_tok_s_{n}"], "tok/s gen", flush=True)
            stop.set(); th.join()
            r["footprint_peak_mib"] = round(base - low[0])
            res[name] = r
            print(name, json.dumps(r), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        finally:
            p.terminate(); p.wait()
            time.sleep(3)


if __name__ == "__main__":
    main()
