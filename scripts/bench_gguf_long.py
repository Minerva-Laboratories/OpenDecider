"""Long-context memory and latency: Mercurius-1-4B vs Qwen3.5-4B GGUFs on the llama.cpp fork (Jetson AGX Orin).
Phase 1 (memory, every context size): start llama-server with -c CTX, record llama.cpp's GPU buffers and the real
footprint (drop in MemAvailable; weights loaded with --no-mmap so they count), plus the footprint while a short
prompt runs. Phase 2 (latency, filled contexts): prompt of CTX - 256 tokens, 32 greedy tokens; prefill time,
generation speed and the peak footprint during the run. Results are appended to --out after every step.
    .venv/bin/python scripts/bench_gguf_long.py --phase memory --out runs/gguf_long.json
    .venv/bin/python scripts/bench_gguf_long.py --phase latency --ctx 65536 131072 262144 --out runs/gguf_long.json"""
import argparse, json, os, re, subprocess, threading, time, urllib.request
from transformers import AutoTokenizer

BIN = "/mnt/ext/opendecider/llama-tq/tom/build/bin/llama-server"
G = "/mnt/ext/opendecider/gguf"
CONFIGS = {
    "mercurius IQ4_NL": ("mercurius-1-4b-IQ4_NL.gguf", ["-fa", "on"]),
    "qwen IQ4_NL + f16 KV": ("qwen3.5-4b-IQ4_NL-pure.gguf", ["-fa", "on"]),
    "qwen IQ4_NL + turbo4 KV": ("qwen3.5-4b-IQ4_NL-pure.gguf", ["-fa", "on", "-ctk", "turbo4", "-ctv", "turbo4"]),
}
BUF = re.compile(r"CUDA0 (model|KV|RS|compute) buffer size = +([\d.]+) MiB")
PORT = 18093


def avail_mib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024


def post(body, timeout=6 * 3600):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def start(f, extra, ctx, log):
    subprocess.run(["sync"]); time.sleep(2)
    base = avail_mib()
    p = subprocess.Popen([BIN, "-m", f"{G}/{f}", "-ngl", "99", "-c", str(ctx), "-b", "2048", "-ub", "512", "--no-mmap",
                          "--host", "127.0.0.1", "--port", str(PORT), "-np", "1", "--cache-ram", "0",
                          "--ctx-checkpoints", "0", "-lv", "4", *extra], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if p.poll() is not None:
            raise RuntimeError(f"server exited ({p.returncode})")
        try:
            if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read():
                break
        except Exception:
            time.sleep(1)
    time.sleep(3)
    return p, base


class Peak:
    def __init__(self):
        self.low = avail_mib(); self.stop = threading.Event()
        self.t = threading.Thread(target=self.run); self.t.start()

    def run(self):
        while not self.stop.is_set():
            self.low = min(self.low, avail_mib()); time.sleep(0.2)

    def done(self):
        self.stop.set(); self.t.join(); return self.low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["memory", "latency"], required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[16384, 65536, 131072, 262144, 524288])
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="runs/gguf_long.json")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("/mnt/ext/opendecider/models/Qwen__Qwen3.5-4B@851bf6e806ef")
    text = "".join(open(p).read() for p in ("docs/RESULTS.md", "README.md", "docs/LANDSCAPE.md", "CLAUDE.md"))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    need = max(a.ctx)
    ids = (ids * (need // len(ids) + 1))[:need]            # repeated text: the cost does not depend on the content
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name, (f, extra) in CONFIGS.items():
        if a.only and a.only not in name:
            continue
        for ctx in a.ctx:
            key = f"{name} @ {ctx}"
            r = res.get(key, {})
            log = open(f"/tmp/bench_long_{PORT}.log", "w")
            try:
                p, base = start(f, extra, ctx, log)
            except Exception as e:
                r[f"{a.phase}_error"] = str(e); res[key] = r
                json.dump(res, open(a.out, "w"), indent=1); print(key, "ERROR", e, flush=True)
                continue
            try:
                bufs = {}
                for m in BUF.finditer(open(log.name).read()):
                    bufs[m.group(1)] = round(bufs.get(m.group(1), 0) + float(m.group(2)), 1)
                bufs["total"] = round(sum(bufs.values()), 1)
                r["buffers_mib"] = bufs
                r["footprint_loaded_mib"] = round(base - avail_mib())
                pk = Peak()
                if a.phase == "memory":
                    post({"prompt": ids[:512], "n_predict": 8, "temperature": 0})
                    r["footprint_short_run_mib"] = round(base - pk.done())
                else:
                    n = ctx - 256
                    o = post({"prompt": ids[:n], "n_predict": 32, "temperature": 0, "top_k": 1, "ignore_eos": True,
                              "cache_prompt": False})
                    t = o["timings"]
                    r.update({"prompt_tokens": n, "prefill_s": round(t["prompt_ms"] / 1000, 1),
                              "prefill_tok_s": round(t["prompt_per_second"], 1),
                              "gen_tok_s": round(t["predicted_per_second"], 2),
                              "footprint_peak_mib": round(base - pk.done())})
                res[key] = r
                json.dump(res, open(a.out, "w"), indent=1)
                print(key, json.dumps(r), flush=True)
            except Exception as e:
                r[f"{a.phase}_error"] = str(e); res[key] = r
                json.dump(res, open(a.out, "w"), indent=1); print(key, "ERROR", e, flush=True)
            finally:
                p.terminate(); p.wait(); time.sleep(5)
                # llama.cpp's breakdown at load and at exit: self = model + context + compute buffers; growth of
                # "unaccounted" device memory between the two = pool allocations made while serving (temporary
                # dequantized K/V, expanded K/V, workspaces), which the buffer report does not include
                rows = re.findall(r"CUDA0 \(Orin\)\s*\|\s*(\d+) = (\d+) \+ \((\d+) =.*\) \+\s*(\d+)", open(log.name).read())
                if len(rows) >= 2:
                    self_mib, un0, un1 = int(rows[0][2]), int(rows[0][3]), int(rows[-1][3])
                    r["device_self_mib"] = self_mib
                    r[f"{a.phase}_pool_growth_mib"] = un1 - un0
                    r[f"{a.phase}_device_peak_est_mib"] = self_mib + max(0, un1 - un0)
                    res[key] = r
                    json.dump(res, open(a.out, "w"), indent=1)
                    print(key, "device self", self_mib, "MiB, pool growth", un1 - un0, "MiB", flush=True)


if __name__ == "__main__":
    main()
