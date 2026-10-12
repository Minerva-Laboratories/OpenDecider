"""Serve a checkpoint over HTTP (POST /v1/decide, /v1/calibrate, /v1/explain, /v1/feedback; GET / playground).

    python -m opendecider.serve --model checkpoints/opendecider-2b [--backbone int8|w8|nf4|awq] [--port 8000]
    python -m opendecider.serve --model checkpoints/opendecider-9b --host 0.0.0.0 \\
        --api-key-file ~/.config/opendecider/api_key --log-dir runs/server        # on the local network

--model is a local checkpoint directory or a Hugging Face repo id. The backbone is downloaded on first use.
Binding anything other than 127.0.0.1 requires an API key file. --log-dir keeps every /v1/decide request and the
feedback sent for it (the request states are stored there: keep it private).
Concurrent /v1/decide requests share one model: requests that arrive while the GPU is busy are batched together
(opendecider.api; --max-batch, --max-batch-tokens).
"""
import argparse
import os


def _build(a):
    from .api import create_app
    from .hub import load_decider
    key = open(os.path.expanduser(a["api_key_file"])).read().strip() if a["api_key_file"] else None
    if a.get("llamacpp_gguf"):
        os.environ["OPENDECIDER_LLAMACPP_GGUF"] = os.path.abspath(a["llamacpp_gguf"])
    dec = load_decider(a["model"], backbone=a["backbone"], device=a["device"], kv_quant=a["kv_quant"])
    if a["warmup"]:
        dec.decide({"state": "warm-up", "questions": {"w": {"type": "noul", "prompt": "Is this a warm-up?"}}})
    info = {"model": os.path.basename(os.path.normpath(a["model"])), "backbone": ("llama.cpp " + os.path.basename(a["llamacpp_gguf"])) if a.get("llamacpp_gguf")
                else (a["backbone"] or "default"),
            "kv_quant": a["kv_quant"] or "default", "max_batch": a["max_batch"]}
    return create_app(dec, api_key=key, log_dir=a["log_dir"], info=info, max_batch=a["max_batch"],
                      max_batch_tokens=a["max_batch_tokens"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="checkpoints/opendecider-2b")
    ap.add_argument("--backbone", default=None, help="backbone variant from the checkpoint's config.json")
    ap.add_argument("--kv-quant", default=None, choices=["none", "int8", "int4"])
    ap.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    ap.add_argument("--llamacpp-gguf", default=None,
                    help="run the backbone in llama.cpp from this GGUF (branched-row models such as the 9B; "
                         "opendecider.llamacpp, deploy/llamacpp/build.sh)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key-file", default=None, help="file holding the API key (required off localhost)")
    ap.add_argument("--log-dir", default=None, help="log requests and feedback here (enables /v1/feedback)")
    ap.add_argument("--warmup", action="store_true", help="run one request before accepting traffic")
    ap.add_argument("--max-batch", type=int, default=16, help="most concurrent requests run as one batch (1: off)")
    ap.add_argument("--max-batch-tokens", type=int, default=16384,
                    help="most state tokens in one batch (padded); a larger request runs alone")
    a = ap.parse_args(argv)
    if a.api_key_file:
        if len(open(os.path.expanduser(a.api_key_file)).read().strip()) < 16:
            ap.error("API key must be at least 16 characters")
    elif a.host not in ("127.0.0.1", "localhost", "::1"):
        ap.error("binding to a network address needs --api-key-file")
    import uvicorn
    uvicorn.run(_build(vars(a)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
