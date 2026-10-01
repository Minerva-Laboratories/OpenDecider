"""Serve a checkpoint over HTTP (POST /v1/decide, /v1/calibrate, /v1/explain).

    python -m opendecider.serve --model checkpoints/opendecider-2b [--backbone int8|w8|nf4|awq] [--port 8000]

--model is a local checkpoint directory or a Hugging Face repo id. The backbone is downloaded on first use.
"""
import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="checkpoints/opendecider-2b")
    ap.add_argument("--backbone", default=None, help="backbone variant from the checkpoint's config.json")
    ap.add_argument("--kv-quant", default=None, choices=["none", "int8", "int4"])
    ap.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    import uvicorn
    from .api import create_app
    from .hub import load_decider
    dec = load_decider(a.model, backbone=a.backbone, device=a.device, kv_quant=a.kv_quant)
    uvicorn.run(create_app(dec), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
