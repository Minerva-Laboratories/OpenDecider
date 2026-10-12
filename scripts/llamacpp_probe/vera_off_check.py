"""Does the 9B's VeRA adapter matter? Typed-decisions test (first 100 cases, 5 questions) and Banking77 8-way with the
adapters on vs removed, int8 weights, eager path (scripts/bench_quant.py helpers)."""
import json, sys
sys.path.insert(0, "src"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from bench_quant import typed, public
from eval.typed_decisions import load_cases
from opendecider.checkpoint import load_model
model, _ = load_model("runs/x9b-clean3/model.pt")
rows = load_cases("test", 100)
res = {"on": {"typed": typed(model, rows), "banking77_8way": public(model, "data/public/banking77_8way.jsonl")}}
print(res, flush=True)
model.vera.remove()
res["off"] = {"typed": typed(model, rows), "banking77_8way": public(model, "data/public/banking77_8way.jsonl")}
print(res, flush=True)
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "vera_off_typed.json", "w"), indent=1)
