"""Smallest end-to-end test of a released checkpoint (downloads the pinned backbone on first run).

    python examples/quickstart.py [checkpoints/opendecider-2b] [backbone variant]
"""
import json
import sys

from opendecider.hub import load_decider

model = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/opendecider-2b"
dec = load_decider(model, backbone=sys.argv[2] if len(sys.argv) > 2 else None)
request = {
    "state": {"ticket": "I was charged twice for my March invoice and want my money back.",
              "customer": {"plan": "pro", "months": 14}},
    "questions": {
        "route": {"type": "choice", "prompt": "Which team handles this?",
                  "options": ["billing", "technical", {"name": "refund", "description": "customer asks for money back"},
                              "other"]},
        "urgent": {"type": "noul", "prompt": "Does this need a reply within the hour?"},
        "severity": {"type": "score", "prompt": "How severe is it?", "levels": ["low", "medium", "high", "critical"]},
    },
}
out = dec.decide(request)
for name, a in out["answers"].items():
    probs = ", ".join(f"{k}: {v:.2f}" for k, v in a["probs"].items())
    print(f"{name:9s} -> {a['value']:9s}  ({probs})  none: {a.get('none', 0):.2f}")
print("timing ms:", json.dumps({k: round(v, 1) for k, v in out["timing_ms"].items()}))
