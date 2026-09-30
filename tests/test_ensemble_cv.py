import json

import numpy as np

from eval.ensemble_cv import main


def test_ensemble_beats_either_source_when_they_are_complementary(tmp_path):
    rng = np.random.default_rng(0)
    a, b = tmp_path / "a", tmp_path / "b"; a.mkdir(); b.mkdir()
    with open(a / "preds_x.jsonl", "w") as fa, open(b / "preds_x.jsonl", "w") as fb:
        for i in range(600):
            y = int(rng.integers(0, 4))
            za = rng.normal(0, 1, 4); zb = rng.normal(0, 1, 4)
            za[y] += 1.2 if i % 2 else 0.0          # A informative on odd items, B on even ones
            zb[y] += 0.0 if i % 2 else 1.2
            for f, z in ((fa, za), (fb, zb)):
                f.write(json.dumps({"id": f"q{i}:0", "label": y, "logits": z.tolist(), "options": list("abcd")}) + "\n")
    main(["--a", str(a), "--b", str(b), "--name", ""])


def test_ensemble_cv_runs_and_improves(tmp_path, capsys):
    test_ensemble_beats_either_source_when_they_are_complementary(tmp_path)
    out = capsys.readouterr().out
    acc = {l.split()[0] + (l.split()[1] if l.split()[1] in ("raw", "temp") else ""): float(l.split("acc")[1].split()[0])
           for l in out.splitlines() if " acc " in l}
    assert acc["ensemble"] > max(acc["Araw"], acc["Braw"])
