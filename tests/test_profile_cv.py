import json

import numpy as np

from eval.profile_cv import main


def test_profile_cv_fixes_overconfidence_out_of_sample(tmp_path):
    rng = np.random.default_rng(0)
    with open(tmp_path / "preds_toy.jsonl", "w") as f:
        for i in range(400):
            y = int(rng.integers(0, 3))
            z = rng.normal(0, 1, 3); z[y] += 1.0
            f.write(json.dumps({"id": str(i), "label": y, "logits": (z * 6).tolist(),       # 6x overconfident
                                "options": ["a", "b", "c"]}) + "\n")
    main([str(tmp_path)])
    r = json.load(open(tmp_path / "profile_cv.json"))["toy"]
    assert r["temperature"]["ece"]["value"] < r["raw"]["ece"]["value"] / 2
    assert r["temperature"]["nll"]["value"] < r["raw"]["nll"]["value"]
    assert "vector" in r


def test_profile_cv_no_vector_bias_when_option_names_differ(tmp_path):
    with open(tmp_path / "preds_mc.jsonl", "w") as f:
        for i in range(40):
            f.write(json.dumps({"id": str(i), "label": i % 2, "logits": [float(i % 3), 0.0],
                                "options": [f"x{i}", f"y{i}"]}) + "\n")
    main([str(tmp_path)])
    assert "vector" not in json.load(open(tmp_path / "profile_cv.json"))["mc"]
