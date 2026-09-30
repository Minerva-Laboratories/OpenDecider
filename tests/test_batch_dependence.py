import json

from eval.batch_dependence import LEVELS, analyse


def test_isolated_model_has_zero_shift_and_contrast_is_negative(tmp_path):
    for lev in LEVELS:
        with open(tmp_path / f"preds_batchdep_{lev}.jsonl", "w") as f:
            for i in range(20):
                d = {"p10": 0.5, "p90": -0.5}.get(lev, 0.0)          # contrast: looks LESS positive among positives
                f.write(json.dumps({"id": f"bd-{i}:0", "label": i % 2, "logits": [1.0 + d if i % 2 == 0 else -1.0 + d, 0.0]}) + "\n")
    res = analyse(str(tmp_path))
    assert res["n_items"] == 20
    assert abs(res["alone"]["shift"]["mean"]) < 1e-12 and abs(res["p50"]["shift"]["mean"]) < 1e-12
    assert res["slope_p90_minus_p10"]["mean"] < 0 and res["alone"]["acc"]["mean"] == 1.0
