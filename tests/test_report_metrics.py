"""Metrics from eval/report_metrics.py on hand-computed cases (cross-checked against scikit-learn once, on 300 random
cases with ties; scikit-learn is not a dependency)."""
import numpy as np
import pytest

from eval.report_metrics import _ap, _auc, _recall_at_fpr, binary_metrics, class_metrics, roc_points

S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
Y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)


def test_binary_metrics_by_hand():
    m, cm = binary_metrics(S, Y)                     # predicted yes: 0.9..0.5 -> TP 3, FP 2, FN 1, TN 4
    assert cm == {"tp": 3, "fn": 1, "fp": 2, "tn": 4}
    assert m["precision"] == pytest.approx(3 / 5) and m["recall"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(4 / 6) and m["f1"] == pytest.approx(2 / 3)
    assert m["mcc"] == pytest.approx((3 * 4 - 2 * 1) / np.sqrt(5 * 4 * 6 * 5))
    assert _auc(S, Y) == pytest.approx(20 / 24)      # positive-negative pairs ranked correctly
    assert _ap(S, Y) == pytest.approx((1 + 2 / 3 + 3 / 4 + 4 / 6) / 4)


def test_ties_form_one_threshold():
    s = np.array([0.8, 0.8, 0.5, 0.5])
    y = np.array([1, 0, 1, 0], dtype=bool)
    assert _ap(s, y) == pytest.approx(0.5 * 0.5 + 0.5 * 0.5)
    assert _ap(s, y) == pytest.approx(_ap(s[[1, 0, 3, 2]], y[[1, 0, 3, 2]]))
    fpr, tpr = roc_points(s, y)
    assert fpr.tolist() == [0, 0.5, 1] and tpr.tolist() == [0, 0.5, 1]


def test_recall_at_fpr_respects_the_budget():
    assert _recall_at_fpr(S, Y, 0.0) == pytest.approx(0.25)      # threshold above the best negative (0.8)
    assert _recall_at_fpr(S, Y, 1 / 6) == pytest.approx(0.75)    # one negative allowed: threshold above 0.5


def test_class_metrics_by_hand():
    gold = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 0, 2])
    m, cm, prec, rec, f1 = class_metrics(pred, gold, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert rec.tolist() == pytest.approx([0.5, 1.0, 0.5]) and prec.tolist() == pytest.approx([0.5, 2 / 3, 1.0])
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx(np.mean([0.5, 0.8, 2 / 3]))
