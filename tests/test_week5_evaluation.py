from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_metric_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_week5_retrieval.py"
    spec = importlib.util.spec_from_file_location("week5_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_precision_at_k_perfect():
    mod = _load_metric_module()
    assert mod.precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_at_k_partial():
    mod = _load_metric_module()
    assert mod.precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5


def test_recall_at_k_partial():
    mod = _load_metric_module()
    assert mod.recall_at_k(["a", "x"], {"a", "b", "c"}, 2) == 1/3


def test_recall_empty_relevant():
    mod = _load_metric_module()
    assert mod.recall_at_k(["a"], set(), 1) == 0.0


def test_mrr_hit_at_first():
    mod = _load_metric_module()
    assert mod.mrr(["a", "b"], {"a"}) == 1.0


def test_mrr_hit_at_third():
    mod = _load_metric_module()
    assert mod.mrr(["x", "y", "a"], {"a"}) == 1/3


def test_average_precision():
    mod = _load_metric_module()
    # 1st hit at position 1, 2nd hit at position 3 → (1/1 + 2/3) / 2
    ap = mod.average_precision(["a", "x", "b"], {"a", "b"})
    assert round(ap, 4) == round((1 + 2/3) / 2, 4)


def test_ndcg_perfect():
    mod = _load_metric_module()
    assert mod.ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_ndcg_zero():
    mod = _load_metric_module()
    assert mod.ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
