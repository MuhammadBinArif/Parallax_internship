from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_sentiment(
    predictions: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """
    Evaluate sentiment predictions against manual labels.
    Returns accuracy, precision, recall, F1 per class.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    valid_pairs = [
        (p, l) for p, l in zip(predictions, labels)
        if l in {"positive", "negative", "neutral"}
    ]

    if not valid_pairs:
        return {"error": "No valid labels found"}

    preds, truths = zip(*valid_pairs)
    preds = list(preds)
    truths = list(truths)

    labels_set = sorted(set(truths) | set(preds))

    report = classification_report(
        truths, preds, labels=labels_set, output_dict=True, zero_division=0
    )

    return {
        "total_samples": len(truths),
        "accuracy": round(accuracy_score(truths, preds), 4),
        "macro_f1": round(f1_score(truths, preds, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(truths, preds, average="weighted", zero_division=0), 4),
        "per_class": report,
        "confusion_matrix": confusion_matrix(truths, preds, labels=labels_set).tolist(),
        "labels_order": labels_set,
    }


def evaluate_topics_manual(
    topic_assignments: list[int],
    texts: list[str],
    n_samples: int = 5,
) -> dict[str, Any]:
    """
    Generate samples for manual topic validation.
    Returns top documents per topic for human review.
    """
    from collections import defaultdict

    topic_docs: dict[int, list[str]] = defaultdict(list)
    for topic_id, text in zip(topic_assignments, texts):
        topic_docs[topic_id].append(text)

    samples = {}
    for topic_id, docs in sorted(topic_docs.items()):
        samples[f"topic_{topic_id}"] = {
            "count": len(docs),
            "samples": [d[:200] for d in docs[:n_samples]],
        }

    return samples


def load_manual_labels(path: Path) -> list[dict[str, Any]]:
    """Load manually labeled evaluation data."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
