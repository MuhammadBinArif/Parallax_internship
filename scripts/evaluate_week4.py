from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week4.sentiment_analysis import VADERSentimentAnalyzer
from src.week4.evaluation import evaluate_sentiment


def main() -> None:
    labels_path = ROOT / "data" / "eval" / "week4_labels" / "sentiment_labels.json"

    if not labels_path.exists():
        raise SystemExit(f"Missing labels file: {labels_path}")

    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(labels_data)} labeled examples")

    analyzer = VADERSentimentAnalyzer()

    predictions = []
    truths = []

    for item in labels_data:
        text = item["text"]
        true_label = item["sentiment"]

        result = analyzer.analyze(text)
        predictions.append(result.label)
        truths.append(true_label)

        match = "✓" if result.label == true_label else "✗"
        print(f"  {match} Predicted: {result.label:10s} | True: {true_label:10s} | Score: {result.score:+.3f}")

    # Evaluate
    metrics = evaluate_sentiment(predictions, truths)

    print(f"\n=== Sentiment Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    # Save results
    output_path = ROOT / "reports" / "week4" / "sentiment_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
