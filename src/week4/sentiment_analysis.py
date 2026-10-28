from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SentimentResult:
    label: str  # positive, negative, neutral
    score: float  # -1.0 to 1.0
    method: str  # vader, textblob, transformer


class VADERSentimentAnalyzer:
    """
    Rule-based sentiment analysis.
    Fast, no model download needed.
    Best for general English text.
    """

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(label="neutral", score=0.0, method="vader")

        # Truncate very long text for efficiency
        truncated = text[:5000]
        scores = self.analyzer.polarity_scores(truncated)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(label=label, score=compound, method="vader")

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(t) for t in texts]


class TextBlobSentimentAnalyzer:
    """
    Simple polarity-based sentiment.
    Good baseline, less accurate than VADER for nuanced text.
    """

    def __init__(self):
        from textblob import TextBlob
        self._TextBlob = TextBlob

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(label="neutral", score=0.0, method="textblob")

        truncated = text[:5000]
        blob = self._TextBlob(truncated)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(label=label, score=polarity, method="textblob")

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        return [self.analyze(t) for t in texts]


class TransformerSentimentAnalyzer:
    """
    Transformer-based sentiment analysis.
    More accurate but slower and requires model download.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        from transformers import pipeline
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1,  # CPU
            truncation=True,
            max_length=512,
        )

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(label="neutral", score=0.0, method="transformer")

        truncated = text[:512]
        result = self.pipe(truncated)[0]

        label_raw = result["label"].lower()
        score = result["score"]

        if "pos" in label_raw:
            label = "positive"
            signed_score = score
        elif "neg" in label_raw:
            label = "negative"
            signed_score = -score
        else:
            label = "neutral"
            signed_score = 0.0

        return SentimentResult(label=label, score=signed_score, method="transformer")

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:512] if t.strip() else "neutral" for t in texts[i:i+batch_size]]
            outputs = self.pipe(batch)
            for out in outputs:
                label_raw = out["label"].lower()
                score = out["score"]
                if "pos" in label_raw:
                    results.append(SentimentResult("positive", score, "transformer"))
                elif "neg" in label_raw:
                    results.append(SentimentResult("negative", -score, "transformer"))
                else:
                    results.append(SentimentResult("neutral", 0.0, "transformer"))
        return results
