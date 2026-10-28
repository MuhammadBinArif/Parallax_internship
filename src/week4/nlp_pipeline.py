from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from src.week4.edge_cases import handle_edge_case, preprocess_for_nlp
from src.week4.sentiment_analysis import (
    SentimentResult,
    VADERSentimentAnalyzer,
)
from src.week4.topic_modeling import LDATopicModeler, TopicResult
from src.week4.ner_extraction import SpacyNERExtractor, NERResult


class NLPPipeline:
    """
    Full NLP analysis pipeline:
    1. Topic modeling
    2. Sentiment analysis
    3. NER extraction
    4. Edge case handling
    """

    def __init__(
        self,
        n_topics: int = 10,
        use_bertopic: bool = False,
        sentiment_method: str = "vader",
    ):
        self.n_topics = n_topics
        self.use_bertopic = use_bertopic
        self.sentiment_method = sentiment_method

        self.topic_modeler: Optional[LDATopicModeler] = None
        self.sentiment_analyzer = VADERSentimentAnalyzer()
        self.ner_extractor: Optional[SpacyNERExtractor] = None

    def fit_topics(self, texts: list[str]) -> None:
        print(f"Fitting topic model on {len(texts)} documents...")
        start = time.perf_counter()

        if self.use_bertopic:
            from src.week4.topic_modeling import BERTopicModeler
            self.topic_modeler = BERTopicModeler(n_topics=self.n_topics)
        else:
            self.topic_modeler = LDATopicModeler(n_topics=self.n_topics)

        self.topic_modeler.fit(texts)
        elapsed = time.perf_counter() - start
        print(f"Topic modeling complete in {elapsed:.2f}s")

    def analyze_document(self, text: str) -> dict[str, Any]:
        """Analyze a single document for topics, sentiment, NER."""
        edge_case = handle_edge_case(text)
        if edge_case:
            return {
                "topic_id": -1,
                "topic_label": f"edge_case:{edge_case}",
                "sentiment_label": "neutral",
                "sentiment_score": 0.0,
                "entities": [],
                "entity_count": 0,
                "edge_case": edge_case,
            }

        cleaned = preprocess_for_nlp(text)

        # Topic
        if self.topic_modeler:
            topic_results = self.topic_modeler.predict([cleaned])
            topic = topic_results[0]
        else:
            topic = TopicResult(topic_id=-1, topic_label="unfitted", top_words=[])

        # Sentiment
        sentiment = self.sentiment_analyzer.analyze(cleaned)

        # NER
        if self.ner_extractor is None:
            self.ner_extractor = SpacyNERExtractor()
        ner = self.ner_extractor.extract(cleaned)

        return {
            "topic_id": topic.topic_id,
            "topic_label": topic.topic_label,
            "topic_probability": topic.probability,
            "sentiment_label": sentiment.label,
            "sentiment_score": round(sentiment.score, 4),
            "sentiment_method": sentiment.method,
            "entities": ner.entities[:10],
            "entity_count": ner.entity_count,
            "has_person": ner.has_person,
            "has_org": ner.has_org,
            "has_location": ner.has_location,
            "edge_case": None,
        }

    def analyze_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Analyze a batch of documents."""
        results = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            for j, text in enumerate(batch):
                idx = i + j
                result = self.analyze_document(text)
                results.append(result)

            if (i + batch_size) % 500 == 0 or i + batch_size >= total:
                print(f"Processed {min(i + batch_size, total)}/{total} documents")

        return results
