from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class TopicResult:
    topic_id: int
    topic_label: str
    top_words: list[str]
    probability: float = 0.0


class LDATopicModeler:
    """
    LDA-based topic modeling.
    Lighter weight than BERTopic, good for CPU.
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_features: int = 5000,
        random_state: int = 42,
    ):
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer

        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words="english",
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=20,
            learning_method="online",
            batch_size=128,
        )
        self.topic_labels: list[str] = []
        self.is_fitted = False

    def fit(self, texts: list[str]) -> "LDATopicModeler":
        print(f"Fitting LDA on {len(texts)} documents...")
        start = time.perf_counter()

        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.lda.fit(doc_term_matrix)

        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_labels = []

        for topic_idx, topic in enumerate(self.lda.components_):
            top_word_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            label = f"Topic {topic_idx}: {', '.join(top_words[:3])}"
            self.topic_labels.append(label)

        elapsed = time.perf_counter() - start
        print(f"LDA fitted in {elapsed:.2f}s")
        self.is_fitted = True
        return self

    def predict(self, texts: list[str]) -> list[TopicResult]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        doc_term_matrix = self.vectorizer.transform(texts)
        topic_distributions = self.lda.transform(doc_term_matrix)

        results = []
        for i, dist in enumerate(topic_distributions):
            topic_id = int(np.argmax(dist))
            probability = float(dist[topic_id])
            top_words = self.topic_labels[topic_id].split(": ")[-1].split(", ")
            results.append(
                TopicResult(
                    topic_id=topic_id,
                    topic_label=self.topic_labels[topic_id],
                    top_words=top_words,
                    probability=probability,
                )
            )
        return results

    def get_topic_labels(self) -> list[str]:
        return self.topic_labels


class BERTopicModeler:
    """
    BERTopic-based topic modeling.
    Uses sentence embeddings (can leverage existing all-MiniLM-L6-v2).
    More accurate but heavier than LDA.
    """

    def __init__(
        self,
        n_topics: Optional[int] = 15,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_topic_size: int = 50,
    ):
        from bertopic import BERTopic
        from sklearn.cluster import KMeans
        from umap import UMAP

        self.embedding_model_name = embedding_model
        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=n_topics,
            min_topic_size=min_topic_size,
            verbose=True,
        )
        self.is_fitted = False

    def fit(self, texts: list[str]) -> "BERTopicModeler":
        print(f"Fitting BERTopic on {len(texts)} documents...")
        start = time.perf_counter()

        self.topics, self.probs = self.topic_model.fit_transform(texts)

        elapsed = time.perf_counter() - start
        print(f"BERTopic fitted in {elapsed:.2f}s")
        print(f"Discovered {len(set(self.topics)) - (1 if -1 in self.topics else 0)} topics")
        self.is_fitted = True
        return self

    def predict(self, texts: list[str]) -> list[TopicResult]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        topics, probs = self.topic_model.transform(texts)
        topic_info = self.topic_model.get_topic_info()

        results = []
        for i, (topic_id, prob) in enumerate(zip(topics, probs)):
            if topic_id == -1:
                label = "Outlier/Noise"
                top_words = []
            else:
                topic_words = self.topic_model.get_topic(topic_id)
                top_words = [w[0] for w in topic_words[:5]] if topic_words else []
                label = f"Topic {topic_id}: {', '.join(top_words[:3])}"

            results.append(
                TopicResult(
                    topic_id=int(topic_id),
                    topic_label=label,
                    top_words=top_words,
                    probability=float(prob) if prob else 0.0,
                )
            )
        return results

    def get_topic_labels(self) -> list[str]:
        topic_info = self.topic_model.get_topic_info()
        labels = []
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                labels.append("Outlier/Noise")
            else:
                words = self.topic_model.get_topic(topic_id)
                top = [w[0] for w in words[:3]] if words else []
                labels.append(f"Topic {topic_id}: {', '.join(top)}")
        return labels
