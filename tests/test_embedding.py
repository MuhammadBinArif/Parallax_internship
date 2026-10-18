import numpy as np
from src.embeddings import SentenceTransformerEmbedder


class DummySentenceTransformer:
    def __init__(self, model_name, device=None):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_embed_returns_expected_shape(monkeypatch):
    monkeypatch.setattr("src.embeddings.SentenceTransformer", DummySentenceTransformer)
    e = SentenceTransformerEmbedder(model_name="dummy")
    r = e.embed(["a", "b", "c"])
    assert r.vectors.shape == (3, 4)
    assert r.count == 3
    assert r.dimension == 4
    assert r.seconds >= 0.0


def test_embed_empty_list(monkeypatch):
    monkeypatch.setattr("src.embeddings.SentenceTransformer", DummySentenceTransformer)
    e = SentenceTransformerEmbedder(model_name="dummy")
    r = e.embed([])
    assert r.vectors.shape == (0, 4)
    assert r.count == 0
    assert r.seconds == 0.0


def test_embed_one_returns_vector(monkeypatch):
    monkeypatch.setattr("src.embeddings.SentenceTransformer", DummySentenceTransformer)
    e = SentenceTransformerEmbedder(model_name="dummy")
    v, s = e.embed_one("hello")
    assert v.shape == (4,)
    assert s >= 0.0
