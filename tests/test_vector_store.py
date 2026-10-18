import pytest
from src.vector_store import ChromaVectorStore


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="unit_test_collection",
    )


def test_upsert_and_query(store):
    emb = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    stats = store.upsert(
        ids=["a", "b", "c"], documents=["alpha", "beta", "gamma"], embeddings=emb,
        metadatas=[{"kind": "x"}, {"kind": "y"}, {"kind": "z"}],
    )
    assert stats["upserted"] == 3
    assert store.count() == 3
    hits = store.query(emb[0], n_results=2)
    assert len(hits) == 2
    assert hits[0].id == "a"
    assert hits[0].document == "alpha"


def test_empty_query_returns_no_hits(store):
    assert store.query([], n_results=3) == []


def test_invalid_embeddings_are_skipped(store):
    stats = store.upsert(
        ids=["good", "bad"], documents=["x", "y"],
        embeddings=[[1.0, 0.0], [float("nan"), 0.0]],
    )
    assert stats["upserted"] == 1
    assert stats["skipped"] == 1
    assert store.count() == 1


def test_duplicate_id_updates_document(store):
    store.upsert(ids=["a"], documents=["old"], embeddings=[[1.0, 0.0]])
    store.upsert(ids=["a"], documents=["new"], embeddings=[[1.0, 0.0]])
    assert store.count() == 1
    hits = store.query([1.0, 0.0], n_results=1)
    assert hits[0].document == "new"
