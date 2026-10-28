from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.week4.vector_db_enrichment import EnrichedVectorStore
from src.week4.nlp_pipeline import NLPPipeline
from src.week4.edge_cases import handle_edge_case


def main() -> None:
    # Load Week 2 chunks (from existing ChromaDB)
    import chromadb
    client = chromadb.PersistentClient(path="chroma_db")

    collections = client.list_collections()
    if not collections:
        raise SystemExit("No ChromaDB collections found. Run Week 2 ingestion first.")

    source_collection = client.get_collection(str(getattr(collections[0], "name", collections[0])))
    count = source_collection.count()
    print(f"Source collection: {source_collection.name}, count: {count}")

    # Get all chunks (limit for CPU)
    limit = min(count, 1000)
    data = source_collection.get(limit=limit, include=["documents", "metadatas", "embeddings"])

    ids = data["ids"]
    documents = data["documents"]
    embeddings = data["embeddings"]
    metadatas = data["metadatas"]

    print(f"Processing {len(ids)} chunks for NLP enrichment...")

    # Run NLP on each chunk
    pipeline = NLPPipeline(n_topics=8, use_bertopic=False)

    # Fit topics on chunk texts
    texts = [d[:500] if d else "" for d in documents]
    pipeline.fit_topics(texts)

    # Analyze
    nlp_results = pipeline.analyze_batch(texts, batch_size=50)

    # Enrich metadata
    enriched_metas = []
    for i, (meta, nlp) in enumerate(zip(metadatas, nlp_results)):
        enriched = dict(meta) if meta else {}
        enriched["topic_id"] = nlp["topic_id"]
        enriched["topic_label"] = nlp.get("topic_label", "")[:100]
        enriched["sentiment_label"] = nlp["sentiment_label"]
        enriched["sentiment_score"] = nlp["sentiment_score"]
        enriched["entity_count"] = nlp.get("entity_count", 0)
        enriched["has_person"] = nlp.get("has_person", False)
        enriched["has_org"] = nlp.get("has_org", False)
        enriched["has_location"] = nlp.get("has_location", False)
        enriched["edge_case"] = nlp.get("edge_case") or "none"
        enriched_metas.append(enriched)

    # Store in new collection
    store = EnrichedVectorStore(
        persist_directory="chroma_db",
        collection_name="week4_enriched",
    )

    print("Upserting enriched chunks...")
    stats = store.upsert_enriched_chunks(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=enriched_metas,
        batch_size=100,
    )

    print(f"\nEnrichment complete:")
    print(f"  Total: {stats['total']}")
    print(f"  Upserted: {stats['upserted']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Collection count: {stats['collection_count']}")

    # Save log
    log_path = ROOT / "reports" / "week4" / "enrichment_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
