from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.chunking import chunk_text
from src.embeddings import SentenceTransformerEmbedder
from src.vector_store import ChromaVectorStore


TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".jsonl", ".json"}

METADATA_KEYS = (
    "id",
    "doc_id",
    "document_id",
    "title",
    "url",
    "date",
    "category",
    "lang",
    "language",
    "source",
    "char_count",
    "word_count",
)

TEXT_KEYS = ("text", "content", "body", "page", "abstract")


def find_text_files(input_path: Path) -> List[Path]:
    """
    Find supported text-like files inside a directory.

    If input_path is a file, return that file.
    """
    if input_path.is_file():
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    )


def _safe_metadata_value(value: Any) -> Any:
    """
    Convert metadata values into types that are safer for ChromaDB.
    """
    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)

    return str(value)


def _metadata_from_object(
    obj: Dict[str, Any],
    path: Path,
    line_number: Optional[int] = None,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build metadata dictionary from a JSON/JSONL document object.
    """
    metadata: Dict[str, Any] = {
        "source": str(path),
        "file_name": path.name,
    }

    if line_number is not None:
        metadata["line_number"] = line_number

    if index is not None:
        metadata["index"] = index

    for key in METADATA_KEYS:
        if key in obj and obj[key] is not None:
            metadata[key] = _safe_metadata_value(obj[key])

    return metadata


def _text_from_object(obj: Dict[str, Any]) -> Optional[str]:
    """
    Extract text from common document fields.
    """
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return None


def load_documents_from_file(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Load one or more documents from a file.

    Supports:
    - .txt, .md, .markdown, .rst
    - .jsonl with lines like {"text": "..."}
    - .json with either:
      - [{"text": "..."}]
      - {"documents": [{"text": "..."}]}
      - {"text": "..."}
    """
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".markdown", ".rst"}:
        text = path.read_text(encoding="utf-8", errors="replace")

        if text.strip():
            yield {
                "text": text,
                "metadata": {
                    "source": str(path),
                    "file_name": path.name,
                    "file_suffix": suffix,
                },
            }
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"Skipping invalid JSONL line {line_number} in {path}: {exc}"
                    )
                    continue

                if not isinstance(obj, dict):
                    continue

                text = _text_from_object(obj)

                if not text:
                    continue

                yield {
                    "text": text,
                    "metadata": _metadata_from_object(
                        obj,
                        path,
                        line_number=line_number,
                    ),
                }
        return

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))

        if isinstance(data, dict):
            if "documents" in data:
                records = data["documents"]
            elif _text_from_object(data):
                records = [data]
            else:
                records = list(data.values())
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")

        for index, obj in enumerate(records, start=1):
            if isinstance(obj, str):
                text = obj
                metadata = {
                    "source": str(path),
                    "file_name": path.name,
                    "index": index,
                }
            elif isinstance(obj, dict):
                text = _text_from_object(obj)
                metadata = _metadata_from_object(obj, path, index=index)
            else:
                continue

            if not text or not str(text).strip():
                continue

            yield {
                "text": str(text),
                "metadata": metadata,
            }
        return

    raise ValueError(f"Unsupported file type: {path}")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON report/file safely.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Week 1 processed corpus into ChromaDB."
    )

    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--collection", default="week2_chunks")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)

    parser.add_argument(
        "--batch-size",
        "--embed-batch-size",
        dest="embed_batch_size",
        type=int,
        default=32,
        help="Number of chunks to embed in one batch.",
    )

    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=100,
        help="Number of chunks to upsert into ChromaDB in one batch.",
    )

    parser.add_argument(
        "--limit-docs",
        type=int,
        default=0,
        help="Limit number of source documents processed. 0 means all documents.",
    )

    parser.add_argument(
        "--log",
        default="reports/week2/ingest_log.json",
        help="Path to ingestion log/report.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/delete Chroma collection before ingesting.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)

    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    total_start = time.perf_counter()

    files = find_text_files(input_path)

    log_payload: Dict[str, Any] = {
        "config": vars(args),
        "files_found": len(files),
        "documents_processed": 0,
        "chunks_created": 0,
        "upserted": 0,
        "skipped": 0,
        "skipped_details": [],
        "file_errors": [],
        "chunking_seconds": 0.0,
        "embedding_seconds": 0.0,
        "upsert_seconds": 0.0,
        "embedding_model": args.model,
        "embedding_dimension": None,
        "collection_count_after_ingest": 0,
        "total_seconds": 0.0,
        "status": "started",
    }

    if not files:
        log_payload["status"] = "no_files"
        write_json(Path(args.log), log_payload)
        print(f"No supported files found in {input_path}")
        return

    print(f"Found {len(files)} supported file(s) in {input_path}")

    try:
        embedder = SentenceTransformerEmbedder(
            model_name=args.model,
            batch_size=args.embed_batch_size,
        )
    except TypeError:
        embedder = SentenceTransformerEmbedder(model_name=args.model)

    try:
        store = ChromaVectorStore(
            persist_directory=args.persist_dir,
            collection_name=args.collection,
            distance_function="cosine",
        )
    except TypeError:
        store = ChromaVectorStore(
            persist_directory=args.persist_dir,
            collection_name=args.collection,
        )

    if args.reset:
        print("Resetting Chroma collection...")
        store.reset()

    buffer: List[Any] = []

    documents_processed = 0
    chunks_created = 0
    upserted = 0
    skipped = 0

    skipped_details: List[Dict[str, Any]] = []

    chunking_seconds = 0.0
    embedding_seconds = 0.0
    upsert_seconds = 0.0

    stop = False

    def flush_buffer() -> None:
        nonlocal upserted
        nonlocal skipped
        nonlocal embedding_seconds
        nonlocal upsert_seconds

        if not buffer:
            return

        ids = [chunk.id for chunk in buffer]
        texts = [chunk.text for chunk in buffer]
        metadatas = [chunk.metadata for chunk in buffer]

        try:
            result = embedder.embed(texts, normalize=True)
        except TypeError:
            result = embedder.embed(texts)

        if hasattr(result, "vectors"):
            vectors = result.vectors
            embedding_seconds += float(getattr(result, "seconds", 0.0))
        elif isinstance(result, tuple) and len(result) >= 2:
            vectors = result[0]
            embedding_seconds += float(result[1])
        else:
            vectors = result
            embedding_seconds += 0.0

        try:
            embedding_list = vectors.tolist()
        except AttributeError:
            embedding_list = list(vectors)

        upsert_start = time.perf_counter()

        stats = store.upsert(
            ids=ids,
            documents=texts,
            embeddings=embedding_list,
            metadatas=metadatas,
            batch_size=args.upsert_batch_size,
        )

        upsert_seconds += time.perf_counter() - upsert_start

        stats = stats or {}

        upserted += int(stats.get("upserted", len(texts)))
        skipped += int(stats.get("skipped", 0))

        for detail in stats.get("skipped_details", []) or []:
            if len(skipped_details) < 20:
                skipped_details.append(detail)

        print(
            f"Flushed batch: chunks={len(texts)}, "
            f"upserted={stats.get('upserted', len(texts))}, "
            f"skipped={stats.get('skipped', 0)}"
        )

        buffer.clear()

    for file_path in files:
        try:
            document_iterator = load_documents_from_file(file_path)
        except Exception as exc:
            message = f"Failed to open {file_path}: {exc}"
            print(message)
            log_payload["file_errors"].append(
                {
                    "file": str(file_path),
                    "error": str(exc),
                }
            )
            continue

        for doc in document_iterator:
            if args.limit_docs > 0 and documents_processed >= args.limit_docs:
                stop = True
                break

            documents_processed += 1

            chunk_start = time.perf_counter()

            chunks = chunk_text(
                doc["text"],
                source=str(file_path),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                metadata=doc["metadata"],
            )

            chunking_seconds += time.perf_counter() - chunk_start

            chunks_created += len(chunks)
            buffer.extend(chunks)

            if len(buffer) >= args.embed_batch_size:
                flush_buffer()

        if stop:
            break

    flush_buffer()

    log_payload.update(
        {
            "status": "ok" if upserted > 0 else "no_chunks",
            "documents_processed": documents_processed,
            "chunks_created": chunks_created,
            "upserted": upserted,
            "skipped": skipped,
            "skipped_details": skipped_details,
            "chunking_seconds": round(chunking_seconds, 6),
            "embedding_seconds": round(embedding_seconds, 6),
            "upsert_seconds": round(upsert_seconds, 6),
            "embedding_dimension": getattr(embedder, "dimension", None),
            "collection_count_after_ingest": store.count(),
            "total_seconds": round(time.perf_counter() - total_start, 6),
        }
    )

    write_json(Path(args.log), log_payload)

    print("Ingestion complete.")
    print(json.dumps(log_payload, indent=2, default=str))


if __name__ == "__main__":
    main()
