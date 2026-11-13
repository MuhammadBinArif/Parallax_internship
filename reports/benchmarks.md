# Benchmark Summary

_Generated: 2026-11-04T21:42:33.644298Z_

## Week 1 — Data Foundation
- Raw documents: **50,000**
- Clean documents: **49,324**
- Environment checks passed: True

## Week 2 — Retrieval
- Chunks indexed: **12825**
- Embedding model: `all-MiniLM-L6-v2` (384-dim)
- Mean retrieval latency: **21.14644956047414 ms**
- Mean embed / search split: 15.720436538977083 ms / 5.4027729204972275 ms
- P95 latency: 25.872278948372692 ms

## Week 3 — Generation
- Provider / model: `groq` / `openai/gpt-oss-20b`
- Mean total latency: **6298.29 ms**
- Mean generation latency: 6251.58 ms
- Mean retrieval latency: 44.97 ms
- Mean tokens: 1126.2
- Category distribution: {'answered': 21, 'insufficient_context': 25, 'out_of_domain': 19}

## Week 4 — NLP Enrichment
- Documents analyzed: 500
- Topics discovered: **10**
- Sentiment accuracy: **0.7**
- Sentiment macro F1: 0.6534
- Enriched chunks: 1000
- Sentiment distribution: {'positive': 328, 'negative': 158, 'neutral': 14}

## Week 5 — API + Evaluation
### Retrieval
- Queries: 9
- **Precision@1: 0.5556**
- Precision@5: 0.4222
- Recall@10: 0.5658
- MAP: 0.2415
- MRR: 0.6714
- nDCG@10: 0.4999

### End-to-End Generation
- Queries: 6
- Category accuracy: **0.6667**
- Citation presence: 1.0
- Mean pipeline latency: 21021.14 ms
- P95 latency: 61775.55 ms

- Total API requests logged: 0

## Aggregate
- Total clean documents: **49,324**
- Total chunks indexed: **12825**
- End-to-end mean latency: **21021.14 ms**
- Retrieval mean latency: 21.14644956047414 ms