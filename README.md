# Internship Repository

Single repository for an **8‑week** internship. Each week adds a new capability on top of the previous one (data → retrieval → generation → evaluation → deployment). To keep this readable as the project grows, the README is organized in three layers:

1. **Global** – setup, repository layout, dependencies, and the rules for adding a new week. Read once.
2. **Weekly logs** – one `## Week N` section per week (objective, what was built, how to run, results, decisions, limitations, checklist).
3. **Template** – a block you copy at the start of each new week so every week looks the same.

> **How to continue next week:** copy the *Weekly Section Template* (see *Conventions & How To Add A New Week*), paste it as a new `## Week N` section, fill it in, flip the week's row in the *Program Roadmap* to `Done`, and update *Repository Layout* only if you add a new top‑level folder. That's it.

---

## Program Roadmap (8 Weeks)

| Week | Topic | Status |
|---|---|---|
| 1 | Environment, data ingestion & preprocessing | ✅ Done |
| 2 | Chunking, embeddings & vector DB (ChromaDB) + latency benchmark | ✅ Done |
| 3 | Hybrid search (BM25 + vector) & reranking | ⬜ *Suggested – replace with assigned topic* |
| 4 | Query understanding, expansion & routing | ⬜ *Suggested – replace with assigned topic* |
| 5 | Retrieval evaluation harness (recall@k, MRR, nDCG) & golden set | ⬜ *Suggested – replace with assigned topic* |
| 6 | End‑to‑end RAG: prompt assembly, generation & citations | ⬜ *Suggested – replace with assigned topic* |
| 7 | Orchestration / caching / optimization (or agentic tool‑use) | ⬜ *Suggested – replace with assigned topic* |
| 8 | Deployment (API), monitoring & final report | ⬜ *Suggested – replace with assigned topic* |

Rows 3–8 are placeholders for continuity. Replace each topic with the actual assigned objective when the week begins, then fill that week's section from the template.

---

## Global Setup (All Weeks)

### 1. Clone the repository

Replace the URL with your actual GitHub repository URL:

```bash
git clone <your-github-repository-url>
cd internship-repo
```

### 2. Create and activate the conda environment

A dedicated environment isolates project dependencies from the base Anaconda environment, which keeps the project reproducible and avoids version conflicts.

Environment name:

```bash
ParallaxLabsInternship
```

```bash
conda create --name ParallaxLabsInternship python=3.11 -y
conda activate ParallaxLabsInternship
python --version   # expected: Python 3.11.x
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. GPU / CUDA (only if required)

If GPU support is needed, install a CUDA‑compatible PyTorch build **before** the rest of the dependencies. Pick the line that matches your driver's CUDA version:

For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies as in step 3.

### 5. What `.gitignore` excludes

Only source code, config examples, dependency files, and small validation reports are committed. The `.gitignore` file excludes:

- large raw and processed data files
- `.env` secret files
- virtual environment / conda artifacts
- Python caches (`__pycache__/`, `*.pyc`) and `.pytest_cache/`
- model caches
- the local vector database directory (`chroma_db/`)
- common OS files (`.DS_Store`, `Thumbs.db`)

---

## Repository Layout (Current)

This reflects the repo after Week 2. New top‑level folders added in later weeks should be appended here.

```text
.
├── data/
│   ├── raw/                         # Week 1 raw download (git-ignored)
│   │   └── wikipedia_raw.jsonl
│   ├── processed/                   # Week 1 clean corpus = Week 2 input (git-ignored)
│   │   ├── clean_corpus.jsonl
│   │   └── clean_corpus.parquet
│   └── queries/                     # Week 2 benchmark queries (committed, small)
│       └── benchmark_queries.json
├── reports/
│   ├── week1/                       # Week 1 logs (committed)
│   │   ├── validation_report.json
│   │   └── verify_env.json
│   └── week2/                       # Week 2 logs (committed)
│       ├── ingest_log.json
│       ├── benchmark.csv
│       └── benchmark_summary.json
├── chroma_db/                       # Week 2 persistent vector DB (git-ignored)
├── scripts/                         # Thin runnable wrappers (one per task)
│   ├── verify_env.py                # Week 1
│   ├── download_dataset.py          # Week 1
│   ├── build_clean_corpus.py        # Week 1
│   ├── ingest_vectors.py            # Week 2
│   ├── search_semantic.py           # Week 2
│   └── benchmark_retrieval.py       # Week 2
├── src/                             # Reusable logic modules
│   ├── __init__.py
│   ├── preprocessing.py             # Week 1
│   ├── chunking.py                  # Week 2
│   ├── embeddings.py                # Week 2
│   ├── vector_store.py              # Week 2
│   ├── ingest.py                    # Week 2 pipeline
│   ├── search.py                    # Week 2 search CLI
│   └── benchmark.py                 # Week 2 benchmark CLI
├── tests/                           # Unit tests (mirror src/)
│   ├── __init__.py
│   ├── test_preprocessing.py        # Week 1
│   ├── test_chunking.py             # Week 2
│   ├── test_embedding.py            # Week 2
│   └── test_vector_store.py         # Week 2
├── .gitignore
├── .env.example                     # Safe to commit
├── .env                             # Secrets (git-ignored)
├── requirements.txt                 # Locked dependencies
└── README.md
```

Auto‑generated, never committed: `__pycache__/`, `.pytest_cache/`, `chroma_db/`, `.venv/`/conda envs, `.env`, large `data/raw` & `data/processed` files.

---

## Conventions & How To Add A New Week

These rules are what make weeks 3–8 "drop‑in". Follow them every week.

### Folder mapping

| You are adding… | Put it in… | Notes |
|---|---|---|
| Reusable logic | `src/<module>.py` | No `argparse`/IO side effects at import time. |
| A runnable task | `scripts/<verb>_<noun>.py` | Thin wrapper that imports from `src/` and calls `main()`. Adds repo root to `sys.path`. |
| Unit tests | `tests/test_<module>.py` | Name must mirror the `src/` module. Mock heavy models (see Week 2 embedding tests). |
| Per‑week logs / evidence | `reports/week<N>/` | Committed. These prove the pipeline ran. |
| Small per‑week inputs (queries, golden sets, configs) | `data/<name>/` | Committed if small. |
| Large derived data | `data/raw` or `data/processed` | Git‑ignored; regenerable via scripts. |
| A persistent local store (DB, index) | top‑level dir e.g. `chroma_db/` | Git‑ignored; document the regen command. |

### Naming rules

- Modules: `snake_case.py`. Tests: `test_<module>.py`. Scripts: `verb_noun.py`.
- Each week owns `reports/week<N>/`. Never overwrite a previous week's reports.
- Keep Week *N* reading from Week *N‑1*'s output (e.g. Week 2 ingests `data/processed/clean_corpus.jsonl`).

### Weekly Section Template (copy this for each new week)

Paste the block below as a new `## Week N — <Topic>` section, fill every heading, then update the Roadmap row and (only if needed) the Layout tree.

````markdown
## Week N — <Topic>

### Objective
<2–4 bullets copied from the week's brief.>

### What I Built
<Numbered subsections, one per component, each with a short purpose + the file path.>

### How To Run
<Exact commands from the repo root, in order. Reference scripts/ wrappers.>

### Results
<Table of concrete metrics. Pull numbers from reports/weekN/*.json — do not estimate.>

### Design Decisions
<Why this approach over alternatives. 3–6 bullets.>

### Known Limitations
<Honest gaps. These often become the next week's objective.>

### Submission Checklist
- [ ] README Week N section complete
- [ ] reports/weekN/ artifacts generated and committed
- [ ] tests pass: `python -m pytest -v`
- [ ] large/data artifacts git-ignored
- [ ] Roadmap row flipped to Done
````

---

## Week 1 — Environment, Data Ingestion & Preprocessing

### Objective

Build a reproducible foundation for a text‑based semantic search pipeline by:

- setting up a clean Python environment
- verifying dependencies and GPU/CUDA availability
- downloading a real‑world text dataset
- implementing modular text preprocessing
- adding unit tests
- producing a validated clean corpus

This week focuses on engineering reliability. The cleaned dataset and modular preprocessing pipeline created here are reused in later weeks for embedding generation, vector storage, and semantic search.

### What I Built

#### 1. Reproducible environment

A dedicated conda environment (`ParallaxLabsInternship`, Python 3.11) isolates project dependencies from the base Anaconda environment. Creation, activation, dependency install, and optional CUDA‑PyTorch install are documented once in **Global Setup** above. The `.gitignore` rules (what is and isn't committed) are also described there.

#### 2. Environment verification script

`scripts/verify_env.py` confirms the project environment is ready before running the pipeline. It checks required package imports (pandas, sentence‑transformers, ChromaDB, spaCy, BeautifulSoup, PyTorch), CUDA/GPU availability, a ChromaDB smoke test, and optional spaCy model availability.

CPU mode:

```bash
python scripts/verify_env.py --require-spacy-model --output-report reports/week1/verify_env.json
```

Strict GPU mode:

```bash
python scripts/verify_env.py --strict-gpu --require-spacy-model --output-report reports/week1/verify_env.json
```

The machine‑readable report is written to `reports/week1/verify_env.json`.

#### 3. Dataset ingestion

`scripts/download_dataset.py` streams a real‑world text dataset and saves raw records to `data/raw/`.

| Item | Value |
|---|---|
| Source | Hugging Face Datasets |
| Dataset | `wikimedia/wikipedia` |
| Config | `20231101.en` |
| Split | `train` |
| Download mode | Streaming |
| Raw documents downloaded | 50,000 |
| Raw output file | `data/raw/wikipedia_raw.jsonl` |

```bash
python scripts/download_dataset.py
```

The raw file is large and intentionally git‑ignored.

#### 4. Modular text preprocessing

Logic lives in `src/preprocessing.py` as reusable functions:

| Function | Purpose |
|---|---|
| `strip_html()` | Removes HTML tags and extracts readable text |
| `normalize_unicode()` | Normalizes Unicode and removes invisible/control characters |
| `clean_whitespace()` | Collapses excessive spaces and newlines |
| `remove_urls_and_emails()` | Removes URLs and email addresses |
| `clean_text()` | Runs the full text cleaning pipeline |
| `is_mostly_english()` | Lightweight English‑language heuristic |
| `extract_text()` | Extracts text from selected document fields |
| `clean_document()` | Cleans one document and returns a standard schema |
| `iter_clean_documents()` | Applies cleaning across many documents |

Each cleaned document uses a consistent schema, which makes the dataset easy to validate and reuse in later weeks:

```json
{
	"id": "document-id",
	"text": "cleaned text",
	"char_count": 1234,
	"word_count": 200,
	"source": "wikimedia/wikipedia",
	"language": "en_heuristic"
}
```

#### 5. Unit tests

`tests/test_preprocessing.py` validates HTML removal, Unicode normalization, whitespace cleanup, URL/email removal, the language heuristic, schema generation, short‑document filtering, invalid‑record handling, and ID assignment/dedup.

```bash
python -m pytest -v
```

All tests should pass before submitting Week 1.

#### 6. Clean corpus generation

`scripts/build_clean_corpus.py` reads the raw JSONL, applies preprocessing, filters invalid documents, and writes cleaned outputs.

```bash
python scripts/build_clean_corpus.py
```

Outputs:

```text
data/processed/clean_corpus.parquet
data/processed/clean_corpus.jsonl
reports/week1/validation_report.json
```

Processed data files are large and git‑ignored; the validation report is small and committed.

### How To Run (Week 1 Pipeline)

#### Step 1 — Verify environment

See commands in *What I Built → 2*. Purpose: confirm packages, GPU (strict mode), ChromaDB smoke test, spaCy; write `reports/week1/verify_env.json`.

#### Step 2 — Download raw dataset

```bash
python scripts/download_dataset.py
```

Purpose: download real‑world text, save raw records, write `data/raw/wikipedia_raw.jsonl`, keep raw separate from processed.

Check the raw file:

```bash
ls -lh data/raw/wikipedia_raw.jsonl
wc -l data/raw/wikipedia_raw.jsonl
```

Validate JSONL structure:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/raw/wikipedia_raw.jsonl")

if not path.exists():
	raise FileNotFoundError(f"File not found: {path}")
	
	valid = 0
	invalid = 0
	
	with path.open("r", encoding="utf-8") as f:
	for line_no, line in enumerate(f, start=1):
		line = line.strip()
		if not line:
			continue
			try:
			json.loads(line)
			valid += 1
			except Exception as exc:
			invalid += 1
			if invalid <= 5:
				print(f"Invalid JSON at line {line_no}: {exc}")
				
				print("Valid JSON lines:", valid)
				print("Invalid JSON lines:", invalid)
				PY
				```
				
				Expected:
				
				```text
				Valid JSON lines: 50000
				Invalid JSON lines: 0
				```
				
				#### Step 3 — Run unit tests
				
				```bash
				python -m pytest -v
				```
				
				Purpose: verify preprocessing behaves correctly, catch silent cleaning bugs, confirm the pipeline is safe to run on the full dataset.
				
				#### Step 4 — Build clean corpus
				
				```bash
				python scripts/build_clean_corpus.py
				```
				
				Purpose: read raw JSONL, apply modular preprocessing, filter short/empty/invalid/non‑English docs, standardize schema, write Parquet + JSONL, write a validation report.
				
				Outputs:
				
				```text
				data/processed/clean_corpus.parquet
				data/processed/clean_corpus.jsonl
				reports/week1/validation_report.json
				```
				
				#### Step 5 — Inspect outputs
				
				Validation report:
				
				```bash
				cat reports/week1/validation_report.json
				```
				
				Look for fields such as:
				
				```json
				{
					"raw_document_count": 50000,
					"clean_document_count": 49324,
					"min_documents_required": 5000,
					"unique_ids": true,
					"null_text_count": 0
				}
				```
				
				First cleaned document:
				
				```bash
				head -n 1 data/processed/clean_corpus.jsonl
				```
				
				Processed Parquet file:
				
				```bash
				python - <<'PY'
				import pandas as pd
				
				df = pd.read_parquet("data/processed/clean_corpus.parquet")
				
				print("Shape:", df.shape)
				print("Columns:", df.columns.tolist())
				print("\nFirst row:")
				print(df.head(1).to_string())
				PY
				```
				
				### Week 1 Results
				
				| Metric | Value |
				|---|---|
				| Raw dataset | `wikimedia/wikipedia` |
				| Dataset config | `20231101.en` |
				| Raw documents downloaded | 50,000 |
				| Clean documents produced | 49,324 |
				| Minimum required clean documents | 5,000 |
				| Minimum character threshold | 120 |
				| Maximum character threshold | 8,000 |
				| Language filter | Enabled |
				| Output format 1 | Parquet |
				| Output format 2 | JSONL |
				| Validation report | `reports/week1/validation_report.json` |
				
				The pipeline produced well over the required 5,000 clean documents. During preprocessing a BeautifulSoup `XMLParsedAsHTMLWarning` appeared on some Wikipedia records containing XML‑like markup; it was non‑fatal and the HTML parser was kept for robustness on messy web text. It can be suppressed in `src/preprocessing.py` if desired.
				
				### Approach & Design Decisions (Week 1)
				
				- **Dedicated conda environment** – isolates dependencies, avoids base‑env conflicts, improves reproducibility.
				- **Locked `requirements.txt`** – allows reliable recreation; unpinned installs risk incompatible versions.
				- **`.gitignore`** – keeps large datasets and secrets out of Git; repo stays clean and portable.
				- **Raw vs processed separation** – raw preserves source records; processed is derived and regenerable; improves lineage and debugging.
				- **Modular preprocessing in `src/`** – reusable, independently testable, extensible in later weeks without rewriting scripts.
				- **Unit tests** – cleaning can fail silently; tests verify HTML/Unicode/whitespace/language/schema behavior.
				- **Both Parquet and JSONL** – Parquet is compact/tabular; JSONL is easy to inspect line‑by‑line; flexibility for later stages.
				- **Heuristic language filter** – sufficient for Week 1 noise removal; a dedicated detector can replace it in production.
				
				### Validation Checks (Week 1)
				
				The corpus builder validates before writing: non‑empty cleaned set, unique document IDs, no null text, all docs within min/max character bounds, and `char_count` matching actual length. Any failure raises an error instead of emitting an invalid dataset.
				
				### Known Limitations (Week 1)
				
				- Language filtering is heuristic, not production‑grade detection.
				- Dataset is a sampled subset, not the full Wikipedia dump.
				- Long documents are truncated to keep processing manageable.
				- GPU verification depends on a compatible NVIDIA driver + CUDA‑enabled PyTorch.
				- Large raw/processed files are intentionally not committed.
				- Some Wikipedia records contain XML‑like markup that triggers non‑fatal BeautifulSoup warnings.
				
				### Troubleshooting (Week 1)
				
				**`ModuleNotFoundError: No module named 'src.preprocessing'`** – run from the repo root (`cd ~/internship-repo`); confirm `src/preprocessing.py` and `src/__init__.py` exist; test with `python -c "from src.preprocessing import iter_clean_documents; print('Import OK')"`.
				
				**`DatasetNotFoundError`** – use a public dataset such as `wikimedia/wikipedia` (config `20231101.en`), which is the repo default.
				
				**Hugging Face download timeout** – the downloader retries automatically; if it crashes after saving, validate before rerunning with `wc -l data/raw/wikipedia_raw.jsonl` (expect `50000`).
				
				**`XMLParsedAsHTMLWarning`** – non‑fatal; optionally suppress at the top of `src/preprocessing.py`:
				
				```python
				import warnings
				try:
				from bs4 import XMLParsedAsHTMLWarning
				warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
				except ImportError:
				pass
				```
				
				**`pytest` cannot import `src`** – run `python -m pytest -v` from the repo root and ensure `src/__init__.py` and `tests/__init__.py` exist. (This import/root caveat is universal across all weeks.)
				
				### Git Submission Checklist (Week 1)
				
				- [ ] Single GitHub repository used for the internship
				- [ ] `README.md` updated with Week 1 progress
				- [ ] `.gitignore` ignores large data files and `.env`
				- [ ] `requirements.txt` exists with locked dependencies
				- [ ] `scripts/verify_env.py` runs successfully
				- [ ] `scripts/download_dataset.py` downloads 5,000+ raw documents
				- [ ] `src/preprocessing.py` implements modular cleaning
				- [ ] `tests/test_preprocessing.py` exists
				- [ ] `python -m pytest -v` passes
				- [ ] `scripts/build_clean_corpus.py` produces 5,000+ clean documents
				- [ ] `data/processed/clean_corpus.parquet` exists locally
				- [ ] `data/processed/clean_corpus.jsonl` exists locally
				- [ ] `reports/week1/validation_report.json` exists
				- [ ] Large data files not staged/committed
				- [ ] Code, tests, reports, README, requirements pushed to GitHub
				
				### Suggested Final Commands (Week 1)
				
				```bash
				cd ~/internship-repo
				python scripts/verify_env.py --require-spacy-model --output-report reports/week1/verify_env.json
				python scripts/download_dataset.py
				python -m pytest -v
				python scripts/build_clean_corpus.py
				ls -lh data/raw data/processed reports/week1
				cat reports/week1/validation_report.json
				```
				
				Stage safe files:
				
				```bash
				git add README.md .gitignore .env.example requirements.txt scripts src tests reports
				git add data/raw/.gitkeep data/processed/.gitkeep data/queries
				```
				
				If `.gitkeep` files are ignored, force‑add only those tiny files:
				
				```bash
				git add -f data/raw/.gitkeep data/processed/.gitkeep
				```
				
				Commit and push:
				
				```bash
				git status
				git commit -m "Week 1: environment setup, data ingestion, preprocessing, tests"
				git branch -M main
				git remote add origin <your-github-repository-url>
				git push -u origin main
				```
				
				---
				
				## Week 2 — Chunking, Embeddings & Vector DB
				
				### Objective
				
				Take the Week 1 clean corpus and make it searchable:
				
				- implement and unit‑test a text chunking strategy (recursive splitting)
				- generate embeddings with `sentence-transformers` and log embedding time
				- set up ChromaDB, ingest chunks/embeddings, implement semantic search
				- benchmark retrieval performance and log latency across queries
				- document the chunking strategy and handle ChromaDB edge cases
				
				Deliverable: a populated vector database with tested chunking and latency‑benchmarked retrieval.
				
				### What I Built
				
				#### 1. Recursive text chunking — `src/chunking.py`
				
				Character‑based recursive splitter that tries natural boundaries in order (paragraph → line → sentence → clause → word → hard‑char fallback), merges small pieces up to `chunk_size`, and adds `chunk_overlap` between neighbors. Defaults: `chunk_size=800`, `chunk_overlap=150`. Deterministic chunk IDs (hash of source+index+text). Handles empty/None/whitespace/unicode/no‑separator inputs.
				
				#### 2. Embeddings — `src/embeddings.py`
				
				Thin `SentenceTransformer` wrapper (`all-MiniLM-L6-v2`, 384‑dim) that normalizes vectors (for cosine search), sanitizes empty strings, and returns timing via `time.perf_counter()`.
				
				#### 3. Vector store — `src/vector_store.py`
				
				Persistent ChromaDB wrapper (cosine space). `upsert()` validates ids/docs/embeddings/metadata, skips NaN/Inf and dimension mismatches with logging, sanitizes metadata to primitives, and batches writes. `query()` clamps `n_results` to collection size, returns hits with distance + similarity score, and safely handles empty collections. `reset()` recreates the collection when the embedding model/dimension changes.
				
				#### 4. Pipeline & CLIs — `src/ingest.py`, `src/search.py`, `src/benchmark.py`
				
				`ingest.py` loads Week 1 documents (supports `.txt/.md/.jsonl/.json`), chunks, embeds, upserts, and writes `reports/week2/ingest_log.json`. `search.py` embeds a query and prints/JSON‑dumps top‑k hits with latency. `benchmark.py` runs warm‑up + repeats over `data/queries/benchmark_queries.json`, separating embed vs search latency, and writes `reports/week2/benchmark.csv` + `benchmark_summary.json` (mean/p50/p95).
				
				#### 5. Wrappers & tests
				
				Runnable wrappers in `scripts/` (`ingest_vectors.py`, `search_semantic.py`, `benchmark_retrieval.py`) add the repo root to `sys.path`. Unit tests in `tests/` cover chunking (size enforcement, fallback, determinism, metadata), embeddings (shape/timing with a mocked model, empty input), and the vector store (upsert/query, duplicate‑id update, NaN skip, empty query).
				
				### How To Run (Week 2)
				
				Run Week 1 first so `data/processed/clean_corpus.jsonl` exists, then from the repo root:
				
				```bash
				python scripts/ingest_vectors.py            # chunk -> embed -> upsert -> reports/week2/ingest_log.json
				python scripts/search_semantic.py --query "What is text chunking?" --top-k 5
				python scripts/benchmark_retrieval.py       # -> reports/week2/benchmark.csv + benchmark_summary.json
				python -m pytest -v                          # all weeks' tests
				```
				
				If you switch embedding models and hit a dimension mismatch, reset the collection:
				
				```bash
				python scripts/ingest_vectors.py --reset
				```
				
				### Results (Week 2)
				
				Fill these from the committed report files — do not estimate.
				
				| Metric | Value |
				|---|---|
				| Status | ok |
				| Files found | 1 |
				| Documents processed | 1000 |
				| Chunks created | 12825 |
				| Embedding model | all-MiniLM-L6-v2 |
				| Embedding dimension | 384 |
				| Chunking seconds | 2.16 |
				| Embedding seconds | 1278.6 |
				| Upsert seconds | 104.92 |
				| Upserted | 12825 |
				| Skipped | 0 |
				| Collection count after ingest | 12825 |
				| Total ingest seconds | 1401.52 |
				| Queries benchmarked | 10 |
				| Repeats per query | 5 |
				| Total measurements | 50 |
				| Mean total latency ms | 21.15 |
				| P50 total latency ms | 20.47 |
				| P95 total latency ms | 25.87 |
				| Mean embed latency ms | 15.72 |
				| Mean search latency ms | 5.4 |
				| Mean hits | 5.0 |
				
				### Design Decisions (Week 2)
				
				- **Character‑based recursive chunking** – simple, dependency‑free, testable baseline before token/semantic chunking.
				- **Normalized embeddings + cosine distance** – makes ChromaDB scores stable and interpretable.
				- **Persistent ChromaDB** – ingestion and search run as separate steps, mirroring a real RAG pipeline.
				- **Separate benchmark script** – keeps latency measurement repeatable and easy to log.
				- **JSONL/JSON loader in ingest** – Week 1 emits `clean_corpus.jsonl`, so Week 2 reads that format directly instead of forcing text files.
				
				### ChromaDB Edge Cases Handled
				
				- Duplicate chunk IDs are prevented by including document identity in the chunk ID namespace. As a defensive safeguard, ChromaDB upsert batches deduplicate repeated IDs before insertion because ChromaDB rejects duplicate IDs within a single batch.
				Empty/whitespace text → no chunks; unicode preserved; no‑separator text → hard split; duplicate IDs → upsert update; empty docs → single space; NaN/Inf embeddings → skipped + logged; dimension mismatch → clear error + `--reset`; empty collection query → no results; unsupported metadata → sanitized to primitives; large ingests → batched; non‑finite query vector → validation error.
				
				### Corpus Scope for Week 2
				
				Week 1 produced approximately 49,324 cleaned Wikipedia documents. For Week 2, embedding generation on CPU is computationally expensive, so the submitted vector database indexes a representative subset of the cleaned corpus using:
				
				```bash
				python scripts/ingest_vectors.py --limit-docs 500 --reset
				
				### Known Limitations (Week 2)
				
				- Chunk size is in characters, not tokenizer tokens.
				- Benchmark latency depends on hardware, warm‑up, and corpus size.
				- ChromaDB HNSW params left at defaults.
				- Metadata filtering supported but lightly tested.
				- Pipeline assumes text/JSONL/JSON; PDF/DOCX need an extractor.
				
				### Submission Checklist (Week 2)
				
				- [ ] `src/` Week 2 modules present and imported cleanly
				- [ ] `scripts/` Week 2 wrappers run from repo root
				- [ ] `tests/test_chunking.py`, `test_embedding.py`, `test_vector_store.py` pass
				- [ ] `reports/week2/ingest_log.json` generated (non‑zero counts)
				- [ ] `reports/week2/benchmark.csv` + `benchmark_summary.json` generated
				- [ ] `data/queries/benchmark_queries.json` committed
				- [ ] `chroma_db/` git‑ignored; regen command documented
				- [ ] README Week 2 section complete with real numbers
				- [ ] Roadmap Week 2 row = Done
				
				---
				
				## Global Dependencies
				
				Exact versions are locked in `requirements.txt`.
				
				| Package | Purpose | Introduced |
				|---|---|---|
				| pandas | Data handling & Parquet export | Week 1 |
				| datasets | Hugging Face dataset download | Week 1 |
				| pyarrow | Parquet support | Week 1 |
				| spacy | Tokenization / NLP support | Week 1 |
				| beautifulsoup4 | HTML stripping | Week 1 |
				| sentence-transformers | Embedding generation | Week 2 |
				| chromadb | Vector storage & semantic search | Week 2 |
				| numpy | Vector/array handling | Week 2 |
				| torch (+CUDA build) | Backend for sentence-transformers | Week 1/2 |
				| pytest | Unit testing | Week 1 |
				| python-dotenv | Environment variable loading | Week 1 |
