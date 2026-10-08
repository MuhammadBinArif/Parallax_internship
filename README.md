# Internship Repository

Single repository for weekly internship submissions.

## Week 1: Environment, Data Ingestion & Preprocessing

### Objective

Build a reproducible foundation for a text-based semantic search pipeline by:

- setting up a clean Python environment
- verifying dependencies and GPU/CUDA availability
- downloading a real-world text dataset
- implementing modular text preprocessing
- adding unit tests
- producing a validated clean corpus

This week focuses on engineering reliability. The cleaned dataset and modular preprocessing pipeline created here will be used in later weeks for embedding generation, vector storage, and semantic search.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                     # Raw downloaded dataset, git-ignored
│   └── processed/               # Cleaned dataset, git-ignored
├── reports/                     # Small validation reports, committed
├── scripts/
│   ├── verify_env.py            # Environment and dependency verification
│   ├── download_dataset.py      # Raw dataset downloader
│   └── build_clean_corpus.py    # Cleans raw data and writes processed corpus
├── src/
│   ├── __init__.py
│   └── preprocessing.py         # Modular text cleaning functions
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py    # Unit tests for preprocessing functions
├── .gitignore                   # Ignores data files, .env, caches, venv/conda artifacts
├── .env.example                 # Example environment variables, safe to commit
├── requirements.txt             # Locked Python dependencies
└── README.md
```

---

## What I Built

### 1. Reproducible Environment

I created a dedicated conda environment for the internship project to avoid dependency conflicts with the base Anaconda environment.

Environment name:

```bash
ParallaxLabsInternship
```

Python version used:

```bash
Python 3.10
```

Create and activate the environment:

```bash
conda create --name ParallaxLabsInternship python=3.11 -y
conda activate ParallaxLabsInternship
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If GPU support is required, install a CUDA-compatible PyTorch build first.

For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The `.gitignore` file excludes:

- large raw and processed data files
- `.env` secret files
- virtual environment folders
- Python caches
- model caches
- common OS files

Only source code, configuration examples, dependency files, and small validation reports are intended to be committed.

---

### 2. Environment Verification Script

The script:

```bash
scripts/verify_env.py
```

is used to verify that the project environment is ready before running the data pipeline.

It checks:

- required package imports
- pandas availability
- sentence-transformers availability
- ChromaDB availability
- spaCy availability
- BeautifulSoup availability
- PyTorch availability
- CUDA/GPU availability
- ChromaDB basic smoke test
- optional spaCy English model availability

Run environment verification in CPU mode:

```bash
python scripts/verify_env.py --require-spacy-model --output-report reports/verify_env.json
```

Run environment verification with strict GPU checking:

```bash
python scripts/verify_env.py --strict-gpu --require-spacy-model --output-report reports/verify_env.json
```

The output report is written to:

```bash
reports/verify_env.json
```

This report can be inspected to confirm which dependencies and hardware features are available.

---

### 3. Dataset Ingestion

The script:

```bash
scripts/download_dataset.py
```

downloads a real-world text dataset and saves the raw records into:

```bash
data/raw/
```

Dataset used:

| Item | Value |
|---|---|
| Source | Hugging Face Datasets |
| Dataset | `wikimedia/wikipedia` |
| Config | `20231101.en` |
| Split | `train` |
| Download mode | Streaming |
| Raw documents downloaded | 50,000 |
| Raw output file | `data/raw/wikipedia_raw.jsonl` |

Run the downloader:

```bash
python scripts/download_dataset.py
```

The raw dataset file is large and is intentionally git-ignored.

---

### 4. Modular Text Preprocessing

The preprocessing logic is implemented in:

```bash
src/preprocessing.py
```

This module contains reusable functions for cleaning and standardizing text documents.

Main functions:

| Function | Purpose |
|---|---|
| `strip_html()` | Removes HTML tags and extracts readable text |
| `normalize_unicode()` | Normalizes Unicode and removes invisible/control characters |
| `clean_whitespace()` | Collapses excessive spaces and newlines |
| `remove_urls_and_emails()` | Removes URLs and email addresses |
| `clean_text()` | Runs the full text cleaning pipeline |
| `is_mostly_english()` | Applies a lightweight English-language heuristic |
| `extract_text()` | Extracts text from selected document fields |
| `clean_document()` | Cleans one document and returns a standard schema |
| `iter_clean_documents()` | Applies cleaning across many documents |

Each cleaned document is converted into a consistent schema:

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

This standard schema makes the dataset easier to validate and reuse in later weeks.

---

### 5. Unit Tests

Unit tests are located in:

```bash
tests/test_preprocessing.py
```

They validate the preprocessing functions, including:

- HTML tag removal
- Unicode normalization
- whitespace cleanup
- URL and email removal
- English language heuristic behavior
- document schema generation
- filtering of short documents
- handling of invalid records
- ID assignment and duplicate handling

Run tests:

```bash
python -m pytest -v
```

All tests should pass before submitting Week 1.

---

### 6. Clean Corpus Generation

The script:

```bash
scripts/build_clean_corpus.py
```

reads the raw JSONL dataset, applies preprocessing, filters invalid documents, and writes cleaned outputs.

Run the corpus builder:

```bash
python scripts/build_clean_corpus.py
```

Outputs:

```bash
data/processed/clean_corpus.parquet
data/processed/clean_corpus.jsonl
reports/validation_report.json
```

The processed data files are large and are git-ignored. The validation report is small and can be committed.

Inspect the validation report:

```bash
cat reports/validation_report.json
```

Inspect the first cleaned document:

```bash
head -n 1 data/processed/clean_corpus.jsonl
```

Inspect the processed Parquet file:

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

---

## Week 1 Results

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
| Validation report | `reports/validation_report.json` |

The pipeline successfully produced more than the required 5,000 clean documents.

During preprocessing, a BeautifulSoup `XMLParsedAsHTMLWarning` appeared while parsing some Wikipedia records containing XML-like markup. This warning was non-fatal. The HTML parser was kept because it is more robust for messy web-derived text. The warning can be suppressed in `src/preprocessing.py` if desired.

---

## Setup Instructions

### 1. Clone the repository

Replace the URL below with your actual GitHub repository URL:

```bash
git clone <your-github-repository-url>
cd internship-repo
```

---

### 2. Create and activate the conda environment

```bash
conda create --name ParallaxLabsInternship python=3.11 -y
conda activate ParallaxLabsInternship
```

Verify Python version:

```bash
python --version
```

Expected:

```bash
Python 3.11.x
```

---

### 3. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

If GPU support is required, install CUDA-compatible PyTorch first, as shown earlier in this README.

---

## How to Run the Week 1 Pipeline

### Step 1: Verify environment

CPU mode:

```bash
python scripts/verify_env.py --require-spacy-model --output-report reports/verify_env.json
```

Strict GPU mode:

```bash
python scripts/verify_env.py --strict-gpu --require-spacy-model --output-report reports/verify_env.json
```

Purpose:

- confirms required packages are installed
- confirms CUDA/GPU availability when strict mode is used
- confirms ChromaDB can run a basic smoke test
- confirms spaCy is available
- writes a machine-readable report to `reports/verify_env.json`

---

### Step 2: Download raw dataset

```bash
python scripts/download_dataset.py
```

Purpose:

- downloads real-world text documents
- saves raw records before cleaning
- writes output to `data/raw/wikipedia_raw.jsonl`
- keeps raw data separate from processed data

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
				
				```bash
				Valid JSON lines: 50000
				Invalid JSON lines: 0
				```
				
				---
				
				### Step 3: Run unit tests
				
				```bash
				python -m pytest -v
				```
				
				Purpose:
				
				- verifies preprocessing functions behave correctly
				- catches silent bugs in text cleaning
				- confirms the pipeline is safe to run on the full dataset
				
				---
				
				### Step 4: Build clean corpus
				
				```bash
				python scripts/build_clean_corpus.py
				```
				
				Purpose:
				
				- reads raw JSONL data
				- applies modular preprocessing
				- filters short, empty, invalid, or non-English documents
				- standardizes document schema
				- writes processed dataset to Parquet and JSONL
				- writes a validation report
				
				Outputs:
				
				```bash
				data/processed/clean_corpus.parquet
				data/processed/clean_corpus.jsonl
				reports/validation_report.json
				```
				
				---
				
				### Step 5: Inspect validation report
				
				```bash
				cat reports/validation_report.json
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

---

## Dependencies

Key dependencies used in Week 1:

| Package | Purpose |
|---|---|
| pandas | Data handling and Parquet export |
| sentence-transformers | Planned embedding generation in future weeks |
| chromadb | Planned vector storage and semantic search in future weeks |
| spacy | Tokenization and NLP support |
| beautifulsoup4 | HTML stripping |
| datasets | Hugging Face dataset downloading |
| pyarrow | Parquet support |
| pytest | Unit testing |
| python-dotenv | Environment variable loading |

Exact package versions are locked in:

```bash
requirements.txt
```

---

## Approach and Design Decisions

### Why use a dedicated conda environment?

A dedicated environment isolates project dependencies and avoids conflicts with the base Anaconda environment. This makes the project easier to reproduce and reduces the chance of version-related errors.

### Why lock dependencies in `requirements.txt`?

Locked dependencies allow the environment to be recreated reliably. Without version pinning, future installs may pull incompatible package versions.

### Why use `.gitignore`?

Large datasets and secret files should not be committed to Git. The `.gitignore` file keeps the repository clean, portable, and safe.

### Why keep raw and processed data separate?

Raw data preserves the original source records. Processed data is derived from raw data and can be regenerated. Separating them improves reproducibility, debugging, and data lineage.

### Why use modular preprocessing?

Text cleaning logic is placed in `src/preprocessing.py` so it can be reused, tested independently, and extended in later weeks without rewriting scripts.

### Why write unit tests?

Preprocessing code can fail silently. Unit tests verify that HTML removal, Unicode normalization, whitespace cleanup, language filtering, and document schema generation behave correctly.

### Why output both Parquet and JSONL?

Parquet is compact and efficient for tabular data. JSONL is easy to inspect line by line and useful for debugging. Providing both formats gives flexibility for later pipeline stages.

### Why use a heuristic language filter?

For Week 1, a lightweight English heuristic is sufficient to remove obvious non-English noise. In a production system, a dedicated language detection model could be used instead.

---

## Validation Checks

The corpus builder performs validation checks before writing outputs.

Checks include:

- cleaned dataset is not empty
- document IDs are unique
- no null text values exist
- all documents meet the minimum character count
- all documents are within the maximum character count
- `char_count` matches actual text length

If any check fails, the script raises an error instead of producing an invalid dataset.

---

## Known Limitations

- Language filtering is heuristic, not a full production-grade language detector.
- The dataset is a sampled subset, not the complete Wikipedia dump.
- Long documents are truncated to keep processing manageable.
- GPU verification depends on the host machine having a compatible NVIDIA driver and CUDA-enabled PyTorch installation.
- Large raw and processed data files are intentionally not committed to Git.
- Some Wikipedia records contain XML-like markup, which can trigger BeautifulSoup parsing warnings. This was treated as non-fatal.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src.preprocessing'`

Make sure you are running commands from the repository root:

```bash
cd ~/internship-repo
```

Confirm the file exists:

```bash
ls -lh src/preprocessing.py
```

Confirm the package marker exists:

```bash
ls -lh src/__init__.py
```

Test the import:

```bash
python -c "from src.preprocessing import iter_clean_documents; print('Import OK')"
```

---

### Dataset download fails with `DatasetNotFoundError`

Use a public dataset such as:

```bash
wikimedia/wikipedia
```

The default downloader in this repository uses:

```bash
wikimedia/wikipedia
```

with config:

```bash
20231101.en
```

---

### Hugging Face download times out

The downloader retries automatically. If the process crashes after saving the file, validate the raw JSONL file before rerunning:

```bash
wc -l data/raw/wikipedia_raw.jsonl
```

Expected:

```bash
50000 data/raw/wikipedia_raw.jsonl
```

---

### `XMLParsedAsHTMLWarning` appears

This warning can appear when parsing messy Wikipedia text. It is non-fatal.

If desired, suppress it in `src/preprocessing.py` near the top:

```python
import warnings

try:
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
pass
```

---

### `pytest` cannot import `src`

Run tests from the repository root:

```bash
python -m pytest -v
```

Make sure these files exist:

```bash
src/__init__.py
tests/__init__.py
```

---

## Git Submission Checklist

Before submitting Week 1, verify:

- [ ] Single GitHub repository is used for the internship
- [ ] `README.md` is updated with Week 1 progress
- [ ] `.gitignore` ignores large data files and `.env`
- [ ] `requirements.txt` exists and contains locked dependencies
- [ ] `scripts/verify_env.py` runs successfully
- [ ] `scripts/download_dataset.py` downloads 5,000+ raw documents
- [ ] `src/preprocessing.py` implements modular cleaning functions
- [ ] `tests/test_preprocessing.py` exists
- [ ] `python -m pytest -v` passes
- [ ] `scripts/build_clean_corpus.py` produces 5,000+ clean documents
- [ ] `data/processed/clean_corpus.parquet` exists locally
- [ ] `data/processed/clean_corpus.jsonl` exists locally
- [ ] `reports/validation_report.json` exists
- [ ] Large data files are not staged or committed
- [ ] Code, tests, reports, README, and requirements are pushed to GitHub

---

## Suggested Final Commands

Run these from the repository root:

```bash
cd ~/internship-repo
```

Verify environment:

```bash
python scripts/verify_env.py --require-spacy-model --output-report reports/verify_env.json
```

Download dataset:

```bash
python scripts/download_dataset.py
```

Run tests:

```bash
python -m pytest -v
```

Build clean corpus:

```bash
python scripts/build_clean_corpus.py
```

Inspect outputs:

```bash
ls -lh data/raw data/processed reports
cat reports/validation_report.json
```

Stage safe files:

```bash
git add README.md .gitignore .env.example requirements.txt scripts src tests reports
git add data/raw/.gitkeep data/processed/.gitkeep
```

If `.gitkeep` files are ignored, force-add only those tiny files:

```bash
git add -f data/raw/.gitkeep data/processed/.gitkeep
```

Check status:

```bash
git status
```

Commit:

```bash
git commit -m "Week 1: environment setup, data ingestion, preprocessing, tests"
```

Rename branch to `main`:

```bash
git branch -M main
```

Add remote:

```bash
git remote add origin <your-github-repository-url>
```

Push:

```bash
git push -u origin main
```

---

## Next Steps

Week 2 will build on this clean corpus by:

- generating document embeddings using `sentence-transformers`
- storing embeddings and metadata in `ChromaDB`
- implementing semantic search over the cleaned corpus
