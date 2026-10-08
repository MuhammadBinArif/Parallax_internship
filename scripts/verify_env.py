#!/usr/bin/env python3
"""
Verify that the Week 1 environment is ready.

Checks:
- Required package imports
- CUDA/GPU availability
- ChromaDB smoke test
- spaCy import and optional model check
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

REQUIRED_MODULES = {
    "pandas": "pandas",
    "sentence_transformers": "sentence-transformers",
    "chromadb": "chromadb",
    "spacy": "spaCy",
    "torch": "PyTorch",
    "bs4": "beautifulsoup4",
    "datasets": "datasets",
    "pyarrow": "pyarrow",
    "pytest": "pytest",
}


def check_import(module_name: str, package_name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "module": module_name,
        "package": package_name,
        "ok": False,
        "version": None,
        "error": None,
    }

    try:
        module = importlib.import_module(module_name)
        result["ok"] = True
        result["version"] = getattr(module, "__version__", "unknown")

        if module_name == "sentence_transformers":
            has_class = hasattr(module, "SentenceTransformer")
            result["has_sentence_transformer"] = has_class
            if not has_class:
                result["ok"] = False
                result["error"] = "SentenceTransformer class not found"

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def check_gpu() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "checked": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "torch_cuda_version": None,
        "cudnn_version": None,
        "error": None,
    }

    try:
        import torch

        result["checked"] = True
        result["torch_cuda_version"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["device_count"] = int(torch.cuda.device_count()) if result["cuda_available"] else 0

        if result["cuda_available"]:
            for i in range(result["device_count"]):
                props = torch.cuda.get_device_properties(i)
                result["devices"].append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    }
                )

            result["cudnn_version"] = torch.backends.cudnn.version()

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def check_chroma() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "error": None,
    }

    try:
        import chromadb

        with tempfile.TemporaryDirectory() as tmpdir:
            client = chromadb.PersistentClient(path=tmpdir)
            collection = client.get_or_create_collection(name="verify_env")

            collection.add(
                ids=["verify-1"],
                embeddings=[[0.1, 0.2, 0.3]],
                documents=["environment check"],
            )

            got = collection.get(ids=["verify-1"])
            queried = collection.query(
                query_embeddings=[[0.1, 0.2, 0.3]],
                n_results=1,
            )

            documents = got.get("documents") or []
            ids = queried.get("ids") or []

            ok = bool(
                documents
                and documents[0] == "environment check"
                and ids
                and ids[0]
                and ids[0][0] == "verify-1"
            )

            result["ok"] = ok

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def check_spacy() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "version": None,
        "model_installed": False,
        "model_runs": False,
        "error": None,
    }

    try:
        import spacy

        result["ok"] = True
        result["version"] = spacy.__version__
        result["model_installed"] = spacy.util.is_package("en_core_web_sm")

        if result["model_installed"]:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            doc = nlp("Hello environment check")
            result["model_runs"] = len(doc) > 0

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Week 1 environment")
    parser.add_argument(
        "--strict-gpu",
        action="store_true",
        help="Fail if CUDA/GPU is not available",
    )
    parser.add_argument(
        "--require-spacy-model",
        action="store_true",
        help="Fail if en_core_web_sm is missing or not working",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "imports": {},
        "gpu": check_gpu(),
        "chroma": check_chroma(),
        "spacy": check_spacy(),
    }

    failures = []

    for module_name, package_name in REQUIRED_MODULES.items():
        result = check_import(module_name, package_name)
        report["imports"][module_name] = result
        if not result["ok"]:
            failures.append(
                f"Import failed: {package_name} ({module_name}) -> {result['error']}"
            )

    if args.strict_gpu and not report["gpu"]["cuda_available"]:
        failures.append(
            "CUDA/GPU not available. Install CUDA-enabled PyTorch or run on a GPU machine."
        )

    if not report["chroma"]["ok"]:
        failures.append(f"ChromaDB smoke test failed -> {report['chroma']['error']}")

    if not report["spacy"]["ok"]:
        failures.append(f"spaCy import failed -> {report['spacy']['error']}")

    if args.require_spacy_model and not (
        report["spacy"]["model_installed"] and report["spacy"]["model_runs"]
    ):
        failures.append(
            "spaCy model en_core_web_sm is not installed/working. "
            "Run: python -m spacy download en_core_web_sm"
        )

    report["passed"] = len(failures) == 0
    report["failures"] = failures

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2, default=str))

    if failures:
        raise AssertionError("Environment verification failed:\n- " + "\n- ".join(failures))

    print("\nEnvironment verification passed.")


if __name__ == "__main__":
    main()
