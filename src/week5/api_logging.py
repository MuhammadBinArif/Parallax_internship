from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Request


class RequestLogger:
    """Thread-safe JSONL logger for FastAPI requests."""

    def __init__(self, path: str = "reports/week5/api_request_logs.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record: dict[str, Any]) -> None:
        record.setdefault(
            "timestamp",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def new_request_id() -> str:
    return uuid.uuid4().hex


async def attach_request_context(request: Request) -> tuple[str, float]:
    """Attach request_id and start time. Returns (request_id, start_time)."""
    request_id = request.headers.get("X-Request-ID") or new_request_id()
    start = time.perf_counter()
    request.state.request_id = request_id
    request.state.start_time = start
    return request_id, start


def build_log_record(
    request: Request,
    status_code: int,
    error_type: Optional[str],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", "unknown")
    start = getattr(request.state, "start_time", None)
    latency_ms = (
        round((time.perf_counter() - start) * 1000.0, 2)
        if start is not None else 0.0
    )

    record: dict[str, Any] = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "client": request.client.host if request.client else None,
        "error_type": error_type,
    }

    if extra:
        record.update(extra)

    return record
