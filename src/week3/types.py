from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Chunk:
    id: str
    text: str
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
