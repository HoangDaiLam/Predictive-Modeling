"""Append-only JSON-lines log of every ingestion attempt (accepted AND rejected).

Temporary stand-in for the PostgreSQL tables of Phase 6. Rejected payloads are
kept raw, so nothing is destroyed.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import settings

_lock = threading.Lock()


def write_event(
    *,
    status: str,
    http_status: int,
    payload: Any,
    record_id: Optional[str] = None,
    vehicle_id: Optional[str] = None,
    source: Optional[str] = None,
    errors: Optional[list] = None,
) -> None:
    event = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "record_id": record_id,
        "status": status,
        "http_status": http_status,
        "vehicle_id": vehicle_id,
        "source": source,
        "errors": errors or [],
        "payload": payload,
    }
    path = Path(settings.ingestion_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")
