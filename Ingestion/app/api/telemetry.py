import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter

from app.schemas.telemetry import TelemetryAccepted, TelemetryIn
from app.services import ingestion_log

logger = logging.getLogger("ingestion")
router = APIRouter(prefix="/api/v1", tags=["telemetry"])


@router.post("/telemetry", status_code=201, response_model=TelemetryAccepted)
def ingest_telemetry(payload: TelemetryIn) -> TelemetryAccepted:
    # By the time this function runs, FastAPI+Pydantic already validated the structure.
    # Malformed requests never reach here (see the 422 handler in main.py).
    record_id = str(uuid.uuid4())
    received_at = datetime.now(timezone.utc)  # server-assigned, never trusted from the client

    ingestion_log.write_event(
        status="accepted",
        http_status=201,
        record_id=record_id,
        vehicle_id=payload.vehicle_id,
        source=payload.source.value,
        payload=payload.model_dump(mode="json"),
    )
    logger.info("accepted record_id=%s vehicle=%s source=%s", record_id, payload.vehicle_id, payload.source.value)
    return TelemetryAccepted(record_id=record_id, received_at=received_at, vehicle_id=payload.vehicle_id)
