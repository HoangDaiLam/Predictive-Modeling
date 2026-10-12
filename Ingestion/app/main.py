import logging

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from app.api.telemetry import router as telemetry_router
from app.config import settings
from app.services import ingestion_log

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingestion")

app = FastAPI(
    title="Data Ingestion -Predictive Modelling Group -. Telemetry Service",
    version="0.3.0",
    description="Ingestion service for Predicitive Modelling Data Pipeline. Currently under development" \
    "\n Developer: Hamza Sahqani",
)
app.include_router(telemetry_router)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness check. Used by us now, and by schedulers/monitors later."""
    return {"status": "ok", "service": "ev-telemetry-ingestion"}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    """Malformed telemetry -> 422, structured errors, and the RAW body is logged."""
    if request.url.path != "/api/v1/telemetry":
        return await request_validation_exception_handler(request, exc)

    errors = []
    for e in exc.errors():
        if e["type"] == "json_invalid":
            field = "(body)"
        else:
            field = ".".join(str(p) for p in e["loc"][1:]) or "(body)"
        errors.append({"field": field, "message": e["msg"], "type": e["type"]})

    raw = exc.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    vehicle_id = raw.get("vehicle_id") if isinstance(raw, dict) else None
    source = raw.get("source") if isinstance(raw, dict) else None

    ingestion_log.write_event(
        status="rejected",
        http_status=422,
        vehicle_id=vehicle_id if isinstance(vehicle_id, str) else None,
        source=source if isinstance(source, str) else None,
        errors=errors,
        payload=raw,
    )
    logger.warning("rejected vehicle=%s errors=%s", vehicle_id, [x["field"] for x in errors])
    return JSONResponse(status_code=422, content={"status": "rejected", "errors": errors})
