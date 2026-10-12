from fastapi import FastAPI

app = FastAPI(
    title="EV Telemetry Ingestion",
    version="0.1.0",
    description="Ingestion service for EV battery energy-consumption prediction.",
)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness check. Used by us now, and by schedulers/monitors later."""
    return {"status": "ok", "service": "ev-telemetry-ingestion"}
