# EV Telemetry Ingestion

Ingestion pipeline for the EV battery energy-consumption prediction project.
Role: Hardware & Data Ingestion Specialist.

Planned flow:
Postman / simulated vehicle -> FastAPI -> schema validation -> feasibility validation
-> PostgreSQL -> validated dataset -> (later) Volvo/Polestar APIs, OBD-II/CAN loggers.

## Status
- [x] Phase 1: telemetry contract (`docs/telemetry_contract.md`)
- [x] Phase 2: project skeleton + `/health`
- [ ] Phase 3: `POST /api/v1/telemetry`

## Setup

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```bash
uvicorn app.main:app --reload
```
Then open http://127.0.0.1:8000/docs

## Test
```bash
pytest -v
```

## Layout
| Path | Purpose |
|---|---|
| `app/main.py` | FastAPI entry point |
| `app/api/` | Route handlers (Phase 3) |
| `app/schemas/` | Pydantic models = the contract in code (Phase 3) |
| `app/validation/` | Range / temporal / physical checks (Phase 5) |
| `app/db/` | SQLAlchemy models, connection (Phase 6) |
| `app/services/` | Ingestion logic, quality metrics (Phases 6-7) |
| `tests/` | pytest tests |
| `docs/` | Telemetry contract and design notes |
| `postman/` | Exported Postman collection (Phase 4) |
| `scripts/` | Simulators and helpers |
| `data/` | Local data files (git-ignored) |
