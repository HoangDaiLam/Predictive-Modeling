# EV Synthetic Dataset + FastAPI

Lightweight shared-data platform for the EV battery energy-consumption ML project.

**Default dataset:** 10,000 rows; each row represents one complete trip, not one second of telemetry.

## 1. Install

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Generate and validate the canonical CSV

```bash
python scripts/generate_dataset.py
```

Optional:

```bash
python scripts/generate_dataset.py --rows 10000 --seed 42 --version v1
```

Creates `data/ev_dataset.csv` and `data/reports/quality_v1.json`. Generation stops if the quality gate fails. Use a new version label for materially changed data, e.g. `v2`.

## 3. Tests

```bash
pytest -q
```

## 4. Start the API

```bash
uvicorn ev_platform.api:app --app-dir src --reload
```

Interactive API documentation: http://127.0.0.1:8000/docs

Endpoints:
- `GET /health`
- `GET /schema`
- `GET /dataset?offset=0&limit=1000` (paginated; maximum 5,000 per page)
- `GET /dataset.csv` (download canonical CSV)
- `GET /dataset/{sample_id}`
- `GET /summary`


## 4.5 Streamlit dashboard

The Streamlit dashboard provides a UI for both dataset generation and API inspection. It does not replace FastAPI or the canonical CSV.

Run it in a separate terminal:

```bash
streamlit run dashboard.py
```

Open the local Streamlit URL printed in the terminal (normally http://localhost:8501).

Dashboard sections:
- **Overview:** local CSV status, API health check, Swagger link and CSV download.
- **Data generation:** choose row count, seed and dataset version; generate data; run in-memory and serialized-CSV quality checks; publish only if both pass; inspect and download the result.
- **FastAPI explorer:** inspect schema and summary, browse paginated rows, retrieve a sample by ID, and download the CSV through the API.

Run FastAPI separately in another terminal before using the API explorer:

```bash
uvicorn ev_platform.api:app --app-dir src --reload
```

The generator page works without FastAPI running. It writes the canonical file to `data/ev_dataset.csv`; FastAPI reads that file on each request, so subsequent API requests see the newly published dataset.

## 5. Teammate access

Preferred for ML: load the canonical CSV directly.

```python
import pandas as pd
df = pd.read_csv("data/ev_dataset.csv")
```

Or retrieve a page through the API:

```python
import requests
import pandas as pd
payload = requests.get(
    "http://127.0.0.1:8000/dataset",
    params={"offset": 0, "limit": 1000}, timeout=30
).json()
page = pd.DataFrame(payload["data"])
```

For all rows via API, loop over offsets or download `/dataset.csv`; one paginated response is not the full dataset.

## 6. Postman

Import `postman/EV_Dataset_API.postman_collection.json`. Set `baseUrl` to the API host. Postman is optional; `/docs` and pytest are the primary checks.

## 7. ML leakage warning

Targets are `energy_consumption_kwh` and `consumption_kwh_per_100km`. Do not include either target or the energy-model audit columns in input features. Exclude `sample_id` and `dataset_version` too. See `docs/DATA_DICTIONARY.md`.

## 8. Physical assumptions and limits

The generator uses a transparent simplified trip-level model: rolling resistance, aerodynamic drag, signed net elevation work, representative stop-start acceleration and regenerative braking, HVAC and auxiliary loads. It derives trip duration from distance and average speed, so those measurements satisfy the average-speed identity by construction. Aerodynamic work uses representative average trip speed rather than a second-by-second speed trace. This is suitable for synthetic ML pipeline development and controlled experiments, but is not calibrated against measured vehicle data and must not be described as real-world validated performance.

The quality report checks integrity and arithmetic consistency; passing it does not prove physical realism or predictive validity. Validate against real measurements if available.

## 9. Architecture decisions

- **Keep:** data-quality metrics, a concise feasibility assessment, optional Postman collection.
- **Update:** versioning, validation, API access and ML leakage controls.
- **Remove:** Supabase/database schema and uploader, live OEM API/OBD ingestion assumptions, database write/auth infrastructure.

See `docs/ARCHITECTURE_DECISIONS.md`.

Use the leakage-safe helper when preparing model inputs:

```python
from ev_platform.features import make_xy
X, y = make_xy(df, target="consumption_kwh_per_100km")
```

It excludes IDs, target fields and energy-model audit components. Categorical variables still need encoding before most scikit-learn estimators.
