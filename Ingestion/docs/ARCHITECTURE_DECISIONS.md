# Architecture decisions: retain, update, remove

## Retain, with updated scope

### Data-quality metrics — retain and strengthen
The quality gate checks required columns, nulls, unique IDs, duplicate rows, plausible numeric ranges, distance-duration-average-speed consistency, positive distance/duration, and consistency between total energy and kWh/100 km. A JSON report is saved under `data/reports/`. Run this gate after every generation and before model training.

### Data feasibility — retain as a concise technical assessment
- 10,000 rows × roughly 40 columns is small for Pandas and scikit-learn.
- CSV is appropriate at this scale, portable, inspectable and easy to version.
- FastAPI offers a shared read-only access contract for dashboards and teammates.
- Synthetic labels are suitable for pipeline development and controlled model comparisons, not proof of real-world vehicle accuracy.
- Use a held-out test set. If repeated vehicle identities/configurations are later introduced, consider group-based splits to avoid leakage.

### Postman — retain as optional API verification
Keep the small Postman collection for endpoint smoke tests and project demonstrations. It is not a datastore, production dependency, or ML requirement. FastAPI `/docs` and automated pytest tests are the primary verification mechanisms.

## Update

### Shared data access
1. One canonical versioned CSV in shared storage or the team repository.
2. FastAPI read-only endpoints for common access and pagination.
3. Direct CSV loading remains supported for local model experiments.
4. Store dataset version and random seed in each row and report.

### Versioning
Git can track a small CSV if repository size and course rules allow it. If datasets grow or are regenerated frequently, store releases in shared cloud storage or Git LFS; keep code, schema and metadata in Git. Never commit credentials.

## Remove
- Supabase/PostgreSQL schema, migrations, RLS policies and database uploader.
- OBD/CAN/live OEM API ingestion assumptions: this phase uses synthetic trip-level samples.
- Database authentication and write endpoints: the shared dataset service is read-only.

## Physical-model caveat
Battery capacity and battery health are retained as vehicle context fields, but do not directly alter this simplified energy-demand target; battery health would require an explicit internal-resistance/thermal model to affect losses credibly. The generator is a transparent simplified trip-level energy approximation. It includes rolling resistance, aerodynamic drag, signed elevation work, stop-start kinetic-energy cycling with regenerative recovery, HVAC and auxiliary loads. Aerodynamic work uses representative trip-average speed rather than integrating a measured second-by-second speed trace. The model is internally consistent, but is not a calibrated digital twin or validated real-vehicle predictor. Passing data-quality checks proves data integrity and arithmetic identities, not physical accuracy.


## Streamlit dashboard

Streamlit is added as an operator-facing interface, not as a replacement for the API or CSV. It supports controlled generation and validation, local CSV inspection/download, API health/schema/summary inspection, paginated record browsing, sample lookup, and API-served CSV download. Run it separately from FastAPI. The generator only replaces the canonical CSV after both in-memory and serialized-file validation pass.
