"""Read-only FastAPI service over the canonical CSV file."""
from pathlib import Path
import os
from fastapi.responses import RedirectResponse
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


def create_app(dataset_path: str | Path | None = None) -> FastAPI:
    path = Path(dataset_path or os.getenv("DATASET_PATH", "data/ev_dataset.csv"))
    app = FastAPI(title="EV Synthetic Dataset API", version="1.0.0", docs_url="/documentation", redoc_url="/reference", description="Read-only API for trip-level synthetic EV data. Docs page is served " \
    "at first")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])

    def load_data() -> pd.DataFrame:
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Dataset not found at {path}; run scripts/generate_dataset.py first.")
        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Unable to read dataset: {exc}")
        
    @app.get("/", include_in_schema=False)
    def redirect_to_docs():
        return RedirectResponse(url="/documentation")
    
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/schema")
    def schema():
        df = load_data()
        return {"row_count": len(df), "dataset_version": str(df.dataset_version.iloc[0]) if len(df) and "dataset_version" in df else None, "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]}

    @app.get("/dataset")
    def dataset(offset: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=5000)):
        df = load_data()
        page = df.iloc[offset:offset + limit]
        page = page.astype(object).where(pd.notna(page), None)
        return {"total": len(df), "offset": offset, "limit": limit, "returned": len(page), "data": page.to_dict(orient="records")}

    @app.get("/dataset.csv")
    def download_csv():
        from fastapi.responses import FileResponse
        if not path.exists():
            raise HTTPException(status_code=503, detail="Dataset file not found.")
        return FileResponse(path, media_type="text/csv", filename=path.name)

    @app.get("/dataset/{sample_id}")
    def get_sample(sample_id: str):
        df = load_data()
        rows = df.loc[df.sample_id == sample_id]
        if rows.empty:
            raise HTTPException(status_code=404, detail="Sample not found.")
        return rows.iloc[0].astype(object).where(pd.notna(rows.iloc[0]), None).to_dict()

    @app.get("/summary")
    def summary():
        df = load_data()
        numeric = df.select_dtypes(include="number")
        categorical = df.select_dtypes(include=["object", "category"])
        return {"row_count": len(df), "column_count": len(df.columns), "dataset_version": str(df.dataset_version.iloc[0]) if len(df) and "dataset_version" in df else None, "numeric_summary": numeric.describe().round(4).to_dict(), "categorical_values": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in categorical if c not in {"sample_id", "dataset_version"}}}

    return app

app = create_app()
