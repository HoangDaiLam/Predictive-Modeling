"""Streamlit control panel for synthetic data generation and FastAPI access."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.ev_platform.generator import generate_dataset
from src.ev_platform.quality import quality_report

DEFAULT_DATASET = ROOT / "data" / "ev_dataset.csv"
REPORT_DIR = ROOT / "data" / "reports"

st.set_page_config(
    page_title="EV Data Platform",
    page_icon="----/----",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    /* Styling for the periodic table chemical logo style */
    .element-element {
        display: inline-block;
        background-color: #005f33;
        color: #ffffff;
        font-weight: bold;
        font-family: 'Courier Nesw', Courier, monospace;
        padding: 4px 12px;
        border: 2px solid #00ff66;
        border-radius: 4px;
        font-size: 2.2rem;
        margin-right: 2px;
        box-shadow: 0 0 10px #00ff66;
    }
    .element-text {
        font-size: 2.2rem;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        color: #ffffff;
        letter-spacing: -1px;
    }
    /* Glow effect for standard metrics */
    [data-testid="stMetricValue"] {
        color: #00ff66 !important;
        text-shadow: 0 0 8px rgba(0, 255, 102, 0.6);
    }
</style>
""", unsafe_allow_html=True)


def local_dataset_status() -> tuple[bool, str, pd.DataFrame | None]:
    if not DEFAULT_DATASET.exists():
        return False, "Canonical CSV has not been generated yet.", None
    try:
        df = pd.read_csv(DEFAULT_DATASET)
        return True, f"Loaded {len(df):,} rows and {len(df.columns)} columns.", df
    except Exception as exc:  # dashboard should display the problem, not crash
        return False, f"Could not read the canonical CSV: {exc}", None


def api_get(base_url: str, endpoint: str, params: dict[str, Any] | None = None):
    url = base_url.rstrip("/") + endpoint
    return httpx.get(url, params=params, timeout=15.0)


def show_quality_report(report: dict[str, Any]) -> None:
    passed = bool(report.get("passed"))
    if passed:
        st.success("All data-quality checks passed.")
    else:
        st.error("The generated dataset failed the quality gate. It was not published.")

    checks = report.get("checks", {})
    if checks:
        rows = [{"Check": name.replace("_", " ").title(), "Result": "PASS" if ok else "FAIL"}
                for name, ok in checks.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{report.get('row_count', 0):,}")
    c2.metric("Columns", f"{report.get('column_count', 0):,}")
    c3.metric("Speed mismatches", report.get("speed_identity_mismatch_count", "—"))
    with st.expander("Full validation report"):
        st.json(report)


st.title("EV Synthetic Data Monitoring Platform")
st.caption("Generate, validate, inspect and serve the canonical trip-level dataset.")

with st.sidebar:
    st.header("Platform controls")
    api_base = st.text_input("FastAPI base URL", value="http://127.0.0.1:8000")
    st.caption("Run FastAPI in a separate terminal. Data generation works even when the API is offline.")
    page = st.radio("Dashboard", ["Overview", "Data generation", "FastAPI explorer"], index=0)
    st.divider()
    ok_local, local_message, local_df = local_dataset_status()
    st.markdown("**Local dataset**")
    if ok_local:
        st.success(local_message)
    else:
        st.warning(local_message)

if page == "Overview":
    st.subheader("Platform overview")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Local CSV")
        if ok_local and local_df is not None:
            st.metric("Trip samples", f"{len(local_df):,}")
            st.metric("Features and targets", len(local_df.columns))
            version = str(local_df["dataset_version"].iloc[0]) if "dataset_version" in local_df else "unknown"
            st.write(f"Dataset version: **{version}**")
            st.download_button(
                "Download canonical CSV",
                data=DEFAULT_DATASET.read_bytes(),
                file_name=DEFAULT_DATASET.name,
                mime="text/csv",
                key="overview_csv_download",
            )
            st.markdown("**Target distribution**")
            counts, edges = np.histogram(local_df["consumption_kwh_per_100km"], bins=12)
            labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges)-1)]
            st.bar_chart(pd.DataFrame({"Trips": counts}, index=labels))
        else:
            st.info("Generate the initial dataset from the Data generation page.")
    with c2:
        st.markdown("#### FastAPI service")
        if st.button("Check API health", key="overview_health"):
            try:
                response = api_get(api_base, "/health")
                response.raise_for_status()
                st.session_state["api_health"] = response.json()
                st.session_state["api_health_error"] = None
            except Exception as exc:
                st.session_state["api_health"] = None
                st.session_state["api_health_error"] = str(exc)
        if st.session_state.get("api_health"):
            st.success(f"API response: {st.session_state['api_health']}")
        elif st.session_state.get("api_health_error"):
            st.error(f"API unavailable: {st.session_state['api_health_error']}")
        else:
            st.caption("Select Check API health to test the configured service.")
        st.link_button("Open FastAPI Swagger docs", api_base.rstrip("/") + "/docs")

    st.divider()
    st.markdown("#### Data flow")
    st.code("Synthetic generator  →  quality gate  →  canonical CSV  →  FastAPI  →  teammates / ML evaluator", language="text")
    st.info("Streamlit is the operator interface. The CSV remains the source of truth; FastAPI serves it, while ML scripts may also load it directly.")

elif page == "Data generation":
    st.subheader("Generate and publish a dataset")
    st.write("Generation runs locally in the Streamlit process. The new CSV replaces the canonical file only after all validation checks pass.")

    with st.form("generation_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_rows = st.number_input("Number of trip samples", min_value=100, max_value=100_000, value=10_000, step=100)
        with c2:
            seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
        with c3:
            version = st.text_input("Dataset version", value="v1", max_chars=32)
        confirm_replace = st.checkbox("I understand this will replace data/ev_dataset.csv after validation", value=False)
        generate = st.form_submit_button("Generate, validate and publish", type="primary", disabled=not confirm_replace)

    if generate:
        if not version.strip() or any(ch in version for ch in "/\\"):
            st.error("Enter a non-empty version label without slashes.")
        else:
            progress = st.progress(0, text="Preparing generator…")
            try:
                progress.progress(15, text="Generating trip samples…")
                generated = generate_dataset(int(n_rows), int(seed), version.strip())
                progress.progress(70, text="Running data-quality checks…")
                report = quality_report(generated)
                if not report["passed"]:
                    st.error("Quality gate failed. Existing canonical CSV was left unchanged.")
                    st.json(report)
                else:
                    progress.progress(85, text="Writing validated dataset…")
                    DEFAULT_DATASET.parent.mkdir(parents=True, exist_ok=True)
                    REPORT_DIR.mkdir(parents=True, exist_ok=True)
                    tmp_path = DEFAULT_DATASET.with_suffix(".csv.tmp")
                    generated.to_csv(tmp_path, index=False, float_format="%.6f")
                    # Validate the serialized file too, catching rounding/serialization issues.
                    round_trip = pd.read_csv(tmp_path)
                    serialized_report = quality_report(round_trip)
                    if not serialized_report["passed"]:
                        tmp_path.unlink(missing_ok=True)
                        st.error("CSV serialization validation failed. Existing dataset was preserved.")
                        st.json(serialized_report)
                    else:
                        tmp_path.replace(DEFAULT_DATASET)
                        report_path = REPORT_DIR / f"quality_{version.strip()}.json"
                        report_path.write_text(json.dumps(serialized_report, indent=2), encoding="utf-8")
                        st.session_state["last_generation_report"] = serialized_report
                        st.session_state["last_generation_version"] = version.strip()
                        st.session_state["last_generation_time"] = datetime.now(timezone.utc).isoformat()
                        progress.progress(100, text="Dataset published successfully.")
                        st.success(f"Published {len(round_trip):,} rows as {version.strip()}.")
                        st.cache_data.clear()
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
            finally:
                progress.empty()

    if st.session_state.get("last_generation_report"):
        st.markdown("#### Latest generation result")
        st.caption(f"Version {st.session_state.get('last_generation_version')} · {st.session_state.get('last_generation_time')}")
        show_quality_report(st.session_state["last_generation_report"])

    ok_local, local_message, local_df = local_dataset_status()
    if ok_local and local_df is not None:
        st.divider()
        st.markdown("#### Current canonical dataset")
        a, b, c = st.columns(3)
        a.metric("Rows", f"{len(local_df):,}")
        b.metric("Mean consumption", f"{local_df['consumption_kwh_per_100km'].mean():.2f} kWh/100 km")
        c.metric("Mean trip distance", f"{local_df['distance_km'].mean():.2f} km")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("**Energy consumption distribution**")
            counts, edges = np.histogram(local_df["consumption_kwh_per_100km"], bins=12)
            labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges)-1)]
            st.bar_chart(pd.DataFrame({"Trips": counts}, index=labels))
        with chart_right:
            st.markdown("**Samples by road type**")
            st.bar_chart(local_df["road_type"].value_counts().sort_index())
        st.markdown("**First 20 records**")
        st.dataframe(local_df.head(20), use_container_width=True, hide_index=True)
        st.download_button("Download current CSV", DEFAULT_DATASET.read_bytes(), "ev_dataset.csv", "text/csv", key="generation_csv_download")

elif page == "FastAPI explorer":
    st.subheader("FastAPI service explorer")
    st.write("This page reads from FastAPI, not directly from the local CSV. Start the API separately to use these controls.")
    c1, c2 = st.columns([1, 3])
    with c1:
        check = st.button("Refresh API status", type="primary")
    if check:
        try:
            r = api_get(api_base, "/health")
            r.raise_for_status()
            st.session_state["api_status_data"] = r.json()
            st.session_state["api_status_error"] = None
        except Exception as exc:
            st.session_state["api_status_data"] = None
            st.session_state["api_status_error"] = str(exc)
    if st.session_state.get("api_status_data"):
        st.success(f"API online: {st.session_state['api_status_data']}")
    elif st.session_state.get("api_status_error"):
        st.error(f"API unavailable: {st.session_state['api_status_error']}")

    tab_schema, tab_summary, tab_records, tab_sample = st.tabs(["Schema", "Summary", "Browse records", "Find sample"])
    with tab_schema:
        if st.button("Load schema", key="load_schema"):
            try:
                r = api_get(api_base, "/schema"); r.raise_for_status()
                st.session_state["schema_response"] = r.json()
            except Exception as exc:
                st.error(f"Could not load schema: {exc}")
        if st.session_state.get("schema_response"):
            schema = st.session_state["schema_response"]
            a, b = st.columns(2)
            a.metric("Rows", f"{schema.get('row_count', 0):,}")
            b.metric("Version", schema.get("dataset_version") or "unknown")
            st.dataframe(pd.DataFrame(schema.get("columns", [])), use_container_width=True, hide_index=True)
    with tab_summary:
        if st.button("Load summary", key="load_summary"):
            try:
                r = api_get(api_base, "/summary"); r.raise_for_status()
                st.session_state["summary_response"] = r.json()
            except Exception as exc:
                st.error(f"Could not load summary: {exc}")
        if st.session_state.get("summary_response"):
            summary = st.session_state["summary_response"]
            a, b = st.columns(2)
            a.metric("Rows", f"{summary.get('row_count', 0):,}")
            b.metric("Columns", summary.get("column_count", 0))
            numeric = summary.get("numeric_summary", {})
            if numeric:
                st.dataframe(pd.DataFrame(numeric), use_container_width=True)
            with st.expander("Categorical values"):
                st.json(summary.get("categorical_values", {}))
    with tab_records:
        p1, p2 = st.columns(2)
        offset = p1.number_input("Offset", min_value=0, value=0, step=1000, key="records_offset")
        limit = p2.selectbox("Rows per page", [100, 500, 1000, 5000], index=2)
        if st.button("Fetch page", key="fetch_page"):
            try:
                r = api_get(api_base, "/dataset", {"offset": int(offset), "limit": int(limit)})
                r.raise_for_status()
                st.session_state["records_page"] = r.json()
            except Exception as exc:
                st.error(f"Could not fetch records: {exc}")
        if st.session_state.get("records_page"):
            page_data = st.session_state["records_page"]
            st.caption(f"Showing {page_data['returned']:,} of {page_data['total']:,} rows · offset {page_data['offset']}")
            st.dataframe(pd.DataFrame(page_data["data"]), use_container_width=True, hide_index=True)
    with tab_sample:
        sample_id = st.text_input("Sample ID", value="TRIP-000001")
        if st.button("Retrieve sample", key="retrieve_sample"):
            try:
                r = api_get(api_base, f"/dataset/{sample_id.strip()}")
                r.raise_for_status()
                st.session_state["sample_response"] = r.json()
                st.session_state["sample_error"] = None
            except Exception as exc:
                st.session_state["sample_response"] = None
                st.session_state["sample_error"] = str(exc)
        if st.session_state.get("sample_error"):
            st.error(st.session_state["sample_error"])
        if st.session_state.get("sample_response"):
            st.json(st.session_state["sample_response"])

    st.divider()
    st.markdown("#### Download through FastAPI")
    if st.button("Fetch CSV from API", key="fetch_api_csv"):
        try:
            r = api_get(api_base, "/dataset.csv")
            r.raise_for_status()
            st.session_state["api_csv_bytes"] = r.content
        except Exception as exc:
            st.error(f"CSV download failed: {exc}")
    if st.session_state.get("api_csv_bytes"):
        st.download_button("Save API-served CSV", st.session_state["api_csv_bytes"], "ev_dataset_from_api.csv", "text/csv", key="api_csv_download")
