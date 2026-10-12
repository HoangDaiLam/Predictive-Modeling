"""Streamlit ingestion monitor.  Run from the project root:

    streamlit run dashboard/app.py

Phase 3: reads the JSON-lines ingestion log. Phase 6/7: swap load_events() for a SQL query.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.config import settings  # noqa: E402
from app.schemas.telemetry import TelemetryIn  # noqa: E402

st.set_page_config(page_title="EV Ingestion Monitor", layout="wide")
st.title("EV Telemetry Ingestion Monitor")


@st.cache_data(ttl=5)
def load_events(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["logged_at"] = pd.to_datetime(df["logged_at"], utc=True)
    return df


with st.sidebar:
    log_path = st.text_input("Ingestion log path", str(settings.ingestion_log_path))
    if st.button("Refresh now"):
        st.cache_data.clear()
    st.caption("Auto-cached for 5 s. Press R in the browser to rerun.")

df = load_events(log_path)
if df.empty:
    st.info("No events yet. Start the API and run `python scripts/seed_demo.py` or send a Postman request.")
    st.stop()

with st.sidebar:
    sources = sorted(df["source"].dropna().unique())
    chosen = st.multiselect("Source", sources, default=sources)
if chosen:
    df = df[df["source"].isin(chosen) | df["source"].isna()]

accepted = df[df["status"] == "accepted"]
rejected = df[df["status"] == "rejected"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total records", len(df))
c2.metric("Accepted", len(accepted))
c3.metric("Rejected", len(rejected))
c4.metric("Acceptance rate", f"{100 * len(accepted) / len(df):.1f}%")

left, right = st.columns(2)
with left:
    st.subheader("Events over time")
    ts = df.groupby([pd.Grouper(key="logged_at", freq="1min"), "status"]).size().unstack(fill_value=0)
    st.bar_chart(ts)
with right:
    st.subheader("Records per source (availability)")
    st.bar_chart(df.groupby(["source", "status"]).size().unstack(fill_value=0))

left, right = st.columns(2)
with left:
    st.subheader("Rejection reasons by field")
    errs = rejected.explode("errors").dropna(subset=["errors"])
    if errs.empty:
        st.write("No rejections.")
    else:
        errs = pd.json_normalize(errs["errors"].tolist())
        st.dataframe(errs.groupby(["field", "type"]).size().rename("count").reset_index()
                     .sort_values("count", ascending=False), hide_index=True, width="stretch")
with right:
    st.subheader("Per-parameter completeness (accepted records)")
    if accepted.empty:
        st.write("No accepted records.")
    else:
        flat = pd.json_normalize(accepted["payload"].tolist())
        fields = list(TelemetryIn.model_fields)
        comp = (flat.reindex(columns=fields).notna().mean() * 100).round(1).sort_values()
        st.dataframe(comp.rename("completeness %").reset_index().rename(columns={"index": "parameter"}),
                     hide_index=True, width="stretch")

st.subheader("Most recent events")
recent = df.sort_values("logged_at", ascending=False).head(50).copy()
recent["error_fields"] = recent["errors"].apply(lambda e: ", ".join(x["field"] for x in e))
st.dataframe(recent[["logged_at", "status", "http_status", "vehicle_id", "source", "error_fields"]],
             hide_index=True, width="stretch")
