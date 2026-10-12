import json
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)
SAMPLE = json.loads((Path(__file__).parent.parent / "scripts" / "sample_valid.json").read_text())


@pytest.fixture(autouse=True)
def tmp_log(tmp_path, monkeypatch):
    path = tmp_path / "log.jsonl"
    monkeypatch.setattr(settings, "ingestion_log_path", path)
    return path


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_valid_record_accepted(tmp_log):
    r = client.post("/api/v1/telemetry", json=SAMPLE)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "accepted" and body["record_id"] and body["received_at"]
    events = lines(tmp_log)
    assert len(events) == 1 and events[0]["status"] == "accepted"


def test_missing_required_field_rejected(tmp_log):
    bad = deepcopy(SAMPLE)
    del bad["soc_pct"]
    r = client.post("/api/v1/telemetry", json=bad)
    assert r.status_code == 422
    assert any(e["field"] == "soc_pct" for e in r.json()["errors"])
    ev = lines(tmp_log)[0]
    assert ev["status"] == "rejected" and ev["payload"]["vehicle_id"] == "SIM-001"  # raw preserved


@pytest.mark.parametrize("field,value", [
    ("speed_kmh", "fast"),
    ("speed_kmh", "50"),
    ("latitude", None),
    ("hvac_on", "yes"),
    ("vehicle_id", 123),
    ("timestamp", "yesterday"),
    ("timestamp", 1759652130),
    ("timestamp", "2026-10-05T08:15:30"),   # no timezone
    ("source", "tesla_api"),
    ("schema_version", "2.0"),
])
def test_bad_values_rejected(field, value):
    bad = deepcopy(SAMPLE)
    bad[field] = value
    r = client.post("/api/v1/telemetry", json=bad)
    assert r.status_code == 422
    assert any(e["field"] == field for e in r.json()["errors"])


def test_unknown_field_rejected():
    bad = deepcopy(SAMPLE)
    bad["speed_kph"] = 50
    assert client.post("/api/v1/telemetry", json=bad).status_code == 422


def test_invalid_json_rejected(tmp_log):
    r = client.post("/api/v1/telemetry", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["errors"][0]["field"] == "(body)"
    assert lines(tmp_log)[0]["payload"] == "{not json"


def test_integer_accepted_for_float_field():
    ok = deepcopy(SAMPLE)
    ok["speed_kmh"] = 50
    assert client.post("/api/v1/telemetry", json=ok).status_code == 201


def test_optional_fields_may_be_omitted():
    minimal = {k: SAMPLE[k] for k in [
        "schema_version", "vehicle_id", "source", "timestamp", "latitude",
        "longitude", "speed_kmh", "soc_pct", "battery_temp_c"]}
    assert client.post("/api/v1/telemetry", json=minimal).status_code == 201
