import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from src.ev_platform.generator import generate_dataset
from src.ev_platform.api import create_app

def test_api(tmp_path):
    path = tmp_path / "dataset.csv"
    generate_dataset(25).to_csv(path, index=False)
    client = TestClient(create_app(path))
    assert client.get("/health").status_code == 200
    page = client.get("/dataset?offset=5&limit=10").json()
    assert page["total"] == 25 and page["returned"] == 10
    assert page["data"][0]["sample_id"] == "TRIP-000006"
    assert client.get("/schema").json()["row_count"] == 25
    assert client.get("/summary").json()["row_count"] == 25
    assert client.get("/dataset/TRIP-000001").status_code == 200
    assert client.get("/dataset/NOT-FOUND").status_code == 404
    assert client.get("/dataset.csv").status_code == 200
