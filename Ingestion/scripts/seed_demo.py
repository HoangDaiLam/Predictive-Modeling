"""Send a mix of valid and deliberately broken records so the dashboard has data.

Usage:  python scripts/seed_demo.py --n 60
"""
import argparse
import random
from datetime import datetime, timedelta, timezone

import httpx


def make_record(i: int, t0: datetime) -> dict:
    return {
        "schema_version": "1.0",
        "vehicle_id": random.choice(["SIM-001", "SIM-002"]),
        "trip_id": "trip-demo",
        "source": "simulator",
        "sequence_number": i,
        "timestamp": (t0 + timedelta(seconds=i)).isoformat().replace("+00:00", "Z"),
        "latitude": 60.17 + i * 1e-4,
        "longitude": 24.94 + i * 1e-4,
        "speed_kmh": round(random.uniform(20, 80), 1),
        "soc_pct": round(80 - i * 0.05, 2),
        "battery_voltage_v": round(random.uniform(380, 400), 1),
        "battery_current_a": round(random.uniform(-20, 90), 1),
        "battery_temp_c": round(random.uniform(20, 28), 1),
        "ambient_temp_c": 3.0 if i % 3 else None,
        "hvac_on": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--url", default="http://127.0.0.1:8000/api/v1/telemetry")
    args = ap.parse_args()
    t0 = datetime.now(timezone.utc) - timedelta(seconds=args.n)
    counts: dict[int, int] = {}
    for i in range(args.n):
        rec = make_record(i, t0)
        if i % 7 == 6:
            del rec["soc_pct"]                    # missing required field
        elif i % 11 == 10:
            rec["speed_kmh"] = "fast"             # wrong type
        elif i % 13 == 12:
            rec["timestamp"] = "yesterday"        # bad timestamp
        r = httpx.post(args.url, json=rec, timeout=5)
        counts[r.status_code] = counts.get(r.status_code, 0) + 1
    print("Done. HTTP status counts:", counts)


if __name__ == "__main__":
    main()
