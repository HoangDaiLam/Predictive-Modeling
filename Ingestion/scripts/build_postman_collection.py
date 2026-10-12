"""Generates the Postman collection, environment and the test-case doc from ONE list of cases.

    python scripts/build_postman_collection.py

Each case declares the expected HTTP status in Phase 4 (structure-only validation, today)
and in Phase 5 (range/temporal/physical validation, later). The collection reads the
`validation_phase` environment variable (4 or 5) and asserts the matching expectation,
so the same collection keeps working as the API gets stricter.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
V = "SIM-{{run_id}}-"  # vehicle_id prefix; run_id changes on every collection run


def base(suffix, seq=1, drop=(), **over):
    d = {
        "schema_version": "1.0", "vehicle_id": V + suffix, "trip_id": "trip-{{run_id}}",
        "source": "simulator", "sequence_number": seq, "timestamp": "{{ts}}",
        "latitude": 60.1699, "longitude": 24.9384, "altitude_m": 12.5, "heading_deg": 87.0,
        "speed_kmh": 52.4, "acceleration_mps2": 0.4, "soc_pct": 78.5,
        "battery_voltage_v": 392.1, "battery_current_a": 45.3, "battery_power_kw": 17.76,
        "battery_temp_c": 24.5, "charging_state": "driving", "hvac_on": True,
        "hvac_power_w": 1200, "ambient_temp_c": 3.0,
    }
    d.update(over)
    for k in drop:
        d.pop(k, None)
    return d


MIN_KEYS = ["schema_version", "vehicle_id", "source", "timestamp", "latitude",
            "longitude", "speed_kmh", "soc_pct", "battery_temp_c"]
OPT_KEYS = [k for k in base("x") if k not in MIN_KEYS]

# (id, title, body, p4, p5, kwargs)   kwargs: off=seconds after t0, ts=JS expr, field, flag, note
C = []
def case(id, title, body, p4, p5, **kw): C.append((id, title, body, p4, p5, kw))

F = {}  # folder titles
F["01"] = "01 Valid telemetry"
case("01a", "Valid, full payload", base("valid"), 201, 201, off=0, note="All contract fields present.")
case("01b", "Valid, required fields only", base("min", drop=OPT_KEYS), 201, 201, off=1,
     note="Optional fields omitted (no V/I, no HVAC). Must still be accepted; completeness is tracked, not enforced.")

F["02"] = "02 Missing fields / bad JSON"
case("02a", "Missing soc_pct", base("miss", drop=["soc_pct"]), 422, 422, off=0, field="soc_pct")
case("02b", "Missing timestamp", base("miss", drop=["timestamp"]), 422, 422, field="timestamp")
case("02c", "Empty object {}", {}, 422, 422, note="Every required field is reported.")
case("02d", "Malformed JSON syntax", '{"vehicle_id": "SIM-1", "speed_kmh": 50,', 422, 422, field="(body)",
     note="Not parseable at all. Raw text is preserved in the log.")

F["03"] = "03 Wrong data types"
case("03a", "speed_kmh = \"fast\"", base("type", speed_kmh="fast"), 422, 422, off=0, field="speed_kmh")
case("03b", "speed_kmh = \"50\" (number as string)", base("type", speed_kmh="50"), 422, 422, off=1, field="speed_kmh",
     note="Strict typing: a string is never silently converted.")
case("03c", "hvac_on = \"yes\"", base("type", hvac_on="yes"), 422, 422, off=2, field="hvac_on")
case("03d", "latitude = null", base("type", latitude=None), 422, 422, off=3, field="latitude")
case("03e", "source = \"tesla_api\" (unknown enum)", base("type", source="tesla_api"), 422, 422, off=4, field="source")

F["04"] = "04 Impossible GPS"
case("04a", "latitude = 95", base("gps", latitude=95.0), 201, 422, off=0, field="latitude",
     note="Phase 4: accepted (types are fine). Phase 5: hard range violation, 422.")
case("04b", "longitude = 200", base("gps", longitude=200.0), 201, 422, off=1, field="longitude")
case("04c", "(0, 0) 'Null Island'", base("gps", latitude=0.0, longitude=0.0), 201, 201, off=2, flag="gps_null_island",
     note="Technically valid coordinates, but the classic 'no GPS fix' value. Phase 5: accepted and FLAGGED.")

F["05"] = "05 SoC out of range"
case("05a", "soc_pct = -5", base("soc", soc_pct=-5.0), 201, 422, off=0, field="soc_pct")
case("05b", "soc_pct = 120", base("soc", soc_pct=120.0), 201, 422, off=1, field="soc_pct")
case("05c", "soc_pct = 100 (boundary)", base("soc", soc_pct=100.0), 201, 201, off=2, note="Boundary values are valid.")

F["06"] = "06 Speed"
case("06a", "speed_kmh = -10", base("speed", speed_kmh=-10.0), 201, 422, off=0, field="speed_kmh",
     note="Contract: speed is magnitude, always >= 0. Reverse driving needs a separate field.")
case("06b", "speed_kmh = 400", base("speed", speed_kmh=400.0), 201, 422, off=1, field="speed_kmh",
     note="Beyond any road EV. The hard limit will be set generously; calibrate per vehicle.")

F["07"] = "07 Impossible battery values"
case("07a", "battery_voltage_v = -50", base("batt", battery_voltage_v=-50.0), 201, 422, off=0, field="battery_voltage_v")
case("07b", "battery_current_a = 99999", base("batt", battery_current_a=99999.0), 201, 422, off=1, field="battery_current_a")
case("07c", "battery_temp_c = 500", base("batt", battery_temp_c=500.0), 201, 422, off=2, field="battery_temp_c")
case("07d", "power inconsistent with V x I", base("batt", battery_power_kw=-80.0), 201, 201, off=3, flag="power_inconsistent",
     note="V=392.1, I=45.3 implies about +17.8 kW, but -80 kW is reported. Each value is individually legal, "
          "so Phase 5 accepts and FLAGS it (we cannot know which value is wrong).")

F["08"] = "08 Invalid timestamps"
case("08a", "timestamp = \"yesterday\"", base("ts", timestamp="yesterday"), 422, 422, field="timestamp")
case("08b", "timestamp without timezone", base("ts", timestamp="2026-10-05T08:15:30"), 422, 422, field="timestamp")
case("08c", "timestamp = unix integer", base("ts", timestamp=1759652130), 422, 422, field="timestamp")
case("08d", "timestamp 2 days in the future", base("ts"), 201, 422, ts="new Date(Date.now()+2*86400000).toISOString()", field="timestamp",
     note="Phase 5: rejected if later than now + 5 min (clock errors).")
case("08e", "timestamp = 1970-01-01", base("ts", timestamp="1970-01-01T00:00:00Z"), 201, 422, field="timestamp",
     note="Classic 'device clock never set' value.")

F["09"] = "09 Duplicate records (run in order)"
case("09a", "First delivery", base("dup", seq=7), 201, 201, off=5)
case("09b", "Identical resend", base("dup", seq=7), 201, 409, off=5,
     note="Same vehicle + source + timestamp. Phase 4: stored a 2nd time (the problem). Phase 5/6: 409 Conflict.")

F["10"] = "10 Out-of-order records (run in order)"
case("10a", "t+0 s, seq 1", base("ooo", seq=1), 201, 201, off=0)
case("10b", "t+30 s, seq 2", base("ooo", seq=2), 201, 201, off=30)
case("10c", "t+15 s, seq 3 (late arrival)", base("ooo", seq=3), 201, 201, off=15, flag="out_of_order",
     note="Older than the latest stored record. Buffered loggers do this legitimately, so Phase 5 accepts and FLAGS it.")

F["11"] = "11 Missing intervals / sampling rate (run in order)"
case("11a", "t+0 s", base("gap", seq=1), 201, 201, off=0)
case("11b", "t+1 s", base("gap", seq=2), 201, 201, off=1)
case("11c", "t+2 s", base("gap", seq=3), 201, 201, off=2)
case("11d", "t+62 s (60 s gap at 1 Hz)", base("gap", seq=4), 201, 201, off=62, flag="gap_detected",
     note="Sequence numbers also jump if the sender skips; here only time does.")
case("11e", "t+62.01 s (10 ms later)", base("gap", seq=5), 201, 201, ts="new Date(new Date(pm.collectionVariables.get('t0')).getTime()+62010).toISOString()",
     flag="sampling_too_fast", note="Faster than the source's declared rate: duplicate-ish or clock problem.")

F["12"] = "12 Sudden physically suspicious changes (run in order)"
case("12a", "SoC baseline 78.5 %", base("soc-jump", seq=1), 201, 201, off=0)
case("12b", "SoC 52 % one second later", base("soc-jump", seq=2, soc_pct=52.0), 201, 201, off=1, flag="soc_jump",
     note="26 % in 1 s is impossible. Legal individually, so FLAGGED (SoC can also be re-calibrated by the BMS).")
case("12c", "Speed baseline 50 km/h", base("spd-jump", seq=1, speed_kmh=50.0), 201, 201, off=0)
case("12d", "Speed 180 km/h one second later", base("spd-jump", seq=2, speed_kmh=180.0), 201, 201, off=1, flag="speed_jump",
     note="Implied acceleration about 36 m/s^2. Thresholds must be calibrated per vehicle.")
case("12e", "GPS baseline (Helsinki)", base("gps-jump", seq=1), 201, 201, off=0)
case("12f", "GPS ~160 km away one second later", base("gps-jump", seq=2, latitude=61.4978, longitude=23.7610), 201, 201, off=1,
     flag="gps_jump", note="About 160 km in 1 s. Typical GPS glitch or a mixed-up vehicle_id.")

INIT_JS = [
    "pm.collectionVariables.set('run_id', Math.random().toString(16).slice(2, 8));",
    "pm.collectionVariables.set('t0', new Date(Date.now() - 10*60*1000).toISOString());",
]


def test_script(id, p4, p5, kw):
    field, flag = kw.get("field"), kw.get("flag")
    return [
        "const phase = parseInt(pm.variables.get('validation_phase') || '4');",
        f"const expected = phase >= 5 ? {p5} : {p4};",
        "const body = (() => { try { return pm.response.json(); } catch (e) { return {}; } })();",
        "pm.test(`HTTP ${expected} expected (validation phase ${phase})`, () => pm.response.to.have.status(expected));",
        "if (expected === 201) pm.test('body.status is accepted', () => pm.expect(body.status).to.eql('accepted'));",
        "if (expected === 422) pm.test('rejected, errors listed', () => {",
        "  pm.expect(body.status).to.eql('rejected'); pm.expect(body.errors).to.be.an('array').that.is.not.empty; });",
        *([f"if (expected === 422) pm.test('error names field {field}', () => pm.expect(body.errors.map(e => e.field)).to.include('{field}'));"] if field else []),
        "if (expected === 409) pm.test('body.status is duplicate', () => pm.expect(body.status).to.eql('duplicate'));",
        *([f"if (phase >= 5) pm.test('quality flag {flag}', () => pm.expect(body.quality_flags || []).to.include('{flag}'));"] if flag else []),
    ]


def make_item(id, title, body, p4, p5, kw):
    if "ts" in kw:
        ts_js = f"pm.variables.set('ts', {kw['ts']});"
    else:
        ts_js = ("pm.variables.set('ts', new Date(new Date(pm.collectionVariables.get('t0')).getTime() + "
                 f"{kw.get('off', 0)}*1000).toISOString());")
    raw = body if isinstance(body, str) else json.dumps(body, indent=2)
    return {
        "name": f"{id} {title}",
        "event": [
            {"listen": "prerequest", "script": {"type": "text/javascript", "exec": [ts_js]}},
            {"listen": "test", "script": {"type": "text/javascript", "exec": test_script(id, p4, p5, kw)}},
        ],
        "request": {
            "method": "POST",
            "header": [{"key": "Content-Type", "value": "application/json"}],
            "body": {"mode": "raw", "raw": raw, "options": {"raw": {"language": "json"}}},
            "url": {"raw": "{{base_url}}/api/v1/telemetry", "host": ["{{base_url}}"], "path": ["api", "v1", "telemetry"]},
            "description": kw.get("note", ""),
        },
    }


init_item = {
    "name": "00 Init new run + health check",
    "event": [
        {"listen": "prerequest", "script": {"type": "text/javascript", "exec": INIT_JS}},
        {"listen": "test", "script": {"type": "text/javascript", "exec": [
            "pm.test('API is up', () => pm.response.to.have.status(200));"]}},
    ],
    "request": {"method": "GET", "header": [], "url": {"raw": "{{base_url}}/health", "host": ["{{base_url}}"], "path": ["health"]},
                "description": "Run first. Generates a fresh run_id (so vehicle IDs are new) and t0 (the simulated trip start, 10 min ago)."},
}

folders = []
for fid, ftitle in F.items():
    folders.append({"name": ftitle, "item": [make_item(i, t, b, p4, p5, kw) for (i, t, b, p4, p5, kw) in C if i.startswith(fid)]})

collection = {
    "info": {"name": "EV Telemetry Simulator", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
             "description": "Postman acting as a simulated vehicle. Run '00 Init' first, then the folders in order (or use the Collection Runner)."},
    "event": [{"listen": "prerequest", "script": {"type": "text/javascript", "exec": [
        "if (!pm.collectionVariables.get('t0')) {", *["  " + l for l in INIT_JS], "}"]}}],
    "variable": [{"key": "base_url", "value": "http://127.0.0.1:8000"}, {"key": "validation_phase", "value": "4"}],
    "item": [init_item, *folders],
}
environment = {"name": "EV Local", "_postman_variable_scope": "environment", "values": [
    {"key": "base_url", "value": "http://127.0.0.1:8000", "enabled": True},
    {"key": "validation_phase", "value": "4", "enabled": True}]}

pm = OUT / "postman"
pm.mkdir(exist_ok=True)
(pm / "EV-Telemetry-Simulator.postman_collection.json").write_text(json.dumps(collection, indent=2), encoding="utf-8")
(pm / "EV-Local.postman_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

# human-readable test-case table
def desc_body(b):
    return "raw text" if isinstance(b, str) else ("{}" if b == {} else "JSON")
rows = ["# Postman test cases and expected responses", "",
        "Generated by `scripts/build_postman_collection.py`. **Phase 4** = structure-only validation (current API). "
        "**Phase 5** = target after range/temporal/physical validation. Flag names are proposed and will be finalised in Phase 5.", "",
        "| ID | Scenario | Phase 4 (now) | Phase 5 (target) | Notes |", "|---|---|---|---|---|"]
st = {201: "201 accepted", 422: "422 rejected", 409: "409 duplicate"}
for (i, t, b, p4, p5, kw) in C:
    p5s = st[p5] + (f" + flag `{kw['flag']}`" if kw.get("flag") else "")
    p4s = st[p4]
    rows.append(f"| {i} | {t} | {p4s} | {p5s} | {kw.get('note','')} |")
(OUT / "docs" / "postman_test_cases.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"{len(C)} cases written")
