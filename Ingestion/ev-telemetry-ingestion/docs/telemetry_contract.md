# Telemetry Contract, schema version 1.0

## Conventions
- **Units are in the field name** (`speed_kmh`, `battery_temp_c`).
- **Timestamps are UTC, ISO 8601 with timezone**, e.g. `2026-10-05T08:15:30.250Z`.
- **Sign convention:** battery current and power are **positive = discharging**, **negative = charging / regeneration**. Adapters for sources using the opposite sign must flip it.
- Raw measurements are never overwritten by derived values.
- Ranges below are generic starting points and **must be calibrated to the actual vehicle**.

## Table A: technical contract

| Parameter | Type | Unit | Required | Target frequency | Valid range (initial, generic) | Raw / Derived |
|---|---|---|---|---|---|---|
| `record_id` | UUID | none | Server-generated | n/a | n/a | Derived (server) |
| `schema_version` | string | none | Required | per record | e.g. `"1.0"` | Raw |
| `vehicle_id` | string | none | Required | per record | 1-64 chars, non-empty | Raw |
| `trip_id` | string | none | Optional | per record | 1-64 chars | Raw or derived |
| `source` | enum | none | Required | per record | `simulator`, `volvo_api`, `obd_ii`, `can_logger`, `other_api` | Raw (adapter-set) |
| `sequence_number` | int | none | Optional | per record | >= 0 | Raw |
| `timestamp` | datetime | UTC | Required | per record | not later than now + 5 min | Raw |
| `received_at` | datetime | UTC | Server-generated | n/a | n/a | Derived (server) |
| `latitude` | float | degrees (WGS84) | Required | 1 Hz | -90 to 90 | Raw |
| `longitude` | float | degrees (WGS84) | Required | 1 Hz | -180 to 180 | Raw |
| `altitude_m` | float | m above sea level | Optional | 1 Hz | -430 to 9000 | Raw |
| `heading_deg` | float | degrees from north | Optional | 1 Hz | 0 <= x < 360 | Raw |
| `speed_kmh` | float | km/h | Required | 1-10 Hz | 0 to ~250 (vehicle-specific) | Raw |
| `acceleration_mps2` | float | m/s^2 (longitudinal) | Optional | 1-10 Hz | about -15 to +15 | Raw (IMU/CAN) or derived from speed |
| `odometer_km` | float | km | Optional | <= 1 Hz | >= 0, non-decreasing | Raw |
| `soc_pct` | float | % | Required | 0.1-1 Hz | 0 to 100 | Raw |
| `battery_voltage_v` | float | V | Required* | 1-10 Hz | > 0; ~250-450 V (400 V packs), ~600-850 V (800 V packs) | Raw |
| `battery_current_a` | float | A (+ discharge) | Required* | 1-10 Hz | vehicle-specific (~ -600 to +800) | Raw |
| `battery_power_kw` | float | kW (+ discharge) | Optional | 1-10 Hz | vehicle-specific (~ -300 to +350) | Raw if reported, else derived |
| `battery_temp_c` | float | degC | Required | 0.1-1 Hz | -40 to 80 | Raw |
| `charging_state` | enum | none | Strongly recommended | on change / 1 Hz | `driving`, `parked`, `charging`, `unknown` | Raw or derived |
| `hvac_on` | bool | none | Optional | 0.1-1 Hz | true/false | Raw |
| `hvac_power_w` | float | W | Optional | 0.1-1 Hz | 0 to ~10000 | Raw or estimated |
| `hvac_setpoint_c` | float | degC | Optional | on change | 10 to 35 | Raw |
| `cabin_temp_c` | float | degC | Optional | 0.1 Hz | -40 to 70 | Raw |
| `ambient_temp_c` | float | degC | Optional | 0.1 Hz | -50 to 60 | Raw (vehicle sensor); weather API enriches separately |
| `drive_mode` | string | none | Optional | on change | free text for now | Raw |
| `regen_level` | string/int | none | Optional | on change | vehicle-specific | Raw |

\* Required to compute energy directly. Many real APIs will not expose these; completeness is tracked in the quality metrics (Phase 7).

## Table B: meaning and importance

| Parameter | Description | Typical source | Why it matters for energy prediction |
|---|---|---|---|
| `record_id` | Unique ID assigned by the server | Server | Traceability; links validation errors to records |
| `schema_version` | Version of this contract | Adapter | Allows schema changes without breaking old data |
| `vehicle_id` | Stable, anonymized vehicle identifier | Vehicle/API | Per-vehicle calibration; grouping for train/test splits |
| `trip_id` | Groups consecutive records into a drive | Client or derived | Energy per trip is likely the ML unit; prevents trip leakage across splits |
| `source` | Adapter that produced the record | Adapter | Reliability differs per source, so quality is tracked per source |
| `sequence_number` | Monotonic sender counter | Device | Best tool for detecting duplicates and gaps |
| `timestamp` | When the measurement was taken (event time) | Vehicle/device | Time axis for integrating power into energy |
| `received_at` | When the server received it | Server | Measures latency; diagnoses delayed or batched uploads |
| `latitude`/`longitude` | Vehicle position | GPS | Route reconstruction, distance, weather/map enrichment |
| `altitude_m` | Elevation | GPS/barometer | Road grade is a major energy driver |
| `heading_deg` | Direction of travel | GPS/IMU | Wind relative to travel direction, map matching |
| `speed_kmh` | Longitudinal speed | CAN/GPS/API | Aerodynamic drag grows with speed squared |
| `acceleration_mps2` | Longitudinal acceleration | IMU/CAN/derived | Driving aggressiveness; separates acceleration and regeneration |
| `odometer_km` | Total distance | CAN/API | Cross-checks GPS distance; energy per km |
| `soc_pct` | State of charge | BMS | Coarse energy estimate; independent validation of V x I |
| `battery_voltage_v` | Pack terminal voltage | BMS | Half of P = V x I |
| `battery_current_a` | Pack current | BMS | Other half of P = V x I; sign encodes regeneration |
| `battery_power_kw` | Pack power | BMS or computed | Direct integrand for energy; cross-check against V x I |
| `battery_temp_c` | Pack average temperature | BMS | Affects internal resistance and usable capacity |
| `charging_state` | Driving / parked / charging | Vehicle | Charging records must not count as consumption |
| `hvac_on`/`hvac_power_w` | Climate system state/power | Vehicle/CAN | Largest auxiliary load, especially in cold climates |
| `hvac_setpoint_c`, `cabin_temp_c` | Climate demand | Vehicle | Explains HVAC power when not measured directly |
| `ambient_temp_c` | Outside air temperature | Vehicle sensor | Drives HVAC load and battery behavior |
| `drive_mode`, `regen_level` | Driver/vehicle settings | Vehicle | Changes how energy is used and recovered |

## Essential variables for energy consumption
- **Tier 1 (compute/validate the target):** `timestamp`, `vehicle_id`, `battery_voltage_v`, `battery_current_a` (or `battery_power_kw`), `soc_pct`, `charging_state`.
- **Tier 2 (main ML features):** `speed_kmh`, `acceleration_mps2`, `latitude`/`longitude`, `altitude_m`, `battery_temp_c`, `ambient_temp_c`, HVAC fields, `trip_id`.
- **Tier 3 (context):** `heading_deg`, `odometer_km`, `drive_mode`, `regen_level`.
- **Fallback when V/I are unavailable:** estimate from SoC change x battery capacity, or API-reported consumption (lower resolution).

## Route-related variables
Ingestion captures only raw GPS, altitude, heading and `trip_id`. Road grade, speed limit, road type, traffic and weather are derived or enriched downstream (Data Enrichment engineer).
