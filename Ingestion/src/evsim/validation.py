from __future__ import annotations


def validate_dataset(dataset) -> None:
    """Raise ValueError when generated records fail core consistency checks."""
    if not dataset.vehicles or not dataset.routes or not dataset.trips:
        raise ValueError("Dataset must contain vehicles, routes and trips")

    vehicle_ids = {v["vehicle_id"] for v in dataset.vehicles}
    route_ids = {r["route_id"] for r in dataset.routes}
    trip_ids = {t["trip_id"] for t in dataset.trips}

    for vehicle in dataset.vehicles:
        if not 0 < vehicle["drivetrain_efficiency"] <= 1:
            raise ValueError("Invalid drivetrain efficiency")
        if not 0 <= vehicle["battery_health_pct"] <= 100:
            raise ValueError("Invalid battery health")

    for trip in dataset.trips:
        if trip["vehicle_id"] not in vehicle_ids:
            raise ValueError(f"Unknown vehicle in trip {trip['trip_id']}")
        if trip["route_id"] not in route_ids:
            raise ValueError(f"Unknown route in trip {trip['trip_id']}")
        if trip["distance_km"] <= 0 or trip["trip_duration_s"] <= 0:
            raise ValueError(f"Invalid distance/duration in {trip['trip_id']}")

    for row in dataset.telemetry:
        if row["trip_id"] not in trip_ids:
            raise ValueError(f"Telemetry references unknown trip {row['trip_id']}")
        if not 0 <= row["soc_pct"] <= 100:
            raise ValueError("SoC outside [0, 100]")
        if row["battery_voltage_v"] <= 0:
            raise ValueError("Battery voltage must be positive")
        if row["speed_kmh"] < 0:
            raise ValueError("Speed cannot be negative")

    if len(dataset.trip_results) != len(dataset.trips):
        raise ValueError("Every trip must have exactly one trip result")
    if len(dataset.telemetry) == 0:
        raise ValueError("Dataset contains no telemetry")
