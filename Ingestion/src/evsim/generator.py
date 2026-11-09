from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import numpy as np


GENERATOR_VERSION = "0.1.0"
G = 9.80665
AIR_DENSITY = 1.225
CRR = 0.010
AUXILIARY_POWER_KW = 0.35
REGEN_EFFICIENCY = 0.65


@dataclass
class GeneratedDataset:
    dataset_run: dict
    vehicles: list[dict]
    routes: list[dict]
    trips: list[dict]
    telemetry: list[dict]
    trip_results: list[dict]


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _make_vehicle(vehicle_id: str, rng: np.random.Generator) -> dict:
    return {
        "vehicle_id": vehicle_id,
        "curb_mass_kg": round(float(rng.uniform(1550, 2450)), 2),
        "battery_capacity_kwh": round(float(rng.choice([58, 75, 82, 95])), 2),
        "battery_health_pct": round(float(rng.uniform(88, 100)), 2),
        "drivetrain_efficiency": round(float(rng.uniform(0.88, 0.95)), 4),
        "motor_efficiency": round(float(rng.uniform(0.90, 0.97)), 4),
        "inverter_efficiency": round(float(rng.uniform(0.96, 0.99)), 4),
        "cd": round(float(rng.uniform(0.21, 0.31)), 3),
        "frontal_area_m2": round(float(rng.uniform(2.1, 2.7)), 3),
        "wheel_radius_m": round(float(rng.uniform(0.31, 0.36)), 3),
        "drivetrain_mode": str(rng.choice(["eco", "normal", "sport"])),
    }


def _make_route(route_id: str, rng: np.random.Generator) -> dict:
    road_type = str(rng.choice(
        ["urban", "suburban", "highway", "hilly"],
        p=[0.35, 0.25, 0.25, 0.15]
    ))
    distance = float(rng.uniform(4, 65))
    if road_type == "urban":
        limit = float(rng.choice([30, 40, 50]))
        traffic = str(rng.choice(["light", "moderate", "heavy"], p=[0.2, 0.5, 0.3]))
        surface = str(rng.choice(["asphalt", "cobblestone"], p=[0.9, 0.1]))
    elif road_type == "highway":
        limit = float(rng.choice([80, 100, 120]))
        traffic = str(rng.choice(["light", "moderate", "heavy"], p=[0.55, 0.35, 0.1]))
        surface = "asphalt"
    elif road_type == "hilly":
        limit = float(rng.choice([50, 60, 80]))
        traffic = str(rng.choice(["light", "moderate"], p=[0.6, 0.4]))
        surface = str(rng.choice(["asphalt", "gravel"], p=[0.8, 0.2]))
    else:
        limit = float(rng.choice([50, 60, 70, 80]))
        traffic = str(rng.choice(["light", "moderate", "heavy"], p=[0.35, 0.5, 0.15]))
        surface = "asphalt"

    gain = float(rng.uniform(0, 12) * distance)
    loss = float(rng.uniform(0, 12) * distance)
    if road_type == "hilly":
        gain *= 2.0
        loss *= 2.0

    return {
        "route_id": route_id,
        "distance_km": round(distance, 3),
        "speed_limit_kmh": limit,
        "road_type": road_type,
        "road_surface": surface,
        "traffic_profile": traffic,
        "elevation_gain_m": round(gain, 2),
        "elevation_loss_m": round(loss, 2),
        "curvature_index": round(float(rng.uniform(0.05, 0.9)), 3),
    }


def _traffic_factor(profile: str) -> float:
    return {"light": 0.92, "moderate": 0.72, "heavy": 0.48}[profile]


def _simulate_trip(
    trip_id: str,
    dataset_id: str,
    vehicle: dict,
    route: dict,
    start_time: datetime,
    interval_s: int,
    rng: np.random.Generator,
) -> tuple[dict, list[dict], dict]:
    distance_m = route["distance_km"] * 1000.0
    speed_limit = route["speed_limit_kmh"]
    traffic_factor = _traffic_factor(route["traffic_profile"])
    ambient = float(rng.uniform(-12, 28))
    precipitation = float(rng.choice([0.0, 0.0, 0.2, 1.5, 4.0]))
    wind = float(rng.uniform(0, 9))
    temp_battery = float(np.clip(ambient + rng.uniform(8, 18), -10, 45))
    capacity_kwh = vehicle["battery_capacity_kwh"] * vehicle["battery_health_pct"] / 100

    # Number of samples is based on a plausible average speed and route length.
    average_speed_guess = max(12.0, speed_limit * traffic_factor)
    duration_s = max(interval_s, int(distance_m / (average_speed_guess / 3.6)))
    n_steps = max(2, math.ceil(duration_s / interval_s))
    dt = duration_s / n_steps

    # Generate a smooth, scenario-dependent speed trace, then normalize distance.
    t = np.linspace(0, 1, n_steps)
    base_speed = speed_limit * traffic_factor
    if route["road_type"] == "urban":
        oscillation = 0.48 * np.sin(2 * np.pi * 3 * t) + 0.18 * np.sin(2 * np.pi * 8 * t)
    elif route["road_type"] == "highway":
        oscillation = 0.05 * np.sin(2 * np.pi * 2 * t)
    else:
        oscillation = 0.18 * np.sin(2 * np.pi * 4 * t)
    speed = np.clip(base_speed * (1 + oscillation) + rng.normal(0, 2.0, n_steps), 0, speed_limit)
    if route["traffic_profile"] == "heavy":
        stop_mask = rng.random(n_steps) < (0.10 if route["road_type"] == "urban" else 0.025)
        speed[stop_mask] = 0
    # Scale positive speed values so integrated distance is close to route distance.
    raw_distance = np.sum(speed / 3.6) * dt
    if raw_distance > 0:
        speed = np.clip(speed * distance_m / raw_distance, 0, speed_limit)
    speed_mps = speed / 3.6
    acceleration = np.gradient(speed_mps, dt)
    # Limit extreme synthetic accelerations.
    acceleration = np.clip(acceleration, -3.0, 2.5)

    grade_mean = (route["elevation_gain_m"] - route["elevation_loss_m"]) / max(distance_m, 1)
    grade = grade_mean + 0.012 * np.sin(2 * np.pi * 2 * t)
    grade = np.clip(grade, -0.12, 0.12)

    mass = vehicle["curb_mass_kg"]
    cd = vehicle["cd"]
    area = vehicle["frontal_area_m2"]
    eta = (
        vehicle["drivetrain_efficiency"]
        * vehicle["motor_efficiency"]
        * vehicle["inverter_efficiency"]
    )

    hvac_kw = max(0.2, (18 - ambient) * 0.11) if ambient < 18 else max(0.2, (ambient - 23) * 0.07)
    hvac_kw = float(np.clip(hvac_kw + rng.normal(0, 0.12), 0.2, 5.0))

    telemetry = []
    battery_energy_kwh = 0.0
    hvac_energy_kwh = 0.0
    regen_energy_kwh = 0.0
    soc = float(rng.uniform(55, 95))
    heading = float(rng.uniform(0, 360))
    cumulative_distance = 0.0

    for i in range(n_steps):
        v = float(speed_mps[i])
        a = float(acceleration[i])
        theta = math.atan(float(grade[i]))

        rolling = mass * G * CRR
        aero = 0.5 * AIR_DENSITY * cd * area * (v + wind * 0.25) ** 2
        grade_force = mass * G * math.sin(theta)
        accel_force = mass * a
        wheel_power_kw = (rolling + aero + grade_force + accel_force) * v / 1000.0

        if wheel_power_kw >= 0:
            traction_kw = wheel_power_kw / max(eta, 0.01)
        else:
            # Negative wheel power is braking; only a portion returns to the battery.
            traction_kw = wheel_power_kw * REGEN_EFFICIENCY

        battery_power_kw = traction_kw + hvac_kw + AUXILIARY_POWER_KW
        # A small sensor/model variation; bounded to avoid non-physical extremes.
        battery_power_kw = float(np.clip(
            battery_power_kw + rng.normal(0, 0.12), -35, 120
        ))
        energy_step_kwh = battery_power_kw * dt / 3600.0
        battery_energy_kwh += energy_step_kwh
        hvac_energy_kwh += hvac_kw * dt / 3600.0
        if battery_power_kw < 0:
            regen_energy_kwh += (-battery_power_kw) * dt / 3600.0

        # Voltage varies modestly with SoC and load. Current sign follows power.
        voltage = float(np.clip(330 + soc * 0.65 - max(battery_power_kw, 0) * 0.12, 280, 430))
        current = battery_power_kw * 1000.0 / voltage
        soc = float(np.clip(soc - (energy_step_kwh / max(capacity_kwh, 1)) * 100, 0, 100))
        temp_battery = float(np.clip(
            temp_battery + max(battery_power_kw, 0) * 0.0007 * dt
            - (temp_battery - ambient) * 0.00008 * dt,
            -20, 65
        ))

        cumulative_distance += v * dt
        timestamp = start_time + timedelta(seconds=i * dt)
        # Synthetic coordinates are illustrative, not a real mapped route.
        latitude = 60.1699 + (cumulative_distance / 111_000.0)
        longitude = 24.9384 + (cumulative_distance / 111_000.0) * 0.35

        telemetry.append({
            "trip_id": trip_id,
            "timestamp": _iso(timestamp),
            "latitude": round(latitude, 7),
            "longitude": round(longitude, 7),
            "speed_kmh": round(v * 3.6, 3),
            "acceleration_mps2": round(a, 4),
            "heading_deg": round(heading % 360, 3),
            "soc_pct": round(soc, 4),
            "battery_voltage_v": round(voltage, 3),
            "battery_current_a": round(current, 3),
            "battery_power_kw": round(battery_power_kw, 4),
            "battery_temperature_c": round(temp_battery, 3),
            "ambient_temperature_c": round(ambient, 3),
            "hvac_power_kw": round(hvac_kw, 3),
        })
        heading = (heading + rng.normal(0, 1.5)) % 360

    end_time = start_time + timedelta(seconds=duration_s)
    actual_distance_km = max(0.1, cumulative_distance / 1000.0)
    average_speed = actual_distance_km / (duration_s / 3600.0)
    max_speed = float(np.max(speed) * 3.6)
    net_energy = max(0.01, battery_energy_kwh)
    trip = {
        "trip_id": trip_id,
        "vehicle_id": vehicle["vehicle_id"],
        "route_id": route["route_id"],
        "start_time": _iso(start_time),
        "end_time": _iso(end_time),
        "trip_duration_s": int(duration_s),
        "distance_km": round(actual_distance_km, 4),
        "average_speed_kmh": round(average_speed, 3),
        "max_speed_kmh": round(max_speed, 3),
        "ambient_temperature_c": round(ambient, 3),
        "precipitation_mm_h": round(precipitation, 3),
        "wind_speed_mps": round(wind, 3),
        "traffic_profile": route["traffic_profile"],
    }
    result = {
        "trip_id": trip_id,
        "energy_consumption_kwh": round(net_energy, 5),
        "consumption_kwh_per_100km": round(net_energy / actual_distance_km * 100, 5),
        "hvac_energy_kwh": round(hvac_energy_kwh, 5),
        "net_regen_energy_kwh": round(regen_energy_kwh, 5),
    }
    return trip, telemetry, result


def generate_dataset(
    num_vehicles: int = 10,
    num_trips: int = 200,
    interval_s: int = 10,
    seed: int = 42,
    dataset_version: str = "v1",
) -> GeneratedDataset:
    if num_vehicles < 1 or num_trips < 1:
        raise ValueError("num_vehicles and num_trips must be at least 1")
    if interval_s < 1:
        raise ValueError("interval_s must be at least 1 second")

    rng = np.random.default_rng(seed)
    dataset_id = None  # Supabase assigns the UUID when dataset_runs is inserted.
    vehicles = [_make_vehicle(f"EV-{i+1:04d}", rng) for i in range(num_vehicles)]
    routes = [_make_route(f"ROUTE-{i+1:04d}", rng) for i in range(max(5, min(num_trips, 50)))]

    trips, telemetry, results = [], [], []
    start_base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for i in range(num_trips):
        vehicle = vehicles[i % len(vehicles)]
        route = routes[int(rng.integers(0, len(routes)))]
        start = start_base + timedelta(hours=i * 2)
        trip_id = f"TRIP-{i+1:06d}"
        trip, trip_telemetry, result = _simulate_trip(
            trip_id, "", vehicle, route, start, interval_s, rng
        )
        trips.append(trip)
        for row in trip_telemetry:
            row["dataset_id"] = dataset_id
        telemetry.extend(trip_telemetry)
        results.append(result)

    dataset_run = {
        "dataset_version": dataset_version,
        "random_seed": int(seed),
        "generator_version": GENERATOR_VERSION,
        "vehicle_count": len(vehicles),
        "route_count": len(routes),
        "trip_count": len(trips),
        "telemetry_count": len(telemetry),
        "notes": (
            "Synthetic data from a simplified longitudinal EV energy model; "
            "coordinates are illustrative and not map-matched."
        ),
    }
    return GeneratedDataset(
        dataset_run=dataset_run,
        vehicles=vehicles,
        routes=routes,
        trips=trips,
        telemetry=telemetry,
        trip_results=results,
    )
