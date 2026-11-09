"""Reproducible trip-level synthetic EV energy dataset generator."""
from __future__ import annotations
import numpy as np
import pandas as pd

G = 9.80665
RHO_AIR = 1.225


def generate_dataset(n_samples: int = 10_000, seed: int = 42, dataset_version: str = "v1") -> pd.DataFrame:
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    rng = np.random.default_rng(seed)

    mass = rng.uniform(1450, 2600, n_samples)
    battery_capacity = rng.choice([58.0, 75.0, 82.0, 95.0], n_samples)
    battery_health = rng.uniform(0.82, 1.0, n_samples)
    cd = rng.uniform(0.22, 0.34, n_samples)
    frontal_area = rng.uniform(2.15, 2.75, n_samples)
    crr = rng.uniform(0.008, 0.015, n_samples)
    drivetrain_mode = rng.choice(["eco", "normal", "sport"], n_samples, p=[0.45, 0.40, 0.15])
    motor_eff = rng.uniform(0.88, 0.96, n_samples)
    # Driving mode affects effective motor efficiency: eco favours efficiency, sport trades some efficiency for response.
    mode_eff_factor = np.select([drivetrain_mode == "eco", drivetrain_mode == "sport"], [1.01, 0.98], default=1.0)
    motor_eff = np.clip(motor_eff * mode_eff_factor, 0.85, 0.97)
    inverter_eff = rng.uniform(0.96, 0.99, n_samples)
    mechanical_eff = rng.uniform(0.96, 0.99, n_samples)
    regen_eff = rng.uniform(0.55, 0.78, n_samples)

    road_type = rng.choice(["urban", "suburban", "highway", "mixed"], n_samples, p=[0.30, 0.25, 0.25, 0.20])
    speed_ranges = {"urban": (30, 50), "suburban": (50, 80), "highway": (90, 120), "mixed": (40, 100)}
    speed_limit = np.array([rng.uniform(*speed_ranges[r]) for r in road_type])
    traffic = rng.beta(2.0, 2.5, n_samples)
    road_surface = rng.choice(["dry_asphalt", "wet_asphalt", "gravel", "snow"], n_samples, p=[0.66, 0.20, 0.09, 0.05])
    surface_factor = pd.Series(road_surface).map({"dry_asphalt": 1.00, "wet_asphalt": 1.02, "gravel": 1.12, "snow": 1.25}).to_numpy()
    cruise = np.select([road_type == "urban", road_type == "suburban", road_type == "highway", road_type == "mixed"], [34.0, 57.0, 94.0, 62.0])
    avg_speed = np.minimum(speed_limit * rng.uniform(0.72, 0.96, n_samples), cruise * (1 - 0.48 * traffic) * rng.uniform(0.88, 1.12, n_samples))
    avg_speed = np.clip(avg_speed, 8.0, 120.0)

    # Construct duration from distance and average speed: v_avg = distance / elapsed time.
    distance = np.clip(rng.lognormal(np.log(18.0), 0.72, n_samples), 2.0, 240.0)
    duration_hours = distance / avg_speed
    duration_minutes = duration_hours * 60.0

    net_grade_pct = np.clip(rng.normal(0.0, 0.35, n_samples), -1.0, 1.0)
    elevation_change_m = distance * 1000.0 * net_grade_pct / 100.0
    curvature = rng.uniform(0.0, 1.0, n_samples)
    stop_count = np.rint((distance / 10.0) * (0.25 + 2.4 * traffic + 0.55 * (road_type == "urban"))).astype(int)
    stop_count = np.clip(stop_count, 0, 100)
    mean_abs_accel = np.clip(rng.normal(0.22 + 0.38 * traffic + 0.10 * curvature, 0.10, n_samples), 0.05, 1.4)
    event_speed = np.clip(avg_speed / 3.6 * rng.uniform(0.45, 0.85, n_samples), 2.0, 20.0)

    temp = np.clip(rng.normal(12.0, 11.0, n_samples), -20, 38)
    snow_depth = np.where(road_surface == "snow", rng.uniform(0.5, 12.0, n_samples), np.where(temp < 0, rng.uniform(0, 1.0, n_samples), 0.0))
    precipitation = np.clip(rng.gamma(1.0, 1.3, n_samples), 0, 12)
    wind_speed = np.clip(rng.weibull(1.8, n_samples) * 5.0, 0, 25)
    wind_factor = rng.uniform(-0.75, 1.0, n_samples)  # negative = tailwind, positive = headwind
    hvac_mode = np.where(temp < 8, "heating", np.where(temp > 24, "cooling", "off"))
    hvac_power = np.where(temp < 8, 0.7 + (8 - temp) * 0.075, np.where(temp > 24, 0.45 + (temp - 24) * 0.055, 0.12))
    auxiliary_power = rng.uniform(0.25, 0.65, n_samples)

    # Battery-side energy model. Wheel work terms are calculated in joules then converted to kWh.
    distance_m = distance * 1000.0
    speed_mps = avg_speed / 3.6
    effective_air_speed = np.maximum(0.0, speed_mps + wind_speed * wind_factor)
    drivetrain_eff = motor_eff * inverter_eff * mechanical_eff
    rolling_wheel_kwh = mass * G * crr * surface_factor * distance_m / 3_600_000.0
    aero_force = 0.5 * RHO_AIR * cd * frontal_area * effective_air_speed**2
    aero_wheel_kwh = aero_force * distance_m / 3_600_000.0
    grade_wheel_kwh = mass * G * elevation_change_m / 3_600_000.0
    road_load_wheel_kwh = rolling_wheel_kwh + aero_wheel_kwh + grade_wheel_kwh
    road_load_battery_kwh = np.where(road_load_wheel_kwh >= 0, road_load_wheel_kwh / drivetrain_eff, road_load_wheel_kwh * regen_eff * drivetrain_eff)

    # Each stop is represented by a representative acceleration and braking cycle.
    kinetic_event_kwh = 0.5 * mass * event_speed**2 / 3_600_000.0
    acceleration_battery_kwh = stop_count * kinetic_event_kwh / drivetrain_eff
    recovered_braking_kwh = stop_count * kinetic_event_kwh * regen_eff * drivetrain_eff
    transient_battery_kwh = acceleration_battery_kwh - recovered_braking_kwh
    hvac_energy_kwh = hvac_power * duration_hours
    auxiliary_energy_kwh = auxiliary_power * duration_hours
    raw_energy = road_load_battery_kwh + transient_battery_kwh + hvac_energy_kwh + auxiliary_energy_kwh
    energy_kwh = np.maximum(raw_energy, 0.05)

    return pd.DataFrame({
        "sample_id": [f"TRIP-{i:06d}" for i in range(1, n_samples + 1)],
        "dataset_version": dataset_version,
        "random_seed": int(seed),
        "vehicle_mass_kg": mass.round(3), "battery_capacity_kwh": battery_capacity,
        "battery_health_fraction": battery_health.round(5), "drag_coefficient": cd.round(5),
        "frontal_area_m2": frontal_area.round(4), "rolling_resistance_coefficient": crr.round(5),
        "motor_efficiency": motor_eff.round(5), "inverter_efficiency": inverter_eff.round(5),
        "mechanical_efficiency": mechanical_eff.round(5), "regenerative_braking_efficiency": regen_eff.round(5),
        "drivetrain_mode": drivetrain_mode, "road_type": road_type, "road_surface": road_surface,
        "distance_km": distance.round(6), "duration_minutes": duration_minutes.round(6),
        "average_speed_kmh": avg_speed.round(6), "speed_limit_kmh": speed_limit.round(3),
        "traffic_index": traffic.round(5), "net_grade_percent": net_grade_pct.round(5),
        "elevation_change_m": elevation_change_m.round(4), "curvature_index": curvature.round(5),
        "stop_count": stop_count, "mean_abs_acceleration_mps2": mean_abs_accel.round(5),
        "acceleration_event_speed_mps": event_speed.round(5), "ambient_temperature_c": temp.round(3),
        "snow_depth_cm": snow_depth.round(3), "precipitation_mm_h": precipitation.round(3),
        "wind_speed_mps": wind_speed.round(3), "wind_direction_factor": wind_factor.round(4),
        "hvac_mode": hvac_mode, "hvac_power_kw": hvac_power.round(4), "auxiliary_power_kw": auxiliary_power.round(4),
        "rolling_energy_wheel_kwh": rolling_wheel_kwh.round(6),
        "aerodynamic_energy_wheel_kwh": aero_wheel_kwh.round(6),
        "grade_energy_wheel_kwh": grade_wheel_kwh.round(6),
        "road_load_energy_battery_kwh": road_load_battery_kwh.round(6),
        "transient_energy_battery_kwh": transient_battery_kwh.round(6),
        "unclipped_energy_kwh": raw_energy.round(6),
        "hvac_energy_kwh": hvac_energy_kwh.round(6), "auxiliary_energy_kwh": auxiliary_energy_kwh.round(6),
        "energy_consumption_kwh": energy_kwh.round(6),
        "consumption_kwh_per_100km": (energy_kwh / distance * 100.0).round(6),
    })
