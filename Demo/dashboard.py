# -*- coding: utf-8 -*-
"""
MODULE: dashboard.py — BƯỚC 5 (v3: chỉ 17 input + 1 output + 5 EDA charts)

Loại bỏ: route comparison, temperature sweep, speed curve, Monte Carlo,
         AI explanation, energy breakdown, 3/4 HVAC metrics.
Giữ: form 17 inputs (data_req Mục 15) + 5 advanced (Mục 2),
     output kWh/km, 5 EDA charts (Mục 13).
"""
import streamlit as st
import pandas as pd

from vehicle_dynamics import Polestar4Dynamics
from winter_hvac import WinterHVACSimulation
from route_optimizer import (
    RouteSegment, RouteCandidate, RouteOptimizationEngine,
)

st.set_page_config(page_title="Polestar 4 — Energy Prediction", page_icon="🔋", layout="wide")
st.title("Polestar 4 Single Motor — Winter Energy Prediction")
st.caption("Predicting energy consumption (kWh/km) based on 17 input features from data_req.md.")

TRIP_DISTANCE_KM = 50.0  # reference trip for single-point prediction

# ============================================================================
# SIDEBAR — 17 INPUTS (data_req Mục 15)
# ============================================================================
with st.sidebar:
    st.header("Trip Conditions")

    st.subheader("Driving (Mục 15: 1-2)")
    speed = st.number_input("Speed (km/h)", 0.0, 200.0, 80.0, 5.0)
    acceleration = st.number_input("Acceleration (m/s²)", -5.0, 5.0, 0.0, 0.1)

    st.subheader("Environment (Mục 15: 3-8)")
    ambient_temp = st.slider("Ambient temperature (°C)", -25.0, 10.0, -10.0, 0.5)
    wind_speed = st.number_input("Wind speed (m/s)", 0.0, 40.0, 5.0, 0.5)
    wind_dir = st.number_input("Wind direction (°, 0=N)", 0.0, 360.0, 0.0, 5.0)
    elevation_gain = st.number_input("Elevation gain (m)", 0.0, 2000.0, 0.0, 10.0)
    road_gradient = st.number_input("Road gradient (%)", -20.0, 20.0, 0.0, 0.5)
    road_quality = st.selectbox("Road quality (Mục 15: 8)",
                                ["dry", "wet", "snow", "ice"])

    st.subheader("Traffic & Load (Mục 15: 9-10)")
    traffic = st.selectbox("Traffic density", ["low", "medium", "high"])
    payload = st.number_input("Payload (kg)", 0.0, 500.0, 150.0, 10.0)

    st.subheader("Battery (Mục 15: 11-13)")
    tire_pressure = st.number_input("Tire pressure (bar)", 1.8, 3.2, 2.8, 0.1)
    soc_start = st.slider("SOC (%)", 5.0, 100.0, 80.0, 1.0)
    battery_temp = st.slider("Battery temperature (°C)", -25.0, 25.0, -10.0, 0.5)

    st.subheader("HVAC & Regen (Mục 15: 14-15)")
    hvac_on = st.checkbox("HVAC ON", value=True)
    regen_avail = st.slider("Regen availability (%)", 0.0, 100.0, 100.0, 5.0)

    st.subheader("Mode & Weather (Mục 15: 16-17)")
    driving_mode = st.selectbox("Driving mode", ["eco", "standard", "performance"], index=1)
    weather = st.selectbox("Weather", ["clear", "rain", "snow", "fog"])

    with st.expander("⚙️ Advanced (Mục 2 supplementary)"):
        tire_type = st.selectbox("Tire type", ["summer", "allseason", "winter"], index=1)
        tire_condition = st.selectbox("Tire condition", ["new", "used", "worn"], index=1)
        climate_target = st.slider("Climate target (°C)", 16.0, 28.0, 22.0, 0.5)
        towing = st.checkbox("Towing", value=False)
        preconditioning = st.checkbox("Preconditioning", value=False)

    run = st.button("Predict Energy Consumption", type="primary", use_container_width=True)

# ============================================================================
# ENGINE
# ============================================================================
vehicle = Polestar4Dynamics(payload_kg=payload, battery_usable_capacity_kwh=92.0)
hvac = WinterHVACSimulation()
engine = RouteOptimizationEngine(vehicle=vehicle, hvac_sim=hvac,
                                  battery_initial_temp_c=battery_temp)

TRAFFIC_MAP = {"low": 0.5, "medium": 2.0, "high": 4.5}

def make_segment(**overrides):
    """Build a single reference segment with user inputs, allowing overrides for EDA."""
    base = dict(
        distance_m=TRIP_DISTANCE_KM * 1000.0,
        avg_speed_kmh=speed, grade_percent=road_gradient,
        ambient_temp_c=ambient_temp,
        acceleration_mps2=acceleration, elevation_m=elevation_gain,
        traffic_density=TRAFFIC_MAP[traffic],
        hvac_on=hvac_on, climate_target_c=climate_target,
        regen_availability_pct=regen_avail, driving_mode=driving_mode,
        weather=weather, towing=towing, preconditioning=preconditioning,
        tire_pressure_bar=tire_pressure, tire_type=tire_type,
        tire_condition=tire_condition, road_surface=road_quality,
        wind_speed_mps=wind_speed, wind_dir_deg=wind_dir, heading_deg=0.0,
    )
    base.update(overrides)
    return RouteSegment(**base)

def predict_kwh_per_km(**overrides):
    seg = make_segment(**overrides)
    route = RouteCandidate("reference", [seg])
    return engine.analyze_route(route).avg_consumption_kwh_per_km

# ============================================================================
# OUTPUT (Mục 15: predicted kWh/km)
# ============================================================================
if run:
    seg = make_segment()
    route = RouteCandidate("Your trip", [seg])
    result = engine.analyze_route(route)

    st.header("Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Energy consumption", f"{result.avg_consumption_kwh_per_km:.4f} kWh/km")
    c2.metric("Total energy (50 km)", f"{result.total_energy_kwh:.2f} kWh")
    c3.metric("Battery consumed", f"{result.battery_percent_consumed:.1f} %")

    st.divider()

    # ========================================================================
    # 5 EDA CHARTS (data_req Mục 13)
    # ========================================================================
    st.header("Exploratory Data Analysis")

    # 1. Temperature vs consumption
    st.subheader("1. Temperature → Consumption")
    temps = [-25, -20, -15, -10, -5, 0, 5, 10]
    temp_data = [(t, predict_kwh_per_km(ambient_temp_c=float(t))) for t in temps]
    st.line_chart(pd.DataFrame(temp_data, columns=["Temperature (°C)", "kWh/km"]).set_index("Temperature (°C)"))

    # 2. Speed vs consumption
    st.subheader("2. Speed → Consumption")
    speeds = [30, 50, 80, 100, 120, 140]
    speed_data = [(v, predict_kwh_per_km(avg_speed_kmh=float(v))) for v in speeds]
    st.line_chart(pd.DataFrame(speed_data, columns=["Speed (km/h)", "kWh/km"]).set_index("Speed (km/h)"))

    # 3. Elevation → Consumption
    st.subheader("3. Elevation → Consumption")
    elevs = [0, 200, 400, 600, 800]
    def elev_to_grade(e):
        return (e / (TRIP_DISTANCE_KM * 1000.0)) * 100.0
    elev_data = [(e, predict_kwh_per_km(grade_percent=elev_to_grade(e))) for e in elevs]
    st.line_chart(pd.DataFrame(elev_data, columns=["Elevation gain (m)", "kWh/km"]).set_index("Elevation gain (m)"))

    # 4. Traffic → Consumption
    st.subheader("4. Traffic → Consumption")
    traffic_data = [
        (label, predict_kwh_per_km(traffic_density=density))
        for label, density in [("Low", 0.5), ("Medium", 2.0), ("High", 4.5)]
    ]
    st.bar_chart(pd.DataFrame(traffic_data, columns=["Traffic", "kWh/km"]).set_index("Traffic"))

    # 5. HVAC ON vs OFF
    st.subheader("5. HVAC → Consumption")
    hvac_data = [
        ("HVAC ON",  predict_kwh_per_km(hvac_on=True)),
        ("HVAC OFF", predict_kwh_per_km(hvac_on=False)),
    ]
    st.bar_chart(pd.DataFrame(hvac_data, columns=["HVAC", "kWh/km"]).set_index("HVAC"))

else:
    st.info("Nhập 17 inputs bên trái và bấm **Predict Energy Consumption**.") 