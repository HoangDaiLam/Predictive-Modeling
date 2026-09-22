# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: dashboard.py
STEP 5 — Streamlit user dashboard
================================================================================
Purpose:
    A visual interface for users to:
        1. Enter/select the Origin and Destination
        2. Input winter weather conditions (temperature, which can be fetched
           automatically via API if keys are configured, or entered manually for offline demo)
        3. View real-time HVAC/BTMS metrics (cabin heating power,
           battery heating power, target temperature)
        4. Compare multiple routes and see the AI Engine's recommended optimal route
           with explanations

Run:
    streamlit run dashboard.py

How the Dashboard communicates with the AI Engine backend:
    The Dashboard does NOT make separate HTTP API calls — it imports the Python
    classes/functions directly from 3 backend modules (vehicle_dynamics,
    winter_hvac, route_optimizer) in the same process (in-process call), because
    Streamlit runs as an independent Python application on the local machine/server.
    This is a "monolithic Streamlit app" architecture — simple and suitable for demo/internal use.

    If you want to split the Dashboard and AI Engine into two independent services
    (for example, backend on a server and dashboard elsewhere), see the guidance in
    README.md under the "Extended architecture with FastAPI" section — in that setup,
    the AI Engine is exposed as a REST API with FastAPI, and the Dashboard calls it over
    HTTP requests instead of importing directly.
================================================================================
"""

import streamlit as st
import pandas as pd

from vehicle_dynamics import Polestar4Dynamics
from winter_hvac import WinterHVACSimulation
from route_optimizer import (
    RouteSegment,
    RouteCandidate,
    RouteOptimizationEngine,
)
from api_integration import APIConfig, APIIntegrationError, generate_mock_elevation_profile

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Polestar 4 — Winter Energy AI Engine",
    page_icon="🔋",
    layout="wide",
)

st.title("Polestar 4 Single Motor — Winter Battery Energy AI Engine")
st.caption(
    "Predicting and optimizing battery consumption for the Polestar 4 Single Motor (Standard Mode) in harsh winter conditions."
)

# ============================================================================
# SIDEBAR: CẤU HÌNH XE & MÔI TRƯỜNG
# ============================================================================
with st.sidebar:
    st.header("This Trip's Number")

    st.subheader("Origin")
    origin_lat = st.number_input("(lat)", value=21.0285, format="%.4f", key="origin_lat")
    origin_lng = st.number_input("(lng)", value=105.8542, format="%.4f", key="origin_lng")

    st.subheader("Destination")
    dest_lat = st.number_input("(lat)", value=20.8133, format="%.4f", key="dest_lat")
    dest_lng = st.number_input("(lng)", value=105.3383, format="%.4f", key="dest_lng")

    st.divider()

    st.subheader("Weather Conditions")
    ambient_temp_c = st.slider(
        "Temperature (°C)", min_value=-25.0, max_value=5.0, value=-10.0, step=0.5
    )
    battery_initial_temp_c = st.slider(
        "Battery Start Temperature (°C, 'cold-soaked')",
        min_value=-25.0, max_value=20.0, value=ambient_temp_c, step=0.5,
    )

    st.divider()

    st.subheader("Car's Details (Advanced)")
    with st.expander("Adjust Polestar 4 weight"):
        payload_kg = st.number_input("Extra weight (kg)", value=150.0, step=10.0)
        battery_capacity_kwh = st.number_input("Current Battery (kWh)", value=92.0, step=1.0)

    st.divider()
    use_real_api = st.checkbox(
        "Use Real API (Google Maps / OpenWeatherMap)",
        value=False,
        help="You need to configure the environment variables GOOGLE_MAPS_API_KEY and OPENWEATHERMAP_API_KEY. "
             "If disabled, the system uses simulated mock data for demo purposes.",
    )

    run_button = st.button("Run and Analyze the best route", type="primary", use_container_width=True)

# ============================================================================
# INITIALIZE BACKEND ENGINE
# ============================================================================
vehicle = Polestar4Dynamics(payload_kg=payload_kg, battery_usable_capacity_kwh=battery_capacity_kwh)
hvac_sim = WinterHVACSimulation()
engine = RouteOptimizationEngine(
    vehicle=vehicle,
    hvac_sim=hvac_sim,
    battery_initial_temp_c=battery_initial_temp_c,
)

# ============================================================================
# SECTION 1: REAL-TIME HVAC / BTMS PARAMETERS
# ============================================================================
st.header("HVAC & BTMS (Real-time)")

hvac_power_result = hvac_sim.compute_cabin_hvac_electric_power_kw(ambient_temp_c)
btms_power_result = hvac_sim.compute_battery_heating_power_kw(ambient_temp_c, battery_initial_temp_c)
capacity_factor = hvac_sim.compute_capacity_fade_factor(battery_initial_temp_c)
resistance_multiplier = hvac_sim.compute_internal_resistance_multiplier(battery_initial_temp_c)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Cabin heating power",
        f"{hvac_power_result['total_hvac_electric_power_kw']:.2f} kW",
        help=f"Current COP of the heat pump: {hvac_power_result['heat_pump_cop']:.2f}",
    )
with col2:
    st.metric(
        "Battery heating power (BTMS)",
        f"{btms_power_result['total_btms_electric_power_kw']:.2f} kW",
    )
with col3:
    st.metric(
        "Effective battery capacity factor",
        f"{capacity_factor*100:.1f}%",
        delta=f"{(capacity_factor-1)*100:.1f}% compare to 25°C",
        delta_color="inverse",
    )
with col4:
    st.metric(
        "Battery internal resistance increase factor",
        f"x{resistance_multiplier:.2f}",
        help="The increase in battery internal resistance at low temperatures leads to higher losses during high‑current charging and discharging.",
    )

with st.expander("Detailed HVAC Power Decomposition"):
    hvac_detail_df = pd.DataFrame([
        {"Component": "Heat Pump", "Electric Power (kW)": hvac_power_result["heat_pump_electric_power_kw"]},
        {"Component": "PTC Backup Heater", "Electric Power (kW)": hvac_power_result["ptc_backup_electric_power_kw"]},
        {"Component": "TOTAL", "Electric Power (kW)": hvac_power_result["total_hvac_electric_power_kw"]},
    ])
    st.dataframe(hvac_detail_df, hide_index=True, use_container_width=True)

st.divider()

# ============================================================================
# PHẦN 2: SO SÁNH TUYẾN ĐƯỜNG
# ============================================================================
st.header("Route Comparison & Optimization")

if run_button:
    with st.spinner("Retrieving terrain and weather data and performing energy consumption calculations…"):

        api_config = APIConfig()
        routes = []

        if use_real_api:
            # ---- Attempt to use the real API ----
            try:
                from route_optimizer import build_route_candidate_from_api
                route1 = build_route_candidate_from_api(
                    "ROUTE 1 (REAL API)", "Direct route computed via Google Routes API",
                    (origin_lat, origin_lng), (dest_lat, dest_lng), api_config,
                )
                routes.append(route1)
                st.success("Successfully retrieved data from the Google Maps and OpenWeatherMap APIs.")
            except APIIntegrationError as e:
                st.warning(f"Failed to retrieve data from the live API ({e}). Falling back to the simulated dataset.")
                use_real_api = False

        if not use_real_api:
            # ---- Simulated data: 2 example routes (direct climb vs flat detour) ----
            mock_profile_direct = generate_mock_elevation_profile(
                origin_lat, origin_lng, dest_lat, dest_lng, num_points=8, hill_amplitude_m=120.0
            )
            mock_profile_flat = generate_mock_elevation_profile(
                origin_lat, origin_lng, dest_lat, dest_lng, num_points=8, hill_amplitude_m=15.0
            )

            def profile_to_segments(profile, avg_speed_kmh, distance_scale_km, temp_c):
                segs = []
                n = len(profile) - 1
                seg_dist_m = (distance_scale_km * 1000.0) / max(n, 1)
                for i in range(n):
                    segs.append(RouteSegment(
                        distance_m=seg_dist_m,
                        avg_speed_kmh=avg_speed_kmh,
                        grade_percent=profile[i].get("grade_percent", 0.0),
                        ambient_temp_c=temp_c,
                    ))
                return segs

            route_direct = RouteCandidate(
                route_name="Route 1: Direct Mountain",
                description="Shorter route with several steep inclines",
                segments=profile_to_segments(mock_profile_direct, 70, 45.0, ambient_temp_c),
            )
            route_flat = RouteCandidate(
                route_name="Route 2: Flat Loop",
                description="The route is longer, but it's almost completely flat.",
                segments=profile_to_segments(mock_profile_flat, 90, 54.0, ambient_temp_c),
            )
            routes = [route_direct, route_flat]

        result = engine.compare_and_optimize(routes)
        st.session_state["last_result"] = result

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    # ---- Overview comparison table ----
    st.subheader("Comparison Table of Routes")

    comparison_rows = []
    optimal_name = result["optimal_route"]["route_name"]
    for r in result["all_results"]:
        comparison_rows.append({
            "Route": ("⭐ " if r["route_name"] == optimal_name else "") + r["route_name"],
            "Description": r["description"],
            "Distance (km)": r["distance_km"],
            "Duration (min)": r["duration_min"],
            "Driving Energy (kWh)": r["driving_energy_kwh"],
            "Winter Overhead (kWh)": r["winter_overhead_kwh"],
            "Total Energy (kWh)": r["total_energy_kwh"],
            "Battery Percent Consumed": r["battery_percent_consumed"],
            "Max Grade (%)": r["max_grade_percent"],
        })

    df_compare = pd.DataFrame(comparison_rows)
    st.dataframe(
        df_compare.style.highlight_min(subset=["Total Energy (kWh)"], color="#d4f7dc"),
        hide_index=True,
        use_container_width=True,
    )

    # ---- Visual comparison chart ----
    chart_df = pd.DataFrame([
        {
            "Route": r["route_name"],
            "Driving Energy": r["driving_energy_kwh"],
            "Winter Overhead (HVAC+BTMS)": r["winter_overhead_kwh"],
        }
        for r in result["all_results"]
    ]).set_index("Route")
    st.bar_chart(chart_df)

    # ---- Explanation of the optimal decision ----
    st.subheader("Explanation of the AI Engine's Decision")
    st.info(result["explanation"])

    # ---- Detailed breakdown of the optimal route ----
    with st.expander("Detailed Breakdown of Segments in the Optimal Route"):
        seg_df = pd.DataFrame(result["optimal_route"]["segment_breakdown"])
        st.dataframe(seg_df, hide_index=True, use_container_width=True)

else:
    st.info("Enter information in the left sidebar and click **'Run Analysis & Optimize Route'** to get started.")

st.divider()
st.caption(
    "Actual results will vary depending on operating conditions, driving style, real-time battery status, and other factors"
)