# Polestar 4 Winter Energy AI Engine

AI Engine + Dashboard for predicting and optimizing battery consumption for **Polestar 4
Single Motor (Standard Mode)** under **extremely cold winter conditions** (sub-zero
temperatures, requiring cabin heating + battery heating).

## 1. System Architecture

```text
polestar4_ai_engine/
├── vehicle_dynamics.py     # STEP 1 — Vehicle physics & energy consumption (Polestar4Dynamics)
├── winter_hvac.py          # STEP 2 — Cabin HVAC + winter battery BTMS (WinterHVACSimulation)
├── api_integration.py      # STEP 3 — Google Maps Elevation/Routes API + OpenWeatherMap integration
├── route_optimizer.py      # STEP 4 — Core AI Engine: route comparison & optimization
├── dashboard.py            # STEP 5 — Dashboard interface (Streamlit)
├── sample_api_response.json # Sample JSON data illustrating the API structure
├── requirements.txt
└── README.md
```

**Data Flow:**

```text
[User] --(origin, destination, temperature)--> [dashboard.py]
                                                        |
                                                        v
                                           [route_optimizer.py] (AI Engine)
                                           /              |              \
                                          v               v               v
                           [vehicle_dynamics.py]  [winter_hvac.py]  [api_integration.py]
                           (force/power/energy   (cabin HVAC +     (elevation, weather,
                            consumption)          battery BTMS)     route alternatives)
                                                        |
                                                        v
                                           [Result: route comparison table
                                            + optimal route + explanation]
                                                        |
                                                        v
                                               [dashboard.py display]
```

The Dashboard **directly imports** the backend Python modules within the same
process (in-process function call) — no separate API server is required for
the basic demo.

## 2. Installation

Python 3.9 or later is required.

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## 3. API Key Configuration (Optional)

If you want to use **REAL** elevation/weather data instead of simulated (mock)
data, set the following two environment variables before running:

```bash
export GOOGLE_MAPS_API_KEY="your_google_maps_api_key"
export OPENWEATHERMAP_API_KEY="your_openweathermap_api_key"
```

* Google Maps API key: **Elevation API** and **Routes API** must be enabled in
  Google Cloud Console (https://console.cloud.google.com/apis/library).
* OpenWeatherMap API key: register for free at https://openweathermap.org/api.

> If the keys are **NOT** configured, the system will still run normally using
> simulated data (`generate_mock_elevation_profile`) — suitable for offline
> demos/testing.

## 4. Running Individual Modules (Backend Logic Testing)

Each module contains a self-test block (`if __name__ == "__main__":`) that can
be run independently to view the results in the console:

```bash
python3 vehicle_dynamics.py    # Test vehicle force/power/energy calculations
python3 winter_hvac.py         # Test HVAC/BTMS power consumption by temperature
python3 api_integration.py     # Test API calls (uses mock data if no key is configured)
python3 route_optimizer.py     # Test comparison & optimization of 2 sample routes
```

## 5. Running the Dashboard

```bash
streamlit run dashboard.py
```

After starting it, open the browser at the address displayed by Streamlit
(default: `http://localhost:8501`).

**Dashboard Usage Steps:**

1. In the left sidebar, enter the coordinates for the **Origin** and
   **Destination**.
2. Adjust the **ambient temperature** and **battery temperature at departure**
   using the sliders to simulate specific winter conditions.
3. (Optional) Adjust the vehicle payload and battery capacity in the
   "Advanced vehicle parameters" section.
4. Click the **"🚀 Run Analysis & Optimize Route"** button.
5. View:

   * Real-time HVAC/BTMS metrics at the top of the page.
   * A route comparison table (the optimal route is marked with ⭐ and
     highlighted in green).
   * A bar chart comparing driving energy vs. winter-related energy losses.
   * A text explanation (in English) of why the AI Engine selected that route.
   * Detailed information for each segment of the optimal route (expand
     the "🔬 Details..." section).

## 6. How the Dashboard Calls the Backend AI Engine (Technical Details)

In `dashboard.py`:

```python
from vehicle_dynamics import Polestar4Dynamics
from winter_hvac import WinterHVACSimulation
from route_optimizer import RouteOptimizationEngine, RouteCandidate, RouteSegment

vehicle = Polestar4Dynamics(payload_kg=..., battery_usable_capacity_kwh=...)
hvac_sim = WinterHVACSimulation()
engine = RouteOptimizationEngine(vehicle=vehicle, hvac_sim=hvac_sim,
                                  battery_initial_temp_c=...)

result = engine.compare_and_optimize([route_1, route_2, ...])
# result["optimal_route"], result["all_results"], result["explanation"]
```

Streamlit reruns the entire script whenever the user interacts with the
interface (buttons, sliders, etc.), so the analysis results are stored in
`st.session_state` to prevent them from being lost when the user opens/closes
other "expander" sections.

## 7. Extended Architecture with FastAPI (When Separating Backend/Frontend)

If you want to deploy the AI Engine as an **independent REST API** (for example,
so that multiple clients — web, mobile, or in-vehicle systems — can access it),
you can wrap `RouteOptimizationEngine` with FastAPI as follows:

```python
# backend_api.py (example extension, not included in this package)
from fastapi import FastAPI
from pydantic import BaseModel
from route_optimizer import RouteOptimizationEngine, RouteCandidate, RouteSegment

app = FastAPI()
engine = RouteOptimizationEngine()

class OptimizeRequest(BaseModel):
    routes: list  # route candidates as JSON

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    routes = [RouteCandidate(
        route_name=r["route_name"],
        description=r.get("description", ""),
        segments=[RouteSegment(**s) for s in r["segments"]],
    ) for r in req.routes]
    return engine.compare_and_optimize(routes)
```

Run the backend:

```bash
uvicorn backend_api:app --reload --port 8000
```

The `dashboard.py` will then replace the `import route_optimizer` section with
an HTTP request:

```python
import requests
response = requests.post("http://localhost:8000/optimize", json={"routes": [...]})
result = response.json()
```

This architecture clearly separates the AI Engine (which can be scaled
independently and deployed on a separate server) from the Dashboard (which
serves only as the presentation layer).

## 8. Model Accuracy Notes

* The physics equations (aerodynamic drag, rolling resistance, grade force,
  kinetic energy, regenerative braking) follow the standard
  "point-mass longitudinal vehicle dynamics" model commonly used in electric
  vehicle engineering.
* The HVAC/BTMS model uses a "lumped thermal mass" approach — a reasonable
  simplification for trip-level energy consumption prediction, but it does not
  replace detailed CFD/thermal simulations used at the design level.
* Vehicle parameters (mass, Cd, battery capacity, etc.) are approximate values
  based on publicly available information for the Polestar 4 Single Motor.
  They should be recalibrated in `Polestar4Dynamics(...)` if more accurate
  manufacturer data is available for a specific market/model year.
* This is a **prediction/simulation tool**, not a real-time measurement system
  operating on an actual vehicle — the results should be used as guidance and
  should not replace the vehicle's actual battery/range instrumentation.
