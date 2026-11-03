# Predictive-Modeling for EV
> **Status:** Work In Progress. <!-- TODO: update (e.g. Data phase / Modelling / Evaluation / Stable) -->

Building machine learning models for predicting the **energy consumption (kWh/km) of the Polestar 4** under different driving, vehicle and environmental conditions. 

<br>

## **Predictive Modeling team**

This repository is currently maintained by the **Predictive Modeling team**, which is responsible for investigating and implementing ML models. 


| Name | Role |
|---|---|
| Dai Lam Hoang | `Project Lead` |
| Hamza Sahqani | `Engineer` |
| Minh Duy Nguyen | `Engineer` |
| Duc Hieu Nguyen | `Engineer` |


---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Getting Started](#3-getting-started)
4. [Data](#4-data)
5. [Modelling Approach](#5-modelling-approach)
6. [Working With Other Teams](#6-working-with-other-teams)
7. [Development](#7-development)


---

## 1. Project Overview

| | |
|---|---|
| **Goal** | Predict vehicle energy consumption under varying conditions |
| **Target variable** | `Energy Consumption (kWh/km)` |
| **Problem type** | WIP (potentially on time-series data) |
| **Vehicle** | Polestar 4 |
| **Possible downstream use** | Driving-range estimation: `Range = Available Battery Energy / Predicted Consumption per km` |

Conceptually:

```
Input conditions → Machine Learning Model → Predicted energy consumption (kWh/km)
```

Candidate input features include speed, acceleration, ambient and battery temperature, state of charge (SOC), HVAC usage, wind, elevation / road gradient, traffic, payload, tyre pressure, regenerative braking and driving mode. The full list and rationale are in [`Data_Req.md`](Data_Req.md).

---

## 2. Repository Structure 
>NOTE!: Subject to Changes
```

Predictive-Modeling/
├── Data/                          # Datasets and data documentation
├── Demo/                          # Demonstration of the model / pipeline
├── Ingestion/
│   └── ev-telemetry-ingestion/    # EV telemetry ingestion
├── Model-Evaluation/              # Model evaluation work
├── .vscode/                       # Shared VS Code settings
├── Data_Req.md                    # Data requirements and data-phase specification
├── .gitignore
├── .gitattributes
└── README.md
```

<!-- TODO: Replace the descriptions below with one precise sentence per folder, plus entry-point files. -->

| Path | Purpose | Notes |
|---|---|---|
| `Data_Req.md` | Defines the data needed, its sources, processing steps, quality checks, limitations and evaluation plan | Primary reference for data questions |
| `Data/` | Datasets and related documentation | <!-- TODO: raw vs processed, formats, size --> |
| `Ingestion/ev-telemetry-ingestion/` | Ingestion of EV telemetry data | <!-- TODO: input source, output location, how to run --> |
| `Model-Evaluation/` | Evaluation of trained models | <!-- TODO: metrics, notebooks/scripts, results --> |
| `Demo/` | POC of the workings of the project | <!-- TODO: how to run --> |

---

## 3. Getting Started

<!-- TODO: Fill in from your actual setup. Delete anything that does not apply. -->

### Prerequisites

- Git
- Python `<version>` <!-- TODO: confirm language/version -->
- `<package manager / environment tool>` <!-- TODO: pip, conda, poetry, etc. -->

### Installation

```bash
git clone https://github.com/HoangDaiLam/Predictive-Modeling.git
cd Predictive-Modeling

# TODO: create environment and install dependencies, e.g.
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
```

### Configuration

<!-- TODO: list required environment variables, API keys or credentials. Never commit secrets. -->

| Variable | Description | Required |
|---|---|---|
| `<VAR_NAME>` | `<description>` | Yes / No |

### Running

```bash
# TODO: ingestion
# TODO: training
# TODO: evaluation
# TODO: demo
```

---

## 4. Data

> This section is intended for teams that need to understand, provide or request data. Full detail is in [`Data_Req.md`](Data_Req.md).

### 4.1 Data we need

| Category | Variables | Unit / format |
|---|---|---|
| Battery | State of charge, battery temperature | %, °C |
| Driving | Speed, acceleration/braking, historical driving pattern | km/h, m/s², time series |
| Environment | Ambient temperature, wind speed/direction, precipitation | °C, m/s, category |
| Road & terrain | Elevation, gradient, road condition | m, %, category |
| Traffic | Traffic density, congestion, stops | Category / numeric |
| Vehicle | Payload, tyre pressure, towing, HVAC / climate usage, preconditioning, regenerative braking, driving mode | kg, bar, on/off, category |
| **Target** | **Energy consumption** | **kWh/km** |

### 4.2 Data types

Numerical, categorical, boolean (on/off, encoded 1/0) and time-series data.

### 4.3 Data source classification

Every dataset must be labelled as one of the following, and **simulated data must never be presented as real vehicle measurements**.

| Class | Description |
|---|---|
| Measured | Directly measured from a vehicle, sensor or experiment |
| External | Collected by another organisation (e.g. weather, traffic, elevation) |
| Manufacturer / technical | From technical documentation (e.g. battery capacity, mass); not real-world consumption |
| Simulated | Generated by a mathematical or computational model |

### 4.4 Processing pipeline

```
Raw data → Collection → Integration → Cleaning → Missing-value handling
→ Outlier checking → Unit conversion → Categorical encoding
→ Feature engineering → Exploratory analysis → Final dataset → ML
```

Datasets from different sources must be aligned on compatible **time, location, trip and vehicle conditions** before being combined.

### 4.5 Standard units

Each variable has one standard unit across the project (e.g. energy in kWh, power in kW, speed in km/h, temperature in °C). Conversions: `1 kWh = 1000 Wh`, `1 m/s = 3.6 km/h`.

### 4.6 Data quality criteria

Completeness, accuracy, consistency, validity (realistic ranges), timeliness, and synchronisation across sources.

### 4.7 Known limitations

- Some vehicle variables may not be publicly available.
- Some datasets may not be Polestar 4-specific.
- Some measurements may need to be simulated.
- Sampling rates differ between sources.
- Some variables contain missing values.
- External data may not perfectly represent actual vehicle conditions.

### 4.8 Data access, ownership and governance

<!-- TODO: complete. These are the questions other teams will ask first. -->

| Item | Details |
|---|---|
| Where data is stored | `<location: Data/ folder, cloud storage, database>` |
| Data owner / steward | `<name or team>` |
| How to request access | `<process, e.g. issue template, email, ticket>` |
| Typical turnaround | `<expected time>` |
| Update frequency | `<static / daily / per experiment>` |
| Formats provided | `<CSV / JSON / Parquet>` |
| Sensitivity / privacy | `<e.g. contains GPS traces, vehicle identifiers? Anonymisation applied?>` |
| Licence / usage restrictions | `<terms for sharing and reuse>` |

### 4.9 Dataset catalogue

<!-- TODO: One row per dataset. Mirrors the Data Requirements Table in Data_Req.md. -->

| Dataset | Description | Source class | Format | Location | Availability |
|---|---|---|---|---|---|
| `<name>` | `<what it contains>` | Measured / External / Manufacturer / Simulated | `<format>` | `<path>` | Available / To verify |

---

## 5. Modelling Approach

- **Task:** supervised regression predicting `kWh/km`.
- **Data splitting:** train / validation / test. Because the data may be time-series, splits must respect time order to **avoid data leakage** (no future observations in training).
- **Evaluation metrics:** MAE (average absolute error), RMSE (penalises large errors more), R² (variance explained). Final metrics will be selected to match project requirements.
- **Models investigated:** <!-- TODO: list models tried and where results live (see Model-Evaluation/) -->
- **Results:** <!-- TODO: summary table or link -->

Feature engineering under consideration includes energy per kilometre, road gradient from elevation, headwind/tailwind/crosswind components, and temperature threshold features (e.g. below 0 °C, below −10 °C).

---

## 6. Working With Other Teams

<!-- TODO: confirm interfaces. Remove rows that do not apply. -->

| If you want to... | Do this |
|---|---|
| **Provide data to us** | Check the variable list in [Section 4.1](#41-data-we-need) and [`Data_Req.md`](Data_Req.md), then contact the data owner with units, sampling rate, format and source class |
| **Request data from us** | Follow the access process in [Section 4.8](#48-data-access-ownership-and-governance) |
| **Use our model outputs** | `<describe interface: files, API, predictions format>` |
| **Report a problem or ask a question** | Open an [issue](https://github.com/HoangDaiLam/Predictive-Modeling/issues) |

When sharing data with us, please include: variable names and meanings, units, sampling rate, time zone and timestamp format, source class, and any known gaps or quality issues.

---

## 7. Development
### For team members only 
1. Create a branch from `main` (e.g. `feature/<short-description>`).
2. Make focused commits with clear messages.
3. Open a [pull request](https://github.com/HoangDaiLam/Predictive-Modeling/pulls) against `main` and request review from a team member.
4. Do **not** commit secrets, credentials or restricted data. Check `.gitignore` before adding data files.

<!-- TODO: add code style, testing and review requirements if you have them. -->

---

