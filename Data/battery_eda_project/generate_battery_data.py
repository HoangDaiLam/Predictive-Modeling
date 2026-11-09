#!/usr/bin/env python3
"""
Generate SYNTHETIC but realistic trip-segment telemetry for the Polestar 4.

Not measured vehicle data. Energy use comes from a road-load physics model
(aero drag, rolling resistance, grade, stop-start kinetic losses, regen,
HVAC with heat-pump COP, battery-temperature efficiency), driven by correlated
inputs (season/time -> weather -> road state -> speed/traffic -> driving pattern)
and per-vehicle / per-driver effects.

Vehicle constants come from published Polestar 4 data (see SPECS/COMMON below); default variant is Long Range Dual Motor;
rolling resistance is calibrated so a WLTP-cycle run reproduces the published
WLTP battery-side consumption (check with --check-wltp). Drivetrain, regen, HVAC,
battery-temperature and driver effects are generic engineering assumptions, not
Polestar data.

Rows are segments within trips, trips belong to vehicles, and the log has
realistic defects (missing values, sensor spikes, duplicate rows). Use
--clean to disable the defects. Split by vehicle_id (not randomly) when modelling.

Usage:
    python generate_battery_data.py
    python generate_battery_data.py --rows 50000 --vehicles 80 --seed 7
    python generate_battery_data.py --clean
    python generate_battery_data.py --variant single
    python generate_battery_data.py --check-wltp
"""

from pathlib import Path
import argparse
import numpy as np
import polars as pl

G = 9.81
YEAR_DAYS = 365
YEAR_START = np.datetime64("2025-01-01T00:00:00")  # a Wednesday
ROAD = ["urban", "suburban", "rural", "motorway"]

# Polestar 4 (coupe-SUV) constants. Source tags:
#   [P]  polestar.com official specification page (Canada market)
#   [R]  Polestar reveal coverage (Cd 0.269; later listings round to 0.26)
#   [T]  third-party spec aggregators (evkx.net, evspecs.org); not official
#   [M]  Polestar-4-specific tyre-pressure table, 255/50R20, 1-3 persons: 39 psi
#   [E]  my estimate / calibration, NOT published
SPECS = {
    "single": dict(kerb_kg=2230, tow_kg=1500, awd=False, wltp_kwh100=15.4),  # [P] [P] ; WLTP [T]
    "dual": dict(kerb_kg=2355, tow_kg=2000, awd=True, wltp_kwh100=16.2),     # [P] [P] ; WLTP [T] (16.2-16.8)
}
COMMON = dict(
    cd=0.269,          # [R]
    frontal_m2=2.41,   # [E] 0.8 * 1.95 m width * 1.544 m height; Polestar publishes no frontal area
    usable_kwh=94.0,   # [T] 100 kWh gross [P]; 94 kWh usable is third-party
    p_cold=2.70,       # [M] bar (39 psi)
    trailer_cda=0.40,  # [E]
    crr0=0.01101,      # [E] calibrated so dual-motor WLTP run = 16.2 kWh/100km (--check-wltp)
)
# (speed variability cv, stops per km) by driving pattern  [E]
PATTERN = {"steady": (0.10, 0.03), "mixed": (0.30, 0.30), "stop_start": (0.60, 1.4)}


# ---------------------------------------------------------------- helpers
def _cat_idx(rng, probs):
    """Sample category indices from a row-wise probability matrix."""
    u = rng.random(len(probs))[:, None]
    return np.minimum((u > np.cumsum(probs, axis=1)).sum(axis=1), probs.shape[1] - 1)


def _trip_cumsum(x, trip):
    """Inclusive within-trip cumulative sum; rows must be contiguous by trip."""
    first = np.r_[True, trip[1:] != trip[:-1]]
    starts = np.flatnonzero(first)
    cs = np.cumsum(x)
    offset = (cs - x)[starts][np.cumsum(first) - 1]
    return cs - offset, starts


def _weather(rng):
    """Daily mean temperature (seasonal + persistent anomaly) and precipitation."""
    doy = np.arange(YEAR_DAYS)
    seasonal = 7 - 12 * np.cos(2 * np.pi * (doy - 15) / YEAR_DAYS)
    rho = 0.75
    anom = np.zeros(YEAR_DAYS)
    anom[0] = rng.normal(0, 4)
    shocks = rng.normal(0, 4 * np.sqrt(1 - rho**2), YEAR_DAYS)
    precip = np.zeros(YEAR_DAYS, dtype=bool)
    precip[0] = rng.random() < 0.33
    for d in range(1, YEAR_DAYS):
        anom[d] = rho * anom[d - 1] + shocks[d]
        precip[d] = rng.random() < (0.55 if precip[d - 1] else 0.22)
    return seasonal + anom, precip


def _fleet(rng, n, S):
    z = rng.normal(0, 1, n)  # latent driver aggressiveness
    z_obs = z + rng.normal(0, 0.6, n)  # noisy proxy -> historical pattern label
    q = np.quantile(z_obs, [0.30, 0.80])
    hist = np.where(
        z_obs < q[0], "efficient", np.where(z_obs > q[1], "energy_intensive", "typical")
    )
    return {
        "z": z,
        "hist": hist,
        "mass": rng.normal(S["kerb_kg"], 25, n),  # wheel/option spread
        "cda": rng.normal(S["cd"] * S["frontal_m2"], 0.01, n),
        "crr": np.clip(rng.normal(S["crr0"], 0.0006, n), 0.008, None),
        "eff": np.exp(rng.normal(0, 0.025, n)),
        "cap": S["usable_kwh"] * np.clip(rng.normal(0.94, 0.03, n), 0.85, 1.0),
        "limit": rng.choice([80.0, 90.0, 100.0], n, p=[0.55, 0.25, 0.20]),
        "p_base": rng.normal(S["p_cold"], 0.06, n),
        "leak": 0.0012 * np.exp(rng.normal(0, 0.3, n)),
        "phase": rng.integers(0, 120, n),
        "tow_ok": rng.random(n) < 0.25,
    }


# ------------------------------------------------------------- physics core
def _energy_per_100km(S, *, T, Tb, speed, slope, mass, cda, crr0, p, cv, stops_km,
                      snow, wet, scale, pre, elapsed_min, wind, z):
    """Deterministic battery-side consumption (kWh/100 km) before noise."""
    rho = 101325 / (287.05 * (T + 273.15))
    v = speed / 3.6
    v_air2 = v**2 * (1 + cv**2) + 2 * v * wind + wind**2
    th = np.arctan(slope / 100)
    crr = (
        crr0
        * (1 + 0.15 * np.maximum(S["p_cold"] - p, 0))
        * (1 + 0.004 * np.maximum(15 - T, 0))
        * np.where(snow, 1.25, np.where(wet, 1.05, 1.0))
    )
    f_net = 0.5 * rho * cda * v_air2 + crr * mass * G * np.cos(th) + mass * G * np.sin(th)

    eta_d = 0.91 - 0.08 * np.exp(-speed / 20) - (0.01 if S["awd"] else 0.0)
    eta_r = 0.65 * np.clip((Tb + 10) / 25, 0.25, 1.0) * np.where(snow, 0.7, 1.0)
    trac = np.where(f_net > 0, f_net / 36 / eta_d, f_net / 36 * eta_r)

    kin = stops_km * 0.5 * mass * np.minimum(1.5 * v, 27.8) ** 2 / 3.6e6 * 100 * (1 / eta_d - eta_r) * np.exp(0.25 * z)

    cop_h = np.clip(2.6 + 0.07 * T, 1.0, 3.2)
    heat = scale * np.maximum(21 - T, 0) * 0.15 / cop_h * (1 - 0.25 * pre * np.exp(-elapsed_min / 40))
    cool = scale * np.maximum(T - 22, 0) * 0.22 / 2.5
    batt_heat = 1.5 * np.clip((10 - Tb) / 20, 0, 1)
    aux = (0.7 + 0.15 * scale + heat + cool + batt_heat) * 100 / speed

    batt_pen = 1 + 0.0045 * np.maximum(20 - Tb, 0) + 0.003 * np.maximum(Tb - 38, 0)
    return (trac + kin) * batt_pen + aux


def wltp_check(variant="dual", crr0=None):
    """Model consumption over WLTP class-3b phases at 23 C, no HVAC, vs published value."""
    S = {**COMMON, **SPECS[variant]}
    if crr0 is not None:
        S["crr0"] = crr0
    phases = [  # km, mean km/h, pattern
        (3.095, 25.7, "stop_start"), (4.756, 44.5, "mixed"),
        (7.162, 60.8, "mixed"), (8.254, 94.0, "steady"),
    ]
    km = np.array([p[0] for p in phases])
    cons = _energy_per_100km(
        S, T=23.0, Tb=25.0, speed=np.array([p[1] for p in phases]), slope=0.0,
        mass=S["kerb_kg"] + 175, cda=S["cd"] * S["frontal_m2"], crr0=S["crr0"], p=S["p_cold"],
        cv=np.array([PATTERN[p[2]][0] for p in phases]),
        stops_km=np.array([PATTERN[p[2]][1] for p in phases]),
        snow=False, wet=False, scale=0.0, pre=False, elapsed_min=30.0, wind=0.0, z=0.0,
    )
    return float((cons * km).sum() / km.sum())


# ------------------------------------------------------------- simulation
def _simulate(rng, n_trips, W, F, S):
    daily_T, daily_precip = W
    V = len(F["z"])

    # ---- trip level
    trip_type = _cat_idx(rng, np.tile([0.50, 0.35, 0.15], (n_trips, 1)))  # city/mixed/long
    n_seg = np.minimum(
        np.select([trip_type == 0, trip_type == 1], [1, 2], 3)
        + rng.poisson(np.select([trip_type == 0, trip_type == 1], [0.9, 0.8], 1.2)),
        6,
    )
    veh = rng.integers(0, V, n_trips)
    day = rng.integers(0, YEAR_DAYS, n_trips)
    weekday = ((day + 2) % 7) < 5
    comp = rng.random(n_trips)
    hour_wd = np.select(
        [comp < 0.28, comp < 0.58, comp < 0.88],
        [rng.normal(7.8, 1.0, n_trips), rng.normal(16.8, 1.4, n_trips), rng.normal(12, 2.5, n_trips)],
        rng.normal(19.5, 1.8, n_trips),
    )
    hour = np.where(weekday, hour_wd, rng.normal(13, 3.5, n_trips)) % 24
    T_trip = daily_T[day] + 3.5 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.2, n_trips)
    T_trip = np.clip(T_trip, -25, 35)

    u = np.clip(np.abs(T_trip - 18) / 14, 0, 1)[:, None]
    p_clim = (1 - u) * np.array([0.45, 0.35, 0.17, 0.03]) + u * np.array([0.03, 0.12, 0.45, 0.40])
    clim_idx = _cat_idx(rng, p_clim)
    p_pre = 0.04 + 0.30 * (T_trip < 5) + 0.10 * ((T_trip >= 5) & (T_trip < 12)) + 0.08 * (T_trip > 26)
    precond = rng.random(n_trips) < p_pre
    garage = rng.random(n_trips) < 0.30
    tow = F["tow_ok"][veh] & (rng.random(n_trips) < 0.15)
    load = 80 + 80 * rng.poisson(0.7, n_trips)  # driver + passengers (kg)
    trailer = np.where(tow, rng.uniform(400, 0.9 * S["tow_kg"], n_trips), 0.0)
    p_charged = rng.random(n_trips) < 0.30
    reserve = rng.uniform(3, 12, n_trips)
    s0_raw = np.where(
        p_charged,
        F["limit"][veh] - rng.uniform(0, 4, n_trips),
        F["limit"][veh] - rng.exponential(26, n_trips),
    )

    # ---- expand to segments
    rep = lambda a: np.repeat(a, n_seg)  # noqa: E731
    trip = rep(np.arange(n_trips))
    N = len(trip)
    v_ = rep(veh)
    d_ = rep(day)
    hr = rep(hour)
    wk = rep(weekday)
    T = np.clip(rep(T_trip) + rng.normal(0, 0.8, N), -25, 35)
    z = F["z"][v_]
    tow_s, pre_s = rep(tow), rep(precond)

    # ---- road type, distance, traffic, road state
    tt = rep(trip_type)
    table = np.array([[0.60, 0.35, 0.05, 0.00], [0.25, 0.30, 0.30, 0.15], [0.03, 0.10, 0.37, 0.50]])
    rt = _cat_idx(rng, table[tt])  # 0 urban,1 suburban,2 rural,3 motorway
    dist = np.clip(rng.gamma(2.0, np.array([2.2, 4.0, 7.5, 15.0])[rt]), 0.5, 60)

    rush = wk & (((hr >= 7) & (hr < 9.2)) | ((hr >= 15.5) & (hr < 18)))
    ph = np.array([0.20, 0.15, 0.05, 0.07])[rt] + rush * np.array([0.25, 0.20, 0.05, 0.15])[rt]
    pm = np.array([0.40, 0.35, 0.25, 0.28])[rt] + rush * 0.15
    tr_idx = _cat_idx(rng, np.column_stack([1 - ph - pm, pm, ph]))
    traffic = np.array(["low", "medium", "high"])[tr_idx]

    precip = daily_precip[d_]
    snow = (precip & (T <= 0.5) & (rng.random(N) < 0.60)) | (~precip & (T < -3) & (rng.random(N) < 0.12))
    wet = ~snow & ((precip & (rng.random(N) < 0.90)) | (~precip & (rng.random(N) < 0.04)))
    road = np.where(snow, "snow_or_ice", np.where(wet, "wet", "dry"))

    # ---- speed (segment average) and driving pattern
    free = np.array([38, 52, 76, 108])[rt]
    f_med = np.array([0.85, 0.85, 0.92, 0.93])[rt]
    f_high = np.array([0.60, 0.65, 0.80, 0.70])[rt]
    f_tr = np.where(tr_idx == 1, f_med, np.where(tr_idx == 2, f_high, 1.0))
    f_road = np.where(snow, 0.82, np.where(wet, 0.95, 1.0))
    speed = free * f_tr * f_road * (1 + 0.06 * z) * np.exp(rng.normal(0, 0.08, N))
    speed = np.where(tow_s, np.minimum(speed * 0.9, 90), speed)
    speed = np.clip(speed, 10, 130)

    P_STOP = np.array([[.15, .40, .75], [.05, .20, .50], [.02, .05, .20], [.00, .02, .10]])
    P_STEADY = np.array([[.25, .10, .00], [.45, .25, .05], [.75, .55, .30], [.95, .80, .50]])
    ps, pst, uu = P_STOP[rt, tr_idx], P_STEADY[rt, tr_idx], rng.random(N)
    pattern = np.where(uu < ps, "stop_start", np.where(uu < ps + pst, "steady", "mixed"))
    pattern = np.where((pattern == "stop_start") & (speed > 60), "mixed", pattern)  # no stop-start at speed

    slope = np.clip(
        np.array([0.9, 0.9, 1.4, 0.7])[rt] * rng.standard_t(5, N) / np.sqrt(5 / 3), -10, 10
    )

    # ---- time within trip, battery temperature
    dur_h = dist / speed
    cum, starts = _trip_cumsum(dur_h, trip)
    elapsed_min = (cum - dur_h / 2) * 60
    start_min = (cum - dur_h) * 60
    warm = 10 * (1 - np.exp(-elapsed_min / 30)) * np.clip(speed / 90, 0.5, 1.3)
    boost = np.where(pre_s, np.where(T < 10, 12, 3), 0) * np.exp(-elapsed_min / 40)
    gar = np.where(rep(garage) & (T < 10), 6, 0) * np.exp(-elapsed_min / 60)
    Tb = np.clip(T + warm + boost + gar + rng.normal(0, 1.2, N), -20, 50)

    # ---- tyres
    p_true = (
        F["p_base"][v_]
        - F["leak"][v_] * ((d_ + F["phase"][v_]) % 120)
        + F["p_base"][v_] * (T - 20) / 293
    )
    p_meas = np.clip(p_true + rng.normal(0, 0.02, N), 1.8, 3.2)

    # ---- physics
    mass = F["mass"][v_] + rep(load) + rep(trailer)
    cda = F["cda"][v_] + np.where(tow_s, S["trailer_cda"], 0.0)
    cv = np.select([pattern == "steady", pattern == "mixed"], [PATTERN["steady"][0], PATTERN["mixed"][0]], PATTERN["stop_start"][0])
    stops_km = np.select([pattern == "steady", pattern == "mixed"], [PATTERN["steady"][1], PATTERN["mixed"][1]], PATTERN["stop_start"][1])
    stops_km = stops_km * np.where(tr_idx == 2, 1.25, 1.0)
    scale = np.array([0.0, 0.35, 0.65, 1.0])[rep(clim_idx)]
    base = _energy_per_100km(
        S, T=T, Tb=Tb, speed=speed, slope=slope, mass=mass, cda=cda, crr0=F["crr"][v_],
        p=p_true, cv=cv, stops_km=stops_km, snow=snow, wet=wet, scale=scale, pre=pre_s,
        elapsed_min=elapsed_min, wind=rng.normal(0, 2.5, N), z=z,
    )
    cons = base * F["eff"][v_] * np.exp(0.04 * z) * np.exp(rng.normal(0, 0.05, N)) + rng.normal(0, 0.6, N)
    cons = np.maximum(cons, 3.0)

    energy = np.maximum(cons * dist / 100 * (1 + rng.normal(0, 0.012, N)), 0.005)
    cap_eff = F["cap"][v_] * (1 - 0.003 * np.maximum(15 - Tb, 0))
    dsoc = energy / cap_eff * 100

    # ---- sequential state of charge within a trip
    total = np.add.reduceat(dsoc, starts)
    s_start = np.maximum(s0_raw, total + reserve)
    ok_trip = s_start <= 100
    cum_soc, _ = _trip_cumsum(dsoc, trip)
    soc_start = rep(s_start) - (cum_soc - dsoc)
    soc_end = soc_start - dsoc
    soc_start = np.clip(soc_start + rng.normal(0, 0.25, N), 0, 100)
    soc_end = np.clip(soc_end + rng.normal(0, 0.25, N), 0, 100)

    ts = YEAR_START + (d_ * 86400 + (hr * 3600).astype(np.int64) + (start_min * 60).astype(np.int64)).astype("timedelta64[s]")

    keep = rep(ok_trip)
    out = {
        "vehicle_id": np.array([f"V{i + 1:03d}" for i in range(V)])[v_],
        "trip_id": np.char.add("T", np.char.zfill(trip.astype(str), 6)),
        "timestamp": ts,
        "distance_km": dist,
        "speed_kmh": speed,
        "traffic_situation": traffic,
        "outside_temperature_c": T,
        "battery_temperature_c": Tb,
        "battery_level_start_pct": soc_start,
        "battery_level_end_pct": soc_end,
        "preconditioning": np.where(pre_s, "on", "off"),
        "climate_settings": np.array(["off", "low", "medium", "high"])[rep(clim_idx)],
        "tyre_pressure_bar": p_meas,
        "road_condition": road,
        "slope_grade_pct": slope,
        "towing": np.where(tow_s, "yes", "no"),
        "current_driving_pattern": pattern,
        "historical_driving_pattern": F["hist"][v_],
        "energy_used_kwh": energy,
        "consumption_kwh_per_100km": cons,
    }
    return {k: a[keep] for k, a in out.items()}


# ------------------------------------------------------------- data defects
def _add_defects(rng, c):
    n = len(c["distance_km"])
    # sensor spikes
    m = rng.random(n) < 0.003
    c["tyre_pressure_bar"] = np.where(m, c["tyre_pressure_bar"] + rng.choice([-1, 1], n) * rng.uniform(0.4, 0.9, n), c["tyre_pressure_bar"])
    m = rng.random(n) < 0.003
    c["slope_grade_pct"] = np.where(m, np.clip(c["slope_grade_pct"] + rng.normal(0, 5, n), -10, 10), c["slope_grade_pct"])
    c["topography"] = np.where(
        c["slope_grade_pct"] > 1.5, "uphill", np.where(c["slope_grade_pct"] < -1.5, "downhill", "mostly_flat")
    )
    # missing values (rates per column; battery temp more often missing when very cold)
    cold = c["outside_temperature_c"] < -10
    rates = {
        "tyre_pressure_bar": 0.03,
        "battery_temperature_c": np.where(cold, 0.045, 0.015),
        "outside_temperature_c": 0.01,
        "slope_grade_pct": 0.02,
        "road_condition": 0.015,
        "preconditioning": 0.01,
        "climate_settings": 0.015,
        "battery_level_end_pct": 0.005,
    }
    for col, r in rates.items():
        mask = rng.random(n) < r
        if col == "slope_grade_pct":
            c["topography"] = c["topography"].astype(object)
            c["topography"][mask] = None
        if c[col].dtype.kind == "f":
            c[col] = np.where(mask, np.nan, c[col])
        else:
            c[col] = c[col].astype(object)
            c[col][mask] = None
    return c


# --------------------------------------------------------------- top level
def generate_data(rows: int, seed: int, vehicles: int = 50, clean: bool = False, variant: str = "dual") -> pl.DataFrame:
    if rows < 1:
        raise ValueError("--rows must be at least 1")
    if vehicles < 1:
        raise ValueError("--vehicles must be at least 1")

    rng = np.random.default_rng(seed)
    S = {**COMMON, **SPECS[variant]}
    W, F = _weather(rng), _fleet(rng, vehicles, S)
    n_dup = 0 if clean else int(0.003 * rows)
    n_unique = rows - n_dup

    factor, c = 1.2, None
    while c is None or len(c["distance_km"]) < n_unique:
        c = _simulate(rng, int(np.ceil(n_unique / 2.4 * factor)) + 5, W, F, S)
        factor *= 1.5
    c = {k: a[:n_unique] for k, a in c.items()}

    order = np.argsort(c["timestamp"], kind="stable")
    c = {k: a[order] for k, a in c.items()}
    c["observation_id"] = np.arange(1, n_unique + 1)

    # derive the ratio from stored (rounded) energy so the identity holds exactly
    c["energy_used_kwh"] = c["energy_used_kwh"].round(3)
    c["consumption_kwh_per_100km"] = c["energy_used_kwh"] / c["distance_km"].round(3) * 100

    if not clean:
        c = _add_defects(rng, c)
    else:
        c["topography"] = np.where(
            c["slope_grade_pct"] > 1.5, "uphill", np.where(c["slope_grade_pct"] < -1.5, "downhill", "mostly_flat")
        )

    idx = np.arange(n_unique)
    if n_dup:
        dup = rng.choice(n_unique, n_dup, replace=False)
        idx = np.concatenate([idx, dup])
        idx = idx[np.argsort(c["observation_id"][idx], kind="stable")]
    c = {k: a[idx] for k, a in c.items()}

    rnd = {
        "distance_km": 3, "speed_kmh": 2, "outside_temperature_c": 2, "battery_temperature_c": 2,
        "battery_level_start_pct": 1, "battery_level_end_pct": 1, "tyre_pressure_bar": 2,
        "slope_grade_pct": 2, "energy_used_kwh": 3, "consumption_kwh_per_100km": 3,
    }
    cols = [
        "observation_id", "vehicle_id", "trip_id", "timestamp", "distance_km", "speed_kmh",
        "traffic_situation", "outside_temperature_c", "battery_temperature_c",
        "battery_level_start_pct", "battery_level_end_pct", "preconditioning",
        "climate_settings", "tyre_pressure_bar", "road_condition", "slope_grade_pct",
        "topography", "towing", "current_driving_pattern", "historical_driving_pattern",
        "energy_used_kwh", "consumption_kwh_per_100km",
    ]
    data = {}
    for k in cols:
        a = c[k]
        if k in rnd:
            a = np.round(a, rnd[k])
            data[k] = pl.Series(k, a).fill_nan(None)
        elif a.dtype == object:
            data[k] = pl.Series(k, a.tolist())
        else:
            data[k] = pl.Series(k, a.astype("datetime64[ms]") if a.dtype.kind == "M" else a)
    return pl.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows (default: 10000)")
    parser.add_argument("--output", type=Path, default=Path("data/battery_data.csv"), help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable data")
    parser.add_argument("--vehicles", type=int, default=50, help="Number of vehicles (default: 50)")
    parser.add_argument("--variant", choices=list(SPECS), default="dual", help="Polestar 4 variant (default: dual)")
    parser.add_argument("--check-wltp", action="store_true", help="Print model vs published WLTP consumption and exit")
    parser.add_argument("--clean", action="store_true", help="Disable missing values, spikes and duplicates")
    args = parser.parse_args()

    if args.check_wltp:
        for k in SPECS:
            print(f"{k}: model {wltp_check(k):.2f} vs published {SPECS[k]['wltp_kwh100']} kWh/100km (battery-side, third-party)")
        return

    df = generate_data(args.rows, args.seed, args.vehicles, args.clean, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.output)

    print(f"Created {df.height:,} synthetic rows: {args.output}")
    print("Reminder: simulated values, not measured vehicle data.")


if __name__ == "__main__":
    main()