# -*- coding: utf-8 -*-
"""
MODULE: route_optimizer.py — BƯỚC 4 (v3+, full)

RouteSegment chứa ĐẦY ĐỦ 22 field (4 base + 6 patch v2 + 12 data_req mới).
Engine có ĐẦY ĐỦ: analyze_route, compare_and_optimize, score_route,
monte_carlo_route, speed_curve, build_route_candidate_from_api.

Dashboard v2 (đang chạy trên máy bạn) sẽ chạy được ngay, không cần sửa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import math
import random
import dataclasses

from vehicle_dynamics import Polestar4Dynamics, air_density_at_temperature
from winter_hvac import WinterHVACSimulation
from api_integration import (
    APIConfig, APIIntegrationError,
    fetch_route_alternatives, fetch_elevation_profile,
    fetch_temperature_profile_along_route, decode_polyline,
    generate_mock_elevation_profile,
)


def _bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lng2 - lng1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# ============================================================================
# RouteSegment — 22 fields
# ============================================================================
@dataclass
class RouteSegment:
    # --- 4 base ---
    distance_m: float
    avg_speed_kmh: float
    grade_percent: float
    ambient_temp_c: float

    # --- 6 patch v2 (data_req: wind, tire, road quality) ---
    wind_speed_mps: float = 0.0
    wind_dir_deg: float = 0.0
    heading_deg: float = 0.0
    tire_pressure_bar: float = 2.8
    tire_type: str = "allseason"
    road_surface: str = "dry"

    # --- 12 v3 (data_req Mục 2 + Mục 15) ---
    acceleration_mps2: float = 0.0
    elevation_m: float = 0.0
    traffic_density: float = 2.0
    hvac_on: bool = True
    climate_target_c: float = 22.0
    regen_availability_pct: float = 100.0
    driving_mode: str = "standard"
    weather: str = "clear"
    towing: bool = False
    preconditioning: bool = False
    historical_speed_avg_kmh: float = 0.0
    tire_condition: str = "used"

    # --- PROPERTIES ---
    @property
    def avg_speed_mps(self) -> float:
        return self.avg_speed_kmh / 3.6

    @property
    def grade_angle_rad(self) -> float:
        return Polestar4Dynamics.grade_percent_to_radians(self.grade_percent)

    @property
    def duration_s(self) -> float:
        return self.distance_m / self.avg_speed_mps if self.avg_speed_mps > 0 else 0.0

    @property
    def wind_along_mps(self) -> float:
        """D1: + = tailwind, − = headwind."""
        if self.wind_speed_mps == 0.0:
            return 0.0
        phi = math.radians(self.heading_deg - self.wind_dir_deg)
        return self.wind_speed_mps * math.cos(phi)

    @property
    def crr_multiplier(self) -> float:
        """D10: pressure × tire_type × tire_condition × road_surface."""
        p = (2.8 / max(self.tire_pressure_bar, 0.5)) ** 0.3
        t = {"summer": 0.95, "allseason": 1.00, "winter": 1.15}.get(self.tire_type, 1.0)
        c = {"new": 1.02, "used": 1.00, "worn": 0.98}.get(self.tire_condition, 1.0)
        s = {"dry": 1.00, "wet": 1.25, "snow": 2.00, "ice": 2.50}.get(self.road_surface, 1.0)
        return p * t * c * s

    @property
    def driving_mode_efficiency_mult(self) -> float:
        return {"eco": 1.03, "standard": 1.00, "performance": 0.97}.get(self.driving_mode, 1.0)

    @property
    def driving_mode_aux_mult(self) -> float:
        return {"eco": 0.80, "standard": 1.00, "performance": 1.20}.get(self.driving_mode, 1.0)

    @property
    def weather_fresh_air_mult(self) -> float:
        return {"clear": 1.00, "rain": 1.20, "snow": 1.30, "fog": 1.40}.get(self.weather, 1.0)

    @property
    def traffic_overhead_factor(self) -> float:
        """Stop-start cycles → extra energy."""
        d = max(self.traffic_density, 0.0)
        return 1.0 + 0.03 * (d ** 1.5)

    @property
    def effective_payload_kg(self) -> float:
        return 500.0 if self.towing else 0.0


@dataclass
class RouteCandidate:
    route_name: str
    segments: List[RouteSegment]
    description: str = ""

    @property
    def total_distance_km(self) -> float:
        return math.fsum(s.distance_m for s in self.segments) / 1000.0

    @property
    def total_duration_s(self) -> float:
        return math.fsum(s.duration_s for s in self.segments)

    @property
    def total_elevation_gain_m(self) -> float:
        return math.fsum(
            s.distance_m * (s.grade_percent / 100.0)
            for s in self.segments if s.grade_percent > 0
        )

    @property
    def max_grade_percent(self) -> float:
        return max((s.grade_percent for s in self.segments), default=0.0)


@dataclass
class RouteAnalysisResult:
    route_name: str
    description: str
    distance_km: float
    duration_min: float
    driving_energy_kwh: float
    winter_overhead_kwh: float
    total_energy_kwh: float
    battery_percent_consumed: float
    effective_usable_capacity_kwh: float
    avg_consumption_kwh_per_km: float
    max_grade_percent: float
    total_elevation_gain_m: float
    segment_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    energy_aero_kwh: float = 0.0
    energy_rolling_kwh: float = 0.0
    energy_grade_kwh: float = 0.0
    energy_hvac_kwh: float = 0.0
    energy_btms_kwh: float = 0.0


# ============================================================================
# ENGINE
# ============================================================================
class RouteOptimizationEngine:
    def __init__(self, vehicle=None, hvac_sim=None,
                 battery_initial_temp_c=None, time_to_precondition_s=900.0):
        self.vehicle = vehicle or Polestar4Dynamics()
        self.hvac_sim = hvac_sim or WinterHVACSimulation()
        self.battery_initial_temp_c = battery_initial_temp_c
        self.time_to_precondition_s = time_to_precondition_s

    # ------------------------------------------------------------------
    def _build_trip_objects(self, seg: RouteSegment):
        """Clone vehicle + hvac theo scenario (mode, towing, weather,
        climate_target, regen_availability). Không sửa object gốc."""
        v = dataclasses.replace(
            self.vehicle,
            payload_kg=self.vehicle.payload_kg + seg.effective_payload_kg,
            drivetrain_efficiency=self.vehicle.drivetrain_efficiency
                                   * seg.driving_mode_efficiency_mult,
            aux_power_baseline_kw=self.vehicle.aux_power_baseline_kw
                                   * seg.driving_mode_aux_mult,
            max_regen_power_kw=self.vehicle.max_regen_power_kw
                                   * max(seg.regen_availability_pct, 0.0) / 100.0,
        )
        h = dataclasses.replace(
            self.hvac_sim,
            cabin_target_temp_c=seg.climate_target_c,
            hvac_fresh_air_flow_m3_per_h=self.hvac_sim.hvac_fresh_air_flow_m3_per_h
                                         * seg.weather_fresh_air_mult,
        )
        return v, h

    # ------------------------------------------------------------------
    def analyze_route(self, route: RouteCandidate) -> RouteAnalysisResult:
        driving_energy_list, winter_overhead_list = [], []
        e_aero = e_roll = e_grade = e_hvac = e_btms = 0.0
        breakdown = []

        for i, seg in enumerate(route.segments):
            vehicle, hvac = self._build_trip_objects(seg)
            rho = air_density_at_temperature(seg.ambient_temp_c)

            # Driving energy
            e_drive = vehicle.compute_segment_energy_kwh(
                distance_m=seg.distance_m,
                avg_speed_mps=seg.avg_speed_mps,
                grade_angle_rad=seg.grade_angle_rad,
                air_density=rho, r_mult=1.0,
                wind_along_mps=seg.wind_along_mps,
                crr_mult=seg.crr_multiplier,
            )
            e_drive *= seg.traffic_overhead_factor

            # Winter overhead
            if seg.hvac_on:
                winter = hvac.compute_total_winter_energy_overhead(
                    ambient_temp_c=seg.ambient_temp_c,
                    trip_duration_s=seg.duration_s,
                    distance_km=seg.distance_m / 1000.0,
                    initial_battery_temp_c=hvac.battery_target_temp_c,
                    time_to_target_s=self.time_to_precondition_s,
                )
                e_winter = winter["total_winter_overhead_kwh"]
                e_hvac  += winter["hvac_energy_kwh"]
                e_btms  += winter["btms_energy_kwh"]
            else:
                e_winter = 0.0

            # Force decomposition
            f = vehicle.compute_total_tractive_force(
                seg.avg_speed_mps, 0.0, seg.grade_angle_rad, rho,
                wind_along_mps=seg.wind_along_mps, crr_mult=seg.crr_multiplier,
            )
            k = seg.distance_m / 3_600_000.0 / vehicle.drivetrain_efficiency
            e_aero  += f["f_aero_N"]  * k
            e_roll  += f["f_roll_N"]  * k
            e_grade += f["f_grade_N"] * k

            driving_energy_list.append(e_drive)
            winter_overhead_list.append(e_winter)
            breakdown.append({
                "segment_index": i,
                "distance_km": round(seg.distance_m / 1000, 3),
                "avg_speed_kmh": seg.avg_speed_kmh,
                "grade_percent": seg.grade_percent,
                "ambient_temp_c": seg.ambient_temp_c,
                "wind_along_mps": round(seg.wind_along_mps, 2),
                "crr_multiplier": round(seg.crr_multiplier, 3),
                "traffic_factor": round(seg.traffic_overhead_factor, 3),
                "driving_energy_kwh": round(e_drive, 4),
                "winter_overhead_kwh": round(e_winter, 4),
                "segment_total_kwh": round(e_drive + e_winter, 4),
            })

        # Warmup cabin + preheat battery — MỘT LẦN, bỏ qua nếu preconditioning
        first = route.segments[0]
        if not first.preconditioning and first.hvac_on:
            winter_overhead_list.append(
                self.hvac_sim.compute_cabin_warmup_energy_kwh(first.ambient_temp_c)
            )
        if not first.preconditioning and self.battery_initial_temp_c is not None:
            winter_overhead_list.append(
                self.hvac_sim.compute_battery_preheat_energy_kwh(self.battery_initial_temp_c)
            )

        e_drive_total = math.fsum(driving_energy_list)
        e_winter_total = math.fsum(winter_overhead_list)
        e_total = e_drive_total + e_winter_total

        ref_temp = (self.battery_initial_temp_c
                    if self.battery_initial_temp_c is not None
                    else first.ambient_temp_c)
        cap_eff = self.hvac_sim.compute_effective_usable_capacity_kwh(
            self.vehicle.battery_usable_capacity_kwh, ref_temp,
        )
        pct = (e_total / cap_eff * 100.0) if cap_eff > 0 else float("inf")
        dist_km = route.total_distance_km
        avg = e_total / dist_km if dist_km > 0 else 0.0

        return RouteAnalysisResult(
            route_name=route.route_name, description=route.description,
            distance_km=round(dist_km, 3),
            duration_min=round(route.total_duration_s / 60.0, 2),
            driving_energy_kwh=round(e_drive_total, 4),
            winter_overhead_kwh=round(e_winter_total, 4),
            total_energy_kwh=round(e_total, 4),
            battery_percent_consumed=round(pct, 3),
            effective_usable_capacity_kwh=round(cap_eff, 3),
            avg_consumption_kwh_per_km=round(avg, 4),
            max_grade_percent=route.max_grade_percent,
            total_elevation_gain_m=round(route.total_elevation_gain_m, 1),
            segment_breakdown=breakdown,
            energy_aero_kwh=round(e_aero, 4),
            energy_rolling_kwh=round(e_roll, 4),
            energy_grade_kwh=round(e_grade, 4),
            energy_hvac_kwh=round(e_hvac, 4),
            energy_btms_kwh=round(e_btms, 4),
        )

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------
    def score_route(
        self,
        result: RouteAnalysisResult,
        soc_start_pct: float = 100.0,
        reserve_pct: float = 10.0,
        w_energy: float = 1.0,
        w_time: float = 0.0,
    ) -> float:
        arrival_soc = soc_start_pct - result.battery_percent_consumed
        if arrival_soc < reserve_pct:
            return float("inf")
        return w_energy * result.total_energy_kwh + w_time * (result.duration_min / 60.0)

    # ------------------------------------------------------------------
    # COMPARE & OPTIMIZE
    # ------------------------------------------------------------------
    def compare_and_optimize(
        self,
        routes: List[RouteCandidate],
        soc_start_pct: float = 100.0,
        reserve_pct: float = 10.0,
        w_energy: float = 1.0,
        w_time: float = 0.0,
    ) -> Dict[str, Any]:
        if not routes:
            raise ValueError("Danh sách routes rỗng, không có gì để tối ưu.")

        results = [self.analyze_route(r) for r in routes]
        scored = [
            (self.score_route(r, soc_start_pct, reserve_pct, w_energy, w_time), r)
            for r in results
        ]
        feasible = [(s, r) for s, r in scored if math.isfinite(s)]
        if feasible:
            feasible.sort(key=lambda x: x[0])
            results_sorted = [r for _, r in feasible]
            needs_charge = False
        else:
            results_sorted = sorted(results, key=lambda r: r.total_energy_kwh)
            needs_charge = True

        optimal = results_sorted[0]
        runner_up = results_sorted[1] if len(results_sorted) > 1 else None
        explanation = self._generate_explanation(optimal, runner_up, results_sorted)

        return {
            "all_results": [self._result_to_dict(r) for r in results],
            "results_ranked": [self._result_to_dict(r) for r in results_sorted],
            "optimal_route": self._result_to_dict(optimal),
            "explanation": explanation,
            "needs_charge": needs_charge,
        }

    @staticmethod
    def _result_to_dict(r: RouteAnalysisResult) -> Dict[str, Any]:
        return {
            "route_name": r.route_name,
            "description": r.description,
            "distance_km": r.distance_km,
            "duration_min": r.duration_min,
            "driving_energy_kwh": r.driving_energy_kwh,
            "winter_overhead_kwh": r.winter_overhead_kwh,
            "total_energy_kwh": r.total_energy_kwh,
            "battery_percent_consumed": r.battery_percent_consumed,
            "effective_usable_capacity_kwh": r.effective_usable_capacity_kwh,
            "avg_consumption_kwh_per_km": r.avg_consumption_kwh_per_km,
            "max_grade_percent": r.max_grade_percent,
            "total_elevation_gain_m": r.total_elevation_gain_m,
            "segment_breakdown": r.segment_breakdown,
            "energy_aero_kwh": r.energy_aero_kwh,
            "energy_rolling_kwh": r.energy_rolling_kwh,
            "energy_grade_kwh": r.energy_grade_kwh,
            "energy_hvac_kwh": r.energy_hvac_kwh,
            "energy_btms_kwh": r.energy_btms_kwh,
        }

    def _generate_explanation(
        self,
        optimal: RouteAnalysisResult,
        runner_up: Optional[RouteAnalysisResult],
        all_ranked: List[RouteAnalysisResult],
    ) -> str:
        lines = [
            f"OPTIMAL ROUTE: '{optimal.route_name}' — "
            f"{optimal.total_energy_kwh:.2f} kWh "
            f"({optimal.battery_percent_consumed:.2f}% battery)."
        ]
        lines.append(
            f"   Distance: {optimal.distance_km:.1f} km | "
            f"Duration: {optimal.duration_min:.0f} min | "
            f"Max grade: {optimal.max_grade_percent:.1f}%"
        )
        if runner_up is None:
            return "\n".join(lines)

        delta = runner_up.total_energy_kwh - optimal.total_energy_kwh
        lines.append("")
        lines.append(
            f"COMPARED TO RUNNER-UP '{runner_up.route_name}' "
            f"({runner_up.total_energy_kwh:.2f} kWh):"
        )
        lines.append(
            f"   Optimal saves {delta:.2f} kWh "
            f"({delta / runner_up.total_energy_kwh * 100:.1f}%)."
        )
        components = {
            "aerodynamic drag":    optimal.energy_aero_kwh    - runner_up.energy_aero_kwh,
            "rolling resistance":  optimal.energy_rolling_kwh - runner_up.energy_rolling_kwh,
            "grade (climbing)":    optimal.energy_grade_kwh   - runner_up.energy_grade_kwh,
            "HVAC (cabin heat)":   optimal.energy_hvac_kwh    - runner_up.energy_hvac_kwh,
            "BTMS (battery heat)": optimal.energy_btms_kwh    - runner_up.energy_btms_kwh,
        }
        adv = {k: v for k, v in components.items() if v < -1e-6}
        pen = {k: v for k, v in components.items() if v >  1e-6}
        if adv:
            k = min(adv, key=adv.get)
            lines.append(f"   → Main advantage: {k} saves {abs(adv[k]):.2f} kWh.")
        if pen:
            k = max(pen, key=pen.get)
            lines.append(f"   → Main penalty: {k} costs {pen[k]:.2f} kWh more.")
        if len(all_ranked) > 2:
            lines.append("")
            lines.append("Ranking (cheapest → most expensive):")
            for i, r in enumerate(all_ranked, start=1):
                lines.append(f"   {i}. {r.route_name}: {r.total_energy_kwh:.2f} kWh")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # MONTE CARLO
    # ------------------------------------------------------------------
    def monte_carlo_route(
        self,
        route: RouteCandidate,
        n_samples: int = 300,
        cv_crr: float = 0.10,
        cv_cd: float = 0.05,
        cv_u: float = 0.15,
        cv_speed: float = 0.05,
        cv_wind: float = 0.20,
        sigma_temp: float = 1.5,
    ) -> Dict[str, float]:
        samples: List[float] = []
        for _ in range(n_samples):
            v = dataclasses.replace(
                self.vehicle,
                rolling_resistance_coeff=self.vehicle.rolling_resistance_coeff
                    * max(random.gauss(1.0, cv_crr), 0.5),
                drag_coefficient_cd=self.vehicle.drag_coefficient_cd
                    * max(random.gauss(1.0, cv_cd), 0.7),
            )
            h = dataclasses.replace(
                self.hvac_sim,
                cabin_u_value_w_m2k=self.hvac_sim.cabin_u_value_w_m2k
                    * max(random.gauss(1.0, cv_u), 0.5),
            )
            noisy_segs = [
                dataclasses.replace(
                    s,
                    avg_speed_kmh=s.avg_speed_kmh * max(random.gauss(1.0, cv_speed), 0.5),
                    ambient_temp_c=s.ambient_temp_c + random.gauss(0.0, sigma_temp),
                    wind_speed_mps=s.wind_speed_mps * max(random.gauss(1.0, cv_wind), 0.0),
                )
                for s in route.segments
            ]
            noisy_route = RouteCandidate(route.route_name, noisy_segs, route.description)
            eng = RouteOptimizationEngine(
                vehicle=v, hvac_sim=h,
                battery_initial_temp_c=self.battery_initial_temp_c,
            )
            samples.append(eng.analyze_route(noisy_route).total_energy_kwh)
        samples.sort()
        n = len(samples)
        return {
            "p10": samples[n // 10], "p50": samples[n // 2],
            "p90": samples[9 * n // 10],
            "mean": math.fsum(samples) / n, "n_samples": n,
        }

    # ------------------------------------------------------------------
    # SPEED CURVE
    # ------------------------------------------------------------------
    def speed_curve(
        self,
        grade_percent: float,
        temp_c: float,
        v_min_kmh: int = 30,
        v_max_kmh: int = 130,
        v_step_kmh: int = 5,
        wind_along_mps: float = 0.0,
        crr_mult: float = 1.0,
    ) -> Dict[str, Any]:
        rho = air_density_at_temperature(temp_c)
        th = Polestar4Dynamics.grade_percent_to_radians(grade_percent)
        p_hvac_w = self.hvac_sim.compute_cabin_hvac_electric_power_kw(temp_c)[
            "total_hvac_electric_power_kw"
        ] * 1000.0
        curve = []
        for v_kmh in range(v_min_kmh, v_max_kmh + 1, v_step_kmh):
            v_mps = v_kmh / 3.6
            p_batt_w = self.vehicle.compute_battery_power_w(
                velocity_mps=v_mps, acceleration_mps2=0.0,
                grade_angle_rad=th, air_density=rho,
                wind_along_mps=wind_along_mps, crr_mult=crr_mult,
            )["p_battery_total_w"]
            kwh_per_km = (p_batt_w + p_hvac_w) / v_mps * 1000.0 / 3_600_000.0
            curve.append((v_kmh, kwh_per_km))
        v_opt, kwh_opt = min(curve, key=lambda x: x[1])
        return {"curve": curve, "v_opt_kmh": v_opt, "kwh_per_km_opt": kwh_opt}


# ============================================================================
# BUILD RouteCandidate TỪ API
# ============================================================================
def build_route_candidate_from_api(
    route_name: str, description: str, origin: tuple, destination: tuple,
    api_config: APIConfig, use_mock_on_failure: bool = True,
    driving_mode: str = "standard", weather: str = "clear",
    tire_pressure_bar: float = 2.8, tire_type: str = "allseason",
    tire_condition: str = "used", road_surface: str = "dry",
    hvac_on: bool = True, climate_target_c: float = 22.0,
    regen_availability_pct: float = 100.0,
    traffic_density: float = 2.0, towing: bool = False,
    preconditioning: bool = False,
) -> RouteCandidate:
    olat, olng = origin
    dlat, dlng = destination
    temp_data = []
    try:
        alts = fetch_route_alternatives(olat, olng, dlat, dlng, api_config)
        primary = alts[0]
        polyline = decode_polyline(primary["encoded_polyline"])
        dist_m = primary["distance_m"]
        dur_s = primary["duration_s"]
        avg_kmh = (dist_m / dur_s * 3.6) if dur_s > 0 else 80.0
        elevation_profile = fetch_elevation_profile(polyline, api_config, samples=100)
        mid = polyline[len(polyline) // 2]
        temp_data = fetch_temperature_profile_along_route(
            [(olat, olng), mid, (dlat, dlng)], api_config,
        )
        avg_temp = math.fsum(t["temperature_c"] for t in temp_data) / len(temp_data)
    except APIIntegrationError as e:
        if not use_mock_on_failure:
            raise
        print(f"[WARN] API failed ({e}). Using MOCK.")
        elevation_profile = generate_mock_elevation_profile(olat, olng, dlat, dlng, 20)
        avg_temp, avg_kmh, temp_data = -8.0, 80.0, []

    from api_integration import _haversine_distance_m

    def near_wind(lat, lng):
        if not temp_data:
            return {"wind_speed_mps": 0.0, "wind_direction_deg": 0.0}
        return min(
            temp_data,
            key=lambda s: (s.get("lat", 0) - lat) ** 2 + (s.get("lng", 0) - lng) ** 2,
        )

    segs = []
    for i in range(len(elevation_profile) - 1):
        p1, p2 = elevation_profile[i], elevation_profile[i + 1]
        d = _haversine_distance_m(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
        if d < 1.0:
            continue
        w = near_wind(p1["lat"], p1["lng"])
        segs.append(RouteSegment(
            distance_m=d, avg_speed_kmh=avg_kmh,
            grade_percent=p1.get("grade_percent", 0.0),
            ambient_temp_c=w.get("temperature_c", avg_temp),
            wind_speed_mps=w.get("wind_speed_mps", 0.0),
            wind_dir_deg=w.get("wind_direction_deg", 0.0),
            heading_deg=_bearing_deg(p1["lat"], p1["lng"], p2["lat"], p2["lng"]),
            tire_pressure_bar=tire_pressure_bar, tire_type=tire_type,
            tire_condition=tire_condition, road_surface=road_surface,
            hvac_on=hvac_on, climate_target_c=climate_target_c,
            regen_availability_pct=regen_availability_pct,
            driving_mode=driving_mode, weather=weather,
            traffic_density=traffic_density, towing=towing,
            preconditioning=preconditioning,
            elevation_m=p1.get("elevation_m", 0.0),
        ))
    return RouteCandidate(route_name, segs, description)


# ============================================================================
# SELF-TEST
# ============================================================================
if __name__ == "__main__":
    print("=== TEST: 2 tuyến, baseline vs wind+wet ===\n")

    route_direct = RouteCandidate(
        route_name="Tuyến 1: Đường đèo",
        description="Ngắn nhưng dốc",
        segments=[
            RouteSegment(distance_m=8000,  avg_speed_kmh=60, grade_percent=8.0,  ambient_temp_c=-10.0),
            RouteSegment(distance_m=12000, avg_speed_kmh=70, grade_percent=5.0,  ambient_temp_c=-10.0),
            RouteSegment(distance_m=10000, avg_speed_kmh=80, grade_percent=-6.0, ambient_temp_c=-9.0),
            RouteSegment(distance_m=5000,  avg_speed_kmh=50, grade_percent=1.0,  ambient_temp_c=-8.5),
        ],
    )
    route_flat = RouteCandidate(
        route_name="Tuyến 2: Đường vòng",
        description="Xa nhưng phẳng",
        segments=[
            RouteSegment(distance_m=15000, avg_speed_kmh=90, grade_percent=0.5,  ambient_temp_c=-9.0),
            RouteSegment(distance_m=18000, avg_speed_kmh=95, grade_percent=-0.5, ambient_temp_c=-9.5),
            RouteSegment(distance_m=9000,  avg_speed_kmh=70, grade_percent=1.0,  ambient_temp_c=-8.0),
        ],
    )

    engine = RouteOptimizationEngine(battery_initial_temp_c=-9.0)

    print("--- BASELINE ---")
    r1 = engine.compare_and_optimize([route_direct, route_flat])
    for r in r1["all_results"]:
        print(f"  {r['route_name']}: {r['total_energy_kwh']:.3f} kWh")

    route_windy = RouteCandidate(
        route_name="Tuyến 1 (wind+wet)",
        description="Gió ngược + đường ướt + lốp non",
        segments=[
            dataclasses.replace(
                s, wind_speed_mps=10.0, wind_dir_deg=180.0, heading_deg=0.0,
                road_surface="wet", tire_pressure_bar=2.4,
            ) for s in route_direct.segments
        ],
    )
    print("\n--- WIND + WET on route 1 ---")
    r2 = engine.compare_and_optimize([route_windy, route_flat])
    for r in r2["all_results"]:
        print(f"  {r['route_name']}: {r['total_energy_kwh']:.3f} kWh")
    print(f"\n  → Impact: +{r2['all_results'][0]['total_energy_kwh'] - r1['all_results'][0]['total_energy_kwh']:.3f} kWh")

    print("\n=== TEST: speed_curve + monte_carlo ===")
    sc = engine.speed_curve(grade_percent=0.0, temp_c=-10.0)
    print(f"  Optimal speed: {sc['v_opt_kmh']} km/h → {sc['kwh_per_km_opt']:.4f} kWh/km")
    mc = engine.monte_carlo_route(route_direct, n_samples=100)
    print(f"  MC: P10={mc['p10']:.2f} P50={mc['p50']:.2f} P90={mc['p90']:.2f}")