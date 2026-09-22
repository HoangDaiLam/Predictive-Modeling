# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: route_optimizer.py
STEP 4 — Route Optimization Algorithm (core AI Engine)
================================================================================
Purpose:
    Combine the 3 built modules:
        - vehicle_dynamics.py  (Polestar4Dynamics)
        - winter_hvac.py       (WinterHVACSimulation)
        - api_integration.py   (route/elevation/weather data)
    to ANALYZE AND COMPARE multiple available routes between points A and B,
    then select the OPTIMAL route in terms of energy consumption (lowest battery use),
    with detailed reasoning for the selection.

Data architecture:
    RouteSegment: a small section of the route (distance, average speed, grade,
                  ambient temperature for that segment)
    RouteCandidate: a complete route = list of many RouteSegment objects
    RouteAnalysisResult: detailed analysis result for one route
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import math

from vehicle_dynamics import Polestar4Dynamics, AIR_DENSITY_SEA_LEVEL, air_density_at_temperature
from winter_hvac import WinterHVACSimulation
from api_integration import (
    APIConfig,
    APIIntegrationError,
    fetch_route_alternatives,
    fetch_elevation_profile,
    fetch_temperature_profile_along_route,
    decode_polyline,
    generate_mock_elevation_profile,
)


# ============================================================================
# ROUTE DATA STRUCTURES
# ============================================================================
@dataclass
class RouteSegment:
    """A small road segment with the parameters needed to calculate energy use."""
    distance_m: float
    avg_speed_kmh: float
    grade_percent: float          # average grade of this segment (%), positive = uphill
    ambient_temp_c: float         # ambient temperature at this segment (°C)

    @property
    def avg_speed_mps(self) -> float:
        return self.avg_speed_kmh / 3.6

    @property
    def grade_angle_rad(self) -> float:
        return Polestar4Dynamics.grade_percent_to_radians(self.grade_percent)

    @property
    def duration_s(self) -> float:
        if self.avg_speed_mps <= 0:
            return 0.0
        return self.distance_m / self.avg_speed_mps


@dataclass
class RouteCandidate:
    """A complete route, used as a candidate for comparison."""
    route_name: str
    segments: List[RouteSegment]
    description: str = ""  # qualitative description (for example: "Direct highway", "Detour avoiding steep grades")

    @property
    def total_distance_km(self) -> float:
        return sum(s.distance_m for s in self.segments) / 1000.0

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.segments)

    @property
    def total_elevation_gain_m(self) -> float:
        """Total elevation gain (only uphill segments count, to describe the route grade profile)."""
        gain = 0.0
        for s in self.segments:
            if s.grade_percent > 0:
                gain += s.distance_m * (s.grade_percent / 100.0)
        return gain

    @property
    def max_grade_percent(self) -> float:
        return max((s.grade_percent for s in self.segments), default=0.0)


@dataclass
class RouteAnalysisResult:
    """Detailed energy analysis result for ONE route."""
    route_name: str
    description: str
    distance_km: float
    duration_min: float
    driving_energy_kwh: float          # energy from vehicle dynamics (Step 1)
    winter_overhead_kwh: float         # energy from HVAC + BTMS (Step 2)
    total_energy_kwh: float            # total energy consumption
    battery_percent_consumed: float    # % battery used (based on effective usable capacity)
    effective_usable_capacity_kwh: float  # usable battery capacity AFTER subtracting cold-related capacity fade
    avg_consumption_kwh_per_km: float
    max_grade_percent: float
    total_elevation_gain_m: float
    segment_breakdown: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# ROUTE ANALYSIS & OPTIMIZATION ENGINE
# ============================================================================
class RouteOptimizationEngine:
    """
    Core AI Engine: receives a list of candidate routes (RouteCandidate),
    calculates the full energy consumption for each route (combining vehicle physics
    and winter effects), and selects the optimal route with an explanation.
    """

    def __init__(
        self,
        vehicle: Optional[Polestar4Dynamics] = None,
        hvac_sim: Optional[WinterHVACSimulation] = None,
        battery_initial_temp_c: Optional[float] = None,
        time_to_precondition_s: float = 900.0,
    ):
        self.vehicle = vehicle or Polestar4Dynamics()
        self.hvac_sim = hvac_sim or WinterHVACSimulation()
        self.battery_initial_temp_c = battery_initial_temp_c
        self.time_to_precondition_s = time_to_precondition_s

    # ------------------------------------------------------------------
    # ANALYZE ONE ROUTE
    # ------------------------------------------------------------------
    def analyze_route(self, route: RouteCandidate) -> RouteAnalysisResult:
        """
        Calculate detailed energy consumption for ONE route by iterating through each
        RouteSegment and:

        1. Use Polestar4Dynamics.compute_segment_energy_kwh() to calculate pure
           vehicle dynamics energy (aero + rolling + grade), with air density (rho)
           adjusted to the actual temperature of that segment (colder air is denser,
           which increases aerodynamic drag).

        2. Use WinterHVACSimulation.compute_total_winter_energy_overhead() to
           calculate the energy loss due to cabin HVAC + battery BTMS during travel
           through that segment.

        3. Sum all segments to determine the total energy of the route.

        At the same time, calculate the EFFECTIVE USABLE BATTERY CAPACITY based on the
        capacity fade factor at the initial battery temperature — because in winter, the
        % battery consumed must be based on the actual usable capacity, not the nominal
        capacity at 25°C.
        """
        driving_energy_kwh_total = 0.0
        winter_overhead_kwh_total = 0.0
        segment_breakdown: List[Dict[str, Any]] = []

        for i, seg in enumerate(route.segments):
            rho = air_density_at_temperature(seg.ambient_temp_c)

            # --- (1) Pure vehicle dynamics energy ---
            driving_energy_kwh = self.vehicle.compute_segment_energy_kwh(
                distance_m=seg.distance_m,
                avg_speed_mps=seg.avg_speed_mps,
                grade_angle_rad=seg.grade_angle_rad,
                air_density=rho,
            )

            # --- (2) Energy loss from HVAC + BTMS on this segment ---
            # Apply BTMS "pull-down" heating only to the FIRST segment of the route
            # (assuming the battery reaches operating temperature afterward);
            # later segments only account for temperature maintenance (holding loss).
            is_first_segment = (i == 0)
            winter_result = self.hvac_sim.compute_total_winter_energy_overhead(
                ambient_temp_c=seg.ambient_temp_c,
                trip_duration_s=seg.duration_s,
                distance_km=seg.distance_m / 1000.0,
                initial_battery_temp_c=(
                    self.battery_initial_temp_c if is_first_segment else self.hvac_sim.battery_target_temp_c
                ),
                time_to_target_s=self.time_to_precondition_s,
            )
            winter_overhead_kwh = winter_result["total_winter_overhead_kwh"]

            driving_energy_kwh_total += driving_energy_kwh
            winter_overhead_kwh_total += winter_overhead_kwh

            segment_breakdown.append({
                "segment_index": i,
                "distance_km": round(seg.distance_m / 1000.0, 3),
                "avg_speed_kmh": seg.avg_speed_kmh,
                "grade_percent": seg.grade_percent,
                "ambient_temp_c": seg.ambient_temp_c,
                "driving_energy_kwh": round(driving_energy_kwh, 4),
                "winter_overhead_kwh": round(winter_overhead_kwh, 4),
                "segment_total_kwh": round(driving_energy_kwh + winter_overhead_kwh, 4),
            })

        total_energy_kwh = driving_energy_kwh_total + winter_overhead_kwh_total

        # --- Effective usable battery capacity based on initial battery temperature ---
        ref_battery_temp = (
            self.battery_initial_temp_c
            if self.battery_initial_temp_c is not None
            else route.segments[0].ambient_temp_c
        )
        effective_capacity_kwh = self.hvac_sim.compute_effective_usable_capacity_kwh(
            nominal_usable_capacity_kwh=self.vehicle.battery_usable_capacity_kwh,
            battery_temp_c=ref_battery_temp,
        )

        battery_percent_consumed = (
            (total_energy_kwh / effective_capacity_kwh) * 100.0 if effective_capacity_kwh > 0 else float("inf")
        )

        distance_km = route.total_distance_km
        avg_consumption = total_energy_kwh / distance_km if distance_km > 0 else 0.0

        return RouteAnalysisResult(
            route_name=route.route_name,
            description=route.description,
            distance_km=round(distance_km, 3),
            duration_min=round(route.total_duration_s / 60.0, 2),
            driving_energy_kwh=round(driving_energy_kwh_total, 4),
            winter_overhead_kwh=round(winter_overhead_kwh_total, 4),
            total_energy_kwh=round(total_energy_kwh, 4),
            battery_percent_consumed=round(battery_percent_consumed, 3),
            effective_usable_capacity_kwh=round(effective_capacity_kwh, 3),
            avg_consumption_kwh_per_km=round(avg_consumption, 4),
            max_grade_percent=route.max_grade_percent,
            total_elevation_gain_m=round(route.total_elevation_gain_m, 1),
            segment_breakdown=segment_breakdown,
        )

    # ------------------------------------------------------------------
    # COMPARE MULTIPLE ROUTES & CHOOSE THE OPTIMAL ROUTE
    # ------------------------------------------------------------------
    def compare_and_optimize(self, routes: List[RouteCandidate]) -> Dict[str, Any]:
        """
        Analyze the full list of candidate routes, sort them by total energy use in
        ascending order, and return:
            - all_results: detailed analysis for ALL routes
            - optimal_route: the optimal route (lowest battery usage)
            - explanation: a detailed natural-language explanation of why this route is better
              than the others.
        """
        if not routes:
            raise ValueError("Route list is empty; there is nothing to optimize.")

        results = [self.analyze_route(r) for r in routes]
        results_sorted = sorted(results, key=lambda r: r.total_energy_kwh)

        optimal = results_sorted[0]
        runner_up = results_sorted[1] if len(results_sorted) > 1 else None

        explanation = self._generate_explanation(optimal, runner_up, results_sorted)

        return {
            "all_results": [self._result_to_dict(r) for r in results],
            "results_ranked": [self._result_to_dict(r) for r in results_sorted],
            "optimal_route": self._result_to_dict(optimal),
            "explanation": explanation,
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
        }

    def _generate_explanation(
        self,
        optimal: RouteAnalysisResult,
        runner_up: Optional[RouteAnalysisResult],
        all_ranked: List[RouteAnalysisResult],
    ) -> str:
        """
        Generate a natural-language explanation for the optimal decision, based on QUANTITATIVE
        analysis of the differences between the optimal route and the nearest competitor
        (runner-up) — for example, a longer route but with fewer steep grades, which still
        results in lower total energy.
        """
        lines = []
        lines.append(
            f"✅ OPTIMAL ROUTE: '{optimal.route_name}' — "
            f"consumes {optimal.total_energy_kwh:.2f} kWh "
            f"({optimal.battery_percent_consumed:.2f}% of effective usable battery capacity "
            f"{optimal.effective_usable_capacity_kwh:.1f} kWh)."
        )
        lines.append(
            f"   - Distance: {optimal.distance_km:.1f} km | Time: {optimal.duration_min:.0f} minutes"
        )
        lines.append(
            f"   - Driving energy: {optimal.driving_energy_kwh:.2f} kWh "
            f"| Winter cabin & battery heating overhead: {optimal.winter_overhead_kwh:.2f} kWh"
        )
        lines.append(
            f"   - Maximum grade: {optimal.max_grade_percent:.1f}% | "
            f"Total elevation gain: {optimal.total_elevation_gain_m:.0f} m"
        )

        if runner_up is not None:
            delta_energy = runner_up.total_energy_kwh - optimal.total_energy_kwh
            delta_distance = optimal.distance_km - runner_up.distance_km
            delta_elevation = runner_up.total_elevation_gain_m - optimal.total_elevation_gain_m

            lines.append("")
            lines.append(
                f"📊 COMPARISON vs. nearest route '{runner_up.route_name}' "
                f"({runner_up.total_energy_kwh:.2f} kWh):"
            )
            lines.append(
                f"   - The optimal route saves {abs(delta_energy):.2f} kWh "
                f"(~{abs(delta_energy)/runner_up.total_energy_kwh*100:.1f}% less)."
            )

            if delta_distance > 0:
                # The optimal route is LONGER but still more efficient -> classic explanation
                lines.append(
                    f"   - Although it is {delta_distance:.1f} km longer than the other route, the optimal route still "
                    f"uses less battery because it has fewer steep sections (total climb is {delta_elevation:.0f} m less). "
                    f"Grade force (F_grade = m·g·sin(θ)) increases almost linearly with slope and vehicle mass, while "
                    f"rolling and aerodynamic resistance over a flat road only increase relatively little — therefore, "
                    f"avoiding steep climbs often provides more energy savings than taking the shortest route."
                )
            elif delta_distance < 0:
                lines.append(
                    f"   - The optimal route is {abs(delta_distance):.1f} km shorter AND has less grade "
                    f"({delta_elevation:.0f} m less climb) — it wins on both criteria and is clearly the best choice."
                )
            else:
                lines.append(
                    "   - The routes are similar in distance; the main difference comes from the slope structure "
                    "and/or temperature differences along the route (which affect HVAC & BTMS power demand)."
                )

        if len(all_ranked) > 2:
            lines.append("")
            lines.append("📋 Full route ranking (from most efficient to least efficient):")
            for idx, r in enumerate(all_ranked, start=1):
                lines.append(f"   {idx}. {r.route_name}: {r.total_energy_kwh:.2f} kWh")

        return "\n".join(lines)


# ============================================================================
# UTILITY: BUILD A RouteCandidate FROM REAL API DATA
# ============================================================================
def build_route_candidate_from_api(
    route_name: str,
    description: str,
    origin: tuple,
    destination: tuple,
    api_config: APIConfig,
    default_avg_speed_kmh: float = 80.0,
    use_mock_on_failure: bool = True,
) -> RouteCandidate:
    """
    Build a complete RouteCandidate by calling the real APIs (Elevation + Weather).
    If the API fails (missing key, network loss, etc.) and use_mock_on_failure=True,
    it will automatically fall back to simulated data (generate_mock_elevation_profile)
    with a warning — ensuring the system still works for demo/testing even without API keys.
    """
    origin_lat, origin_lng = origin
    dest_lat, dest_lng = destination

    try:
        path_points = [(origin_lat, origin_lng), (dest_lat, dest_lng)]
        elevation_profile = fetch_elevation_profile(path_points, api_config, samples=20)
        temp_data = fetch_temperature_profile_along_route(
            [(origin_lat, origin_lng), (dest_lat, dest_lng)], api_config
        )
        avg_temp_c = sum(t["temperature_c"] for t in temp_data) / len(temp_data)
    except APIIntegrationError as e:
        if not use_mock_on_failure:
            raise
        print(f"[WARN] API call failed ({e}). Falling back to mock data for demo purposes.")
        elevation_profile = generate_mock_elevation_profile(origin_lat, origin_lng, dest_lat, dest_lng, num_points=20)
        avg_temp_c = -8.0  # default winter temperature assumption when no real data is available

    segments: List[RouteSegment] = []
    for i in range(len(elevation_profile) - 1):
        p1 = elevation_profile[i]
        p2 = elevation_profile[i + 1]

        from api_integration import _haversine_distance_m
        dist_m = _haversine_distance_m(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
        if dist_m < 1.0:
            continue

        segments.append(RouteSegment(
            distance_m=dist_m,
            avg_speed_kmh=default_avg_speed_kmh,
            grade_percent=p1.get("grade_percent", 0.0),
            ambient_temp_c=avg_temp_c,
        ))

    return RouteCandidate(route_name=route_name, description=description, segments=segments)


# ============================================================================
# SELF-TEST BLOCK — EXAMPLE COMPARISON OF 2 ROUTES (manual data)
# ============================================================================
if __name__ == "__main__":
    print("=== TEST: Compare 2 winter routes, -10°C ===\n")

    # Route 1: direct mountain road, shorter but steeper
    route_direct = RouteCandidate(
        route_name="Route 1: Direct mountain road",
        description="Shorter but with grades up to 8%",
        segments=[
            RouteSegment(distance_m=8000, avg_speed_kmh=60, grade_percent=8.0, ambient_temp_c=-10.0),
            RouteSegment(distance_m=12000, avg_speed_kmh=70, grade_percent=5.0, ambient_temp_c=-10.0),
            RouteSegment(distance_m=10000, avg_speed_kmh=80, grade_percent=-6.0, ambient_temp_c=-9.0),
            RouteSegment(distance_m=5000, avg_speed_kmh=50, grade_percent=1.0, ambient_temp_c=-8.5),
        ],
    )

    # Route 2: longer but flatter road through town
    route_flat = RouteCandidate(
        route_name="Route 2: Flat detour",
        description="Longer but almost flat",
        segments=[
            RouteSegment(distance_m=15000, avg_speed_kmh=90, grade_percent=0.5, ambient_temp_c=-9.0),
            RouteSegment(distance_m=18000, avg_speed_kmh=95, grade_percent=-0.5, ambient_temp_c=-9.5),
            RouteSegment(distance_m=9000, avg_speed_kmh=70, grade_percent=1.0, ambient_temp_c=-8.0),
        ],
    )

    engine = RouteOptimizationEngine(battery_initial_temp_c=-9.0)
    result = engine.compare_and_optimize([route_direct, route_flat])

    print(result["explanation"])

    print("\n\n=== CHI TIẾT TỪNG TUYẾN (JSON-like) ===")
    for r in result["all_results"]:
        print(f"\n{r['route_name']}:")
        print(f"  Tổng năng lượng: {r['total_energy_kwh']} kWh | % pin: {r['battery_percent_consumed']}%")
        print(f"  Driving: {r['driving_energy_kwh']} kWh | Winter overhead: {r['winter_overhead_kwh']} kWh")