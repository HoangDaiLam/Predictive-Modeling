# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: route_optimizer.py
BƯỚC 4 — Route Optimization Algorithm (AI Engine trung tâm)
================================================================================
Mục đích:
    Kết hợp 3 module đã xây dựng:
        - vehicle_dynamics.py  (Polestar4Dynamics)
        - winter_hvac.py       (WinterHVACSimulation)
        - api_integration.py   (dữ liệu route/elevation/weather)
    để PHÂN TÍCH VÀ SO SÁNH nhiều tuyến đường khả dụng giữa điểm A và B,
    từ đó chọn ra tuyến đường TỐI ƯU về năng lượng (tiêu thụ pin thấp nhất),
    kèm giải thích chi tiết lý do lựa chọn.

Kiến trúc dữ liệu:
    RouteSegment: 1 đoạn nhỏ của tuyến đường (khoảng cách, tốc độ TB, độ dốc,
                  nhiệt độ môi trường tại đoạn đó)
    RouteCandidate: 1 tuyến đường hoàn chỉnh = danh sách nhiều RouteSegment
    RouteAnalysisResult: kết quả phân tích chi tiết cho 1 tuyến đường
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
# CẤU TRÚC DỮ LIỆU TUYẾN ĐƯỜNG
# ============================================================================
@dataclass
class RouteSegment:
    """Một đoạn đường nhỏ với các thông số cần thiết để tính năng lượng."""
    distance_m: float
    avg_speed_kmh: float
    grade_percent: float          # độ dốc trung bình đoạn này (%), dương = lên dốc
    ambient_temp_c: float         # nhiệt độ môi trường tại đoạn này (°C)

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
    """Một tuyến đường hoàn chỉnh, ứng viên để so sánh."""
    route_name: str
    segments: List[RouteSegment]
    description: str = ""  # mô tả định tính (ví dụ: "Cao tốc trực tiếp", "Đường vòng tránh đèo")

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
    """Kết quả phân tích năng lượng chi tiết cho MỘT tuyến đường."""
    route_name: str
    description: str
    distance_km: float
    duration_min: float
    driving_energy_kwh: float          # năng lượng do động lực học xe (Bước 1)
    winter_overhead_kwh: float         # năng lượng do HVAC + BTMS (Bước 2)
    total_energy_kwh: float            # tổng năng lượng tiêu thụ
    battery_percent_consumed: float    # % pin tiêu thụ (trên dung lượng khả dụng hiệu dụng)
    effective_usable_capacity_kwh: float  # dung lượng pin khả dụng SAU khi trừ capacity fade do lạnh
    avg_consumption_kwh_per_km: float
    max_grade_percent: float
    total_elevation_gain_m: float
    segment_breakdown: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# ENGINE PHÂN TÍCH & TỐI ƯU TUYẾN ĐƯỜNG
# ============================================================================
class RouteOptimizationEngine:
    """
    AI Engine trung tâm: nhận vào danh sách các tuyến đường ứng viên
    (RouteCandidate), tính toán năng lượng tiêu thụ đầy đủ cho từng tuyến
    (kết hợp vật lý xe + ảnh hưởng mùa đông), và đưa ra quyết định tuyến
    đường tối ưu kèm giải thích.
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
    # PHÂN TÍCH MỘT TUYẾN ĐƯỜNG DUY NHẤT
    # ------------------------------------------------------------------
    def analyze_route(self, route: RouteCandidate) -> RouteAnalysisResult:
        """
        Tính toán chi tiết năng lượng tiêu thụ cho MỘT tuyến đường, bằng
        cách lặp qua từng RouteSegment và:

        1. Dùng Polestar4Dynamics.compute_segment_energy_kwh() để tính năng
           lượng động lực học thuần tuý (aero + rolling + grade), với khối
           lượng riêng không khí (rho) được hiệu chỉnh theo nhiệt độ thực tế
           của đoạn đường đó (khí lạnh đặc hơn -> aero drag cao hơn).

        2. Dùng WinterHVACSimulation.compute_total_winter_energy_overhead()
           để tính năng lượng hao hụt do HVAC cabin + BTMS pin trong thời
           gian di chuyển qua đoạn đó.

        3. Cộng dồn (sum) toàn bộ các đoạn để ra tổng năng lượng tuyến đường.

        Đồng thời tính TOÁN DUNG LƯỢNG PIN KHẢ DỤNG HIỆU DỤNG (effective
        usable capacity) dựa trên capacity fade factor tại nhiệt độ pin ban
        đầu — vì trong mùa đông, % pin tiêu thụ phải tính trên dung lượng
        THỰC TẾ khai thác được, không phải dung lượng danh định lúc 25°C.
        """
        driving_energy_list: List[float] = []
        winter_overhead_list: List[float] = []
        segment_breakdown: List[Dict[str, Any]] = []

        for i, seg in enumerate(route.segments):
            rho = air_density_at_temperature(seg.ambient_temp_c)

            # --- (1) Năng lượng động lực học thuần tuý ---
            driving_energy_kwh = self.vehicle.compute_segment_energy_kwh(
                distance_m=seg.distance_m,
                avg_speed_mps=seg.avg_speed_mps,
                grade_angle_rad=seg.grade_angle_rad,
                air_density=rho,
            )

            # --- (2) Năng lượng hao hụt do HVAC + BTMS trên đoạn này ---
            # Chỉ áp dụng BTMS "pull-down" (làm nóng pin từ lạnh) cho ĐOẠN ĐẦU
            # TIÊN của tuyến (giả định pin đã đạt nhiệt độ vận hành sau đó);
            # các đoạn sau chỉ tính chi phí "duy trì" nhiệt độ (holding loss).
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

            driving_energy_list.append(driving_energy_kwh)
            winter_overhead_list.append(winter_overhead_kwh)

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

        driving_energy_kwh_total = math.fsum(driving_energy_list)
        winter_overhead_kwh_total = math.fsum(winter_overhead_list)
        total_energy_kwh = math.fsum([
            driving_energy_kwh_total, winter_overhead_kwh_total
        ])

        # --- Dung lượng pin khả dụng hiệu dụng theo nhiệt độ pin ban đầu ---
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
    # SO SÁNH NHIỀU TUYẾN & CHỌN TUYẾN TỐI ƯU
    # ------------------------------------------------------------------
    def compare_and_optimize(self, routes: List[RouteCandidate]) -> Dict[str, Any]:
        """
        Phân tích toàn bộ danh sách tuyến đường ứng viên, sắp xếp theo tổng
        năng lượng tiêu thụ tăng dần, và trả về:
            - all_results: kết quả phân tích chi tiết của TẤT CẢ các tuyến
            - optimal_route: tuyến đường tối ưu (tiêu thụ pin ít nhất)
            - explanation: giải thích chi tiết bằng ngôn ngữ tự nhiên (tiếng Việt)
              lý do tuyến này tối ưu hơn các tuyến còn lại.
        """
        if not routes:
            raise ValueError("Danh sách routes rỗng, không có gì để tối ưu.")

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
        Sinh giải thích ngôn ngữ tự nhiên (tiếng Việt) cho quyết định tối ưu,
        dựa trên PHÂN TÍCH ĐỊNH LƯỢNG các yếu tố khác biệt giữa tuyến tối ưu
        và tuyến gần nhất (runner-up) — ví dụ: quãng đường dài hơn nhưng ít
        dốc hơn nên tổng năng lượng vẫn thấp hơn.
        """
        lines = []
        lines.append(
            f"✅ Optimal Route Name/TUYẾN ĐƯỜNG TỐI ƯU: '{optimal.route_name}' — "
            f"Optimal Total Energy/tiêu thụ {optimal.total_energy_kwh:.2f} kWh "
            f"({optimal.battery_percent_consumed:.2f}% Battery Percent Consumed) "
            f"Effective Usable Capacity: {optimal.effective_usable_capacity_kwh:.1f} kWh)."
        )
        lines.append(
            f"   - Quãng đường: {optimal.distance_km:.1f} km | Thời gian: {optimal.duration_min:.0f} phút"
        )
        lines.append(
            f"   - Năng lượng động lực học (di chuyển): {optimal.driving_energy_kwh:.2f} kWh "
            f"| Hao hụt do sưởi cabin & pin mùa đông: {optimal.winter_overhead_kwh:.2f} kWh"
        )
        lines.append(
            f"   - Độ dốc lớn nhất trên tuyến: {optimal.max_grade_percent:.1f}% | "
            f"Tổng độ cao phải leo: {optimal.total_elevation_gain_m:.0f} m"
        )

        if runner_up is not None:
            delta_energy = runner_up.total_energy_kwh - optimal.total_energy_kwh
            delta_distance = optimal.distance_km - runner_up.distance_km
            delta_elevation = runner_up.total_elevation_gain_m - optimal.total_elevation_gain_m

            lines.append("")
            lines.append(
                f"📊 SO SÁNH với tuyến gần nhất '{runner_up.route_name}' "
                f"({runner_up.total_energy_kwh:.2f} kWh):"
            )
            lines.append(
                f"   - Tuyến tối ưu tiết kiệm hơn {abs(delta_energy):.2f} kWh "
                f"(~{abs(delta_energy)/runner_up.total_energy_kwh*100:.1f}% ít hơn)."
            )

            if delta_distance > 0:
                # Tuyến tối ưu DÀI HƠN nhưng vẫn tiết kiệm hơn -> giải thích lý do kinh điển
                lines.append(
                    f"   - Dù DÀI HƠN {delta_distance:.1f} km so với tuyến kia, tuyến tối ưu vẫn "
                    f"tiết kiệm pin hơn vì có ít đoạn dốc hơn (leo tổng cộng ít hơn "
                    f"{delta_elevation:.0f} m). Lực leo dốc (F_grade = m·g·sin(θ)) tăng gần như "
                    f"TUYẾN TÍNH theo độ dốc và khối lượng xe, trong khi lực cản lăn/khí động học "
                    f"trên quãng đường bằng phẳng tăng thêm chỉ tương đối nhỏ — vì vậy tránh dốc "
                    f"cao thường mang lại lợi ích năng lượng lớn hơn việc đi đường ngắn nhất."
                )
            elif delta_distance < 0:
                lines.append(
                    f"   - Tuyến tối ưu NGẮN HƠN {abs(delta_distance):.1f} km VÀ ít dốc hơn "
                    f"({delta_elevation:.0f} m leo ít hơn) — chiến thắng trên cả hai tiêu chí, "
                    f"nên là lựa chọn rõ ràng."
                )
            else:
                lines.append(
                    "   - Quãng đường tương đương nhau, chênh lệch năng lượng chủ yếu đến từ "
                    "cấu trúc độ dốc và/hoặc chênh lệch nhiệt độ dọc tuyến (ảnh hưởng đến "
                    "công suất HVAC & BTMS)."
                )

        if len(all_ranked) > 2:
            lines.append("")
            lines.append("📋 Toàn bộ xếp hạng các tuyến (từ tiết kiệm nhất đến tốn nhất):")
            for idx, r in enumerate(all_ranked, start=1):
                lines.append(f"   {idx}. {r.route_name}: {r.total_energy_kwh:.2f} kWh")

        return "\n".join(lines)


# ============================================================================
# HÀM TIỆN ÍCH: XÂY DỰNG RouteCandidate TỪ DỮ LIỆU API THẬT
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
    Xây dựng một RouteCandidate hoàn chỉnh bằng cách gọi các API thật
    (Elevation + Weather). Nếu API lỗi (thiếu key, mất mạng...) và
    use_mock_on_failure=True, sẽ tự động fallback sang dữ liệu mô phỏng
    (generate_mock_elevation_profile) kèm cảnh báo — đảm bảo hệ thống
    vẫn hoạt động được để demo/test ngay cả khi chưa cấu hình API key.
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
        print(f"[WARN] Gọi API thất bại ({e}). Chuyển sang dữ liệu MÔ PHỎNG (mock) để demo.")
        elevation_profile = generate_mock_elevation_profile(origin_lat, origin_lng, dest_lat, dest_lng, num_points=20)
        avg_temp_c = -8.0  # nhiệt độ mùa đông giả định mặc định khi không có dữ liệu thật

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
# KHỐI TỰ KIỂM TRA — VÍ DỤ SO SÁNH 2 TUYẾN ĐƯỜNG (dữ liệu thủ công)
# ============================================================================
if __name__ == "__main__":
    print("=== TEST: So sánh 2 tuyến đường mùa đông, -10°C ===\n")

    # Tuyến 1: Đường trực tiếp qua đèo, ngắn nhưng dốc đứng
    route_direct = RouteCandidate(
        route_name="Tuyến 1: Đường đèo trực tiếp",
        description="Ngắn hơn nhưng có đoạn dốc lên tới 8%",
        segments=[
            RouteSegment(distance_m=8000, avg_speed_kmh=60, grade_percent=8.0, ambient_temp_c=-10.0),
            RouteSegment(distance_m=12000, avg_speed_kmh=70, grade_percent=5.0, ambient_temp_c=-10.0),
            RouteSegment(distance_m=10000, avg_speed_kmh=80, grade_percent=-6.0, ambient_temp_c=-9.0),
            RouteSegment(distance_m=5000, avg_speed_kmh=50, grade_percent=1.0, ambient_temp_c=-8.5),
        ],
    )

    # Tuyến 2: Đường vòng xa hơn nhưng phẳng, qua thị trấn
    route_flat = RouteCandidate(
        route_name="Tuyến 2: Đường vòng bằng phẳng",
        description="Xa hơn nhưng gần như không dốc",
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