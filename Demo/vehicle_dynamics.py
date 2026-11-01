# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: vehicle_dynamics.py
BƯỚC 1 — Vehicle Dynamics & Energy Physics Engine
================================================================================
Mục đích:
    Mô phỏng chính xác động lực học và tiêu thụ năng lượng của xe điện
    Polestar 4 Single Motor (RWD), chế độ lái Standard Mode.

PATCH v2:
    - compute_aero_drag_force() thêm wind_along_mps (headwind/tailwind)
    - compute_rolling_resistance_force() thêm crr_mult (tire/road correction)
    - Propagate 2 param này qua compute_total_tractive_force, compute_wheel_power_w,
      compute_battery_power_w, compute_segment_energy_kwh.
    - Default wind=0, crr_mult=1.0 → hành vi y hệt code cũ.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ------------------------------------------------------------------------
# HẰNG SỐ VẬT LÝ TOÀN CỤC
# ------------------------------------------------------------------------
GRAVITY: float = 9.80665
AIR_DENSITY_SEA_LEVEL: float = 1.225


def air_density_at_temperature(temp_celsius: float, pressure_pa: float = 101325.0) -> float:
    """rho = P / (R_specific * T) — không khí lạnh đặc hơn."""
    R_SPECIFIC_AIR = 287.058
    return pressure_pa / (R_SPECIFIC_AIR * (temp_celsius + 273.15))


def trapezoidal_integral(times: List[float], values: List[float]) -> float:
    """Tích phân hình thang O(h²)."""
    if len(times) < 2:
        return 0.0
    return math.fsum(
        (times[i + 1] - times[i]) * (values[i] + values[i + 1]) * 0.5
        for i in range(len(times) - 1)
    )


@dataclass
class Polestar4Dynamics:
    """Polestar 4 Single Motor (Long Range Single Motor, RWD)."""

    curb_mass_kg: float = 2350.0
    payload_kg: float = 150.0
    frontal_area_m2: float = 2.75
    drag_coefficient_cd: float = 0.27
    wheel_radius_m: float = 0.36
    rolling_resistance_coeff: float = 0.009

    motor_peak_power_kw: float = 200.0
    drivetrain_efficiency: float = 0.91
    max_regen_power_kw: float = 150.0

    battery_gross_capacity_kwh: float = 94.0
    battery_usable_capacity_kwh: float = 92.0
    battery_nominal_voltage_v: float = 400.0
    battery_internal_resistance_ohm: float = 0.05

    aux_power_baseline_kw: float = 0.45
    drag_coefficient_regen_efficiency: float = 0.70

    def __post_init__(self):
        self.total_mass_kg: float = self.curb_mass_kg + self.payload_kg

    # ====================================================================
    # 1. LỰC CẢN KHÔNG KHÍ — PATCH: + wind_along_mps
    # ====================================================================
    def compute_aero_drag_force(
        self,
        velocity_mps: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
        wind_along_mps: float = 0.0,
    ) -> float:
        """
        F_aero = 0.5 * rho * Cd * A * v_rel * |v_rel|

        v_rel = velocity_mps − wind_along_mps
            wind_along_mps > 0  → tailwind (giảm cản, tiết kiệm pin)
            wind_along_mps < 0  → headwind (tăng cản, tốn pin)
        Dùng v_rel*|v_rel| (không phải v_rel²) để giữ dấu — an toàn khi xe
        chạy chậm hơn gió (v_rel âm → lực đẩy xe về phía trước).

        Default wind_along_mps=0.0 → v_rel = v, kết quả y hệt code cũ.
        """
        v_rel = velocity_mps - wind_along_mps
        f_aero = 0.5 * air_density * self.drag_coefficient_cd * self.frontal_area_m2 * v_rel * abs(v_rel)
        return f_aero

    # ====================================================================
    # 2. LỰC MA SÁT LĂN — PATCH: + crr_mult
    # ====================================================================
    def compute_rolling_resistance_force(
        self,
        grade_angle_rad: float = 0.0,
        crr_mult: float = 1.0,
    ) -> float:
        """
        F_roll = Crr * crr_mult * m * g * cos(theta)

        crr_mult: hệ số hiệu chỉnh từ tire pressure × tire type × road surface.
        Default 1.0 → hành vi cũ.
        """
        f_roll = (
            self.rolling_resistance_coeff
            * crr_mult
            * self.total_mass_kg
            * GRAVITY
            * math.cos(grade_angle_rad)
        )
        return f_roll

    # ====================================================================
    # 3. LỰC LEO DỐC
    # ====================================================================
    def compute_grade_force(self, grade_angle_rad: float) -> float:
        return self.total_mass_kg * GRAVITY * math.sin(grade_angle_rad)

    @staticmethod
    def grade_percent_to_radians(grade_percent: float) -> float:
        return math.atan(grade_percent / 100.0)

    # ====================================================================
    # 4. LỰC GIA TỐC
    # ====================================================================
    def compute_acceleration_force(self, acceleration_mps2: float) -> float:
        return self.total_mass_kg * acceleration_mps2

    def compute_kinetic_energy_delta(self, v_start_mps: float, v_end_mps: float) -> float:
        return 0.5 * self.total_mass_kg * (v_end_mps ** 2 - v_start_mps ** 2)

    # ====================================================================
    # 5. TỔNG HỢP LỰC — PATCH: + wind_along_mps, + crr_mult
    # ====================================================================
    def compute_total_tractive_force(
        self,
        velocity_mps: float,
        acceleration_mps2: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
        wind_along_mps: float = 0.0,
        crr_mult: float = 1.0,
    ) -> Dict[str, float]:
        f_aero  = self.compute_aero_drag_force(velocity_mps, air_density, wind_along_mps)
        f_roll  = self.compute_rolling_resistance_force(grade_angle_rad, crr_mult)
        f_grade = self.compute_grade_force(grade_angle_rad)
        f_accel = self.compute_acceleration_force(acceleration_mps2)
        f_total = f_aero + f_roll + f_grade + f_accel
        return {
            "f_aero_N": f_aero,
            "f_roll_N": f_roll,
            "f_grade_N": f_grade,
            "f_accel_N": f_accel,
            "f_total_N": f_total,
        }

    # ====================================================================
    # 6. CÔNG SUẤT BÁNH XE — PATCH: + wind_along_mps, + crr_mult
    # ====================================================================
    def compute_wheel_power_w(
        self,
        velocity_mps: float,
        acceleration_mps2: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
        wind_along_mps: float = 0.0,
        crr_mult: float = 1.0,
    ) -> float:
        forces = self.compute_total_tractive_force(
            velocity_mps, acceleration_mps2, grade_angle_rad, air_density,
            wind_along_mps=wind_along_mps, crr_mult=crr_mult,
        )
        return forces["f_total_N"] * velocity_mps

    # ====================================================================
    # 7. CÔNG SUẤT PIN — PATCH: + wind_along_mps, + crr_mult
    # ====================================================================
    def compute_battery_power_w(
        self,
        velocity_mps: float,
        acceleration_mps2: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
        include_aux_load: bool = True,
        r_mult: float = 1.0,
        wind_along_mps: float = 0.0,
        crr_mult: float = 1.0,
    ) -> Dict[str, float]:
        p_wheel = self.compute_wheel_power_w(
            velocity_mps, acceleration_mps2, grade_angle_rad, air_density,
            wind_along_mps=wind_along_mps, crr_mult=crr_mult,
        )
        p_aux_w = self.aux_power_baseline_kw * 1000.0 if include_aux_load else 0.0

        if p_wheel >= 0:
            p_battery_traction = p_wheel / self.drivetrain_efficiency
            i_pack = p_battery_traction / self.battery_nominal_voltage_v
            p_battery_traction += i_pack * i_pack * self.battery_internal_resistance_ohm * r_mult
            mode = "DRIVING"
        else:
            max_regen_w = -abs(self.max_regen_power_kw) * 1000.0 / max(r_mult, 1.0)
            p_wheel_clamped = max(p_wheel, max_regen_w)
            p_battery_traction = p_wheel_clamped * self.drag_coefficient_regen_efficiency
            i_pack = abs(p_battery_traction) / self.battery_nominal_voltage_v
            p_battery_traction -= i_pack * i_pack * self.battery_internal_resistance_ohm * r_mult
            mode = "REGEN_BRAKING"

        p_battery_total = p_battery_traction + p_aux_w
        return {
            "p_wheel_w": p_wheel,
            "p_battery_traction_w": p_battery_traction,
            "p_aux_w": p_aux_w,
            "p_battery_total_w": p_battery_total,
            "mode": mode,
        }

    # ====================================================================
    # 8. DRIVE CYCLE — không đổi signature (wind/crr = default bên trong)
    # ====================================================================
    def simulate_drive_cycle(
        self,
        time_s: List[float],
        velocity_profile_mps: List[float],
        grade_profile_rad: Optional[List[float]] = None,
        air_density_profile: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        n = len(time_s)
        assert n == len(velocity_profile_mps)
        if grade_profile_rad is None:
            grade_profile_rad = [0.0] * n
        if air_density_profile is None:
            air_density_profile = [AIR_DENSITY_SEA_LEVEL] * n

        p_battery_series, p_regen_series = [], []
        v_series = list(velocity_profile_mps)

        for i in range(n):
            if i < n - 1 and (time_s[i + 1] - time_s[i]) > 0:
                dt_i = time_s[i + 1] - time_s[i]
                a_i = (velocity_profile_mps[i + 1] - velocity_profile_mps[i]) / dt_i
            else:
                a_i = 0.0
            power_result = self.compute_battery_power_w(
                velocity_mps=velocity_profile_mps[i],
                acceleration_mps2=a_i,
                grade_angle_rad=grade_profile_rad[i],
                air_density=air_density_profile[i],
            )
            p_battery_series.append(power_result["p_battery_total_w"])
            p_regen_series.append(-min(0.0, power_result["p_battery_traction_w"]))

        total_energy_j       = trapezoidal_integral(time_s, p_battery_series)
        total_regen_energy_j = trapezoidal_integral(time_s, p_regen_series)
        total_distance_m     = trapezoidal_integral(time_s, v_series)

        total_energy_kwh = total_energy_j / 3_600_000.0
        total_regen_kwh  = total_regen_energy_j / 3_600_000.0
        distance_km      = total_distance_m / 1000.0
        pct = (total_energy_kwh / self.battery_usable_capacity_kwh) * 100.0

        return {
            "total_energy_consumed_kwh": round(total_energy_kwh, 4),
            "total_regen_recovered_kwh": round(total_regen_kwh, 4),
            "distance_km": round(distance_km, 3),
            "battery_percent_consumed": round(pct, 3),
            "avg_consumption_kwh_per_km": round(total_energy_kwh / distance_km, 4) if distance_km > 0 else 0.0,
        }

    # ====================================================================
    # 9. SEGMENT ENERGY — PATCH: + wind_along_mps, + crr_mult
    # ====================================================================
    def compute_segment_energy_kwh(
        self,
        distance_m: float,
        avg_speed_mps: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
        r_mult: float = 1.0,
        wind_along_mps: float = 0.0,
        crr_mult: float = 1.0,
    ) -> float:
        if avg_speed_mps <= 0:
            return 0.0
        time_s = distance_m / avg_speed_mps
        power_result = self.compute_battery_power_w(
            velocity_mps=avg_speed_mps,
            acceleration_mps2=0.0,
            grade_angle_rad=grade_angle_rad,
            air_density=air_density,
            r_mult=r_mult,
            wind_along_mps=wind_along_mps,
            crr_mult=crr_mult,
        )
        return power_result["p_battery_total_w"] * time_s / 3_600_000.0


if __name__ == "__main__":
    car = Polestar4Dynamics()

    print("=== TEST 1: 100 km/h, đường bằng, no wind, crr nominal ===")
    v_100 = 100 / 3.6
    f = car.compute_total_tractive_force(v_100, 0.0, 0.0)
    for k, val in f.items():
        print(f"  {k}: {val:.2f}")

    print("\n=== TEST 2: same, but 10 m/s headwind (wind_along=-10) ===")
    f2 = car.compute_total_tractive_force(v_100, 0.0, 0.0, wind_along_mps=-10.0)
    print(f"  f_aero with headwind: {f2['f_aero_N']:.2f} N  "
          f"(was {f['f_aero_N']:.2f} N → +{(f2['f_aero_N']/f['f_aero_N']-1)*100:.1f}%)")

    print("\n=== TEST 3: same, but wet road (crr_mult=1.25) ===")
    f3 = car.compute_total_tractive_force(v_100, 0.0, 0.0, crr_mult=1.25)
    print(f"  f_roll wet: {f3['f_roll_N']:.2f} N  (was {f['f_roll_N']:.2f} N)")

    print("\n=== TEST 4: 50km @ 90km/h, +4% grade, 8m/s headwind, wet ===")
    theta = Polestar4Dynamics.grade_percent_to_radians(4.0)
    e = car.compute_segment_energy_kwh(
        distance_m=50_000, avg_speed_mps=90/3.6, grade_angle_rad=theta,
        wind_along_mps=-8.0, crr_mult=1.25,
    )
    e_baseline = car.compute_segment_energy_kwh(
        distance_m=50_000, avg_speed_mps=90/3.6, grade_angle_rad=theta,
    )
    print(f"  With wind+wet:  {e:.3f} kWh")
    print(f"  Baseline:       {e_baseline:.3f} kWh  → +{(e/e_baseline-1)*100:.1f}%")