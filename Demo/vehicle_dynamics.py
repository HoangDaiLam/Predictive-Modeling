# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: vehicle_dynamics.py
STEP 1 — Vehicle Dynamics & Energy Physics Engine
================================================================================
Purpose:
        Simulate the dynamics and energy consumption of the Polestar 4 Single Motor
        (RWD) electric vehicle in Standard drive mode.

Author role: EV Dynamics Engineer (simulation based on user requirements)

Technical notes:
        - All physical formulas are kept intact and are not simplified.
        - Units are standardized to SI (meters, kg, seconds, Newtons, Watts, Joules),
            converting to kWh only at the final output stage for the user.
        - Vehicle parameters are set to configurable default values
            (the constructor allows full override), because manufacturer data can vary
            by version and market.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

 # ------------------------------------------------------------------------
 # GLOBAL PHYSICAL CONSTANTS
 # ------------------------------------------------------------------------
GRAVITY: float = 9.80665            # standard gravitational acceleration (m/s^2)
AIR_DENSITY_SEA_LEVEL: float = 1.225  # air density at 15°C, sea level (kg/m^3)


def air_density_at_temperature(temp_celsius: float, pressure_pa: float = 101325.0) -> float:
    """
    Calculate air density (rho) from the actual temperature using the ideal gas law:
    rho = P / (R_specific * T)

    This is important in winter because cold air is denser than warm air,
    which increases aerodynamic drag (F_aero) relative to the assumption
    rho = 1.225 kg/m^3 (which is only valid at 15°C).

    Args:
        temp_celsius: ambient temperature (°C)
        pressure_pa: atmospheric pressure (Pa), default 1 atm

    Returns:
        rho (kg/m^3)
    """
    R_SPECIFIC_AIR = 287.058  # specific gas constant for dry air (J/(kg·K))
    temp_kelvin = temp_celsius + 273.15
    rho = pressure_pa / (R_SPECIFIC_AIR * temp_kelvin)
    return rho


@dataclass
class Polestar4Dynamics:
    """
    OOP class representing the full physical parameters and dynamics model of the
    Polestar 4 Single Motor (Long Range Single Motor, RWD).

    Default values are based on approximate published specifications from the
    manufacturer for the Single Motor version (can be adjusted in the constructor
    if more precise Polestar data is available for a specific market/model year).
    """

    # ---------------- Mass and geometry parameters ----------------
    curb_mass_kg: float = 2350.0          # vehicle curb mass (kg)
    payload_kg: float = 150.0             # additional payload (people + luggage), default for 2 occupants
    frontal_area_m2: float = 2.75         # frontal area A (m^2)
    drag_coefficient_cd: float = 0.27     # aerodynamic drag coefficient Cd (Single Motor, no active spoiler)
    wheel_radius_m: float = 0.36          # wheel radius (corresponding to 20–21 inch tires) (m)
    rolling_resistance_coeff: float = 0.009  # rolling resistance coefficient Crr (low rolling resistance EV tires)

    # ---------------- Drivetrain parameters (Single Motor) ----------------
    motor_peak_power_kw: float = 200.0     # rear motor rated power (kW)
    drivetrain_efficiency: float = 0.91    # combined drivetrain efficiency
                                            # (inverter + motor + single-speed gearbox + differential)
    max_regen_power_kw: float = 150.0      # maximum regenerative braking power

    # ---------------- Battery parameters ----------------
    battery_gross_capacity_kwh: float = 94.0   # gross battery capacity
    battery_usable_capacity_kwh: float = 92.0  # usable battery capacity (after BMS buffer)
    battery_nominal_voltage_v: float = 400.0   # nominal battery system voltage

    # ---------------- Auxiliary electrical system efficiency ----------------
    aux_power_baseline_kw: float = 0.45    # baseline 12V/electrical auxiliary load
                                            # (lights, display, basic cooling pump, ECU, ...)
    drag_coefficient_regen_efficiency: float = 0.70
    # ^ conversion efficiency from kinetic energy to electrical energy recharged into the battery during regenerative braking
    #   (including losses through the motor acting as a generator, inverter and BMS charge losses)

    def __post_init__(self):
        self.total_mass_kg: float = self.curb_mass_kg + self.payload_kg

    # ====================================================================
    # 1. AERODYNAMIC DRAG FORCE
    # ====================================================================
    def compute_aero_drag_force(self, velocity_mps: float, air_density: float = AIR_DENSITY_SEA_LEVEL) -> float:
        """
        Aerodynamic drag force formula:
            F_aero = 0.5 * rho * Cd * A * v^2

        Where:
            rho  : air density (kg/m^3) — use air_density_at_temperature() for more accurate winter simulations.
            Cd   : aerodynamic drag coefficient (dimensionless)
            A    : frontal area (m^2)
            v    : relative speed between the vehicle and the air (m/s)
                   (assuming no crosswind or headwind, v = vehicle speed)

        Note: this force always resists motion (always positive when v > 0),
        and it increases with the square of speed — which is why high-speed driving
        consumes significantly more energy than average-speed cruising.
        """
        v = abs(velocity_mps)
        f_aero = 0.5 * air_density * self.drag_coefficient_cd * self.frontal_area_m2 * (v ** 2)
        return f_aero

    # ====================================================================
    # 2. ROLLING RESISTANCE FORCE
    # ====================================================================
    def compute_rolling_resistance_force(self, grade_angle_rad: float = 0.0) -> float:
        """
        Rolling resistance force formula:
            F_roll = Crr * m * g * cos(theta)

        Where:
            Crr   : tire rolling resistance coefficient
            m     : total vehicle mass (kg) = curb_mass + payload
            g     : gravitational acceleration (m/s^2)
            theta : road grade angle (rad). cos(theta) adjusts for the component of gravity
                    perpendicular to the road surface, because rolling resistance depends on
                    normal force rather than total weight.

        At small grade angles (typical roads are usually under 15°), cos(theta) ≈ 1,
        but we keep the full formula to maintain accuracy in steep-hill simulations.
        """
        f_roll = (
            self.rolling_resistance_coeff
            * self.total_mass_kg
            * GRAVITY
            * math.cos(grade_angle_rad)
        )
        return f_roll

    # ====================================================================
    # 3. GRADE / GRAVITATIONAL FORCE
    # ====================================================================
    def compute_grade_force(self, grade_angle_rad: float) -> float:
        """
        Gravitational force along the grade:
            F_grade = m * g * sin(theta)

        theta > 0  => uphill (vehicle must generate extra tractive force, consuming energy)
        theta < 0  => downhill (gravity assists the vehicle and may allow regenerative recovery)

        theta is calculated from grade percentage (grade %) or from elevation data divided by
        horizontal distance via the grade_percent_to_radians() helper below.
        """
        f_grade = self.total_mass_kg * GRAVITY * math.sin(grade_angle_rad)
        return f_grade

    @staticmethod
    def grade_percent_to_radians(grade_percent: float) -> float:
        """
        Convert grade percentage (grade %, as used by Google Maps/GPS) into radians.

        Definition: grade% = (delta_elevation / delta_horizontal_distance) * 100
        => tan(theta) = grade% / 100
        => theta = arctan(grade% / 100)
        """
        return math.atan(grade_percent / 100.0)

    # ====================================================================
    # 4. INERTIAL / ACCELERATION FORCE
    # ====================================================================
    def compute_acceleration_force(self, acceleration_mps2: float) -> float:
        """
        Newton's second law:
            F_accel = m * a

        a > 0: the vehicle is accelerating (consumes extra energy to build kinetic energy)
        a < 0: the vehicle is decelerating (kinetic energy is released and may be partially recovered
               through regenerative braking — see compute_regenerative_energy())

        We can also calculate the kinetic energy change directly:
            delta_KE = 0.5 * m * (v_end^2 - v_start^2)
        The compute_kinetic_energy_delta() function below uses this expression to cross-check the
        force-based path integration.
        """
        f_accel = self.total_mass_kg * acceleration_mps2
        return f_accel

    def compute_kinetic_energy_delta(self, v_start_mps: float, v_end_mps: float) -> float:
        """
        Change in kinetic energy: delta_KE = 0.5 * m * (v_end^2 - v_start^2)   [Joule]
        """
        return 0.5 * self.total_mass_kg * (v_end_mps ** 2 - v_start_mps ** 2)

    # ====================================================================
    # 5. FORCE AND INSTANTANEOUS POWER SUMMATION
    # ====================================================================
    def compute_total_tractive_force(
        self,
        velocity_mps: float,
        acceleration_mps2: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
    ) -> Dict[str, float]:
        """
        Combine the tractive force required at the wheel:
            F_total = F_aero + F_roll + F_grade + F_accel

        This is the longitudinal force balance equation for a point-mass vehicle model,
        a standard formulation in electric vehicle engineering.

        Returns a dict with detailed force components for analysis/debugging.
        """
        f_aero = self.compute_aero_drag_force(velocity_mps, air_density)
        f_roll = self.compute_rolling_resistance_force(grade_angle_rad)
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

    def compute_wheel_power_w(
        self,
        velocity_mps: float,
        acceleration_mps2: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
    ) -> float:
        """
        Wheel power (before drivetrain efficiency losses):
            P_wheel = F_total * v

        If P_wheel > 0: the vehicle requires tractive effort (motor operates in driving mode)
        If P_wheel < 0: the vehicle requires braking force (motor can operate in regenerative
                        braking mode — see compute_regenerative_energy())
        """
        forces = self.compute_total_tractive_force(velocity_mps, acceleration_mps2, grade_angle_rad, air_density)
        p_wheel = forces["f_total_N"] * velocity_mps
        return p_wheel

    # ====================================================================
    # 6. BATTERY POWER (including drivetrain efficiency + regenerative braking)
    # ====================================================================
    def compute_battery_power_w(
        self,
        velocity_mps: float,
        acceleration_mps2: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
        include_aux_load: bool = True,
    ) -> Dict[str, float]:
        """
        Convert wheel power (P_wheel) into power drawn from the battery (P_battery),
        distinguishing between two operating modes:

        (a) DRIVING MODE (P_wheel >= 0):
            P_battery = P_wheel / eta_drivetrain + P_aux
            (divide by efficiency because of losses in the motor + inverter + gearbox)

        (b) REGENERATIVE BRAKING MODE (P_wheel < 0):
            P_battery = P_wheel * eta_regen_total + P_aux
            (multiply by efficiency because this is energy being fed back into the battery,
             with losses during conversion from kinetic energy to electrical energy; limited by
             the motor/inverter max_regen_power_kw)

        P_aux: baseline auxiliary electrical load (excluding HVAC — HVAC is added separately
               in winter_hvac.py to keep the pure vehicle physics distinct from thermal systems).

        Returns positive power = battery discharging (consumption),
        negative power = battery charging (recovery).
        """
        p_wheel = self.compute_wheel_power_w(velocity_mps, acceleration_mps2, grade_angle_rad, air_density)
        p_aux_w = self.aux_power_baseline_kw * 1000.0 if include_aux_load else 0.0

        if p_wheel >= 0:
            # Chế độ kéo: tổn hao truyền động làm pin phải cấp NHIỀU hơn P_wheel
            p_battery_traction = p_wheel / self.drivetrain_efficiency
            mode = "DRIVING"
        else:
            # Chế độ phanh tái sinh: giới hạn theo công suất tái sinh tối đa của motor
            max_regen_w = -abs(self.max_regen_power_kw) * 1000.0
            p_wheel_clamped = max(p_wheel, max_regen_w)  # không vượt quá giới hạn phần cứng
            p_battery_traction = p_wheel_clamped * self.drag_coefficient_regen_efficiency
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
    # 7. ENERGY CONSUMPTION SIMULATION FOR A DRIVING CYCLE
    # ====================================================================
    def simulate_drive_cycle(
        self,
        time_s: List[float],
        velocity_profile_mps: List[float],
        grade_profile_rad: Optional[List[float]] = None,
        air_density_profile: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Numerically integrate battery power over time to calculate the total energy consumed
        over a discretized drive cycle using a simplified rectangular/Euler method on each step dt:

            E_battery = sum( P_battery(t_i) * dt_i )   với dt_i = t_(i+1) - t_i

        Instantaneous acceleration at each step is approximated as:
            a_i = (v_(i+1) - v_i) / dt_i

        Args:
            time_s: list of time stamps (seconds), increasing over the cycle
            velocity_profile_mps: vehicle speed at each time stamp (m/s)
            grade_profile_rad: grade angle at each time stamp (rad); default 0 (flat road)
            air_density_profile: rho at each time stamp (kg/m^3); default sea-level reference

        Returns:
            Dict with total energy consumed (kWh), recovered regenerative energy (kWh),
            distance traveled (km), and battery percentage consumed (% of usable capacity).
        """
        n = len(time_s)
        assert n == len(velocity_profile_mps), "time_s and velocity_profile_mps must have the same length"

        if grade_profile_rad is None:
            grade_profile_rad = [0.0] * n
        if air_density_profile is None:
            air_density_profile = [AIR_DENSITY_SEA_LEVEL] * n

        total_energy_j = 0.0          # tổng năng lượng ròng rút từ pin (Joule)
        total_regen_energy_j = 0.0    # tổng năng lượng tái sinh thu hồi (Joule, giá trị dương)
        total_distance_m = 0.0

        for i in range(n - 1):
            dt = time_s[i + 1] - time_s[i]
            if dt <= 0:
                continue

            v_i = velocity_profile_mps[i]
            v_ip1 = velocity_profile_mps[i + 1]
            a_i = (v_ip1 - v_i) / dt

            grade_i = grade_profile_rad[i]
            rho_i = air_density_profile[i]

            power_result = self.compute_battery_power_w(
                velocity_mps=v_i,
                acceleration_mps2=a_i,
                grade_angle_rad=grade_i,
                air_density=rho_i,
            )
            p_battery = power_result["p_battery_total_w"]

            energy_step_j = p_battery * dt  # Joule = Watt * giây
            total_energy_j += energy_step_j

            if power_result["mode"] == "REGEN_BRAKING" and power_result["p_battery_traction_w"] < 0:
                total_regen_energy_j += -power_result["p_battery_traction_w"] * dt

            total_distance_m += v_i * dt

        total_energy_kwh = total_energy_j / 3_600_000.0
        total_regen_kwh = total_regen_energy_j / 3_600_000.0
        distance_km = total_distance_m / 1000.0

        battery_percent_consumed = (total_energy_kwh / self.battery_usable_capacity_kwh) * 100.0

        return {
            "total_energy_consumed_kwh": round(total_energy_kwh, 4),
            "total_regen_recovered_kwh": round(total_regen_kwh, 4),
            "distance_km": round(distance_km, 3),
            "battery_percent_consumed": round(battery_percent_consumed, 3),
            "avg_consumption_kwh_per_km": round(total_energy_kwh / distance_km, 4) if distance_km > 0 else 0.0,
        }

    # ====================================================================
    # 8. ENERGY CALCULATION FOR A SIMPLIFIED ROAD SEGMENT (steady-state segment)
    # ====================================================================
    def compute_segment_energy_kwh(
        self,
        distance_m: float,
        avg_speed_mps: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
    ) -> float:
        """
        Estimate the energy consumption for ONE ROAD SEGMENT at a constant average speed
        (steady-state, a = 0) — used for route optimization in Step 4, where we have
        segment-level grade and distance data from map APIs but no detailed second-by-second
        speed profile.

            time_s = distance_m / avg_speed_mps
            E = P_battery(steady-state) * time_s

        This is a valid steady-state model because it assumes the vehicle travels at a
        constant average speed over the segment (a=0), which is suitable for long highway or
        intercity road segments. For urban segments with repeated stops, use simulate_drive_cycle().
        """
        if avg_speed_mps <= 0:
            return 0.0

        time_s = distance_m / avg_speed_mps
        power_result = self.compute_battery_power_w(
            velocity_mps=avg_speed_mps,
            acceleration_mps2=0.0,
            grade_angle_rad=grade_angle_rad,
            air_density=air_density,
        )
        energy_j = power_result["p_battery_total_w"] * time_s
        energy_kwh = energy_j / 3_600_000.0
        return energy_kwh


# ============================================================================
# KHỐI TỰ KIỂM TRA (self-test) khi chạy trực tiếp file này
# ============================================================================
if __name__ == "__main__":
    car = Polestar4Dynamics()

    print("=== TEST 1: Lực & công suất ở 100 km/h, đường bằng, vận tốc ổn định ===")
    v_100kmh = 100 / 3.6
    forces = car.compute_total_tractive_force(v_100kmh, 0.0, 0.0)
    for k, val in forces.items():
        print(f"  {k}: {val:.2f}")
    power = car.compute_battery_power_w(v_100kmh, 0.0, 0.0)
    print(f"  Công suất pin: {power['p_battery_total_w']/1000:.2f} kW ({power['mode']})")

    print("\n=== TEST 2: Đoạn đường 50km, tốc độ TB 90km/h, dốc lên 4% ===")
    theta = Polestar4Dynamics.grade_percent_to_radians(4.0)
    e_kwh = car.compute_segment_energy_kwh(distance_m=50_000, avg_speed_mps=90/3.6, grade_angle_rad=theta)
    print(f"  Năng lượng tiêu thụ: {e_kwh:.3f} kWh (~{e_kwh/car.battery_usable_capacity_kwh*100:.2f}% pin)")