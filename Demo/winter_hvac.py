# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: winter_hvac.py
STEP 2 — Winter Climate & Thermal Management Engine
================================================================================
Purpose:
        Simulate the effect of winter conditions (sub-zero temperatures) on energy
        consumption for the Polestar 4 through two related but distinct thermal systems:
                1. Cabin HVAC (heating the passenger compartment)
                2. BTMS - Battery Thermal Management System (battery heating)
        This also models reduced battery performance (capacity fade / internal
        resistance losses) caused by low temperatures.

Technical notes:
        - The Polestar 4 uses a heat pump as the primary cabin heating source
            (more efficient than pure PTC electric resistance heating), with COP
            (Coefficient of Performance) decreasing as ambient temperature drops.
        - The cabin thermal model uses a simplified "lumped thermal mass" approach
            (one thermal mass representing cabin air + cabin interior).
        - The BTMS model uses empirical heating power lookup tables by temperature
            band, reflecting general published behavior of modern EV lithium-ion BTMS systems.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class WinterHVACSimulation:
    """
    OOP class modeling the cabin HVAC + battery BTMS system for the Polestar 4
    under cold winter conditions (-10°C to 0°C, and potentially lower).
    """

    # ---------------- Cabin thermal parameters (lumped thermal model) ----------------
    cabin_volume_m3: float = 3.2          # approximate cabin volume (m^3)
    cabin_surface_area_m2: float = 11.5   # outer cabin heat-transfer surface area (m^2)
    cabin_u_value_w_m2k: float = 2.5      # overall heat-transfer coefficient of body/glazing (W/(m^2·K))
                                           # (blend between well-insulated body ~0.8-1.2 and large glass area
                                           # with poorer insulation ~5.5-5.8, characteristic of the Polestar 4's
                                           # large glasshouse / no rear glass)
    hvac_fresh_air_flow_m3_per_h: float = 180.0
    # ^ fresh-air flow that the HVAC system continuously draws in and heats to prevent
    #   condensation/fogging of windows and maintain cabin air quality, at average fan speed.
    #   This is the actual air flow passing through the heater core, unlike passive leakage,
    #   so we use direct volumetric flow instead of ACH (Air Changes per Hour), which only
    #   describes passive leakage.
    air_density_kg_m3: float = 1.25
    air_specific_heat_j_kgk: float = 1005.0  # specific heat capacity of air (J/(kg·K))

    cabin_target_temp_c: float = 22.0     # target cabin temperature (normal heat mode)

    # ---------------- Heat pump parameters for cabin HVAC ----------------
    heat_pump_cop_at_0c: float = 3.0      # COP of the heat pump at 0°C
    heat_pump_cop_at_neg10c: float = 1.8  # COP decreases as ambient temperature drops (harder to extract heat from the air)
    heat_pump_max_power_kw: float = 6.0   # maximum cabin heat output of the heat pump (kW thermal output)
    ptc_backup_heater_kw: float = 4.0     # auxiliary PTC resistance heater power when the heat pump is insufficient

    # ---------------- BTMS (Battery Thermal Management) parameters ----------------
    battery_mass_kg: float = 550.0        # battery pack mass (kg), including cells + casing + frame
    battery_specific_heat_j_kgk: float = 950.0  # representative battery pack specific heat capacity (J/(kg·K))
    battery_target_temp_c: float = 20.0   # optimal battery operating temperature (°C)
    battery_min_safe_temp_c: float = 5.0  # below this threshold, BTMS must actively heat before heavy charge/discharge
    btms_heater_max_power_kw: float = 6.0 # maximum battery heating power (resistance heater / auxiliary heat pump)
    btms_insulation_loss_w_per_k: float = 8.0  # heat loss from the battery pack to the environment (W/K), pack with good insulation

    # ---------------- Battery performance degradation with temperature ----------------
    # Empirical lookup table: effective battery capacity factor and internal resistance
    # multiplier versus battery temperature. Linear interpolation is used between points.
    capacity_fade_lookup_temp_c: Tuple[float, ...] = (-20, -10, 0, 10, 20, 25)
    capacity_fade_factor: Tuple[float, ...] = (0.68, 0.78, 0.88, 0.95, 0.99, 1.00)
    # ^ for example, at -10°C the battery can only effectively use about 78% of its nominal
    #   capacity because electrochemical reaction rates slow and electrolyte viscosity increases.

    internal_resistance_multiplier_lookup: Tuple[float, ...] = (2.6, 1.9, 1.35, 1.1, 1.0, 1.0)
    # ^ increased internal resistance -> higher I^2*R losses during high-current charging/discharging at low temperature.

    # ========================================================================
    # A. CABIN HEATING POWER CALCULATION (HVAC)
    # ========================================================================
    def _interpolate_heat_pump_cop(self, ambient_temp_c: float) -> float:
        """
        Linearly interpolate the heat-pump COP versus ambient temperature using the two
        calibrated points at 0°C and -10°C:

            COP(T) = COP_neg10 + (COP_0 - COP_neg10) * (T - (-10)) / (0 - (-10))

                Outside the range [-10, 0], clamp the value to prevent unrealistic extrapolation:
                        - If T < -10°C: COP drops further linearly but not below 1.0
                            (below 1.0 would mean worse than pure resistance heating, which is not realistic
                            for modern heat pumps, so we enforce a floor of 1.0)
                        - If T > 0°C: COP continues to increase with the same slope, capped at a reasonable maximum of 4.0
        """
        slope = (self.heat_pump_cop_at_0c - self.heat_pump_cop_at_neg10c) / (0 - (-10))
        cop = self.heat_pump_cop_at_neg10c + slope * (ambient_temp_c - (-10))
        cop = max(1.0, min(cop, 4.0))
        return cop

    def compute_cabin_heat_loss_w(self, ambient_temp_c: float, cabin_temp_c: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate the total heat loss from the cabin (W) that must be offset to maintain the
        target temperature, consisting of two components:

        (1) Heat conduction/convection through the body and glass (Conduction/Convection loss):
                Q_conduction = U * A * deltaT
            U: overall heat-transfer coefficient (W/m^2·K)
            A: heat-transfer surface area (m^2)
            deltaT: temperature difference inside vs. outside the cabin (K = °C)

        (2) Heating the incoming fresh-air stream (Fresh-air heating load):
                mdot_air = (flow_m3_per_h / 3600) * rho_air     [kg/s]
                Q_air = mdot_air * c_p_air * deltaT              [W]
            This is the heat required to warm the cold outdoor air continuously drawn in by the HVAC system
            to prevent windshield fogging and maintain cabin air quality, unlike passive leakage losses.

        Q_total = Q_conduction + Q_air
        """
        if cabin_temp_c is None:
            cabin_temp_c = self.cabin_target_temp_c

        delta_t = cabin_temp_c - ambient_temp_c
        delta_t = max(delta_t, 0.0)  # if the outside air is warmer than the target, no heating is needed

        q_conduction_w = self.cabin_u_value_w_m2k * self.cabin_surface_area_m2 * delta_t

        mdot_air_kg_s = (self.hvac_fresh_air_flow_m3_per_h / 3600.0) * self.air_density_kg_m3
        q_air_w = mdot_air_kg_s * self.air_specific_heat_j_kgk * delta_t

        q_total_w = q_conduction_w + q_air_w

        return {
            "q_conduction_w": q_conduction_w,
            "q_air_infiltration_w": q_air_w,
            "q_total_heat_loss_w": q_total_w,
            "delta_t_k": delta_t,
        }

    def compute_cabin_hvac_electric_power_kw(self, ambient_temp_c: float, cabin_temp_c: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate the electrical power (kW) that the HVAC system needs to draw from the battery
        to offset the heat loss, including heat-pump efficiency (COP):

            P_electric_heat_pump = Q_heat_loss / COP(T_ambient)

        If the heat demand exceeds the maximum heat-pump capacity
        (heat_pump_max_power_kw), the shortfall is supplied by the auxiliary PTC resistance heater
        (PTC efficiency is approximately 1.0, i.e., COP=1, because electricity is converted directly into heat):

            Q_shortfall = Q_heat_loss - Q_heat_pump_max
            P_ptc = Q_shortfall / 1.0   (nếu Q_shortfall > 0)

        Total electrical power consumed for cabin heating:
            P_hvac_total = P_electric_heat_pump + P_ptc
        """
        heat_loss = self.compute_cabin_heat_loss_w(ambient_temp_c, cabin_temp_c)
        q_total_w = heat_loss["q_total_heat_loss_w"]

        cop = self._interpolate_heat_pump_cop(ambient_temp_c)
        heat_pump_max_output_w = self.heat_pump_max_power_kw * 1000.0

        if q_total_w <= heat_pump_max_output_w:
            q_from_heat_pump_w = q_total_w
            q_from_ptc_w = 0.0
        else:
            q_from_heat_pump_w = heat_pump_max_output_w
            q_from_ptc_w = q_total_w - heat_pump_max_output_w
            # Giới hạn PTC theo công suất PTC tối đa cấu hình
            q_from_ptc_w = min(q_from_ptc_w, self.ptc_backup_heater_kw * 1000.0)

        p_electric_heat_pump_w = q_from_heat_pump_w / cop
        p_electric_ptc_w = q_from_ptc_w / 1.0  # PTC hiệu suất ~100% (điện trở thuần)

        p_hvac_total_kw = (p_electric_heat_pump_w + p_electric_ptc_w) / 1000.0

        return {
            "heat_loss_w": q_total_w,
            "heat_pump_cop": round(cop, 3),
            "heat_pump_electric_power_kw": round(p_electric_heat_pump_w / 1000.0, 3),
            "ptc_backup_electric_power_kw": round(p_electric_ptc_w / 1000.0, 3),
            "total_hvac_electric_power_kw": round(p_hvac_total_kw, 3),
        }

    # ========================================================================
    # B. BTMS - BATTERY HEATING CALCULATION
    # ========================================================================
    def compute_battery_heating_power_kw(
        self,
        ambient_temp_c: float,
        current_battery_temp_c: Optional[float] = None,
        time_to_target_s: float = 900.0,
    ) -> Dict[str, float]:
        """
        Calculate the electrical power (kW) needed for BTMS to warm the battery to its
        optimal operating temperature (battery_target_temp_c), using a lumped capacitance model:

        (1) Heat required to RAISE the battery temperature (sensible heating):
                Q_raise = m_battery * c_p_battery * (T_target - T_current)
            (only counted if T_current < T_target; if the battery is already warm enough, Q_raise = 0)

            Average power required to reach the target within time_to_target_s:
                P_raise = Q_raise / time_to_target_s

        (2) Heat required to compensate for CONTINUOUS heat loss to the environment while
            maintaining temperature (steady-state holding loss) through the battery pack insulation:
                Q_hold_loss = k_insulation * (T_target - T_ambient)
            k_insulation: pack heat-loss coefficient (W/K), with the battery pack insulated and
            potentially exposed to cold ambient air in the underbody or cabin.

        Total BTMS electrical power (assuming heater efficiency ~95%, with minor losses in wiring/control):
                P_btms_total = (P_raise + Q_hold_loss) / eta_heater
        """
        if current_battery_temp_c is None:
            # Default assumption: battery temperature is close to ambient when the car
            # has been parked overnight in cold weather (cold-soaked battery)
            current_battery_temp_c = ambient_temp_c

        eta_heater = 0.95

        delta_t_raise = max(self.battery_target_temp_c - current_battery_temp_c, 0.0)
        q_raise_j = self.battery_mass_kg * self.battery_specific_heat_j_kgk * delta_t_raise
        p_raise_w = q_raise_j / time_to_target_s if time_to_target_s > 0 else 0.0

        delta_t_hold = max(self.battery_target_temp_c - ambient_temp_c, 0.0)
        q_hold_loss_w = self.btms_insulation_loss_w_per_k * delta_t_hold

        p_btms_raw_w = p_raise_w + q_hold_loss_w
        p_btms_total_w = p_btms_raw_w / eta_heater

        # Giới hạn theo công suất heater tối đa phần cứng
        p_btms_total_w = min(p_btms_total_w, self.btms_heater_max_power_kw * 1000.0)

        return {
            "battery_temp_before_c": current_battery_temp_c,
            "battery_target_temp_c": self.battery_target_temp_c,
            "power_to_raise_temp_kw": round(p_raise_w / 1000.0, 3),
            "power_to_hold_temp_kw": round(q_hold_loss_w / 1000.0, 3),
            "total_btms_electric_power_kw": round(p_btms_total_w / 1000.0, 3),
        }

    # ========================================================================
    # C. BATTERY PERFORMANCE DEGRADATION IN LOW TEMPERATURES
    # ========================================================================
    @staticmethod
    def _linear_interp(x: float, xp: Tuple[float, ...], fp: Tuple[float, ...]) -> float:
        """Manual piecewise-linear interpolation function (without numpy),
        with clamping at both ends of the lookup table."""
        if x <= xp[0]:
            return fp[0]
        if x >= xp[-1]:
            return fp[-1]
        for i in range(len(xp) - 1):
            if xp[i] <= x <= xp[i + 1]:
                ratio = (x - xp[i]) / (xp[i + 1] - xp[i])
                return fp[i] + ratio * (fp[i + 1] - fp[i])
        return fp[-1]

    def compute_capacity_fade_factor(self, battery_temp_c: float) -> float:
        """
        Return the effective usable capacity factor (0.0 - 1.0) of the battery at the
        specified temperature, interpolated from the experimental lookup table
        capacity_fade_lookup_temp_c / capacity_fade_factor.

        Physical meaning: at low temperatures, the diffusion rate of Li+ ions in the
        electrolyte and electrodes slows, reducing the immediate usable capacity
        (not permanent degradation; capacity recovers when the battery warms up).
        """
        return self._linear_interp(
            battery_temp_c, self.capacity_fade_lookup_temp_c, self.capacity_fade_factor
        )

    def compute_internal_resistance_multiplier(self, battery_temp_c: float) -> float:
        """
        Return the internal resistance multiplier at the specified battery temperature,
        used to estimate additional I^2*R losses when the battery operates (charges/discharges)
        at low temperatures.
        """
        return self._linear_interp(
            battery_temp_c,
            self.capacity_fade_lookup_temp_c,
            self.internal_resistance_multiplier_lookup,
        )

    def compute_effective_usable_capacity_kwh(self, nominal_usable_capacity_kwh: float, battery_temp_c: float) -> float:
        """
        Effective usable capacity in cold conditions:
            C_effective = C_nominal * capacity_fade_factor(T_battery)
        """
        factor = self.compute_capacity_fade_factor(battery_temp_c)
        return nominal_usable_capacity_kwh * factor

    # ========================================================================
    # D. AGGREGATION: ENERGY LOSS FROM HVAC + BTMS OVER TIME/DISTANCE
    # ========================================================================
    def compute_total_winter_energy_overhead(
        self,
        ambient_temp_c: float,
        trip_duration_s: float,
        distance_km: float,
        initial_battery_temp_c: Optional[float] = None,
        time_to_target_s: float = 900.0,
    ) -> Dict[str, float]:
        """
        Aggregate the total electrical energy loss (kWh) from the cabin HVAC + battery BTMS
        system over a trip based on trip duration.

        Reasonable simplified assumption: HVAC and BTMS power remain close to constant
        over short to medium trips because ambient temperature and cabin/battery target
        conditions remain nearly constant => Energy = Power * time.

        (For very long trips crossing multiple climate regions, split the trip into smaller
        sections and call this function for each section — see route_optimizer.py.)
        """
        hvac_result = self.compute_cabin_hvac_electric_power_kw(ambient_temp_c)
        btms_result = self.compute_battery_heating_power_kw(
            ambient_temp_c, initial_battery_temp_c, time_to_target_s
        )

        hvac_power_kw = hvac_result["total_hvac_electric_power_kw"]
        btms_power_kw = btms_result["total_btms_electric_power_kw"]

        trip_duration_h = trip_duration_s / 3600.0

        hvac_energy_kwh = hvac_power_kw * trip_duration_h
        btms_energy_kwh = btms_power_kw * trip_duration_h
        total_overhead_kwh = hvac_energy_kwh + btms_energy_kwh

        overhead_per_km = total_overhead_kwh / distance_km if distance_km > 0 else 0.0

        return {
            "ambient_temp_c": ambient_temp_c,
            "hvac_power_kw": hvac_power_kw,
            "btms_power_kw": btms_power_kw,
            "trip_duration_h": round(trip_duration_h, 4),
            "hvac_energy_kwh": round(hvac_energy_kwh, 4),
            "btms_energy_kwh": round(btms_energy_kwh, 4),
            "total_winter_overhead_kwh": round(total_overhead_kwh, 4),
            "overhead_per_km_kwh": round(overhead_per_km, 4),
            "capacity_fade_factor": round(
                self.compute_capacity_fade_factor(initial_battery_temp_c or ambient_temp_c), 4
            ),
            "internal_resistance_multiplier": round(
                self.compute_internal_resistance_multiplier(initial_battery_temp_c or ambient_temp_c), 4
            ),
        }


# ============================================================================
# KHỐI TỰ KIỂM TRA
# ============================================================================
if __name__ == "__main__":
    hvac_sim = WinterHVACSimulation()

    print("=== TEST: Conditions -10°C, trip duration 45 minutes, 40 km ===")
    result = hvac_sim.compute_total_winter_energy_overhead(
        ambient_temp_c=-10.0,
        trip_duration_s=45 * 60,
        distance_km=40.0,
        initial_battery_temp_c=-8.0,
    )
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== TEST: Compare capacity fade factors by temperature ===")
    for t in [-20, -10, -5, 0, 10, 20]:
        f = hvac_sim.compute_capacity_fade_factor(t)
        r = hvac_sim.compute_internal_resistance_multiplier(t)
        print(f"  T={t:>4}°C -> capacity_factor={f:.3f}, internal_resistance_x={r:.3f}")