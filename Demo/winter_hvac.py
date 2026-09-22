# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: winter_hvac.py
BƯỚC 2 — Winter Climate & Thermal Management Engine
================================================================================
Mục đích:
    Mô phỏng ảnh hưởng của thời tiết mùa đông (nhiệt độ âm) lên tiêu thụ
    năng lượng của Polestar 4 thông qua 2 hệ thống nhiệt độc lập nhưng
    liên quan:
        1. HVAC khoang cabin (sưởi ấm hành khách)
        2. BTMS - Battery Thermal Management System (sưởi pin)
    Đồng thời mô hình hoá sự suy giảm hiệu suất pin (capacity fade /
    internal resistance losses) do nhiệt độ thấp.

Ghi chú kỹ thuật:
    - Polestar 4 sử dụng bơm nhiệt (heat pump) làm nguồn sưởi chính cho
      cabin (hiệu quả hơn sưởi điện trở PTC thuần tuý), với COP (Coefficient
      of Performance) giảm dần khi nhiệt độ ngoài trời giảm.
    - Mô hình nhiệt cabin dùng mô hình "lumped thermal mass" đơn giản hoá
      (1 khối nhiệt dung đại diện cho không khí + nội thất cabin).
    - Mô hình BTMS dùng công suất sưởi pin theo bảng tra cứu thực nghiệm
      (empirical lookup) theo dải nhiệt độ, mô phỏng theo dữ liệu công bố
      chung của các hệ thống BTMS pin Lithium-ion trên EV hiện đại.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class WinterHVACSimulation:
    """
    Class OOP mô phỏng hệ thống HVAC cabin + BTMS pin cho Polestar 4
    trong điều kiện mùa đông lạnh giá (-10°C đến 0°C, có thể mở rộng thấp hơn).
    """

    # ---------------- Thông số nhiệt cabin (lumped thermal model) ----------------
    cabin_volume_m3: float = 3.2          # thể tích khoang cabin xấp xỉ (m^3)
    cabin_surface_area_m2: float = 11.5   # diện tích bề mặt truyền nhiệt cabin ra ngoài (m^2)
    cabin_u_value_w_m2k: float = 2.5      # hệ số truyền nhiệt tổng hợp vỏ xe/kính (W/(m^2·K))
                                           # (giá trị blend giữa thân xe cách nhiệt tốt ~0.8-1.2
                                           # và diện tích kính lớn dẫn nhiệt kém ~5.5-5.8, đặc thù
                                           # xe có diện tích kính lớn như Polestar 4 - không kính hậu)
    hvac_fresh_air_flow_m3_per_h: float = 180.0
    # ^ lưu lượng khí tươi mà hệ thống HVAC hút vào và làm nóng liên tục để
    #   chống đọng sương/mờ kính (defog) + duy trì chất lượng không khí, ở mức
    #   quạt gió trung bình. Đây là dòng khí THỰC SỰ đi qua giàn sưởi, khác với
    #   rò rỉ khí bị động — vì vậy dùng lưu lượng thể tích trực tiếp thay vì
    #   hệ số ACH (Air Changes per Hour) vốn chỉ mô tả rò rỉ thụ động.
    air_density_kg_m3: float = 1.25
    air_specific_heat_j_kgk: float = 1005.0  # nhiệt dung riêng không khí (J/(kg·K))

    cabin_target_temp_c: float = 22.0     # nhiệt độ cabin mục tiêu (chế độ sưởi bình thường)

    # ---------------- Thông số bơm nhiệt (Heat Pump) cho HVAC cabin ----------------
    heat_pump_cop_at_0c: float = 3.0      # COP của bơm nhiệt tại 0°C
    heat_pump_cop_at_neg10c: float = 1.8  # COP giảm khi trời càng lạnh (khó trích nhiệt từ không khí)
    heat_pump_max_power_kw: float = 6.0   # công suất sưởi tối đa của bơm nhiệt (kW nhiệt đầu ra)
    ptc_backup_heater_kw: float = 4.0     # sưởi điện trở PTC dự phòng khi bơm nhiệt không đủ công suất

    # ---------------- Thông số BTMS (Battery Thermal Management) ----------------
    battery_mass_kg: float = 550.0        # khối lượng pack pin (kg), bao gồm cell + vỏ + khung
    battery_specific_heat_j_kgk: float = 950.0  # nhiệt dung riêng trung bình pack pin (J/(kg·K))
    battery_target_temp_c: float = 20.0   # nhiệt độ vận hành tối ưu của pin (°C)
    battery_min_safe_temp_c: float = 5.0  # dưới ngưỡng này, BTMS phải sưởi tích cực trước khi sạc/xả mạnh
    btms_heater_max_power_kw: float = 6.0 # công suất sưởi pin tối đa (heater điện trở/heat pump phụ)
    btms_insulation_loss_w_per_k: float = 8.0  # tổn thất nhiệt pack pin ra môi trường (W/K), pack có cách nhiệt tốt

    # ---------------- Suy giảm hiệu suất pin theo nhiệt độ ----------------
    # Bảng tra cứu thực nghiệm (empirical lookup table): hệ số hiệu dụng dung lượng
    # pin (effective capacity factor) và hệ số tăng nội trở (internal resistance
    # multiplier) theo nhiệt độ pin. Nội suy tuyến tính giữa các mốc.
    capacity_fade_lookup_temp_c: Tuple[float, ...] = (-20, -10, 0, 10, 20, 25)
    capacity_fade_factor: Tuple[float, ...] = (0.68, 0.78, 0.88, 0.95, 0.99, 1.00)
    # ^ ví dụ: ở -10°C, pin chỉ khai thác được ~78% dung lượng danh định do
    #   tốc độ phản ứng điện hoá chậm lại + độ nhớt chất điện giải tăng.

    internal_resistance_multiplier_lookup: Tuple[float, ...] = (2.6, 1.9, 1.35, 1.1, 1.0, 1.0)
    # ^ nội trở tăng -> tổn hao I^2*R tăng khi sạc/xả dòng lớn ở nhiệt độ thấp.

    # ========================================================================
    # A. TÍNH TOÁN CÔNG SUẤT SƯỞI CABIN (HVAC)
    # ========================================================================
    def _interpolate_heat_pump_cop(self, ambient_temp_c: float) -> float:
        """
        Nội suy tuyến tính hệ số hiệu suất COP của bơm nhiệt theo nhiệt độ
        môi trường, giữa 2 mốc đã hiệu chuẩn (0°C và -10°C):

            COP(T) = COP_neg10 + (COP_0 - COP_neg10) * (T - (-10)) / (0 - (-10))

        Ngoài dải [-10, 0] ta clamp (giới hạn) để tránh ngoại suy sai lệch:
            - Nếu T < -10°C: COP giảm tuyến tính thêm nhưng không dưới 1.0
              (dưới 1.0 nghĩa là kém hơn sưởi điện trở thuần, không thực tế
              với bơm nhiệt hiện đại nên ta chặn sàn ở 1.0)
            - Nếu T > 0°C: COP tăng tiếp theo cùng độ dốc, chặn trần hợp lý ở 4.0
        """
        slope = (self.heat_pump_cop_at_0c - self.heat_pump_cop_at_neg10c) / (0 - (-10))
        cop = self.heat_pump_cop_at_neg10c + slope * (ambient_temp_c - (-10))
        cop = max(1.0, min(cop, 4.0))
        return cop

    def compute_cabin_heat_loss_w(self, ambient_temp_c: float, cabin_temp_c: Optional[float] = None) -> Dict[str, float]:
        """
        Tính tổng nhiệt lượng thất thoát ra khỏi cabin (W) cần bù đắp để
        duy trì nhiệt độ mục tiêu, gồm 2 thành phần:

        (1) Dẫn nhiệt/đối lưu qua vỏ xe & kính (Conduction/Convection loss):
                Q_conduction = U * A * deltaT
            U: hệ số truyền nhiệt tổng hợp (W/m^2·K)
            A: diện tích bề mặt truyền nhiệt (m^2)
            deltaT: chênh lệch nhiệt độ trong/ngoài cabin (K = °C)

        (2) Làm nóng dòng khí tươi nạp vào cabin (Fresh-air heating load):
                mdot_air = (flow_m3_per_h / 3600) * rho_air     [kg/s]
                Q_air = mdot_air * c_p_air * deltaT              [W]
            Đây là tải nhiệt để hâm nóng luồng khí lạnh từ ngoài trời được
            hệ thống HVAC hút vào liên tục (bắt buộc để chống mờ/đọng sương
            kính lái theo luật an toàn), khác với tổn thất rò rỉ thụ động.

        Q_total = Q_conduction + Q_air
        """
        if cabin_temp_c is None:
            cabin_temp_c = self.cabin_target_temp_c

        delta_t = cabin_temp_c - ambient_temp_c
        delta_t = max(delta_t, 0.0)  # nếu ngoài trời ấm hơn mục tiêu thì không cần sưởi

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
        Tính CÔNG SUẤT ĐIỆN (kW) mà hệ thống HVAC cần rút từ pin để bù đắp
        nhiệt lượng thất thoát, có xét đến hiệu suất bơm nhiệt (COP):

            P_electric_heat_pump = Q_heat_loss / COP(T_ambient)

        Nếu nhu cầu nhiệt vượt quá công suất tối đa của bơm nhiệt
        (heat_pump_max_power_kw), phần còn thiếu được bù bởi sưởi điện trở
        PTC phụ trợ (hiệu suất PTC xấp xỉ 1.0, tức COP=1, vì chuyển hoá điện
        trực tiếp thành nhiệt):

            Q_shortfall = Q_heat_loss - Q_heat_pump_max
            P_ptc = Q_shortfall / 1.0   (nếu Q_shortfall > 0)

        Tổng công suất điện tiêu thụ cho cabin:
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
    # B. TÍNH TOÁN BTMS - SƯỞI PIN
    # ========================================================================
    def compute_battery_heating_power_kw(
        self,
        ambient_temp_c: float,
        current_battery_temp_c: Optional[float] = None,
        time_to_target_s: float = 900.0,
    ) -> Dict[str, float]:
        """
        Tính công suất điện (kW) cần thiết để BTMS làm ấm pin lên nhiệt độ
        vận hành tối ưu (battery_target_temp_c), theo mô hình nhiệt dung
        tập trung (lumped capacitance model):

        (1) Nhiệt lượng cần cấp để NÂNG NHIỆT ĐỘ pin (sensible heating):
                Q_raise = m_battery * c_p_battery * (T_target - T_current)
            (chỉ tính nếu T_current < T_target; nếu pin đã đủ ấm, Q_raise = 0)

            Công suất trung bình cần thiết để đạt mục tiêu trong thời gian
            time_to_target_s:
                P_raise = Q_raise / time_to_target_s

        (2) Nhiệt lượng cần bù đắp TỔN THẤT LIÊN TỤC ra môi trường trong lúc
            duy trì nhiệt độ (steady-state holding loss), qua lớp cách nhiệt
            pack pin:
                Q_hold_loss = k_insulation * (T_target - T_ambient)
            k_insulation: hệ số tổn thất nhiệt của pack (W/K), pack pin có
            vỏ cách nhiệt + có thể đặt trong khoang gầm hở ra khí lạnh.

        Tổng công suất điện BTMS (giả định heater hiệu suất ~95%, một phần nhỏ
        tổn hao qua dây dẫn/điều khiển):
                P_btms_total = (P_raise + Q_hold_loss) / eta_heater
        """
        if current_battery_temp_c is None:
            # Giả định mặc định: nhiệt độ pin xấp xỉ nhiệt độ môi trường khi
            # xe đỗ qua đêm ngoài trời lạnh (cold-soaked battery)
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
    # C. SUY GIẢM HIỆU SUẤT PIN DO NHIỆT ĐỘ THẤP
    # ========================================================================
    @staticmethod
    def _linear_interp(x: float, xp: Tuple[float, ...], fp: Tuple[float, ...]) -> float:
        """Hàm nội suy tuyến tính từng đoạn thủ công (không phụ thuộc numpy),
        có clamp ở 2 đầu bảng tra cứu."""
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
        Trả về hệ số dung lượng hiệu dụng (0.0 - 1.0) của pin tại nhiệt độ
        cho trước, nội suy từ bảng thực nghiệm capacity_fade_lookup_temp_c /
        capacity_fade_factor.

        Ý nghĩa vật lý: ở nhiệt độ thấp, tốc độ khuếch tán ion Li+ trong chất
        điện giải và điện cực chậm lại, làm giảm dung lượng KHẢ DỤNG tức thời
        (không phải suy giảm vĩnh viễn - capacity hồi phục khi pin ấm lên).
        """
        return self._linear_interp(
            battery_temp_c, self.capacity_fade_lookup_temp_c, self.capacity_fade_factor
        )

    def compute_internal_resistance_multiplier(self, battery_temp_c: float) -> float:
        """
        Trả về hệ số nhân nội trở tại nhiệt độ pin cho trước, dùng để ước
        tính thêm tổn hao I^2*R khi pin hoạt động (sạc/xả) ở nhiệt độ thấp.
        """
        return self._linear_interp(
            battery_temp_c,
            self.capacity_fade_lookup_temp_c,
            self.internal_resistance_multiplier_lookup,
        )

    def compute_effective_usable_capacity_kwh(self, nominal_usable_capacity_kwh: float, battery_temp_c: float) -> float:
        """
        Dung lượng khả dụng hiệu dụng trong điều kiện lạnh:
            C_effective = C_nominal * capacity_fade_factor(T_battery)
        """
        factor = self.compute_capacity_fade_factor(battery_temp_c)
        return nominal_usable_capacity_kwh * factor

    # ========================================================================
    # D. TỔNG HỢP: HAO HỤT ĐIỆN NĂNG DO HVAC + BTMS THEO THỜI GIAN/QUÃNG ĐƯỜNG
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
        Tổng hợp toàn bộ hao hụt điện năng (kWh) do hệ thống HVAC cabin + BTMS
        pin trong suốt một chuyến đi, dựa trên thời lượng chuyến đi.

        Giả định đơn giản hoá hợp lý: công suất HVAC & BTMS gần như không đổi
        trong suốt chuyến đi ngắn/trung bình (vì nhiệt độ môi trường & mục
        tiêu cabin/pin không đổi) => Energy = Power * time.

        (Với chuyến đi rất dài qua nhiều vùng khí hậu khác nhau, nên chia nhỏ
        theo từng đoạn và gọi lại hàm này cho từng đoạn — xem route_optimizer.py.)
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

    print("=== TEST: Điều kiện -10°C, chuyến đi 45 phút, 40km ===")
    result = hvac_sim.compute_total_winter_energy_overhead(
        ambient_temp_c=-10.0,
        trip_duration_s=45 * 60,
        distance_km=40.0,
        initial_battery_temp_c=-8.0,
    )
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== TEST: So sánh hệ số suy giảm dung lượng theo nhiệt độ ===")
    for t in [-20, -10, -5, 0, 10, 20]:
        f = hvac_sim.compute_capacity_fade_factor(t)
        r = hvac_sim.compute_internal_resistance_multiplier(t)
        print(f"  T={t:>4}°C -> capacity_factor={f:.3f}, internal_resistance_x={r:.3f}")