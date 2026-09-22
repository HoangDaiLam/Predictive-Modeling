# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: vehicle_dynamics.py
BƯỚC 1 — Vehicle Dynamics & Energy Physics Engine
================================================================================
Mục đích:
    Mô phỏng chính xác động lực học và tiêu thụ năng lượng của xe điện
    Polestar 4 Single Motor (RWD), chế độ lái Standard Mode.

Tác giả vai trò: EV Dynamics Engineer (mô phỏng theo yêu cầu người dùng)

Ghi chú kỹ thuật:
    - Toàn bộ công thức vật lý được giữ nguyên, KHÔNG rút gọn.
    - Đơn vị chuẩn hoá theo hệ SI (mét, kg, giây, Newton, Watt, Joule),
      chỉ quy đổi sang kWh khi xuất kết quả cuối cùng cho người dùng.
    - Các thông số xe được đặt làm giá trị mặc định có thể cấu hình lại
      (constructor cho phép override toàn bộ), vì số liệu nhà sản xuất
      có thể thay đổi theo phiên bản/thị trường.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ------------------------------------------------------------------------
# HẰNG SỐ VẬT LÝ TOÀN CỤC
# ------------------------------------------------------------------------
GRAVITY: float = 9.80665            # gia tốc trọng trường chuẩn (m/s^2)
AIR_DENSITY_SEA_LEVEL: float = 1.225  # khối lượng riêng không khí ở 15°C, mực nước biển (kg/m^3)


def air_density_at_temperature(temp_celsius: float, pressure_pa: float = 101325.0) -> float:
    """
    Tính khối lượng riêng không khí (rho) theo nhiệt độ thực tế, dùng phương
    trình khí lý tưởng: rho = P / (R_specific * T)

    Điều này QUAN TRỌNG trong mùa đông vì không khí lạnh đặc hơn không khí ấm,
    làm tăng lực cản khí động học (F_aero) so với giả định rho = 1.225 kg/m^3
    (vốn chỉ đúng ở 15°C).

    Args:
        temp_celsius: nhiệt độ môi trường (°C)
        pressure_pa: áp suất khí quyển (Pa), mặc định 1 atm

    Returns:
        rho (kg/m^3)
    """
    R_SPECIFIC_AIR = 287.058  # hằng số khí riêng của không khí khô (J/(kg·K))
    temp_kelvin = temp_celsius + 273.15
    rho = pressure_pa / (R_SPECIFIC_AIR * temp_kelvin)
    return rho


@dataclass
class Polestar4Dynamics:
    """
    Class OOP đại diện cho toàn bộ thông số vật lý & mô hình động lực học
    của Polestar 4 Single Motor (Long Range Single Motor, RWD).

    Thông số mặc định dựa trên số liệu công bố gần đúng của nhà sản xuất
    cho phiên bản Single Motor (có thể hiệu chỉnh qua constructor nếu có
    số liệu chính xác hơn từ Polestar cho thị trường/model-year cụ thể).
    """

    # ---------------- Thông số khối lượng & hình học ----------------
    curb_mass_kg: float = 2350.0          # khối lượng bản thân xe (kg)
    payload_kg: float = 150.0             # tải trọng bổ sung (người + hành lý), mặc định 2 người
    frontal_area_m2: float = 2.75         # diện tích mặt cắt ngang chính diện A (m^2)
    drag_coefficient_cd: float = 0.27     # hệ số cản gió Cd (Single Motor, không active spoiler)
    wheel_radius_m: float = 0.36          # bán kính bánh xe (tương ứng lốp 20-21 inch) (m)
    rolling_resistance_coeff: float = 0.009  # hệ số cản lăn Crr (lốp EV low rolling resistance)

    # ---------------- Thông số hệ truyền động (Single Motor) ----------------
    motor_peak_power_kw: float = 200.0     # công suất định mức động cơ sau (kW)
    drivetrain_efficiency: float = 0.91    # hiệu suất truyền động tổng hợp
                                            # (inverter + motor + hộp số 1 cấp + vi sai)
    max_regen_power_kw: float = 150.0      # công suất phanh tái sinh tối đa

    # ---------------- Thông số pin ----------------
    battery_gross_capacity_kwh: float = 94.0   # dung lượng pin tổng (gross)
    battery_usable_capacity_kwh: float = 92.0  # dung lượng pin khả dụng (usable, sau buffer BMS)
    battery_nominal_voltage_v: float = 400.0   # điện áp danh định hệ thống pin

    # ---------------- Hiệu suất phụ trợ hệ thống điện ----------------
    aux_power_baseline_kw: float = 0.45    # công suất phụ tải cơ bản 12V/hệ thống điện tử
                                            # (đèn, màn hình, bơm nước làm mát cơ bản, ECU...)
    drag_coefficient_regen_efficiency: float = 0.70
    # ^ hiệu suất chuyển đổi năng lượng động năng -> điện năng nạp lại pin khi phanh tái sinh
    #   (đã tính đến tổn hao qua motor hoạt động như máy phát + inverter + BMS charge losses)

    def __post_init__(self):
        self.total_mass_kg: float = self.curb_mass_kg + self.payload_kg

    # ====================================================================
    # 1. LỰC CẢN KHÔNG KHÍ (Aerodynamic Drag Force)
    # ====================================================================
    def compute_aero_drag_force(self, velocity_mps: float, air_density: float = AIR_DENSITY_SEA_LEVEL) -> float:
        """
        Công thức lực cản khí động học:
            F_aero = 0.5 * rho * Cd * A * v^2

        Trong đó:
            rho  : khối lượng riêng không khí (kg/m^3) — dùng air_density_at_temperature()
                   khi mô phỏng mùa đông để có kết quả chính xác hơn.
            Cd   : hệ số cản gió (không thứ nguyên)
            A    : diện tích mặt cắt ngang (m^2)
            v    : vận tốc tương đối giữa xe và không khí (m/s)
                   (giả định không có gió ngang/ngược, v = vận tốc xe)

        Lưu ý: lực này luôn cản trở chuyển động (luôn dương khi v > 0),
        và tăng theo BÌNH PHƯƠNG vận tốc — đây là lý do tốc độ cao tiêu
        hao pin nhanh hơn nhiều so với tốc độ trung bình.
        """
        v = abs(velocity_mps)
        f_aero = 0.5 * air_density * self.drag_coefficient_cd * self.frontal_area_m2 * (v ** 2)
        return f_aero

    # ====================================================================
    # 2. LỰC MA SÁT LĂN (Rolling Resistance Force)
    # ====================================================================
    def compute_rolling_resistance_force(self, grade_angle_rad: float = 0.0) -> float:
        """
        Công thức lực cản lăn:
            F_roll = Crr * m * g * cos(theta)

        Trong đó:
            Crr   : hệ số cản lăn của lốp xe
            m     : tổng khối lượng xe (kg) = curb_mass + payload
            g     : gia tốc trọng trường (m/s^2)
            theta : góc dốc của mặt đường (rad). cos(theta) hiệu chỉnh thành
                    phần trọng lực VUÔNG GÓC với mặt đường (áp lực pháp
                    tuyến lên lốp), vì lực ma sát lăn tỉ lệ với lực pháp tuyến
                    chứ không phải trọng lượng toàn phần.

        Ở góc dốc nhỏ (đường thực tế thường < 15°), cos(theta) ≈ 1, nhưng
        ta vẫn giữ đầy đủ công thức để đảm bảo độ chính xác khi mô phỏng
        đèo dốc lớn.
        """
        f_roll = (
            self.rolling_resistance_coeff
            * self.total_mass_kg
            * GRAVITY
            * math.cos(grade_angle_rad)
        )
        return f_roll

    # ====================================================================
    # 3. LỰC LEO DỐC (Grade / Gravitational Force)
    # ====================================================================
    def compute_grade_force(self, grade_angle_rad: float) -> float:
        """
        Công thức lực trọng trường theo phương dốc:
            F_grade = m * g * sin(theta)

        theta > 0  => đường lên dốc (xe phải sinh thêm lực đẩy, tốn năng lượng)
        theta < 0  => đường xuống dốc (trọng lực hỗ trợ xe, có thể tái sinh)

        theta được tính từ độ dốc phần trăm (grade %) hoặc từ dữ liệu độ cao
        (elevation) chia cho khoảng cách ngang, thông qua hàm tiện ích
        grade_percent_to_radians() bên dưới.
        """
        f_grade = self.total_mass_kg * GRAVITY * math.sin(grade_angle_rad)
        return f_grade

    @staticmethod
    def grade_percent_to_radians(grade_percent: float) -> float:
        """
        Quy đổi độ dốc theo phần trăm (grade %, ví dụ Google Maps/GPS dùng)
        sang góc radian.

        Định nghĩa grade %:  grade% = (delta_elevation / delta_horizontal_distance) * 100
        => tan(theta) = grade% / 100
        => theta = arctan(grade% / 100)
        """
        return math.atan(grade_percent / 100.0)

    # ====================================================================
    # 4. LỰC GIA TỐC / ĐỘNG NĂNG (Inertial / Acceleration Force)
    # ====================================================================
    def compute_acceleration_force(self, acceleration_mps2: float) -> float:
        """
        Định luật II Newton:
            F_accel = m * a

        a > 0: xe đang tăng tốc (tiêu thụ thêm năng lượng để tích luỹ động năng)
        a < 0: xe đang giảm tốc (động năng được giải phóng, có thể thu hồi
               một phần qua phanh tái sinh — xem compute_regenerative_energy())

        Ngoài ra ta có thể tính trực tiếp biến thiên động năng:
            delta_KE = 0.5 * m * (v_end^2 - v_start^2)
        Hàm compute_kinetic_energy_delta() bên dưới dùng công thức này để
        kiểm chứng chéo (cross-check) với tích phân lực theo quãng đường.
        """
        f_accel = self.total_mass_kg * acceleration_mps2
        return f_accel

    def compute_kinetic_energy_delta(self, v_start_mps: float, v_end_mps: float) -> float:
        """
        Biến thiên động năng: delta_KE = 0.5 * m * (v_end^2 - v_start^2)   [Joule]
        """
        return 0.5 * self.total_mass_kg * (v_end_mps ** 2 - v_start_mps ** 2)

    # ====================================================================
    # 5. TỔNG HỢP LỰC & CÔNG SUẤT TỨC THỜI
    # ====================================================================
    def compute_total_tractive_force(
        self,
        velocity_mps: float,
        acceleration_mps2: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
    ) -> Dict[str, float]:
        """
        Tổng hợp lực kéo cần thiết tại bánh xe:
            F_total = F_aero + F_roll + F_grade + F_accel

        Đây là phương trình cân bằng lực dọc trục xe theo mô hình
        "point-mass longitudinal vehicle dynamics" tiêu chuẩn trong
        ngành kỹ thuật ô tô điện.

        Trả về dict chi tiết từng thành phần lực để phục vụ phân tích/debug.
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
        Công suất tại bánh xe (trước khi qua hiệu suất truyền động):
            P_wheel = F_total * v

        Nếu P_wheel > 0: xe cần lực kéo (motor hoạt động ở chế độ driving)
        Nếu P_wheel < 0: xe cần lực hãm (motor có thể hoạt động ở chế độ
                          phanh tái sinh — xem compute_regenerative_energy())
        """
        forces = self.compute_total_tractive_force(velocity_mps, acceleration_mps2, grade_angle_rad, air_density)
        p_wheel = forces["f_total_N"] * velocity_mps
        return p_wheel

    # ====================================================================
    # 6. CÔNG SUẤT PIN (đã tính hiệu suất truyền động + phanh tái sinh)
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
        Chuyển công suất bánh xe (P_wheel) thành công suất rút ra từ pin
        (P_battery), có phân biệt 2 chế độ vận hành:

        (a) CHẾ ĐỘ KÉO (Driving, P_wheel >= 0):
            P_battery = P_wheel / eta_drivetrain + P_aux
            (chia cho hiệu suất vì có tổn hao qua motor + inverter + hộp số)

        (b) CHẾ ĐỘ PHANH TÁI SINH (Regenerative Braking, P_wheel < 0):
            P_battery = P_wheel * eta_regen_total + P_aux
            (nhân với hiệu suất vì đây là dòng năng lượng NẠP LẠI pin,
             luôn có tổn hao trong quá trình chuyển đổi ngược động năng
             -> điện năng; giới hạn bởi max_regen_power_kw của motor/inverter)

        P_aux: phụ tải điện cơ bản (không tính HVAC — HVAC được cộng riêng
               ở module winter_hvac.py để tách bạch vật lý thuần tuý xe
               khỏi hệ thống nhiệt).

        Trả về công suất dương = pin đang XẢ (tiêu thụ),
        công suất âm = pin đang SẠC (thu hồi năng lượng).
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
    # 7. MÔ PHỎNG TIÊU THỤ NĂNG LƯỢNG THEO CHU TRÌNH VẬN TỐC (DRIVE CYCLE)
    # ====================================================================
    def simulate_drive_cycle(
        self,
        time_s: List[float],
        velocity_profile_mps: List[float],
        grade_profile_rad: Optional[List[float]] = None,
        air_density_profile: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Tích phân số (numerical integration) công suất pin theo thời gian
        để tính tổng năng lượng tiêu thụ trên một chu trình lái (drive cycle)
        rời rạc hoá, dùng phương pháp hình thang (trapezoidal rule) đơn giản
        hoá thành hình chữ nhật (rectangular/Euler) trên từng bước dt:

            E_battery = sum( P_battery(t_i) * dt_i )   với dt_i = t_(i+1) - t_i

        Gia tốc tức thời tại mỗi bước được xấp xỉ:
            a_i = (v_(i+1) - v_i) / dt_i

        Args:
            time_s: danh sách mốc thời gian (giây), tăng dần
            velocity_profile_mps: vận tốc xe tại từng mốc thời gian (m/s)
            grade_profile_rad: góc dốc tại từng mốc (rad); mặc định 0 (đường bằng)
            air_density_profile: rho tại từng mốc (kg/m^3); mặc định chuẩn mực nước biển

        Returns:
            Dict chứa tổng năng lượng tiêu thụ (kWh), năng lượng tái sinh (kWh),
            quãng đường (km), và phần trăm pin tiêu thụ (% of usable capacity).
        """
        n = len(time_s)
        assert n == len(velocity_profile_mps), "time_s và velocity_profile_mps phải cùng độ dài"

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
    # 8. TÍNH NĂNG LƯỢNG CHO MỘT ĐOẠN ĐƯỜNG ĐƠN GIẢN HOÁ (steady-state segment)
    # ====================================================================
    def compute_segment_energy_kwh(
        self,
        distance_m: float,
        avg_speed_mps: float,
        grade_angle_rad: float,
        air_density: float = AIR_DENSITY_SEA_LEVEL,
    ) -> float:
        """
        Ước tính năng lượng tiêu thụ cho MỘT ĐOẠN ĐƯỜNG ở tốc độ trung bình
        ổn định (steady-state, a = 0) — dùng cho bài toán tối ưu tuyến đường
        (route optimization) ở Bước 4, nơi ta có dữ liệu độ dốc & khoảng
        cách theo từng đoạn (segment) từ API bản đồ, nhưng không có chu trình
        vận tốc chi tiết theo giây.

            time_s = distance_m / avg_speed_mps
            E = P_battery(steady-state) * time_s

        Đây là mô hình steady-state hợp lệ vì giả định xe di chuyển ở tốc độ
        trung bình ổn định trên đoạn đường đó (a=0), phù hợp cho đoạn cao tốc/
        tỉnh lộ dài. Với đoạn đô thị nhiều dừng-đi, nên dùng simulate_drive_cycle().
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