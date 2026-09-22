# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: api_integration.py
BƯỚC 3 — Elevation & Weather API Integration
================================================================================
Mục đích:
    Kết nối tới các API bản đồ/thời tiết bên ngoài để lấy:
        1. Tuyến đường (route) giữa điểm đi (origin) và điểm đến (destination)
           — dùng Google Routes API (hoặc Directions API cũ).
        2. Hồ sơ độ cao dọc tuyến đường (elevation profile) — dùng Google
           Elevation API, từ đó suy ra góc dốc theta cho từng đoạn đường.
        3. Nhiệt độ môi trường hiện tại tại khu vực tuyến đường đi qua —
           dùng OpenWeatherMap Current Weather API.

Yêu cầu API keys (đặt trong biến môi trường, KHÔNG hard-code trong source):
    GOOGLE_MAPS_API_KEY
    OPENWEATHERMAP_API_KEY

Ghi chú:
    - Toàn bộ hàm gọi API đều có cơ chế xử lý lỗi (try/except) đầy đủ:
      lỗi mạng, lỗi timeout, lỗi HTTP status, lỗi parse JSON, API trả về
      status lỗi nghiệp vụ (vd: ZERO_RESULTS, REQUEST_DENIED...).
    - Khi không có API key hoặc lỗi kết nối, các hàm sẽ raise một exception
      tuỳ biến (APIIntegrationError) kèm thông điệp rõ ràng, để lớp gọi
      phía trên (route_optimizer.py / dashboard.py) có thể fallback sang
      dữ liệu mô phỏng (mock data) hoặc thông báo cho người dùng.
    - File sample_api_response.json (cùng thư mục) minh hoạ cấu trúc JSON
      mẫu trả về từ các API này, dùng cho việc test không cần gọi mạng thật.
================================================================================
"""

from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # sẽ raise lỗi rõ ràng khi thực sự gọi API nếu thiếu thư viện


# ============================================================================
# EXCEPTION TUỲ BIẾN
# ============================================================================
class APIIntegrationError(Exception):
    """Exception chung cho mọi lỗi liên quan đến gọi API bên ngoài
    (network, timeout, HTTP status lỗi, API business-logic lỗi, thiếu key...)."""
    pass


# ============================================================================
# CẤU HÌNH API
# ============================================================================
@dataclass
class APIConfig:
    google_maps_api_key: Optional[str] = None
    openweathermap_api_key: Optional[str] = None
    request_timeout_s: float = 10.0
    max_retries: int = 2
    retry_backoff_s: float = 1.5

    def __post_init__(self):
        if self.google_maps_api_key is None:
            self.google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if self.openweathermap_api_key is None:
            self.openweathermap_api_key = os.environ.get("OPENWEATHERMAP_API_KEY")


# ============================================================================
# TIỆN ÍCH: GỌI HTTP CÓ RETRY + XỬ LÝ LỖI CHUẨN HOÁ
# ============================================================================
def _http_get_with_retry(url: str, params: Dict[str, Any], config: APIConfig, api_name: str) -> Dict[str, Any]:
    """
    Hàm helper dùng chung cho mọi lời gọi GET tới API bên ngoài, với:
        - Retry tự động (config.max_retries lần) khi gặp lỗi mạng/timeout
        - Backoff tăng dần giữa các lần retry
        - Chuẩn hoá mọi lỗi thành APIIntegrationError để lớp trên xử lý thống nhất
    """
    if requests is None:
        raise APIIntegrationError(
            f"[{api_name}] Thư viện 'requests' chưa được cài đặt. "
            f"Hãy chạy: pip install requests"
        )

    last_exception: Optional[Exception] = None

    for attempt in range(1, config.max_retries + 2):  # +1 lần gọi gốc + max_retries lần retry
        try:
            response = requests.get(url, params=params, timeout=config.request_timeout_s)
        except requests.exceptions.Timeout as e:
            last_exception = e
            _log_retry(api_name, attempt, "Timeout khi gọi API")
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            _log_retry(api_name, attempt, "Lỗi kết nối mạng (ConnectionError)")
        except requests.exceptions.RequestException as e:
            last_exception = e
            _log_retry(api_name, attempt, f"Lỗi request không xác định: {e}")
        else:
            # Có response, kiểm tra HTTP status code
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    raise APIIntegrationError(
                        f"[{api_name}] Không thể parse JSON từ response (HTTP 200). Lỗi: {e}"
                    )
            elif response.status_code in (429,):
                # Rate limit -> đáng để retry
                last_exception = APIIntegrationError(f"[{api_name}] HTTP 429 - Rate limited")
                _log_retry(api_name, attempt, "Bị giới hạn tần suất gọi (429), thử lại...")
            elif 500 <= response.status_code < 600:
                # Lỗi server -> đáng để retry
                last_exception = APIIntegrationError(f"[{api_name}] HTTP {response.status_code} - Lỗi server")
                _log_retry(api_name, attempt, f"Lỗi server API ({response.status_code}), thử lại...")
            else:
                # Lỗi client (400, 401, 403, 404...) -> KHÔNG retry, raise ngay
                raise APIIntegrationError(
                    f"[{api_name}] HTTP {response.status_code}: {response.text[:300]}"
                )

        if attempt <= config.max_retries:
            time.sleep(config.retry_backoff_s * attempt)

    raise APIIntegrationError(
        f"[{api_name}] Gọi API thất bại sau {config.max_retries + 1} lần thử. "
        f"Lỗi cuối cùng: {last_exception}"
    )


def _log_retry(api_name: str, attempt: int, message: str) -> None:
    print(f"[WARN][{api_name}] Lần thử {attempt}: {message}")


# ============================================================================
# 1. GOOGLE ROUTES API — LẤY CÁC TUYẾN ĐƯỜNG THAY THẾ (ALTERNATIVES)
# ============================================================================
def fetch_route_alternatives(
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float,
    config: APIConfig,
) -> List[Dict[str, Any]]:
    """
    Gọi Google Routes API (computeRoutes) để lấy DANH SÁCH các tuyến đường
    thay thế (alternative routes) giữa 2 toạ độ, mỗi tuyến gồm:
        - polyline (chuỗi điểm toạ độ dọc tuyến)
        - distance_m (tổng quãng đường, mét)
        - duration_s (thời gian di chuyển ước tính, giây)

    API endpoint tham khảo: POST https://routes.googleapis.com/directions/v2:computeRoutes

    Returns:
        List các dict, mỗi dict mô tả 1 tuyến đường (đã chuẩn hoá cấu trúc).

    Raises:
        APIIntegrationError nếu thiếu API key, lỗi mạng, hoặc API trả về lỗi.
    """
    if not config.google_maps_api_key:
        raise APIIntegrationError(
            "Thiếu GOOGLE_MAPS_API_KEY. Vui lòng đặt biến môi trường hoặc "
            "truyền vào APIConfig(google_maps_api_key=...)."
        )

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": config.google_maps_api_key,
        "X-Goog-FieldMask": (
            "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,"
            "routes.legs.steps.distanceMeters,routes.legs.steps.staticDuration"
        ),
    }
    body = {
        "origin": {"location": {"latLng": {"latitude": origin_lat, "longitude": origin_lng}}},
        "destination": {"location": {"latLng": {"latitude": destination_lat, "longitude": destination_lng}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "computeAlternativeRoutes": True,
        "languageCode": "vi-VN",
        "units": "METRIC",
    }

    if requests is None:
        raise APIIntegrationError("Thư viện 'requests' chưa được cài đặt.")

    try:
        response = requests.post(url, headers=headers, json=body, timeout=config.request_timeout_s)
    except requests.exceptions.RequestException as e:
        raise APIIntegrationError(f"[Google Routes API] Lỗi kết nối: {e}")

    if response.status_code != 200:
        raise APIIntegrationError(
            f"[Google Routes API] HTTP {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        raise APIIntegrationError(f"[Google Routes API] Lỗi parse JSON: {e}")

    routes = data.get("routes", [])
    if not routes:
        raise APIIntegrationError(
            "[Google Routes API] Không tìm thấy tuyến đường nào (routes rỗng). "
            "Kiểm tra lại toạ độ origin/destination."
        )

    normalized_routes = []
    for idx, r in enumerate(routes):
        normalized_routes.append({
            "route_index": idx,
            "distance_m": r.get("distanceMeters", 0),
            "duration_s": _parse_google_duration(r.get("duration", "0s")),
            "encoded_polyline": r.get("polyline", {}).get("encodedPolyline", ""),
        })

    return normalized_routes


def _parse_google_duration(duration_str: str) -> float:
    """Google Routes API trả về duration dạng chuỗi '1234s'. Hàm này parse
    thành số giây (float)."""
    try:
        return float(str(duration_str).rstrip("s"))
    except (ValueError, AttributeError):
        return 0.0


# ============================================================================
# 2. GOOGLE ELEVATION API — LẤY HỒ SƠ ĐỘ CAO DỌC TUYẾN ĐƯỜNG
# ============================================================================
def fetch_elevation_profile(
    path_coordinates: List[Tuple[float, float]],
    config: APIConfig,
    samples: int = 100,
) -> List[Dict[str, float]]:
    """
    Gọi Google Elevation API để lấy hồ sơ độ cao (elevation profile) dọc
    theo một tuyến đường, từ danh sách toạ độ path (lat, lng).

    API endpoint: GET https://maps.googleapis.com/maps/api/elevation/json
        ?path=lat1,lng1|lat2,lng2|...&samples=N&key=API_KEY

    Sau khi có elevation tại N điểm lấy mẫu cách đều nhau dọc tuyến đường,
    ta tính GÓC DỐC theta giữa 2 điểm mẫu liên tiếp:

        delta_elevation = elevation[i+1] - elevation[i]
        delta_horizontal = haversine_distance(point[i], point[i+1])
        theta_i = arctan(delta_elevation / delta_horizontal)

    Args:
        path_coordinates: danh sách tối thiểu 2 toạ độ (lat, lng) định nghĩa
                           đường path (thường lấy từ polyline đã decode)
        samples: số điểm lấy mẫu độ cao dọc tuyến (Google Elevation API path
                 request lấy mẫu đều dọc theo path)

    Returns:
        List các dict {lat, lng, elevation_m, grade_angle_rad, grade_percent}
        (grade_angle_rad/grade_percent của điểm i tính so với điểm i+1)

    Raises:
        APIIntegrationError nếu thiếu key, lỗi mạng, hoặc API trả lỗi.
    """
    if not config.google_maps_api_key:
        raise APIIntegrationError(
            "Thiếu GOOGLE_MAPS_API_KEY cho Elevation API."
        )
    if len(path_coordinates) < 2:
        raise APIIntegrationError("path_coordinates cần tối thiểu 2 điểm.")

    path_str = "|".join(f"{lat},{lng}" for lat, lng in path_coordinates)
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "path": path_str,
        "samples": samples,
        "key": config.google_maps_api_key,
    }

    data = _http_get_with_retry(url, params, config, api_name="Google Elevation API")

    status = data.get("status")
    if status != "OK":
        error_message = data.get("error_message", "Không có thông tin lỗi chi tiết.")
        raise APIIntegrationError(
            f"[Google Elevation API] status={status}. Chi tiết: {error_message}"
        )

    results = data.get("results", [])
    if not results:
        raise APIIntegrationError("[Google Elevation API] results rỗng.")

    profile: List[Dict[str, float]] = []
    for i, point in enumerate(results):
        lat = point["location"]["lat"]
        lng = point["location"]["lng"]
        elevation_m = point["elevation"]

        grade_angle_rad = 0.0
        grade_percent = 0.0

        if i < len(results) - 1:
            next_point = results[i + 1]
            next_elevation = next_point["elevation"]
            horizontal_dist_m = _haversine_distance_m(
                lat, lng, next_point["location"]["lat"], next_point["location"]["lng"]
            )
            delta_elevation = next_elevation - elevation_m

            if horizontal_dist_m > 1e-3:  # tránh chia cho 0 khi 2 điểm trùng nhau
                grade_angle_rad = math.atan(delta_elevation / horizontal_dist_m)
                grade_percent = (delta_elevation / horizontal_dist_m) * 100.0

        profile.append({
            "lat": lat,
            "lng": lng,
            "elevation_m": elevation_m,
            "grade_angle_rad": grade_angle_rad,
            "grade_percent": grade_percent,
        })

    return profile


def _haversine_distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Công thức Haversine tính khoảng cách cung tròn (great-circle distance)
    giữa 2 toạ độ địa lý trên bề mặt Trái Đất (xấp xỉ hình cầu):

        a = sin^2(deltaLat/2) + cos(lat1)*cos(lat2)*sin^2(deltaLng/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        d = R_earth * c

    R_earth = 6,371,000 m (bán kính trung bình Trái Đất)
    """
    R_EARTH_M = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_EARTH_M * c


# ============================================================================
# 3. OPENWEATHERMAP API — LẤY NHIỆT ĐỘ MÔI TRƯỜNG HIỆN TẠI
# ============================================================================
def fetch_current_temperature(
    lat: float,
    lng: float,
    config: APIConfig,
    units: str = "metric",
) -> Dict[str, Any]:
    """
    Gọi OpenWeatherMap Current Weather API để lấy nhiệt độ môi trường hiện
    tại tại một toạ độ cụ thể (thường gọi tại điểm đi, điểm đến, và một vài
    điểm trung gian dọc tuyến để có bức tranh nhiệt độ đầy đủ hơn).

    API endpoint: GET https://api.openweathermap.org/data/2.5/weather
        ?lat={lat}&lon={lng}&appid={API_KEY}&units=metric

    Returns:
        Dict {temperature_c, feels_like_c, humidity_pct, wind_speed_mps,
              weather_description, raw_response}

    Raises:
        APIIntegrationError nếu thiếu key, lỗi mạng, hoặc API trả lỗi.
    """
    if not config.openweathermap_api_key:
        raise APIIntegrationError(
            "Thiếu OPENWEATHERMAP_API_KEY. Vui lòng đặt biến môi trường hoặc "
            "truyền vào APIConfig(openweathermap_api_key=...)."
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lng,
        "appid": config.openweathermap_api_key,
        "units": units,
    }

    data = _http_get_with_retry(url, params, config, api_name="OpenWeatherMap API")

    # OpenWeatherMap dùng mã lỗi "cod" (có thể là int hoặc str tuỳ trường hợp)
    cod = str(data.get("cod", ""))
    if cod not in ("200",):
        error_message = data.get("message", "Không có thông tin lỗi chi tiết.")
        raise APIIntegrationError(f"[OpenWeatherMap API] cod={cod}. Chi tiết: {error_message}")

    try:
        main = data["main"]
        weather_list = data.get("weather", [{}])
        result = {
            "temperature_c": main["temp"],
            "feels_like_c": main.get("feels_like"),
            "humidity_pct": main.get("humidity"),
            "wind_speed_mps": data.get("wind", {}).get("speed"),
            "weather_description": weather_list[0].get("description", "") if weather_list else "",
            "raw_response": data,
        }
    except KeyError as e:
        raise APIIntegrationError(f"[OpenWeatherMap API] Thiếu trường dữ liệu bắt buộc: {e}")

    return result


def fetch_temperature_profile_along_route(
    sampled_points: List[Tuple[float, float]],
    config: APIConfig,
) -> List[Dict[str, Any]]:
    """
    Lấy nhiệt độ tại NHIỀU điểm dọc tuyến đường (ví dụ: điểm đi, 1-2 điểm
    trung gian, điểm đến) để mô hình hoá thay đổi nhiệt độ theo hành trình
    (ví dụ: đi từ đồng bằng ấm hơn lên vùng núi lạnh hơn).

    Nếu một điểm bị lỗi khi gọi API, hàm KHÔNG dừng toàn bộ mà ghi nhận lỗi
    cho điểm đó và tiếp tục các điểm còn lại (graceful degradation), miễn là
    có ít nhất 1 điểm lấy được dữ liệu thành công.
    """
    results = []
    errors = []

    for (lat, lng) in sampled_points:
        try:
            temp_data = fetch_current_temperature(lat, lng, config)
            results.append({"lat": lat, "lng": lng, **temp_data})
        except APIIntegrationError as e:
            errors.append({"lat": lat, "lng": lng, "error": str(e)})

    if not results:
        raise APIIntegrationError(
            f"[OpenWeatherMap API] Không lấy được nhiệt độ tại BẤT KỲ điểm nào. "
            f"Chi tiết lỗi: {errors}"
        )

    if errors:
        print(f"[WARN] Một số điểm lấy nhiệt độ thất bại ({len(errors)}/{len(sampled_points)}): {errors}")

    return results


# ============================================================================
# 4. GIẢI MÃ POLYLINE (Google Encoded Polyline Algorithm Format)
# ============================================================================
def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    """
    Giải mã chuỗi polyline đã mã hoá (Google Encoded Polyline Algorithm
    Format) thành danh sách toạ độ (lat, lng).

    Thuật toán tiêu chuẩn của Google, xử lý theo từng byte 5-bit với cờ
    "continuation bit", áp dụng zig-zag decoding cho số âm.
    """
    points: List[Tuple[float, float]] = []
    index = lat = lng = 0
    length = len(encoded)

    while index < length:
        # Giải mã delta latitude
        result = 1
        shift = 0
        while True:
            b = ord(encoded[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        lat += (~result >> 1) if (result & 1) != 0 else (result >> 1)

        # Giải mã delta longitude
        result = 1
        shift = 0
        while True:
            b = ord(encoded[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        lng += (~result >> 1) if (result & 1) != 0 else (result >> 1)

        points.append((lat * 1e-5, lng * 1e-5))

    return points


# ============================================================================
# 5. MOCK DATA / FALLBACK — dùng khi không có API key hoặc để test offline
# ============================================================================
def generate_mock_elevation_profile(
    origin_lat: float, origin_lng: float,
    destination_lat: float, destination_lng: float,
    num_points: int = 20,
    hill_amplitude_m: float = 80.0,
) -> List[Dict[str, float]]:
    """
    Sinh dữ liệu độ cao MÔ PHỎNG (không gọi API thật) để phục vụ demo/test
    khi chưa cấu hình API key. Dùng hàm sin tổng hợp để tạo địa hình đồi núi
    giả lập hợp lý dọc theo đường thẳng nối origin -> destination.
    """
    profile = []
    total_dist = _haversine_distance_m(origin_lat, origin_lng, destination_lat, destination_lng)

    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        lat = origin_lat + t * (destination_lat - origin_lat)
        lng = origin_lng + t * (destination_lng - origin_lng)
        # Địa hình giả lập: tổ hợp 2 tần số sin để tạo đồi/dốc tự nhiên hơn
        elevation = (
            50.0
            + hill_amplitude_m * math.sin(t * math.pi * 3)
            + (hill_amplitude_m * 0.4) * math.sin(t * math.pi * 7 + 1.0)
        )
        profile.append({"lat": lat, "lng": lng, "elevation_m": elevation, "t_fraction": t})

    # Tính góc dốc giữa các điểm liên tiếp
    for i in range(len(profile) - 1):
        d_elev = profile[i + 1]["elevation_m"] - profile[i]["elevation_m"]
        d_horiz = (total_dist / (num_points - 1)) if num_points > 1 else 1.0
        profile[i]["grade_angle_rad"] = math.atan(d_elev / d_horiz) if d_horiz > 0 else 0.0
        profile[i]["grade_percent"] = (d_elev / d_horiz) * 100.0 if d_horiz > 0 else 0.0
    profile[-1]["grade_angle_rad"] = 0.0
    profile[-1]["grade_percent"] = 0.0

    return profile


# ============================================================================
# KHỐI TỰ KIỂM TRA (chạy offline bằng mock data, KHÔNG cần API key thật)
# ============================================================================
if __name__ == "__main__":
    print("=== TEST: Mock elevation profile (Hà Nội -> Hoà Bình, offline) ===")
    mock_profile = generate_mock_elevation_profile(
        origin_lat=21.0285, origin_lng=105.8542,   # Hà Nội
        destination_lat=20.8133, destination_lng=105.3383,  # Hoà Bình
        num_points=10,
    )
    for p in mock_profile:
        print(f"  lat={p['lat']:.4f}, lng={p['lng']:.4f}, elev={p['elevation_m']:.1f}m, "
              f"grade={p.get('grade_percent', 0):.2f}%")

    print("\n=== TEST: Xử lý lỗi khi thiếu API key ===")
    cfg = APIConfig(google_maps_api_key=None, openweathermap_api_key=None)
    try:
        fetch_elevation_profile([(21.0, 105.8), (20.8, 105.3)], cfg)
    except APIIntegrationError as e:
        print(f"  Bắt lỗi thành công (đúng như kỳ vọng): {e}")

    print("\n=== TEST: Giải mã polyline mẫu ===")
    sample_polyline = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
    decoded = decode_polyline(sample_polyline)
    print(f"  Số điểm giải mã: {len(decoded)}")
    print(f"  Điểm đầu: {decoded[0]}")