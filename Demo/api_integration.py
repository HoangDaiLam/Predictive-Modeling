# -*- coding: utf-8 -*-
"""
================================================================================
MODULE: api_integration.py
BƯỚC 3 — Elevation & Weather API Integration
================================================================================
PATCH v2:
    - Thêm _http_post_with_retry() dùng cho Google Routes API (trước đây không retry).
    - fetch_current_temperature() trả thêm wind_direction_deg.
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
except ImportError:
    requests = None


class APIIntegrationError(Exception):
    pass


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


def _log_retry(api_name: str, attempt: int, message: str) -> None:
    print(f"[WARN][{api_name}] Lần thử {attempt}: {message}")


# ============================================================================
# GET helper (giữ nguyên từ v1)
# ============================================================================
def _http_get_with_retry(url: str, params: Dict[str, Any], config: APIConfig, api_name: str) -> Dict[str, Any]:
    if requests is None:
        raise APIIntegrationError(f"[{api_name}] Thư viện 'requests' chưa cài.")
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.max_retries + 2):
        try:
            response = requests.get(url, params=params, timeout=config.request_timeout_s)
        except requests.exceptions.Timeout as e:
            last_exception = e
            _log_retry(api_name, attempt, "Timeout khi gọi API")
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            _log_retry(api_name, attempt, "Lỗi kết nối mạng")
        except requests.exceptions.RequestException as e:
            last_exception = e
            _log_retry(api_name, attempt, f"Lỗi request: {e}")
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    raise APIIntegrationError(f"[{api_name}] Parse JSON fail: {e}")
            elif response.status_code == 429 or 500 <= response.status_code < 600:
                last_exception = APIIntegrationError(f"[{api_name}] HTTP {response.status_code}")
                _log_retry(api_name, attempt, f"HTTP {response.status_code}, retry...")
            else:
                raise APIIntegrationError(
                    f"[{api_name}] HTTP {response.status_code}: {response.text[:300]}"
                )
        if attempt <= config.max_retries:
            time.sleep(config.retry_backoff_s * attempt)

    raise APIIntegrationError(
        f"[{api_name}] Thất bại sau {config.max_retries + 1} lần. Last: {last_exception}"
    )


# ============================================================================
# PATCH: POST helper — retry tương tự GET
# ============================================================================
def _http_post_with_retry(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    config: APIConfig,
    api_name: str,
) -> Dict[str, Any]:
    """POST version — cùng logic retry + backoff + phân loại lỗi như GET."""
    if requests is None:
        raise APIIntegrationError(f"[{api_name}] Thư viện 'requests' chưa cài.")
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.max_retries + 2):
        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=config.request_timeout_s,
            )
        except requests.exceptions.Timeout as e:
            last_exception = e
            _log_retry(api_name, attempt, "Timeout khi gọi API")
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            _log_retry(api_name, attempt, "Lỗi kết nối mạng")
        except requests.exceptions.RequestException as e:
            last_exception = e
            _log_retry(api_name, attempt, f"Lỗi request: {e}")
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    raise APIIntegrationError(f"[{api_name}] Parse JSON fail: {e}")
            elif response.status_code == 429 or 500 <= response.status_code < 600:
                last_exception = APIIntegrationError(f"[{api_name}] HTTP {response.status_code}")
                _log_retry(api_name, attempt, f"HTTP {response.status_code}, retry...")
            else:
                raise APIIntegrationError(
                    f"[{api_name}] HTTP {response.status_code}: {response.text[:300]}"
                )
        if attempt <= config.max_retries:
            time.sleep(config.retry_backoff_s * attempt)

    raise APIIntegrationError(
        f"[{api_name}] Thất bại sau {config.max_retries + 1} lần. Last: {last_exception}"
    )


# ============================================================================
# 1. GOOGLE ROUTES API — dùng POST helper mới
# ============================================================================
def fetch_route_alternatives(
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float,
    config: APIConfig,
) -> List[Dict[str, Any]]:
    if not config.google_maps_api_key:
        raise APIIntegrationError(
            "Thiếu GOOGLE_MAPS_API_KEY. Vui lòng đặt biến môi trường."
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

    # PATCH: dùng helper có retry thay vì gọi trực tiếp
    data = _http_post_with_retry(url, headers, body, config, api_name="Google Routes API")

    routes = data.get("routes", [])
    if not routes:
        raise APIIntegrationError(
            "[Google Routes API] Không tìm thấy tuyến đường nào. Kiểm tra lại toạ độ."
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
    try:
        return float(str(duration_str).rstrip("s"))
    except (ValueError, AttributeError):
        return 0.0


# ============================================================================
# 2. GOOGLE ELEVATION API
# ============================================================================
def fetch_elevation_profile(
    path_coordinates: List[Tuple[float, float]],
    config: APIConfig,
    samples: int = 100,
) -> List[Dict[str, float]]:
    if not config.google_maps_api_key:
        raise APIIntegrationError("Thiếu GOOGLE_MAPS_API_KEY cho Elevation API.")
    if len(path_coordinates) < 2:
        raise APIIntegrationError("path_coordinates cần tối thiểu 2 điểm.")

    path_str = "|".join(f"{lat},{lng}" for lat, lng in path_coordinates)
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {"path": path_str, "samples": samples, "key": config.google_maps_api_key}

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
            if horizontal_dist_m > 1e-3:
                grade_angle_rad = math.atan(delta_elevation / horizontal_dist_m)
                grade_percent = (delta_elevation / horizontal_dist_m) * 100.0

        profile.append({
            "lat": lat, "lng": lng, "elevation_m": elevation_m,
            "grade_angle_rad": grade_angle_rad, "grade_percent": grade_percent,
        })
    return profile


def _haversine_distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R_EARTH_M = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_EARTH_M * c


# ============================================================================
# 3. OPENWEATHERMAP — PATCH: + wind_direction_deg
# ============================================================================
def fetch_current_temperature(
    lat: float,
    lng: float,
    config: APIConfig,
    units: str = "metric",
) -> Dict[str, Any]:
    if not config.openweathermap_api_key:
        raise APIIntegrationError("Thiếu OPENWEATHERMAP_API_KEY.")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat, "lon": lng,
        "appid": config.openweathermap_api_key,
        "units": units,
    }

    data = _http_get_with_retry(url, params, config, api_name="OpenWeatherMap API")

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
            "wind_direction_deg": data.get("wind", {}).get("deg", 0.0),   # PATCH
            "weather_description": weather_list[0].get("description", "") if weather_list else "",
            "raw_response": data,
        }
    except KeyError as e:
        raise APIIntegrationError(f"[OpenWeatherMap API] Thiếu trường: {e}")

    return result


def fetch_temperature_profile_along_route(
    sampled_points: List[Tuple[float, float]],
    config: APIConfig,
) -> List[Dict[str, Any]]:
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
            f"Chi tiết: {errors}"
        )
    if errors:
        print(f"[WARN] Một số điểm thất bại ({len(errors)}/{len(sampled_points)})")
    return results


# ============================================================================
# 4. POLYLINE DECODE
# ============================================================================
def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    index = lat = lng = 0
    length = len(encoded)
    while index < length:
        result = 1; shift = 0
        while True:
            b = ord(encoded[index]) - 63 - 1
            index += 1
            result += b << shift
            shift += 5
            if b < 0x1f:
                break
        lat += (~result >> 1) if (result & 1) != 0 else (result >> 1)

        result = 1; shift = 0
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
# 5. MOCK FALLBACK
# ============================================================================
def generate_mock_elevation_profile(
    origin_lat: float, origin_lng: float,
    destination_lat: float, destination_lng: float,
    num_points: int = 20,
    hill_amplitude_m: float = 80.0,
) -> List[Dict[str, float]]:
    profile = []
    total_dist = _haversine_distance_m(origin_lat, origin_lng, destination_lat, destination_lng)
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        lat = origin_lat + t * (destination_lat - origin_lat)
        lng = origin_lng + t * (destination_lng - origin_lng)
        elevation = (
            50.0
            + hill_amplitude_m * math.sin(t * math.pi * 3)
            + (hill_amplitude_m * 0.4) * math.sin(t * math.pi * 7 + 1.0)
        )
        profile.append({"lat": lat, "lng": lng, "elevation_m": elevation, "t_fraction": t})

    for i in range(len(profile) - 1):
        d_elev = profile[i + 1]["elevation_m"] - profile[i]["elevation_m"]
        d_horiz = (total_dist / (num_points - 1)) if num_points > 1 else 1.0
        profile[i]["grade_angle_rad"] = math.atan(d_elev / d_horiz) if d_horiz > 0 else 0.0
        profile[i]["grade_percent"] = (d_elev / d_horiz) * 100.0 if d_horiz > 0 else 0.0
    profile[-1]["grade_angle_rad"] = 0.0
    profile[-1]["grade_percent"] = 0.0
    return profile


if __name__ == "__main__":
    print("=== TEST: Mock elevation profile (Hà Nội → Hoà Bình) ===")
    mock_profile = generate_mock_elevation_profile(
        origin_lat=21.0285, origin_lng=105.8542,
        destination_lat=20.8133, destination_lng=105.3383,
        num_points=10,
    )
    for p in mock_profile:
        print(f"  lat={p['lat']:.4f}, elev={p['elevation_m']:.1f}m, grade={p.get('grade_percent', 0):.2f}%")

    print("\n=== TEST: Missing API key ===")
    cfg = APIConfig(google_maps_api_key=None, openweathermap_api_key=None)
    try:
        fetch_elevation_profile([(21.0, 105.8), (20.8, 105.3)], cfg)
    except APIIntegrationError as e:
        print(f"  Caught: {e}")