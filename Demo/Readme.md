# Polestar 4 Winter Energy AI Engine

AI Engine + Dashboard dự đoán và tối ưu hoá tiêu thụ pin cho **Polestar 4
Single Motor (Standard Mode)** trong điều kiện **mùa đông lạnh giá** (nhiệt độ
âm, cần sưởi cabin + sưởi pin).

## 1. Kiến trúc hệ thống

```
polestar4_ai_engine/
├── vehicle_dynamics.py     # BƯỚC 1 — Vật lý xe & tiêu thụ năng lượng (Polestar4Dynamics)
├── winter_hvac.py          # BƯỚC 2 — HVAC cabin + BTMS pin mùa đông (WinterHVACSimulation)
├── api_integration.py      # BƯỚC 3 — Kết nối Google Maps Elevation/Routes API + OpenWeatherMap
├── route_optimizer.py      # BƯỚC 4 — AI Engine trung tâm: so sánh & tối ưu tuyến đường
├── dashboard.py            # BƯỚC 5 — Dashboard giao diện (Streamlit)
├── sample_api_response.json # Dữ liệu JSON mẫu minh hoạ cấu trúc API
├── requirements.txt
└── README.md
```

**Luồng dữ liệu:**

```
[Người dùng] --(origin, destination, nhiệt độ)--> [dashboard.py]
                                                        |
                                                        v
                                          [route_optimizer.py] (AI Engine)
                                          /              |              \
                                         v                v               v
                          [vehicle_dynamics.py]  [winter_hvac.py]  [api_integration.py]
                          (lực/công suất/năng     (HVAC cabin +     (elevation, weather,
                           lượng tiêu thụ)         BTMS pin)         route alternatives)
                                                        |
                                                        v
                                          [Kết quả: bảng so sánh tuyến
                                           + tuyến tối ưu + giải thích]
                                                        |
                                                        v
                                              [dashboard.py hiển thị]
```

Dashboard **import trực tiếp** các module Python backend trong cùng một tiến
trình (in-process function call) — không cần chạy một server API riêng cho
demo cơ bản.

## 2. Cài đặt

Yêu cầu Python 3.9 trở lên.

```bash
# 1. Tạo virtual environment (khuyến nghị)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Cài thư viện
pip install -r requirements.txt
```

## 3. Cấu hình API keys (tuỳ chọn)

Nếu muốn dùng dữ liệu độ dốc/thời tiết THẬT thay vì dữ liệu mô phỏng (mock),
đặt 2 biến môi trường sau trước khi chạy:

```bash
export GOOGLE_MAPS_API_KEY="your_google_maps_api_key"
export OPENWEATHERMAP_API_KEY="your_openweathermap_api_key"
```

- Google Maps API key: cần bật **Elevation API** và **Routes API** trong
  Google Cloud Console (https://console.cloud.google.com/apis/library).
- OpenWeatherMap API key: đăng ký miễn phí tại https://openweathermap.org/api.

> Nếu KHÔNG cấu hình key, hệ thống vẫn chạy được bình thường bằng dữ liệu mô
> phỏng (`generate_mock_elevation_profile`) — phù hợp để demo/test offline.

## 4. Chạy từng module riêng lẻ (kiểm tra logic backend)

Mỗi module có khối tự kiểm tra (`if __name__ == "__main__":`) để chạy độc
lập và xem kết quả console:

```bash
python3 vehicle_dynamics.py    # Test tính lực/công suất/năng lượng xe
python3 winter_hvac.py         # Test công suất HVAC/BTMS theo nhiệt độ
python3 api_integration.py     # Test gọi API (dùng mock nếu chưa có key)
python3 route_optimizer.py     # Test so sánh & tối ưu 2 tuyến đường mẫu
```

## 5. Chạy Dashboard

```bash
streamlit run dashboard.py
```

Sau khi chạy, mở trình duyệt tại địa chỉ Streamlit in ra (mặc định
`http://localhost:8501`).

**Các bước sử dụng Dashboard:**
1. Ở thanh bên trái (sidebar), nhập toạ độ **Điểm xuất phát** và **Điểm đến**.
2. Điều chỉnh **nhiệt độ môi trường** và **nhiệt độ pin lúc khởi hành**
   bằng thanh trượt để mô phỏng điều kiện mùa đông cụ thể.
3. (Tuỳ chọn) Tuỳ chỉnh tải trọng xe, dung lượng pin trong phần "Thông số xe
   (nâng cao)".
4. Bấm nút **"🚀 Chạy phân tích & Tối ưu tuyến đường"**.
5. Xem:
   - Các chỉ số HVAC/BTMS thời gian thực ở đầu trang.
   - Bảng so sánh các tuyến đường (tuyến tối ưu được đánh dấu ⭐ và tô màu xanh).
   - Biểu đồ cột so sánh năng lượng di chuyển vs. hao hụt mùa đông.
   - Giải thích bằng văn bản (tiếng Việt) lý do AI Engine chọn tuyến đó.
   - Chi tiết từng đoạn đường của tuyến tối ưu (mở phần "🔬 Chi tiết...").

## 6. Cách Dashboard gọi AI Engine backend (chi tiết kỹ thuật)

Trong `dashboard.py`:

```python
from vehicle_dynamics import Polestar4Dynamics
from winter_hvac import WinterHVACSimulation
from route_optimizer import RouteOptimizationEngine, RouteCandidate, RouteSegment

vehicle = Polestar4Dynamics(payload_kg=..., battery_usable_capacity_kwh=...)
hvac_sim = WinterHVACSimulation()
engine = RouteOptimizationEngine(vehicle=vehicle, hvac_sim=hvac_sim,
                                  battery_initial_temp_c=...)

result = engine.compare_and_optimize([route_1, route_2, ...])
# result["optimal_route"], result["all_results"], result["explanation"]
```

Streamlit chạy lại toàn bộ script mỗi khi người dùng tương tác (nút bấm,
thanh trượt...), nên kết quả phân tích được lưu vào `st.session_state` để
không bị mất khi người dùng mở/đóng các phần "expander" khác.

## 7. Kiến trúc mở rộng với FastAPI (khi cần tách Backend/Frontend)

Nếu muốn triển khai AI Engine như một **REST API độc lập** (ví dụ: để nhiều
client khác nhau — web, mobile, hệ thống trong xe — cùng gọi), có thể bọc
`RouteOptimizationEngine` bằng FastAPI như sau:

```python
# backend_api.py (ví dụ mở rộng, không có sẵn trong gói này)
from fastapi import FastAPI
from pydantic import BaseModel
from route_optimizer import RouteOptimizationEngine, RouteCandidate, RouteSegment

app = FastAPI()
engine = RouteOptimizationEngine()

class OptimizeRequest(BaseModel):
    routes: list  # danh sách route candidates dạng JSON

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    routes = [RouteCandidate(
        route_name=r["route_name"],
        description=r.get("description", ""),
        segments=[RouteSegment(**s) for s in r["segments"]],
    ) for r in req.routes]
    return engine.compare_and_optimize(routes)
```

Chạy backend:
```bash
uvicorn backend_api:app --reload --port 8000
```

Khi đó, `dashboard.py` sẽ thay phần `import route_optimizer` bằng lời gọi
HTTP:

```python
import requests
response = requests.post("http://localhost:8000/optimize", json={"routes": [...]})
result = response.json()
```

Kiến trúc này tách biệt rõ AI Engine (có thể scale độc lập, deploy trên
server riêng) khỏi Dashboard (chỉ là lớp hiển thị).

## 8. Ghi chú về độ chính xác mô hình

- Các công thức vật lý (lực cản khí động học, ma sát lăn, lực leo dốc, động
  năng, phanh tái sinh) tuân theo mô hình động lực học dọc trục xe tiêu chuẩn
  ("point-mass longitudinal vehicle dynamics") dùng phổ biến trong ngành kỹ
  thuật ô tô điện.
- Mô hình HVAC/BTMS dùng phương pháp "lumped thermal mass" (khối nhiệt tập
  trung) — đơn giản hoá hợp lý cho bài toán dự đoán năng lượng cấp độ
  chuyến đi, không thay thế mô phỏng CFD/nhiệt học chi tiết cấp độ thiết kế.
- Thông số xe (khối lượng, Cd, dung lượng pin...) là giá trị gần đúng công
  bố công khai cho Polestar 4 Single Motor — nên hiệu chỉnh lại trong
  `Polestar4Dynamics(...)` nếu có số liệu chính xác hơn từ nhà sản xuất cho
  thị trường/năm sản xuất cụ thể.
- Đây là công cụ **dự đoán/mô phỏng**, không phải hệ thống đo lường thời gian
  thực trên xe thật — kết quả nên được dùng như tham khảo định hướng, không
  thay thế đồng hồ đo pin/quãng đường thực tế trên xe.