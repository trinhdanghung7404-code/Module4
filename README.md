# Module 4: Ceramic Defect Inspection & Return Verification System (Module4_AIReturn)

Hệ thống phát hiện lỗi và thẩm định sản phẩm gốm sứ hoàn trả tự động dựa trên mô hình nền tảng **DINOv2** kết hợp **Cổng kiểm chứng đối chiếu vi sai** (Differential Inspection Gate).

---

## 1. Cấu trúc thư mục chuẩn hệ thống

```
Module4_AIReturn/
├── app.py                      # FastAPI server & entrypoint
├── config.py                   # Cấu hình toàn cục (paths, credentials, server settings)
├── run_server.bat              # Script khởi động máy chủ một chạm
├── requirements.txt            # Danh mục thư viện phụ thuộc
├── core/                       # Nhân xử lý AI & thuật toán cốt lõi
│   ├── __init__.py
│   ├── pipeline.py             # ReferencePipeline (DINOv2 + Cổng kiểm chứng đối chiếu vi sai)
│   ├── preprocessor.py         # Tiền xử lý (tách nền BiRefNet/u2net, crop, quét lóa sáng)
│   └── matcher.py              # Ghép mẫu ghost, kiểm định chất lượng ảnh tải lên
├── data/                       # Dữ liệu mẫu tham chiếu
│   ├── profiles/               # Hồ sơ vân tay DINOv2 (.pt) của từng góc sản phẩm
│   ├── ghost_overlays/         # Khung bóng mờ căn chỉnh camera di động (_nb.png)
│   └── reference_images/       # Ảnh gốc chuẩn sản phẩm
├── models/                     # Checkpoints mô hình AI
│   └── specsegv3_chkpt.h5      # Specular segmentation checkpoint
├── templates/                  # Giao diện Web Jinja2
│   ├── capture.html            # Giao diện camera di động (auto-capture, hướng dẫn)
│   ├── desktop.html            # Giao diện máy tính giám sát phiên & QR code
│   └── result.html             # Giao diện hiển thị kết quả kiểm định đối chiếu
├── tools/                      # Công cụ dòng lệnh CLI
│   ├── inspect_image.py        # Soi lỗi nhanh 1 ảnh bất kỳ độc lập
│   └── inspect_image.bat       # Chạy công cụ soi lỗi nhanh trên Windows
├── utils/                      # Tiện ích bổ trợ hệ thống
│   ├── __init__.py
│   ├── qr_code.py              # Tạo mã QR Base64 & PNG
│   ├── tunnel.py               # Khởi tạo HTTPS Cloudflare tunnel cho iOS Safari
│   └── cloudflared.exe         # Binary Cloudflared
└── uploads/                    # Thư mục lưu trữ ảnh chụp theo từng phiên
```

---

## 2. Luồng hoạt động hệ thống

```
                    ┌─────────────────────────┐
                    │     Desktop Client      │
                    │      (desktop.html)     │
                    └────────────┬────────────┘
                                 │ 1. Tạo session & hiển thị QR
                                 ▼
                    ┌─────────────────────────┐
                    │      Capture Server     │
                    │         (app.py)        │
                    └────────────┬────────────┘
                                 │ 2. Quét QR kết nối qua HTTPS Tunnel
                                 ▼
                    ┌─────────────────────────┐
                    │      Mobile Camera      │
                    │      (capture.html)     │
                    └────────────┬────────────┘
                                 │ 3. Căn khung bóng mờ & Auto-Capture 4 góc
                                 │ 4. Kiểm tra độ nét & loại bỏ ảnh giả (bàn phím/sàn nhà)
                                 ▼
                    ┌─────────────────────────┐
                    │    Core Preprocessor    │
                    │   (core/preprocessor)   │
                    └────────────┬────────────┘
                                 │ 5. Tách nền xám 40 + Tight Crop
                                 │ 6. Quét vùng lóa sáng (Specular / Saturated Mask)
                                 ▼
                    ┌─────────────────────────┐
                    │   Reference Pipeline    │
                    │     (core/pipeline)     │
                    └────────────┬────────────┘
                                 │ 7. Khớp vân tay DINOv2 với hồ sơ chuẩn (.pt)
                                 │ 8. Cổng kiểm chứng đối chiếu vi sai
                                 ▼
                    ┌─────────────────────────┐
                    │     Result Dashboard    │
                    │      (result.html)      │
                    └─────────────────────────┘
```

---

## 3. Cài đặt & Khởi động

### 3.1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10+:
```bash
pip install -r requirements.txt
```

### 3.2. Chạy Capture Server
Nhấp đúp chuột vào file `run_server.bat` hoặc thực hiện lệnh:
```bash
python app.py
```
Máy chủ sẽ tự động thiết lập đường truyền HTTPS an toàn (hỗ trợ Safari iOS) và mở cổng lắng nghe tại `http://localhost:8000`.

### 3.3. Soi khuyết tật ảnh độc lập bằng CLI
Nhấp đúp vào `tools/inspect_image.bat` hoặc chạy:
```bash
python tools/inspect_image.py [đường_dẫn_ảnh] [mã_sản_phẩm]
```
Ví dụ:
```bash
python tools/inspect_image.py C:\anh_loi.jpg B1_01_front
```
Nếu không truyền tham số, công cụ sẽ mở chế độ tương tác kéo-thả ảnh trực tiếp.

---

## 4. Đặc tả kỹ thuật Cổng kiểm chứng vi sai (ReferencePipeline)

1. **Feature Extractor**: DINOv2 (`dinov2_vits14`), trích xuất patch embeddings 14x14 không phụ thuộc vào texture truyền thống.
2. **Background Removal**: BiRefNet SOTA (hoặc ONNX Runtime u2netp fallback) đảm bảo cắt sát viền và chống viền tương phản giả.
3. **Alignment**: SIFT keypoint matching kết hợp RANSAC Affine/Homography transformation.
4. **Local Search**: Tìm kiếm ma trận lân cận vi mô 5x5 để bù trừ độ dãn nở quang học và sai số góc chụp di động.
5. **Differential Inspection**: Loại trừ 100% hoa văn gốc của bình, chỉ kích hoạt cảnh báo khi phát hiện cấu trúc khuyết tật thật (sứt mẻ, rạn nứt, vết ố, lỗ châm kim).
