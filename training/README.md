# Mô-đun Huấn luyện Mô hình Phân loại Khuyết tật (training/)
## DINOv2 + Ceramic Defect MLP Head (CD-MLP)

Mô-đun này được thiết kế **hoàn toàn độc lập** trong thư mục `training/`, không làm ảnh hưởng đến luồng code chính của máy chủ kiểm định (`app.py`, `core/pipeline.py`).

---

## 1. Cấu trúc thư mục

```
training/
├── model.py              # Định nghĩa mạng CeramicDefectHead (384 -> 128 -> 32 -> 4)
├── dataset_generator.py  # Thuật toán cấy khuyết tật 3D quang học & lưu "Sổ địa chỉ"
├── extract_features.py   # Trích xuất vector DINOv2 theo sổ địa chỉ & mẫu âm men lành
├── train.py              # Huấn luyện mô hình, đánh giá, tự động vẽ biểu đồ đồ án
├── run_train.bat         # Script khởi chạy 1-chạm trên Windows
├── figures/              # Nơi lưu trữ 3 biểu đồ chuẩn học thuật phục vụ Báo cáo:
│   ├── loss_curve.png        # Đường cong mất mát qua 25 Epochs
│   ├── accuracy_curve.png    # Đường cong độ chính xác Train vs Validation
│   └── confusion_matrix.png  # Ma trận nhầm lẫn 4 lớp khuyết tật
├── ground_truth_defects.json # Sổ địa chỉ lưu tọa độ [x, y, w, h] các khuyết tật sinh ra
├── dataset_features.pt       # Tensor chứa đặc trưng vi sai và nhãn phục vụ train
└── defect_head.pt            # Trọng số mô hình sau khi huấn luyện xong (~0.21 MB)
```

---

## 2. Các lớp khuyết tật phân loại

| Nhãn (Label ID) | Tên kỹ thuật | Mô tả khuyết tật | Xử lý giao diện |
| :---: | :--- | :--- | :--- |
| **0** | `Normal` | Men lành nguyên vẹn / Lóa sáng nhẹ | Bỏ qua (không báo lỗi) |
| **1** | `Crack` | Vết rạn nứt kết cấu / rạn tóc | Báo động đỏ |
| **2** | `Chip` | Vết sứt mẻ vành miệng / chân đế | Cảnh báo vàng |
| **3** | `Stain` | Vết ố vàng / lem mực / loang màu | Cảnh báo cam |

---

## 3. Cách khởi chạy huấn luyện

### Cách 1: Chạy file Batch 1-chạm
Nhấp đúp chuột vào file `training/run_train.bat`.

### Cách 2: Chạy dòng lệnh
```bash
cd C:\Users\Admin\Documents\Module4_AIReturn
python training/train.py
```

*Lưu ý: Nếu chưa có file dữ liệu `dataset_features.pt`, script `train.py` sẽ tự động kích hoạt bộ cấy lỗi và trích xuất vector một cách hoàn toàn tự động.*
