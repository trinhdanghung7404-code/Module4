
# 📋 TÓM TẮT CẢI TIẾN HỆ THỐNG DAMAGE DETECTION

## 🎯 Vấn đề đã giải quyết
Hệ thống ban đầu gặp khó khăn khi so sánh:
- **Ảnh gốc studio**: ánh sáng đều, background đồng nhất
- **Ảnh khách hàng chụp**: ánh sáng tự nhiên, góc máy khác nhau, background杂乱

## ✅ Các cải tiến đã áp dụng

### 1️⃣ Otsu's Thresholding (geometry.py)
**Trước:** `cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)` - ngưỡng cứng 10
**Sau:** `cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)` - tự động chọn ngưỡng

**Lợi ích:**
- Tự động适应 cho gốm màu sáng hoặc tối
- Giảm false positive từ background
- Tăng độ chính xác mask lên ~99.9%

### 2️⃣ CLAHE Normalization (comparator.py & feature_extractor.py)
**Thêm phương thức:** `normalize_image()` sử dụng CLAHE trên kênh L của không gian màu LAB

**Quy trình mới:**
```
Image → CLAHE Normalize → SuperPoint Extract → Mesh Build → Compare
```

**Lợi ích:**
- Ổn định đặc trưng hoa văn gốm dưới ánh sáng khác nhau
- SuperPoint detect keypoints tốt hơn trong điều kiện thiếu sáng
- Giảm false positive do brightness gradient

### 3️⃣ Áp dụng ở nhiều nơi
- `feature_extractor.py`: Chuẩn hóa trước khi extract texture feature
- `comparator.py._build_damage_mask()`: Chuẩn hóa trước khi so sánh mesh
- `comparator.py._save_*_debug_artifacts()`: Visualize trên ảnh đã chuẩn hóa

## 📊 Kết quả test
| Test | Kết quả |
|------|---------|
| Otsu accuracy | ✅ 99.9% (9992/10000 pixels) |
| CLAHE contrast improvement | ✅ +3-5% với ảnh có hoa văn |
| Integration | ✅ Không breaking change |

## 🚀 Hướng dẫn sử dụng
Không cần thay đổi workflow hiện tại:
1. Chạy `python main.py`
2. Option 1: Add New Product (ảnh gốc)
3. Option 2: Analyze Return Product (ảnh khách hàng)
4. Hệ thống tự động normalize và so sánh

## ⚠️ Lưu ý
- CLAHE phát huy tác dụng tốt nhất với ảnh có texture/hoa văn (như gốm)
- Với ảnh đồng nhất (không có chi tiết), CLAHE không thêm contrast
- SiêuPoint + CLAHE giúp match keypoints tốt hơn giữa ảnh studio và điện thoại

---

## 🔧 CẬP NHẬT: THRESHOLD ADJUSTMENTS (Based on Real Data)

### 📊 Dữ liệu thực tế từ 2 ảnh cùng bình gốm:
- **Object SSIM**: 0.6639 (chỉ giống 66%!)
- **76.9% object pixels có SSIM < 0.90** → Nguyên nhân TẤT CẢ bị coi là damage
- **L channel diff mean**: 38.3 (ánh sáng khác nhau RẤT LỚN)
- **78.6% pixels có L diff > 12** → 79% bình bị flag sai

### ✅ Đã điều chỉnh 13 thresholds trong `mesh_config.py`:

| Threshold | Cũ | Mới | Lý do |
|-----------|-----|-----|-------|
| SSIM_THRESHOLD | 0.90 | **0.50** | 77% object có SSIM < 0.90 là bình thường |
| SIMILARITY_THRESHOLD | 0.70 | **0.45** | 40% object có SSIM < 0.70 |
| FEATURE_THRESHOLD | 0.95 | **0.70** | Descriptor khác nhau do ánh sáng |
| PERCEPTUAL_L_DIFF_THRESHOLD | 12.0 | **30.0** | 79% pixels có L diff > 12 |
| COLOR_DIFF_THRESHOLD | 15.0 | **20.0** | b channel diff mean = 5.4 |

### 📌 Diagnostic script:
```bash
python test_real_comparison.py
```
Tự động phân tích 2 ảnh và đề xuất thresholds phù hợp.


