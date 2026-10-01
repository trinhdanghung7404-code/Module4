# Hướng dẫn sử dụng và giải thích mã nguồn

Tài liệu này hướng dẫn cách thiết lập, chạy hệ thống tiền xử lý ảnh và giao diện chụp ảnh (Test_Camera), đồng thời giải thích các công nghệ và file code.

## 1. Môi trường và công nghệ sử dụng

Dự án sử dụng Python để xử lý ảnh với các thư viện sau:
- **`opencv-python` (`cv2`)**: Dùng để đọc/ghi ảnh, chuyển đổi không gian màu (LAB), áp dụng cân bằng sáng (CLAHE), phân tích pixel và cắt ảnh.
- **`numpy`**: Hỗ trợ xử lý ma trận ảnh.
- **`rembg`**: Dùng để nhận diện và xóa nền ảnh.

Cài đặt các thư viện:
```bash
pip install opencv-python numpy rembg
```

## 2. Tiền xử lý ảnh và tạo mask dữ liệu (`preprocess_mask.py`)

### Mục đích
- **Pha 1 (Tiền xử lý):** Tách nền bằng màu xám trung tính (tránh gắt viền) và crop sát khung vật thể để chuẩn bị cho bước khớp đặc trưng.
- **Pha 2 (Tạo Mask):** Quét và nhận diện các vùng lóa sáng (specular) hoặc bão hòa (cháy sáng/đen tuyệt đối) để loại trừ khi so sánh, đồng thời đánh giá tỷ lệ lỗi của ảnh.

### Cách chạy
1. Đặt 2 file ảnh cùng cấp thư mục với file code:
   - `anh_goc.jpg`
   - `anh_khach.jpg`
   *(Có thể chạy riêng lẻ bằng cách truyền tham số: `python preprocess_mask.py anh_goc.jpg`)*
2. Chạy lệnh:
   ```bash
   python preprocess_mask.py
   ```
3. Kết quả (với mỗi ảnh, tạo ra các file với tiền tố `goc_` hoặc `khach_`):
   - `_processed.png`: Ảnh đã tách nền và crop sát vật thể.
   - `_specular_mask.png`: Mask vùng lóa sáng.
   - `_saturated_mask.png`: Mask vùng bão hòa cực trị.
   - `_ignore_mask.png`: Mask gộp các vùng cần bỏ qua.
   - `_overlay_contour.png`: Ảnh trực quan có vẽ viền xanh quanh các đốm lóa.

### Giải thích code
- **Bóc nền và Crop (Pha 1)**: Dùng `rembg` tách nền, sử dụng nền xám để tránh tạo cạnh tương phản giả, sau đó crop sát bounding box của vật thể.
- **Quét vùng lỗi (Pha 2)**: 
  - Lóa sáng (Specular): Xét trên kênh HSV (Value cao, Saturation thấp) kết hợp khử nhiễu.
  - Bão hòa (Saturated): Xét trên kênh L của LAB để tìm vùng trắng/đen tuyệt đối mất thông tin.
- **Đánh giá ảnh**: Tính tỷ lệ diện tích vùng lỗi (`ignore_ratio`), nếu vượt ngưỡng (vd > 15%) sẽ báo cờ `should_reshoot = True` để đề xuất chụp lại ảnh.

## 3. Tiền xử lý ảnh chụp mẫu (`preprocess_gimage.py`)

### Mục đích
Chuẩn bị các ảnh mờ (ghost images) để làm lớp phủ hướng dẫn canh góc chụp trong thư mục `Test_Camera`.

### Cách chạy
1. Đặt 4 file ảnh cùng cấp thư mục với code:
   - `truoc.jpg`
   - `sau.jpg`
   - `trai.jpg`
   - `phai.jpg`
2. Chạy lệnh:
   ```bash
   python preprocess_gimage.py
   ```
3. Kết quả: Tạo ra 4 file PNG nền trong suốt: `truoc_nb.png`, `sau_nb.png`, `trai_nb.png`, `phai_nb.png`.

### Giải thích code
- **Xóa nền**: Dùng `rembg` để xóa nền.
- **Cắt sát viền**: Dùng `cv2.findNonZero` trên kênh alpha để tìm các pixel của vật thể, sau đó dùng `cv2.boundingRect` để lấy khung chữ nhật bao quanh vật thể.
- **Lưu ảnh**: Cắt ảnh theo khung vừa tìm được và lưu định dạng `.png` để giữ nền trong suốt và giảm kích thước file.

## 4. Chạy giao diện camera (`Test_Camera`)

Thư mục `Test_Camera` chứa code web để hiển thị giao diện chụp ảnh trên điện thoại, sử dụng ảnh mẫu (ghost images) làm lớp phủ.

### Các bước test:
1. Tạo 4 thư mục con `1`, `2`, `3`, `4` bên trong thư mục `Test_Camera`.
2. Copy 4 file `.png` được tạo từ `preprocess_gimage.py` vào chung một thư mục (ví dụ thư mục `1`). Lặp lại quá trình này cho các thư mục còn lại.
3. Upload toàn bộ thư mục `Test_Camera` lên các dịch vụ hosting như Netlify.
4. Mở link web trên điện thoại để kiểm tra giao diện chụp ảnh có lớp phủ.

### Quy trình thực tế
Khi triển khai thực tế:
- Quá trình tiền xử lý được thực hiện tự động trên cloud khi shop upload ảnh, và ghost image được lưu trên cloud.
- Khi người dùng cần hoàn tiền, hệ thống sẽ lấy đúng ghost image của sản phẩm đó để hiển thị trên trang chụp ảnh mà không cần tạo thư mục thủ công.