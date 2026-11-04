# HƯỚNG DẪN CHỤP ẢNH & QUY TRÌNH KIỂM ĐỊNH TỰ ĐỘNG (MODULE4_AIRETURN)

## 1. Quy cách chụp ảnh sản phẩm bằng điện thoại

### A. Chụp ảnh mẫu xuất kho (Master References):
* Cầm điện thoại chụp chính diện rõ nét hoa văn của bình (nền phòng, mặt bàn bình thường, không cần tách nền trước).
* Đầy đủ 4 góc: Mặt Trước (`truoc`), Mặt Sau (`sau`), Mặt Trái (`trai`), Mặt Phải (`phai`).
* Tải lên qua API `/api/product/upload` hoặc đặt vào thư mục `data/ghost_overlays/<product_id>/`.

### B. Chụp ảnh kiểm định hoàn trả (Return Inspection):
* Mở giao diện chụp bằng cách quét mã QR từ trang chủ desktop `/` hoặc truy cập `/capture/{session_id}`.
* Căn chỉnh sản phẩm theo khung bóng mờ (ghost overlay) trên màn hình điện thoại.
* Giữ yên camera trong 1 giây để hệ thống tự động kích hoạt tính năng Auto-Capture.
* Hệ thống sẽ tự động lọc rung lắc (độ nét quang học) và chống gian lận (loại bỏ ảnh bàn phím, mặt bàn, sàn nhà).

---

## 2. Quy trình xử lý lỗi của hệ thống

1. **Pha 1 (Tách nền & Chuẩn hóa):** Tách nền bằng BiRefNet/u2netp, tô màu xám trung tính 40 (chống nhiễu biên), cắt sát bounding box.
2. **Pha 2 (Quét lóa sáng):** Nhận diện vùng phản xạ gương (specular glare) và bão hòa sáng (saturated mask) để loại trừ khỏi diện tích nghi ngờ.
3. **Pha 3 (Soi khuyết tật đối chiếu vi sai):** Trích xuất patch embeddings 14x14 bằng DINOv2 backbone, đối chiếu với hồ sơ vân tay mẫu (`data/profiles/`), loại trừ 100% hoa văn gốc và chỉ giữ lại khuyết tật vật lý thật.
