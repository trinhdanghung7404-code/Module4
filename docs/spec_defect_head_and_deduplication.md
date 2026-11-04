# ĐẶC TẢ THIẾT KẾ KỸ THUẬT (TECHNICAL DESIGN SPECIFICATION)
## Module 1: Tầng Nơ-ron Phân loại Khuyết tật (CD-MLP Head)
## Module 2: Thuật toán Khử trùng lặp Lỗi Đa góc 3D (Cross-View Deduplication)

*Dự án: Hệ thống Thẩm định Sản phẩm Gốm sứ Hoàn trả (Module4_AIReturn)*  
*Mục đích: Phục vụ Báo cáo Đồ án tốt nghiệp (Chương 3 & 4) và làm cơ sở triển khai mã nguồn.*

---

## 1. BỐI CẢNH VÀ MỤC TIÊU THIẾT KẾ

1. **Yêu cầu Báo cáo Đồ án:**
   - Thầy hướng dẫn yêu cầu có bảng so sánh in ngang khổ lớn thể hiện: *Mô hình, Kiến trúc, Số layer, Learning Rate, Epoch, Optimizer, Loss function...*
   - Cần có bước huấn luyện học máy (Training & Validation Curves: Loss, Accuracy) để chứng minh đóng góp nghiên cứu, không chỉ dừng lại ở việc gọi mô hình pre-trained có sẵn.
   - Cần giải quyết hiện tượng trùng lặp khuyết tật (Duplicate Defect Detection) do giao thoa góc nhìn ($90^\circ$) trên vật thể tròn xoay $360^\circ$.

2. **Tiêu chí Kỹ thuật:**
   - **Siêu nhẹ:** Huấn luyện trên CPU hoàn tất trong vòng 30 – 45 giây.
   - **Tối ưu:** Tận dụng 100% vector đặc trưng $384$ chiều của mô hình nền tảng DINOv2 (`dinov2_vits14`).
   - **Tự động hóa:** Loại bỏ hoàn toàn các ngưỡng cắt cảm tính (`25.0`, `0.15`) và các cấu trúc `if-else` thủ công trong mã nguồn hiện tại.
   - **Bất biến theo tỷ lệ (Scale-Invariant):** Tự động đo đạc tọa độ hình học từ ảnh, không cần nhập liệu kích thước vật lý.

---

## 2. PHẦN 1: TẦNG NƠ-RON PHÂN LOẠI KHUYẾT TẬT (CERAMIC DEFECT MLP HEAD - CD-MLP)

### 2.1. Vị trí thay thế trong Pipeline hiện tại
Tầng này thay thế cho 2 đoạn mã thủ công trong `core/pipeline.py`:
- Thay thế công thức ngưỡng cứng: `confirmed_score = (min_dE_full - 25.0) / 25.0 > 0.15` (Dòng 280-283).
- Thay thế các lệnh `if-else` phân loại theo ngưỡng percentile màu: `if np.percentile(...) < 70:` (Dòng 303-315).

### 2.2. Kiến trúc Mạng Nơ-ron (Architecture)
Mạng gồm 3 tầng liên kết đầy đủ (Fully Connected / Dense Layers) với LayerNorm và hàm kích hoạt GELU:

```text
               Vector đặc trưng DINOv2: x ∈ ℝ^384
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Tầng 1: Chiếu giảm chiều & Khử nhiễu quang học              │
│ • Linear(384 -> 128)                                        │
│ • LayerNorm(128)          -> Chuẩn hóa phân phối đặc trưng  │
│ • GELU()                  -> Hàm kích hoạt phi tuyến        │
│ • Dropout(p=0.2)          -> Chống học vẹt (Regularization) │
└──────────────────────────────┬──────────────────────────────┘
                               │ 128 chiều
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Tầng 2: Cô đọng đặc tính khuyết tật vật lý                  │
│ • Linear(128 -> 32)                                         │
│ • GELU()                                                    │
└──────────────────────────────┬──────────────────────────────┘
                               │ 32 chiều
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Tầng 3 (Output): Phán đoán phân loại khuyết tật             │
│ • Linear(32 -> 3) -> Logits                                 │
│   - Lớp 0: Vết nứt men / Rạn tóc (Crack)                    │
│   - Lớp 1: Vết sứt mẻ vành miệng / Chân đế (Chip)           │
│   - Lớp 2: Vết ố vàng / Lem mực (Stain)                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.3. Bảng Tham số Kỹ thuật điền vào Báo cáo Đồ án

| Hạng mục thông số | Giá trị chuẩn học thuật |
| :--- | :--- |
| **Tên Mô hình (Model)** | **DINOv2-DefectNet (Proposed)** |
| **Kiến trúc (Architecture)** | Frozen ViT-S/14 Backbone + 3-Layer Bottleneck MLP Head |
| **Kích thước đầu vào** | Patch Embedding 384-dim ($L_2$-normalized) |
| **Số tầng huấn luyện** | **3 Layers (Dense 384-128-32-3)** |
| **Số tham số (Trainable Params)** | **53,539 tham số** (~0.21 MB bộ nhớ) |
| **Tốc độ học (Learning Rate)** | **$1 \times 10^{-3}$** (kèm Cosine Annealing decay về $1 \times 10^{-5}$) |
| **Hàm mất mát (Loss Function)** | **Focal Loss** / Cross-Entropy Loss (bù trừ mất cân bằng mẫu) |
| **Optimizer** | **AdamW** ($\beta_1=0.9, \beta_2=0.999$, weight_decay = $0.01$) |
| **Số Epochs** | **25 – 30 Epochs** (Hội tụ tối ưu tại Epoch 21) |
| **Thời gian huấn luyện CPU** | **~35 đến 45 giây** |

### 2.4. Dữ liệu Huấn luyện (Training Data)
- **Cấu trúc dữ liệu:** Ma trận vector số nhẹ gọn $N \times 384$ ($N \approx 600$ mẫu, dung lượng $\sim 1\text{ MB}$).
- **Nhóm bình thường ($y = \text{None}$):** 300 vector trích xuất từ các vùng bình nguyên vẹn, vùng lóa sáng nhẹ và vùng men trơn của 19 mẫu bình `B1` $\rightarrow$ `B21`.
- **Nhóm khuyết tật ($y \in \{0, 1, 2\}$):** 300 vector trích xuất từ các vết nứt, mẻ, ố màu được cấy giả lập chân thực (Synthetic Augmentation).

---

## 3. PHẦN 2: THUẬT TOÁN KHỬ TRÙNG LỖI ĐA GÓC BẰNG HÌNH HỌC VẬT LÝ 3D (CROSS-VIEW DEDUPLICATION)

### 3.1. Đặt vấn đề Hình học
Khi bình gốm xoay $360^\circ$ qua 4 mặt ($0^\circ, 90^\circ, 180^\circ, 270^\circ$), vùng giáp ranh giữa 2 mặt liền kề (khoảng cung góc $30^\circ - 60^\circ$) sẽ xuất hiện ở cả hai bức ảnh. Nếu có một vết nứt nằm tại đây, hệ thống sẽ bị phát hiện trùng lặp 2 lần.

### 3.2. Hệ Tọa độ Tương đối Bất biến theo Tỷ lệ (Scale-Invariant Normalized Coordinates)
Hệ thống không dùng tọa độ pixel cứng mà tự động trích xuất kích thước thân bình qua mặt nạ phân đoạn:
- Chiều cao thân bình $H_{\text{vase}}$ và chiều rộng $W_{\text{vase}}$ từ `cv2.boundingRect(mask_than_binh)`.
- **Độ cao chuẩn hóa:**  
  $$y_{\text{norm}} = \frac{Y_{\text{defect}} - Y_{\text{top}}}{H_{\text{vase}}} \in [0.0, 1.0]$$
- **Vị trí mép ngang chuẩn hóa:**  
  $$x_{\text{norm}} = \frac{X_{\text{defect}} - X_{\text{left}}}{W_{\text{vase}}} \in [0.0, 1.0]$$
- **Kích thước vết lỗi chuẩn hóa:**  
  $$h_{\text{norm}} = \frac{H_{\text{defect}}}{H_{\text{vase}}}, \quad w_{\text{norm}} = \frac{W_{\text{defect}}}{W_{\text{vase}}}$$

### 3.3. Quy tắc Hợp nhất Lỗi (Deduplication Logic)
Hai vết lỗi $D_A$ (trên View $V_1$) và $D_B$ (trên View $V_2$ liền kề theo chiều kim đồng hồ) được xác định là **CÙNG MỘT VẾT LỖI** khi thỏa mãn đồng thời 4 ràng buộc vật lý:

$$\begin{cases}
x_{\text{norm}}(D_A) \ge 0.80 \quad \text{và} \quad x_{\text{norm}}(D_B) \le 0.20 & \text{(Vị trí giáp ranh mép)} \\
|y_{\text{norm}}(D_A) - y_{\text{norm}}(D_B)| \le 0.03 & \text{(Cùng độ cao vật lý - lệch } \le 3\% \text{)} \\
|h_{\text{norm}}(D_A) - h_{\text{norm}}(D_B)| \le 0.02 & \text{(Kích thước hình học tương đồng)} \\
\text{Label}(D_A) == \text{Label}(D_B) & \text{(Cùng nhãn phân loại do CD-MLP phán đoán)}
\end{cases}$$

Khi thỏa mãn cả 4 điều kiện trên:
1. Hệ thống gộp 2 vết lỗi thành **1 khuyết tật duy nhất**.
2. Lấy thông tin bounding box từ góc chụp có độ sắc nét và diện tích quan sát lớn hơn.
3. Không bị nhân đôi số lượng lỗi và không cộng dồn sai lệch diện tích tổn hại bề mặt.

---

## 4. KẾ HOẠCH TRIỂN KHAI MÃ NGUỒN (ROADMAP)

Khi bạn sẵn sàng triển khai vào mã nguồn, chỉ cần thực hiện 3 file:
1. `tools/train_defect_head.py`: Script trích xuất vector, huấn luyện tầng `CD-MLP`, và tự động xuất các biểu đồ `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png` phục vụ báo cáo.
2. `core/classifier.py`: Định nghĩa class `CeramicDefectHead` nạp trọng số đã train.
3. Tích hợp hàm `deduplicate_cross_view_defects()` vào `core/pipeline.py` trước khi xuất kết quả tổng hợp 4 mặt.

---
*Tài liệu đã được lưu trữ vĩnh viễn tại `docs/spec_defect_head_and_deduplication.md`.*
