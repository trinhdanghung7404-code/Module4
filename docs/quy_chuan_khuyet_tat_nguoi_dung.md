# QUY CHUẨN ĐẶC TẢ KHUYẾT TẬT DO NGƯỜI DÙNG & VẬN CHUYỂN
## (USER & TRANSIT-CAUSED CERAMIC DEFECT SPECIFICATION)

* **Hệ thống:** Máy chủ kiểm định hoàn trả gốm sứ (`Module4_AIReturn`)
* **Tôn chỉ kiểm định:** Bình gốm khi xuất kho được bảo đảm **100% hoàn hảo và khớp tuyệt đối với hồ sơ mẫu chuẩn** (đã qua KCS xưởng loại trừ toàn bộ lỗi lò nung như lỗ kim, co men, méo dáng). Mọi khuyết tật xuất hiện khi khách hoàn trả đều phát sinh từ **quá trình vận chuyển (Transit)** hoặc **quá trình sử dụng/bảo quản của khách hàng (User Handling)**.

---

## TỔNG QUAN PHÂN LOẠI 3 NHÓM (11 LOẠI KHUYẾT TẬT THỰC TẾ)

```text
                               KHUYẾT TẬT HOÀN TRẢ GỐM SỨ
                                           │
         ┌─────────────────────────────────┼────────────────────────────────┐
         ▼                                 ▼                                ▼
[NHÓM A: NỨT VỠ KẾT CẤU]        [NHÓM B: SỨT MẺ & TRẦY XƯỚC]       [NHÓM C: NHIỄM BẨN & VẾT Ố]
- A1: Nứt kết cấu lớn            - B1: Mẻ vành / Chân đế (Notch)    - C1: Mực bút dạ / Bút bi
- A2: Rạn tóc men                - B2: Mẻ bong tróc men thân        - C2: Ố ngấm nước chè qua rãnh
- A3: Va đập hình sao            - B3: Xước sâu rạch thủng men      - C3: Keo băng dính bám bụi
                                 - B4: Xước to cọ xát (Mờ sương)
                                 - B5: Xước dăm quầng xoáy búi sắt
```

---

## CHI TIẾT ĐẶC TẢ 11 LOẠI KHUYẾT TẬT

---

### NHÓM A: TỔN HẠI NỨT VỠ KẾT CẤU (FRACTURE & CRACKING)

#### 1. [A1] Vết nứt kết cấu lớn (Macro Structural Crack)
* **Hành vi gây ra:** Thùng hàng bị ném quăng quật khi vận chuyển, hoặc khách làm rơi bình xuống nền cứng.
* **Hình thái học:** Rãnh nứt dài ($60 - 250\text{mm}$), rộng $0.3 - 1.5\text{mm}$ ($2 - 5\text{px}$). Đứt gãy xuyên cả 3 lớp: men $\to$ hoa văn $\to$ xương gốm. Thường phân 2-3 nhánh rễ cây tự nhiên.
* **Đặc tính quang học:** Đáy rãnh nứt tối sẫm do đổ bóng sâu (Ambient Occlusion). Sườn mép nứt hướng sáng có viền bắt sáng trắng chói sắc lẹm $1\text{px}$.
* **Ảnh hưởng hoa văn:** Cắt đứt gãy hoàn toàn các nét vẽ nghệ nhân.
* **Dấu hiệu AI:** $\Delta E > 45$, vector DINOv2 vi sai $\Delta f > 0.65$.

#### 2. [A2] Vết rạn tóc men (Hairline Glaze Crack)
* **Hành vi gây ra:** Khách đổ nước sôi trực tiếp vào bình gây sốc nhiệt, hoặc va chạm chấn động nhẹ.
* **Hình thái học:** Siêu mảnh ($1.0 - 1.5\text{px}$), chạy dài uốn lượn mềm mại theo ứng suất kéo của thành bình cong. Chỉ nứt lớp men ngoài, xương gốm còn nguyên.
* **Đặc tính quang học:** Không có bóng rãnh sâu. Nhìn nghiêng có ánh sáng khúc xạ men thủy tinh óng ánh.
* **Ảnh hưởng hoa văn:** Nét vẽ hoa văn bên dưới vẫn nhìn thấy rõ, chỉ có đường rạn cắt ngang phía trên.
* **Dấu hiệu AI:** Phát hiện bằng lưới vi sai cục bộ Local Search DINOv2 kết hợp gradient góc xiên.

#### 3. [A3] Vết nứt mạng nhện do va đập (Star Impact Crack)
* **Hành vi gây ra:** Bị vật nhọn đập thẳng vào thân bình (đầu búa nhỏ, góc bàn, va chạm điểm tốc độ cao).
* **Hình thái học:** Lỗ dập nát trung tâm ($8 - 15\text{mm}$) kèm 3-6 tia nứt nan hoa tỏa tròn ra xung quanh ($25 - 60\text{mm}$) uốn lượn 3D.
* **Đặc tính quang học:** Hốc tâm tối sẫm do gốm bị nghiền nát, các tia nứt nan hoa bắt sáng phản xạ tỏa tròn.
* **Ảnh hưởng hoa văn:** Phá hủy hoàn toàn hoa văn tại tâm và chia cắt hoa văn theo các nan hoa.

---

### NHÓM B: TỔN HẠI SỨT MẺ & TRẦY XƯỚC (CHIPPING & ABRASION)

#### 4. [B1] Vết sứt mẻ vành miệng & chân đế (Rim & Base Chipping - Silhouette Notch)
* **Hành vi gây ra:** Khách mở/đậy nắp mạnh tay, cấn miệng bình khi cắm hoa, hoặc đặt mạnh đáy bình xuống sàn gạch/mặt đá.
* **Hình thái học:** Vỡ hình vỏ sò sắc cạnh. **Cắn lẹm khuyết vào đường viền bình (Silhouette Notch)**, kích thước $8 - 35\text{mm}$.
* **Đặc tính quang học:** Bóc toàn bộ men, lộ **xương gốm đất nung Bát Tràng (Biscuit)** nhám mờ màu trắng ngà/xám sáng (#E8E3D8). Phía trên có bóng lún, phía dưới mép vỡ bắt sáng.
* **Ảnh hưởng hoa văn:** Mất hẳn mảng hoa văn viền miệng hoặc viền chân đế.

#### 5. [B2] Vết bong tróc men trong thân bình (Surface Glaze Spall / Flaking)
* **Hành vi gây ra:** Vật cứng gõ vào thân bình làm lớp men bóng bung ra từng mảng mà không làm méo đường viền ngoài.
* **Hình thái học:** Nằm trong lòng thân bình, hình vảy sò bất đối xứng ($6 - 20\text{mm}$).
* **Đặc tính quang học:** Vùng nhám mờ màu cốt gốm ngà nằm giữa nền men bóng láng. Gờ men xung quanh phản xạ ánh sáng đồng tâm.
* **Ảnh hưởng hoa văn:** Nét vẽ nghệ nhân tại vị trí đó bị bóc đi cùng lớp men, trơ lại nền cốt đất nhám.

#### 6. [B3] Vết xước sâu rạch thủng men (Deep Gouge / Incised Scratch)
* **Hành vi gây ra:** Mũi đinh, lưỡi dao, chìa khóa kim loại, cạnh sắc của mảnh vỡ cào xước với áp lực tì đè mạnh.
* **Hình thái học:** Rãnh chữ V sắc lẹm, hẹp ($1.5 - 3.5\text{px}$), rạch xuyên men vào xương gốm. Hai bên mép rãnh có men vỡ vụn vảy cá (micro-spalling).
* **Đặc tính quang học:** Đáy rãnh đen tối sâu (bóng đổ hốc lún). Mép men đối diện ánh sáng có đường phản xạ trắng chói $1\text{px}$.
* **Ảnh hưởng hoa văn:** Cắt đứt hoàn toàn hoa văn trên đường rạch.

#### 7. [B4] Vết xước to cọ xát diện rộng (Surface Scuff & Abrasion Track)
* **Hành vi gây ra:** Bình bị cọ vào thành thùng carton cứng, cọ xát mạt cát thạch anh, hoặc hai bình cọ vào nhau trên xe tải.
* **Hình thái học:** Dải rộng ($10 - 40\text{px}$), nông trên bề mặt ngoài cùng của men bóng ($< 0.05\text{mm}$).
* **Đặc tính quang học:** **Mất độ bóng gương $\to$ Tạo vệt sương mờ đục trắng (Milky Haze)** do tán xạ ánh sáng. Không có đáy rãnh đen.
* **Ảnh hưởng hoa văn:** Hoa văn bên dưới **không bị đứt**, chỉ bị mờ nhòe đi như nhìn qua kính mờ.

#### 8. [B5] Vết xước dăm quầng xoáy (Swirl Marks / Micro-Scratching)
* **Hành vi gây ra:** Khách hàng chùi rửa bình bằng búi sắt cọ xoong hoặc giẻ lau dính cát thô.
* **Hình thái học:** Hàng trăm vết xước mảnh li ti ($1\text{px}$) đan chéo hoặc xoáy tròn theo chuyển động tay lau.
* **Đặc tính quang học:** Dưới ánh đèn chiếu điểm tạo thành quầng sáng tán xạ dạng mạng nhện (spiderweb halo), làm mờ nước men bóng.

---

### NHÓM C: NHIỄM BẨN & VẾT Ố BỀ MẶT (CONTAMINATION & STAINING)

#### 9. [C1] Vết mực bút dạ / Bút bi / Bút lông dầu (Marker & Ink Scribbles)
* **Hành vi gây ra:** Trẻ con vẽ bậy, nhân viên kho vận dùng bút lông dầu đánh dấu nhầm lên men, hoặc bút bi va quẹt khi ký nhận.
* **Hình thái học:** Nét vẽ liền mạch, màu đen mực dầu, xanh bút bi, đỏ bút nhớ ($3 - 100\text{px}$). Men bên dưới nguyên vẹn.
* **Đặc tính quang học:** Hấp thụ ánh sáng mạnh, màu đậm đặc, nằm đè lên trên cả men trắng lẫn hoa văn vẽ tay.
* **Dấu hiệu AI:** $\Delta E > 50$, DINOv2 phân loại chính xác nhãn `STAIN`.

#### 10. [C2] Vết ố ngấm nước chè / Rỉ sắt qua rãnh nứt (Capillary Seepage Stains)
* **Hành vi gây ra:** Khách ngâm nước chè, cắm hoa để lâu ngày, nước bẩn ngấm qua các vết rạn tóc hoặc ngấm từ chân đế xốp.
* **Hình thái học:** Loang mờ hữu cơ ($20 - 60\text{mm}$) dọc theo các khe nứt rạn.
* **Đặc tính quang học:** Màu nâu trà hoặc vàng rỉ sắt, gradient đậm ở khe nứt và mờ dần ra men xung quanh. Phân tử màu nằm sâu dưới men, không lau sạch được.

#### 11. [C3] Vết keo băng dính bám bụi thoái hóa (Tape & Adhesive Residue)
* **Hành vi gây ra:** Dán băng dính niêm phong hoặc băng keo xốp trực tiếp lên thân bình khi đóng gói trả hàng, lớp keo bị chảy dính chặt vào men.
* **Hình thái học:** Mảng chữ nhật hoặc dải dài có mép nham nhở, bề mặt bắt bụi đen li ti.
* **Đặc tính quang học:** Mảng nhờ nhờ đục, mất bóng gương, có vết gợn dơ bẩn bề mặt.

---

## BẢNG MA TRẬN ĐỐI CHIẾU 11 KHUYẾT TẬT ĐỂ TRIỂN KHAI MÔ PHỎNG

| Mã | Tên khuyết tật | Cấp độ | Chiều sâu vật lý | Đặc tính quang học cốt lõi | Ảnh hưởng hoa văn |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **A1** | Nứt kết cấu lớn | Vĩ mô | Xuyên thấu xương gốm | Rãnh đen sâu + Gờ bắt sáng $1\text{px}$ | Đứt gãy hoàn toàn |
| **A2** | Rạn tóc men | Vi mô | Chỉ nứt lớp men ngoài | Nét thanh, óng ánh men, không rãnh | Nhìn thấy rõ dưới rạn |
| **A3** | Va đập hình sao | Vĩ mô | Lỗ vỡ + Nứt nan hoa | Hốc tối dập nát + Tia nứt tỏa tròn 3D | Phá hủy tâm va chạm |
| **B1** | Mẻ vành / Chân đế | Vĩ mô | Cắn lẹm thành bình | Lõm khuyết đường viền, lộ cốt đất ngà | Mất mảng hoa văn viền |
| **B2** | Mẻ bong men thân | Trung bình | Lõm bằng độ dày men | Đảo nhám mờ giữa biển men bóng | Trơ cốt đất, đứt nét vẽ |
| **B3** | Xước sâu rạch men | Vi mô | Khía rãnh chữ V vào cốt | Rãnh đen hẹp + Mép vảy cá + Gờ sáng | Cắt đứt dứt khoát nét vẽ |
| **B4** | Xước to cọ xát | Vĩ mô | Cực nông trên mặt men | Vệt mờ sương (Haze), mất bóng gương | Hoa văn mờ nhưng liên tục |
| **B5** | Xước dăm quầng xoáy | Vi mô | Xước bề mặt siêu mịn | Quầng mạng nhện tán xạ ánh đèn | Mờ nước men tổng thể |
| **C1** | Mực bút dạ / Bút bi | Tự do | Nằm trên mặt men | Màu mực đen/xanh/đỏ đè lên men | Che phủ hoa văn |
| **C2** | Ố ngấm nước chè | Trung bình | Thấm mao dẫn dưới men | Gradient nâu trà lan tỏa từ khe nứt | Nhuộm màu hoa văn |
| **C3** | Keo băng dính bẩn | Vĩ mô | Màng keo ngoài mặt men | Mảng nhờ đục dính bụi nham nhở | Mờ hoa văn bên dưới |
