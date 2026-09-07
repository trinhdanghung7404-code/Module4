Damage Detection — Adaptive Mesh Region Matching

Dự án này phát hiện vùng hư hỏng giữa ảnh sản phẩm gốc và ảnh sản phẩm hoàn trả bằng cách kết hợp feature matching, alignment và so khớp vùng theo lưới tam giác thích nghi.

Pipeline chính dùng SuperPoint để trích xuất keypoint/descriptor, lọc match, căn chỉnh ảnh hoàn trả về ảnh sản phẩm, sau đó chia vật thể thành các vùng tam giác và đánh giá mức khác biệt theo từng vùng. Kết quả cuối cùng là mask hư hỏng, ảnh preview và các thống kê/debug phục vụ kiểm chứng.

---

## 1. Mục tiêu

Dự án hướng đến bài toán kiểm tra hư hỏng sản phẩm sau hoàn trả:

- So sánh ảnh sản phẩm chuẩn với ảnh sản phẩm hoàn trả.
- Căn chỉnh hai ảnh trước khi so sánh để giảm sai lệch do góc chụp hoặc dịch chuyển.
- Phát hiện vùng nghi ngờ hư hỏng thay vì chỉ tính sai khác toàn ảnh.
- Tạo debug output để dễ đánh giá từng bước của pipeline.

---

## 2. Ý tưởng tổng quan

Thay vì dùng local window matching cố định trên toàn ảnh, pipeline hiện tại dùng **Adaptive Mesh Region Matching**:

1. Trích xuất đặc trưng ảnh bằng SuperPoint.
2. Match keypoint giữa ảnh gốc và ảnh hoàn trả.
3. Lọc match tốt và tìm inlier bằng RANSAC/TPS.
4. Dùng các inlier thật làm đỉnh lưới tương ứng giữa hai ảnh.
5. Tạo mesh tam giác trên ảnh sản phẩm.
6. Tính similarity cho từng tam giác bằng nhiều metric cổ điển.
7. Chỉ refine các tam giác đáng nghi theo priority queue.
8. Tạo damage map/mask từ các vùng có similarity thấp.
9. Áp dụng morphology và component validation để giảm nhiễu.
10. Xuất ảnh debug, overlay và thống kê.

---

## 3. Kiến trúc pipeline

```text
Product Image + Return Image
        │
        ▼
SuperPoint Feature Extraction
        │
        ▼
Feature Matching + Ratio Test
        │
        ▼
RANSAC / TPS Alignment
        │
        ▼
Adaptive Mesh Construction
        │
        ▼
Triangle Region Similarity
        │
        ▼
Priority-based Mesh Refinement
        │
        ▼
Damage Map + Validation
        │
        ▼
Mask / Preview / Debug Statistics
```

---

## 4. Các thành phần chính

| File / module | Vai trò |
|---|---|
| `product_pipeline.py` | Xử lý ảnh sản phẩm gốc, chuẩn bị dữ liệu đầu vào và luồng so sánh. |
| `return_pipeline.py` | Xử lý ảnh sản phẩm hoàn trả, gọi pipeline phát hiện hư hỏng. |
| `mesh_damage_detector.py` | Module trung tâm của Adaptive Mesh Region Matching; nhận ảnh đã align, keypoints, descriptors và inlier matches để tạo damage map. |
| `mesh_config.py` | Chứa cấu hình debug, ngưỡng similarity, số tam giác tối đa, độ sâu refine và số worker. |
| `mesh_builder.py` | Xây dựng mesh tam giác ban đầu từ các cặp keypoint tương ứng. |
| `mesh_refiner.py` / `triangle_refiner.py` | Chia nhỏ tam giác nghi ngờ để tăng độ chi tiết vùng kiểm tra. |
| `priority_scheduler.py` / `priority_mesh_scheduler.py` | Điều phối refine bằng priority queue, ưu tiên tam giác có similarity thấp. |
| `region_extractor.py` | Tạo mask tam giác, crop vùng ảnh tương ứng trong product/return image. |
| `region_similarity.py` | Tính similarity vùng bằng SSIM, NCC, gradient magnitude và gradient orientation. |
| `triangle_similarity.py` | Tính điểm similarity cho từng tam giác, kết hợp region similarity và descriptor similarity. |
| `similarity_cache.py` | Lưu cache kết quả similarity để tránh tính lặp. |
| `component_validator.py` | Kiểm tra connected components sau khi tạo mask để loại vùng nhiễu. |
| `scratch_test_sp.py` / `scratch_verify.py` | Script thử nghiệm, kiểm tra nhanh SuperPoint hoặc logic pipeline. |

---

## 5. Luồng xử lý chi tiết

### 5.1. Đọc ảnh đầu vào

Ảnh đầu vào thường gồm:

- `images/1.jpg`: ảnh sản phẩm gốc.
- `images/2.jpg`: ảnh sản phẩm hoàn trả.

Ảnh được đọc bằng OpenCV, sau đó chuyển qua các bước feature extraction và alignment.

### 5.2. Trích xuất keypoint bằng SuperPoint

Pipeline dùng mô hình SuperPoint để lấy:

- `product_keypoints`
- `return_keypoints`
- `product_descriptors`
- `return_descriptors`

Các descriptor này được dùng để match feature và tính feature similarity tại các đỉnh mesh.

### 5.3. Feature matching

Các keypoint giữa hai ảnh được match bằng descriptor. Sau đó pipeline lọc match bằng ratio test, chọn các match tốt và tiếp tục lọc bằng RANSAC/TPS để giữ lại inlier.

Các thông tin thường được log:

- Raw matches.
- Good matches.
- Inlier count.
- Match ratio.
- Inlier ratio.
- Mean/median match distance.

### 5.4. Alignment bằng TPS

Ảnh hoàn trả được căn chỉnh về không gian ảnh sản phẩm. TPS giúp xử lý biến dạng cục bộ tốt hơn so với affine/homography trong trường hợp vật thể có sai lệch nhẹ do góc chụp hoặc hình dạng.

Kết quả alignment thường được lưu trong thư mục:

```text
debug_outputs/alignment/<product_id>_vs_<return_id>/
```

Các ảnh debug quan trọng:

- `01_product.png`
- `02_return.png`
- `03_keypoints_product.png`
- `04_keypoints_return.png`
- `05_raw_matches.png`
- `06_good_matches.png`
- `07_inlier_matches.png`
- `08_aligned_return.png`
- `09_overlay_alignment.png`
- `10_difference_before_alignment.png`
- `11_difference_after_alignment.png`

### 5.5. Tạo paired keypoints cho mesh

`MeshDamageDetector` chỉ sử dụng các cặp SuperPoint inlier thật để tạo đỉnh mesh. Không nên tự thêm grid point dạng identity `(x, y) -> (x, y)` nếu điểm đó không có correspondence thật, vì ảnh return có thể bị dịch hoặc biến dạng nhẹ.

Cách này giúp mesh phản ánh đúng quan hệ hình học giữa hai ảnh hơn.

### 5.6. Xây dựng mesh tam giác

Từ các cặp keypoint tương ứng, pipeline xây dựng mesh tam giác trên ảnh product. Mỗi tam giác có vùng tương ứng trên ảnh return thông qua các đỉnh matched.

Mỗi tam giác được xem như một region để so sánh.

### 5.7. Tính similarity vùng

`RegionSimilarity` kết hợp nhiều metric:

- **SSIM**: đo tương đồng cấu trúc.
- **NCC**: đo tương quan cường độ pixel.
- **Gradient Magnitude**: đo khác biệt biên/cường độ cạnh.
- **Gradient Orientation**: đo khác biệt hướng gradient.

Trọng số mặc định:

```python
{
    "ssim": 0.40,
    "ncc": 0.30,
    "gradient_magnitude": 0.20,
    "gradient_orientation": 0.10,
}
```

Điểm similarity càng cao thì hai vùng càng giống nhau. Vùng có similarity thấp và feature similarity thấp được xem là ứng viên hư hỏng.

### 5.8. Priority-based mesh refinement

Không refine toàn bộ mesh. Scheduler chỉ chọn các tam giác nghi ngờ nhất, thường là tam giác có similarity thấp, còn đủ diện tích và chưa vượt quá độ sâu refine.

Cách này giúp:

- Giảm chi phí tính toán.
- Tập trung chi tiết vào vùng nghi ngờ.
- Tránh số lượng tam giác tăng quá lớn.

Các tham số chính nằm trong `mesh_config.py`:

```python
MAX_TOTAL_TRIANGLES = 2000
TOP_K = 24
MAX_DEPTH = 3
MIN_TRIANGLE_AREA = 80.0
SIMILARITY_THRESHOLD = 0.7
FEATURE_THRESHOLD = 0.95
SSIM_THRESHOLD = 0.90
MIN_AREA_THRESHOLD = 15
DIFF_THRESHOLD = 25
SIMILARITY_WORKERS = 4
```

### 5.9. Tạo damage map

Các tam giác bị đánh dấu là damaged được chuyển thành mask. Tùy phiên bản, damage map có thể:

- Fill toàn bộ tam giác nghi ngờ, hoặc
- Chỉ giữ pixel thật sự khác biệt bên trong tam giác.

Phiên bản nên ưu tiên là pixel-level damage map để tránh tô toàn bộ vùng trang trí/hoa văn không hỏng.

### 5.10. Morphology và validation

Sau khi có mask thô, pipeline áp dụng:

- Morphological open/close để giảm noise.
- Connected component analysis.
- Validation dựa trên diện tích, feature similarity, SSIM, gradient và support.

Mục tiêu là loại bỏ các vùng sai khác do texture, ánh sáng, alignment hoặc họa tiết không phải hư hỏng.

---

## 6. Output

Pipeline tạo các output chính:

```text
debug_outputs/
├── <product_id>_vs_<return_id>_damage_mask.png
├── <product_id>_vs_<return_id>_preview.png
├── superpoint/
│   └── <product_id>_vs_<return_id>/
├── alignment/
│   └── <product_id>_vs_<return_id>/
├── local_matching/
│   └── <product_id>_vs_<return_id>/
└── mesh/
    └── <product_id>_vs_<return_id>/
```

Các thống kê thường có:

- Difference area.
- Largest damage area.
- Damage score.
- Components before/after validation.
- Pixels removed by validation.
- SuperPoint keypoints/descriptors.
- Raw/good/inlier matches.
- Alignment method.
- TPS control points.
- Triangle count.
- Average/median/min/max SSIM.

---

## 7. Cài đặt

### 7.1. Yêu cầu hệ thống

- DbB postgre
- Python `>= 3.10`.
- Khuyến nghị dùng môi trường ảo.
- Có thể chạy CPU; GPU sẽ tăng tốc nếu dùng PyTorch CUDA.

### 7.2. Tạo virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Lưu ý: `opencv-contrib-python` được dùng thay vì `opencv-python` để hỗ trợ đầy đủ các hàm OpenCV mở rộng. Không nên cài đồng thời cả hai package này trong cùng environment.

Tải db postpress rồi tự tùy chỉnh

Tạo db 

tự config file database.py


<img width="331" height="181" alt="image" src="https://github.com/user-attachments/assets/683ea357-2de3-4f4c-91a2-070c48281fd4" />

-- =========================
-- 1. TẠO SEQUENCE
-- =========================

CREATE SEQUENCE IF NOT EXISTS public.return_image_analysis_id_seq
    START WITH 1
    INCREMENT BY 1;

CREATE SEQUENCE IF NOT EXISTS public.product_image_feature_id_seq
    START WITH 1
    INCREMENT BY 1;


-- =========================
-- 2. TẠO BẢNG return_image_analysis
-- =========================

CREATE TABLE IF NOT EXISTS public.return_image_analysis
(
    id integer NOT NULL DEFAULT nextval('public.return_image_analysis_id_seq'::regclass),
    product_id integer,
    image_path text,
    object_area double precision,
    perimeter double precision,
    width integer,
    height integer,
    difference_area double precision,
    largest_damage_area double precision,
    created_at timestamp without time zone DEFAULT now(),
    damage_score double precision,
    CONSTRAINT return_image_analysis_pkey PRIMARY KEY (id)
);

ALTER TABLE public.return_image_analysis
    OWNER TO postgres;

ALTER SEQUENCE public.return_image_analysis_id_seq
    OWNED BY public.return_image_analysis.id;


-- =========================
-- 3. TẠO BẢNG product_image_feature
-- =========================

CREATE TABLE IF NOT EXISTS public.product_image_feature
(
    id integer NOT NULL DEFAULT nextval('public.product_image_feature_id_seq'::regclass),
    image_path text,
    object_area double precision,
    perimeter double precision,
    width integer,
    height integer,
    texture_feature jsonb,
    created_at timestamp without time zone DEFAULT now(),
    CONSTRAINT product_image_feature_pkey PRIMARY KEY (id)
);

ALTER TABLE public.product_image_feature
    OWNER TO postgres;

ALTER SEQUENCE public.product_image_feature_id_seq
    OWNED BY public.product_image_feature.id;

---

## 8. Cách chạy

Chạy main.py


```

Pipeline
Option 1: thêm ảnh
Option 2: chọn ảnh return

Chạy lại
Kích hoạt lại moi trường ảo, ví dụ:
C:\Users\hung\Documents\Module4\image-damage-detection\venv\Scripts\ + activate
python main.py
---

## 9. Cấu hình quan trọng

Các ngưỡng nên tinh chỉnh trong `mesh_config.py`:

| Tham số | Ý nghĩa | Gợi ý |
|---|---|---|
| `DEBUG` | Bật/tắt lưu ảnh debug. | Tắt khi chạy batch lớn. |
| `MAX_TOTAL_TRIANGLES` | Giới hạn số tam giác tối đa. | Tăng nếu cần chi tiết hơn. |
| `TOP_K` | Số tam giác nghi ngờ được ưu tiên refine. | Tăng để bắt nhiều vùng nhỏ hơn. |
| `MAX_DEPTH` | Độ sâu refine tối đa. | Tăng sẽ tốn thời gian hơn. |
| `MIN_TRIANGLE_AREA` | Diện tích nhỏ nhất để tiếp tục refine. | Giảm nếu cần bắt hư hỏng nhỏ. |
| `SIMILARITY_THRESHOLD` | Ngưỡng similarity để coi vùng là đáng nghi. | Giảm để pipeline bảo thủ hơn. |
| `FEATURE_THRESHOLD` | Ngưỡng descriptor similarity để loại vùng có feature quá giống. | Tăng/giảm tùy texture sản phẩm. |
| `SSIM_THRESHOLD` | Ngưỡng SSIM để loại vùng quá giống. | Thường giữ quanh `0.90`. |
| `MIN_AREA_THRESHOLD` | Diện tích component nhỏ nhất. | Giảm nếu cần bắt vết nhỏ. |
| `DIFF_THRESHOLD` | Ngưỡng khác biệt pixel/gradient. | Giảm để nhạy hơn, tăng để ít nhiễu hơn. |
| `SIMILARITY_WORKERS` | Số luồng tính similarity. | Đặt theo số core CPU. |

---

## 10. Gợi ý debug

### 10.1. Mask bị rỗ hoặc đứt đoạn

Thử:

- Giảm `MIN_AREA_THRESHOLD`.
- Giảm `DIFF_THRESHOLD`.
- Kiểm tra morphology open có làm mất chi tiết nhỏ không.
- Kiểm tra alignment trước/sau trong debug output.

### 10.2. Mask tô quá rộng

Thử:

- Ưu tiên pixel-level damage map thay vì fill toàn tam giác.
- Tăng `DIFF_THRESHOLD`.
- Tăng `MIN_AREA_THRESHOLD`.
- Siết validation bằng SSIM/feature similarity.

### 10.3. Nhiều vùng trang trí bị nhận là hư hỏng

Thử:

- Tăng trọng số SSIM hoặc NCC.
- Tăng `FEATURE_THRESHOLD`/điều kiện feature support.
- Kiểm tra ánh sáng giữa hai ảnh.
- Kiểm tra TPS alignment có làm lệch hoa văn không.

### 10.4. Không phát hiện hư hỏng nhỏ

Thử:

- Tăng `TOP_K`.
- Tăng `MAX_DEPTH`.
- Giảm `MIN_TRIANGLE_AREA`.
- Giảm `SIMILARITY_THRESHOLD` hoặc `DIFF_THRESHOLD` tùy loại lỗi.

---

## 11. Quy ước thư mục đề xuất

```text
image-damage-detection/
├── images/
│   ├── 1.jpg
│   └── 2.jpg
├── debug_outputs/
├── product_pipeline.py
├── return_pipeline.py
├── mesh_damage_detector.py
├── mesh_config.py
├── mesh_builder.py
├── mesh_refiner.py
├── triangle_refiner.py
├── priority_scheduler.py
├── priority_mesh_scheduler.py
├── region_extractor.py
├── region_similarity.py
├── triangle_similarity.py
├── similarity_cache.py
├── component_validator.py
├── requirements.txt
└── README.md
```

---

## 12. Kiểm thử nhanh

Sau khi chạy pipeline, kiểm tra:

1. Ảnh `aligned_return` có khớp với ảnh product không.
2. `inlier_count` có đủ lớn không.
3. `inlier_ratio` có hợp lý không.
4. `damage_mask` có bám đúng vùng hư hỏng không.
5. `preview` có dễ quan sát không.
6. Component validation có loại nhầm vùng hư hỏng thật không.

---

## 13. Hướng phát triển tiếp theo

- Thêm CLI chuẩn bằng `argparse`.
- Thêm batch mode cho nhiều ảnh return.
- Lưu kết quả ra JSON/CSV.
- Tách config sang YAML để dễ tuning.
- Thêm test case có ground truth mask để đo IoU/F1.
- Thêm chế độ so sánh nhiều ảnh product reference.
- Tối ưu TPS/alignment cho ảnh có ít keypoint.
- Tối ưu pixel-level damage extraction trong vùng tam giác để giảm false positive trên hoa văn.

---

## 14. Ghi chú

- Nếu chạy trên server không cần GUI, có thể thay `opencv-contrib-python` bằng `opencv-contrib-python-headless` trong `requirements.txt`.
- Nếu dùng GPU, cài PyTorch theo hướng dẫn chính thức phù hợp với CUDA của máy.
- Không nên commit toàn bộ `debug_outputs/` nếu dung lượng lớn; nên thêm vào `.gitignore`.
