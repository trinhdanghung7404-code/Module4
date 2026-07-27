"""
===============================================================================
MODULE GHÉP: PHA 1 (TIỀN XỬ LÝ) + PHA 2 (MASK DỮ LIỆU - QUÉT LÓA SÁNG)
===============================================================================
Ghép từ 3 file: preprocess_minimal.py, detect_uncertain_regions.py, main4.py

Lưu ý về các phần đã thay đổi so với bản gốc:
- "Resize đồng bộ" và "Cân sáng nhẹ" vẫn giữ code đầy đủ nhưng COMMENT LẠI
  (không chạy mặc định) để cô lập biến số khi đang test riêng Pha 2 (detect
  lóa sáng) -- giữ ảnh ở trạng thái gần nguyên bản giúp đối chiếu kết quả detect
  dễ hơn, không lẫn với thay đổi do resize/cân sáng gây ra.
  Muốn test lại 2 bước này: bỏ comment hàm + bỏ comment dòng gọi trong main().
- "Vùng bóng đổ" (detect_shadow_mask) đã BỎ HẲN khỏi pipeline này theo yêu cầu
  (ngưỡng cũ chưa calibrate tốt, dễ bắt nhầm cả phần đổ khối tự nhiên của mặt
  cong). Có thể bổ sung lại sau khi có thêm dữ liệu thật để tinh chỉnh ngưỡng.
"""

import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io
import sys


# ===============================================================================
# PHA 1: TIỀN XỬ LÝ
# (Chỉ gồm các bước AN TOÀN, không sửa pixel bên trong vật thể, không làm biến
#  dạng ảnh -- tách nền + align crop theo bounding box vật thể)
# ===============================================================================

def remove_background_gray(bgr_img, gray_value=40):
    """
    Tách nền bằng rembg, tô nền bằng màu XÁM TRUNG TÍNH (không dùng đen tuyệt đối).
    Lý do dùng xám: đen tuyệt đối [0,0,0] tạo cạnh tương phản quá gắt ngay tại biên
    vật thể/nền, dễ bị các thuật toán so sánh (SuperPoint, SIFT) hiểu nhầm thành
    1 cạnh vật lý thật -> gây nhiễu ngược lại cho bước so sánh sau này.
    """
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    result_bytes = remove(buf.getvalue())

    result_img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    result_np = np.array(result_img)

    rgb = result_np[:, :, :3].astype(np.float64)
    alpha = result_np[:, :, 3:4] / 255.0
    bg = np.full_like(rgb, gray_value, dtype=np.float64)

    composited = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    composited_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
    object_mask = result_np[:, :, 3]
    return composited_bgr, object_mask


def crop_to_object(bgr_img, object_mask, margin_ratio=0.05):
    """
    Crop ảnh (VÀ mask) về đúng bounding box của vật thể (+ margin nhỏ).
    Đây là cách "align" AN TOÀN nhất cho vật thể có mặt cong (bình gốm):
    không warp/xoay ép làm méo hoa văn, chỉ đưa ảnh về đúng khung chứa vật thể.
    Trả về cả mask đã crop cùng toạ độ để dùng ở các bước sau.
    """
    ys, xs = np.where(object_mask > 10)
    if len(xs) == 0:
        return bgr_img, object_mask  # fallback nếu mask rỗng
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    h, w = bgr_img.shape[:2]
    mw = int((x1 - x0) * margin_ratio)
    mh = int((y1 - y0) * margin_ratio)

    x0 = max(0, x0 - mw)
    y0 = max(0, y0 - mh)
    x1 = min(w, x1 + mw)
    y1 = min(h, y1 + mh)

    return bgr_img[y0:y1, x0:x1], object_mask[y0:y1, x0:x1]


# --- [TẠM TẮT] Resize đồng bộ về kích thước cố định ---------------------------
# Kỹ thuật KHÔNG sai, chỉ tạm comment để cô lập biến số khi debug riêng Pha 2.
# Muốn bật lại để test: bỏ comment hàm dưới + bỏ comment dòng gọi trong main().
#
# def resize_synchronized(img_a, img_b, mask_a, mask_b, output_size=(1024, 1024)):
#     """Resize đồng bộ 2 ảnh (và mask) về cùng kích thước chuẩn."""
#     img_a_r = cv2.resize(img_a, output_size, interpolation=cv2.INTER_AREA)
#     img_b_r = cv2.resize(img_b, output_size, interpolation=cv2.INTER_AREA)
#     mask_a_r = cv2.resize(mask_a, output_size, interpolation=cv2.INTER_NEAREST)
#     mask_b_r = cv2.resize(mask_b, output_size, interpolation=cv2.INTER_NEAREST)
#     return img_a_r, img_b_r, mask_a_r, mask_b_r


# --- [TẠM TẮT] Cân sáng nhẹ (match mean/std kênh L theo ảnh tham chiếu) -------
# Kỹ thuật KHÔNG sai, chỉ tạm comment cùng lý do trên: giữ ảnh ở độ sáng gốc
# (chưa chỉnh) giúp thấy rõ đốm lóa/vùng bão hoà thật do ánh sáng thật khi test
# Pha 2, tránh lẫn với thay đổi sáng do thuật toán cân sáng gây ra.
# Muốn bật lại để test: bỏ comment hàm dưới + bỏ comment dòng gọi trong main().
#
# def match_lightness(source_bgr, reference_bgr, source_mask=None, reference_mask=None):
#     """Dịch mean/std kênh L của source theo reference, tính trên object_mask thật."""
#     src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
#     ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
#     l_src, a_src, b_src = cv2.split(src_lab)
#     l_ref, _, _ = cv2.split(ref_lab)
#
#     def mean_std_on_mask(channel, mask):
#         if mask is not None:
#             vals = channel[mask > 10]
#         else:
#             vals = channel.flatten()
#         if len(vals) == 0:
#             return channel.mean(), channel.std() + 1e-6
#         return vals.mean(), vals.std() + 1e-6
#
#     mean_src, std_src = mean_std_on_mask(l_src, source_mask)
#     mean_ref, std_ref = mean_std_on_mask(l_ref, reference_mask)
#
#     l_matched = (l_src - mean_src) * (std_ref / std_src) + mean_ref
#     l_matched = np.clip(l_matched, 0, 255)
#     matched_lab = cv2.merge([l_matched.astype(np.uint8), a_src.astype(np.uint8), b_src.astype(np.uint8)])
#     return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


# ===============================================================================
# PHA 2: MASK DỮ LIỆU -- DETECT VÙNG NGHI NGỜ (QUÉT LÓA SÁNG)
# (Chỉ giữ: đốm sáng specular (kế thừa từ main4.py, có bổ sung morphology + lọc
#  blob nhỏ) + vùng bão hoà cực trị. Vùng bóng đổ đã bị loại bỏ khỏi pipeline này.)
# ===============================================================================

def detect_specular_mask(
    bgr_img,
    object_mask,
    min_area=10
):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # So sánh độ sáng với vùng xung quanh
    local_v = cv2.GaussianBlur(v, (31, 31), 0)
    diff = cv2.subtract(v, local_v)

    # 1. Lõi lóa: phải rất sáng và nổi bật
    strong_glare = (
        (v >= 252) &
        (s <= 15) &
        (diff >= 12)
    ).astype(np.uint8) * 255

    # 2. Phần lóa mềm xung quanh
    soft_glare = (
        (v >= 235) &
        (s <= 40) &
        (diff >= 6)
    ).astype(np.uint8) * 255

    # Chỉ cho vùng mềm lan khoảng 5 pixel quanh lõi
    expand_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)
    )
    allowed_area = cv2.dilate(strong_glare, expand_kernel)

    soft_glare = cv2.bitwise_and(soft_glare, allowed_area)

    # Gộp lõi và vùng mềm
    mask = cv2.bitwise_or(strong_glare, soft_glare)
    mask = cv2.bitwise_and(mask, object_mask)

    # Chỉ nối các khe nhỏ
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = _filter_small_blobs(mask, min_area)
    return mask


def detect_saturated_mask(bgr_img, object_mask, near_white=250, near_black=5):
    """
    Vùng bão hoà cực trị: cháy sáng gần trắng tuyệt đối hoặc đen tuyệt đối.
    Đây là vùng ĐÃ MẤT THÔNG TIN VĨNH VIỄN -- không có thuật toán nào phục hồi
    được, nên luôn phải ignore ở bước so sánh, không phụ thuộc đã cân sáng chưa.
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    mask = ((l >= near_white) | (l <= near_black)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, object_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _filter_small_blobs(mask, min_area):
    """Loại các vùng nhỏ lẻ tẻ (khả năng cao là nhiễu hạt, không phải vùng thật)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean, [c], -1, 255, -1)
    return clean


def draw_specular_contours(bgr_img, specular_mask, color=(0, 255, 0), thickness=1):
    """
    Vẽ viền mỏng quanh vùng lóa để kiểm tra trực quan bằng mắt
    (kế thừa cách trình bày kết quả từ main4.py).
    """
    contours, _ = cv2.findContours(specular_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = bgr_img.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def build_ignore_mask(bgr_img, object_mask):
    """
    Gộp specular + saturated thành 1 ignore_mask duy nhất, kèm tỉ lệ diện tích
    để cảnh báo khi ảnh quá tệ (nên yêu cầu chụp lại thay vì kết luận vội).
    """
    specular = detect_specular_mask(bgr_img, object_mask)
    saturated = detect_saturated_mask(bgr_img, object_mask)

    ignore_mask = cv2.bitwise_or(specular, saturated)

    object_area = int((object_mask > 10).sum())
    ignore_area = int((ignore_mask > 10).sum())
    ignore_ratio = ignore_area / object_area if object_area > 0 else 0.0

    return {
        "specular_mask": specular,
        "saturated_mask": saturated,
        "ignore_mask": ignore_mask,
        "ignore_ratio": ignore_ratio,
        "should_reshoot": ignore_ratio > 0.15,  # ngưỡng gợi ý, cần calibrate thêm
    }

# ===============================================================================
# MAIN: chạy thử toàn bộ pipeline trên 1 hoặc 2 ảnh
# ===============================================================================

def process_and_detect(img_path, label="anh", output_dir="."):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

    # ---- PHA 1: tiền xử lý (2 bước an toàn, luôn chạy) ----
    nobg, mask = remove_background_gray(bgr)
    cropped, mask_cropped = crop_to_object(nobg, mask)

    # ---- [TẠM TẮT] resize đồng bộ + cân sáng nhẹ ----
    # Muốn test 2 bước này: bỏ comment 2 hàm ở PHA 1 phía trên, rồi bỏ comment
    # đúng 2 dòng dưới đây (lưu ý match_lightness cần 1 ảnh reference thật,
    # ví dụ ảnh gốc shop, không phải chính ảnh nó -- ví dụ dưới chỉ minh hoạ cú pháp):
    #
    # cropped, _, mask_cropped, _ = resize_synchronized(cropped, cropped, mask_cropped, mask_cropped)
    # cropped = match_lightness(cropped, anh_tham_chieu, mask_cropped, mask_tham_chieu)

    # ---- PHA 2: detect vùng nghi ngờ (quét lóa sáng) ----
    result = build_ignore_mask(cropped, mask_cropped)
    overlay = draw_specular_contours(cropped, result["specular_mask"])

    cv2.imwrite(f"{output_dir}/{label}_processed.png", cropped)
    cv2.imwrite(f"{output_dir}/{label}_specular_mask.png", result["specular_mask"])
    cv2.imwrite(f"{output_dir}/{label}_saturated_mask.png", result["saturated_mask"])
    cv2.imwrite(f"{output_dir}/{label}_ignore_mask.png", result["ignore_mask"])
    cv2.imwrite(f"{output_dir}/{label}_overlay_contour.png", overlay)

    print(f"[{label}] ignore_ratio = {result['ignore_ratio']*100:.2f}%  "
          f"should_reshoot = {result['should_reshoot']}")
    return result


if __name__ == "__main__":
    # ---- CẤU HÌNH MẶC ĐỊNH ----
    # Input: anh_goc.jpg + anh_khach.jpg (thiếu 1 trong 2 thì vẫn chạy cái còn lại)
    # Output: prefix tự động là goc_ hoặc khach_ dựa trên tên input file
    
    default_files = {
        "anh_goc.jpg": "goc",      # input_file: output_prefix
        "anh_khach.jpg": "khach",
    }
    
    # Nếu truyền tham số từ command line, chỉ chạy file đó
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        # Tự động extract prefix từ tên file
        # ví dụ: "anh_khach.jpg" → "khach", "anh_goc.jpg" → "goc"
        if "khach" in img_path.lower():
            label = "khach"
        elif "goc" in img_path.lower():
            label = "goc"
        else:
            label = sys.argv[2] if len(sys.argv) > 2 else "anh"
        
        process_and_detect(img_path, label)
    else:
        # Chạy mặc định: cả 2 file anh_goc.jpg + anh_khach.jpg (nếu tồn tại)
        for img_file, prefix in default_files.items():
            try:
                process_and_detect(img_file, prefix)
            except FileNotFoundError:
                print(f"⚠️  Không tìm thấy {img_file}, bỏ qua...")
