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
from PIL import Image
import io
import sys
import os
from typing import Tuple, Optional, Union, Dict, Any

_HAS_REMBG = False
remove = None
if os.environ.get("USE_REMBG", "1") == "1":
    try:
        from rembg import remove
        _HAS_REMBG = True
    except Exception:
        _HAS_REMBG = False
        remove = None


# ===============================================================================
# PHA 1: TIỀN XỬ LÝ
# (Chỉ gồm các bước AN TOÀN, không sửa pixel bên trong vật thể, không làm biến
#  dạng ảnh -- tách nền + align crop theo bounding box vật thể)
# ===============================================================================

_onnx_segmenter_session = None

def get_onnx_segmenter_session():
    """Tải ONNX session trực tiếp (u2netp) để tách nền mà không cần numba/rembg."""
    global _onnx_segmenter_session
    if _onnx_segmenter_session is None:
        try:
            import onnxruntime as ort
            candidates = [
                os.path.expanduser(r"~/.rembg/models/u2netp/u2netp.onnx"),
                os.path.expanduser(r"~/.rembg/models/bria-rmbg/bria-rmbg.onnx"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "u2netp.onnx"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    _onnx_segmenter_session = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
                    break
        except Exception as e:
            print(f"[WARN] Khong the tai ONNX model: {e}")
    return _onnx_segmenter_session


def remove_background_gray(bgr_img, gray_value=40):
    """
    Tách nền, tô nền bằng màu XÁM TRUNG TÍNH (không dùng đen tuyệt đối).
    Lý do dùng xám: đen tuyệt đối [0,0,0] tạo cạnh tương phản quá gắt ngay tại biên
    vật thể/nền, dễ bị các thuật toán so sánh (SuperPoint, SIFT) hiểu nhầm thành
    1 cạnh vật lý thật -> gây nhiễu ngược lại cho bước so sánh sau này.
    """
    orig_h, orig_w = bgr_img.shape[:2]
    object_mask = None

    # 1. Thử rembg nếu khả dụng
    if _HAS_REMBG and remove is not None:
        try:
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            result_bytes = remove(buf.getvalue())
            result_img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
            result_np = np.array(result_img)
            object_mask = result_np[:, :, 3]
        except Exception:
            object_mask = None

    # 2. Thử mô hình ONNX u2netp trực tiếp (an toàn trên Windows AppLocker)
    if object_mask is None:
        session = get_onnx_segmenter_session()
        if session is not None:
            try:
                img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(img_rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm_img = (resized - mean) / std
                input_tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, :, :, :]

                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                outputs = session.run([output_name], {input_name: input_tensor})
                pred = outputs[0][0, 0, :, :]

                ma, mi = np.max(pred), np.min(pred)
                pred_norm = (pred - mi) / (ma - mi + 1e-8)
                mask = cv2.resize(pred_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                object_mask = (mask > 0.45).astype(np.uint8) * 255
            except Exception as e:
                print(f"[WARN] ONNX inference loi: {e}")
                object_mask = None

    # 3. Fallback thuần OpenCV Otsu
    if object_mask is None:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        object_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    # Tô nền bằng màu xám trung tính (gray_value)
    alpha = (object_mask.astype(np.float64) / 255.0)[:, :, np.newaxis]
    rgb = bgr_img.astype(np.float64)
    bg = np.full_like(rgb, gray_value, dtype=np.float64)

    composited = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return composited, object_mask


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
    inner_mask = cv2.erode(object_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.bitwise_and(mask, inner_mask)

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
    inner_mask = cv2.erode(object_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mask = cv2.bitwise_and(mask, inner_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _filter_small_blobs(mask, min_area):
    """Loại các vùng nhỏ lẻ tẻ (khả năng cao là nhiễu hạt, không phải vùng thật)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
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

def process_and_detect(img_path: str, label: str = "customer_front", output_dir: str = ".", target_size: int = 1024) -> Dict[str, Any]:
    """
    Standardized Image Preprocessing and Optical Quality Gate (1024x1024 Pure White Canvas):
    1. High-fidelity background removal (RGBA transparency preservation or lightweight AI RemBG).
    2. Synchronous bounding-box centering and scaling to target_size x target_size (pure white [255, 255, 255]).
    3. Specular glare and saturated pixel scanning directly on the 1024x1024 canvas.
    4. Export all standardized maps strictly at 1024x1024 with English naming conventions.
    """
    canvas_bgr, canvas_mask = clean_rembg_image(img_path, target_size=target_size, return_mask=True)
    if canvas_bgr is None:
        raise FileNotFoundError(f"[ERROR] Failed to read or preprocess image: {img_path}")

    result = build_ignore_mask(canvas_bgr, canvas_mask)
    overlay = draw_specular_contours(canvas_bgr, result["specular_mask"])

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/{label}_processed.png", canvas_bgr)
    cv2.imwrite(f"{output_dir}/{label}_specular_mask.png", result["specular_mask"])
    cv2.imwrite(f"{output_dir}/{label}_saturated_mask.png", result["saturated_mask"])
    cv2.imwrite(f"{output_dir}/{label}_ignore_mask.png", result["ignore_mask"])
    cv2.imwrite(f"{output_dir}/{label}_overlay_contour.png", overlay)

    print(f"[PREPROCESS] [{label}] Standardized shape: {canvas_bgr.shape[:2]} | ignore_ratio: {result['ignore_ratio']*100:.2f}% | reshoot_required: {result['should_reshoot']}")
    return result


if __name__ == "__main__":
    default_files = {
        "reference_front.jpg": "reference_front",
        "customer_front.jpg": "customer_front",
    }
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        input_label = sys.argv[2] if len(sys.argv) > 2 else "sample"
        process_and_detect(input_path, input_label)
    else:
        for img_file, prefix in default_files.items():
            if os.path.exists(img_file):
                process_and_detect(img_file, prefix)


# ===============================================================================
# STANDARDIZED CANVAS PREPROCESSOR (1024x1024 WHITE CANVAS VIA REMBG / RGBA)
# ===============================================================================

_REMBG_SESSION = None

def get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            from rembg import new_session
            _REMBG_SESSION = new_session('u2net')
        except Exception:
            try:
                from rembg import new_session
                _REMBG_SESSION = new_session('u2netp')
            except Exception:
                _REMBG_SESSION = False
    return _REMBG_SESSION


def clean_rembg_image(
    img_path: str,
    target_size: int = 1024,
    return_mask: bool = False
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    High-fidelity background removal and canvas standardization (target_size x target_size):
    1. Native RGBA image (transparent _nb.png from Cloudinary):
       Instantaneous transparency preservation (0 RAM overhead, 0 AI latency).
    2. Standard 3-channel BGR:
       If USE_REMBG=1, runs RemBG. Otherwise uses robust C++ contrast morphology segmentation.
    3. Synchronously crop both BGR and binary Mask tightly along object bounding box via cv2.boundingRect.
    4. Aspect-ratio preserving scaling onto fixed pure white canvas (255, 255, 255) with 48px padding.
    """
    raw_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        return (None, None) if return_mask else None

    clean_bgr = None
    mask = None

    # CASE A: Native RGBA image with pre-existing transparency
    if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
        alpha = raw_img[:, :, 3]
        if np.any(alpha < 240):
            bgr = raw_img[:, :, :3]
            alpha_f = alpha.astype(np.float32) / 255.0
            clean_bgr = (bgr.astype(np.float32) * alpha_f[:, :, None] + 255.0 * (1.0 - alpha_f[:, :, None])).astype(np.uint8)
            mask = (alpha > 128).astype(np.uint8) * 255

    # CASE B: Standard RGB/BGR image requiring background segmentation
    if clean_bgr is None:
        bgr = raw_img[:, :, :3] if len(raw_img.shape) == 3 and raw_img.shape[2] == 4 else raw_img
        
        # Safe downscaling to avoid memory pressure on large smartphone captures
        h_orig, w_orig = bgr.shape[:2]
        max_dim = max(h_orig, w_orig)
        if max_dim > 1024:
            scale_safe = 1024.0 / float(max_dim)
            bgr = cv2.resize(bgr, (int(w_orig * scale_safe), int(h_orig * scale_safe)), interpolation=cv2.INTER_AREA)

        if os.environ.get("USE_REMBG", "1") == "1":
            session = get_rembg_session()
            if session:
                try:
                    from rembg import remove
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    out_pil = remove(Image.fromarray(rgb), session=session, only_mask=True)
                    mask = np.array(out_pil)
                except Exception as e:
                    print(f"[WARN] RemBG execution failed ({e}), using morphology fallback.")
                    mask = None
        
        if mask is not None:
            # Cut table contact shadow (bottom 4% of object bounding box resting on table)
            nz = cv2.findNonZero(mask)
            if nz is not None:
                bx_t, by_t, bw_t, bh_t = cv2.boundingRect(nz)
                table_cut_y = by_t + int(bh_t * 0.96)
                mask[table_cut_y:, :] = 0

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                mask_clean = np.zeros_like(mask)
                cv2.drawContours(mask_clean, [c], -1, 255, -1)
                mask = mask_clean
                
            # Sharp alpha masking to completely eliminate gray halo around rim and foot
            alpha_f = (mask > 128).astype(np.float32)
            clean_bgr = (bgr.astype(np.float32) * alpha_f[:, :, None] + 255.0 * (1.0 - alpha_f[:, :, None])).astype(np.uint8)
        else:
            # High-performance C++ morphology background segmentation (zero memory overhead)
            corner_bgr = np.median(bgr[:25, :25], axis=(0, 1)).astype(np.uint8)
            diff_map = cv2.absdiff(bgr, corner_bgr)
            gray_diff = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            alpha_f = (mask > 128).astype(np.float32)
            clean_bgr = (bgr.astype(np.float32) * alpha_f[:, :, None] + 255.0 * (1.0 - alpha_f[:, :, None])).astype(np.uint8)

    # Crop tightly to object bounding box via cv2.boundingRect (zero memory allocation)
    mask_cropped = None
    if mask is not None:
        bx, by, bw, bh = cv2.boundingRect(mask)
        if bw > 20 and bh > 20:
            margin = 2 # Tight crop to prevent table/room border leakage
            y_min = max(0, by - margin)
            x_min = max(0, bx - margin)
            y_max = min(clean_bgr.shape[0], by + bh + margin)
            x_max = min(clean_bgr.shape[1], bx + bw + margin)
            clean_bgr = clean_bgr[y_min:y_max, x_min:x_max]
            mask_cropped = mask[y_min:y_max, x_min:x_max]
        else:
            mask_cropped = mask.copy()
    else:
        mask_cropped = np.full(clean_bgr.shape[:2], 255, dtype=np.uint8)

    # Center and scale onto fixed target_size x target_size pure white canvas
    h, w = clean_bgr.shape[:2]
    padding = 48
    scale = min((target_size - 2 * padding) / float(h), (target_size - 2 * padding) / float(w))
    nh, nw = int(h * scale), int(w * scale)
    resized_bgr = cv2.resize(clean_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask_cropped, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas_bgr = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)

    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    canvas_bgr[y_off:y_off+nh, x_off:x_off+nw] = resized_bgr
    canvas_mask[y_off:y_off+nh, x_off:x_off+nw] = resized_mask
    canvas_bgr[canvas_mask == 0] = [255, 255, 255]

    if return_mask:
        return canvas_bgr, canvas_mask
    return canvas_bgr
