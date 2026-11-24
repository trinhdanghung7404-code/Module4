"""
core/matcher.py
===============
Template matching, ghost overlay generation, and upload validation
to prevent fake submissions (blur, foreign objects, keyboard, etc.).
"""

import os
import cv2
import numpy as np
from typing import Optional, Dict, Any

from core.preprocessor import get_onnx_segmenter_session


def process_single_ghost_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Tách nền trực tiếp và cắt sát viền alpha để tạo ảnh bóng mờ (ghost overlay).
    Sử dụng ONNX session nếu khả dụng, fallback bằng OpenCV Otsu.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    session = get_onnx_segmenter_session()
    alpha = None

    if session is not None:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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
            alpha = (mask > 0.45).astype(np.uint8) * 255
        except Exception as e:
            print(f"[WARN] ONNX ghost segmentation error: {e}, using Otsu fallback")
            alpha = None

    if alpha is None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        alpha = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    coords = cv2.findNonZero(alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = bgra[y:y+h, x:x+w]
    else:
        cropped = bgra

    return cropped


def compute_similarity_from_bgr(ghost_path: str, cap_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Đánh giá độ tương đồng thực tế giữa ảnh mẫu và ma trận ảnh chụp BGR:
    - Multi-scale template matching xác định vùng vật thể.
    - So sánh cấu trúc NCC + đường nét viền Sobel + histogram cường độ + histogram màu HSV.
    """
    try:
        if not os.path.exists(ghost_path):
            return None
        ghost_bgra = cv2.imread(ghost_path, cv2.IMREAD_UNCHANGED)
        if ghost_bgra is None or cap_bgr is None:
            return None

        if len(ghost_bgra.shape) == 3 and ghost_bgra.shape[2] == 4:
            ghost_bgr = ghost_bgra[:, :, :3]
            alpha = ghost_bgra[:, :, 3]
        elif len(ghost_bgra.shape) == 3:
            ghost_bgr = ghost_bgra
            # Tự động tách mặt nạ bình nếu ảnh tham chiếu có nền trắng thương mại điện tử
            if np.all(ghost_bgr[:15, :15] > 235) and np.all(ghost_bgr[-15:, -15:] > 235):
                alpha = ((ghost_bgr < 235).any(axis=2).astype(np.uint8)) * 255
                alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            else:
                alpha = np.ones(ghost_bgra.shape[:2], dtype=np.uint8) * 255
        else:
            ghost_bgr = cv2.cvtColor(ghost_bgra, cv2.COLOR_GRAY2BGR)
            if np.all(ghost_bgr[:15, :15] > 235) and np.all(ghost_bgr[-15:, -15:] > 235):
                alpha = (ghost_bgra < 235).astype(np.uint8) * 255
                alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            else:
                alpha = np.ones(ghost_bgra.shape[:2], dtype=np.uint8) * 255

        H, W = cap_bgr.shape[:2]
        gh, gw = ghost_bgra.shape[:2]

        rel_base = H / float(gh)
        ds = 0.5
        s_cap = cv2.resize(cap_bgr, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_AREA)

        best_v = -1
        best_l = (0, 0)
        best_s = 0.60

        for s in [0.35, 0.42, 0.50, 0.58, 0.65, 0.72, 0.80, 0.88]:
            actual_s = rel_base * s
            tg_w = int(gw * actual_s)
            tg_h = int(gh * actual_s)
            if tg_w >= W or tg_h >= H or tg_w < 20 or tg_h < 20:
                continue

            small_g = cv2.resize(ghost_bgr, (int(tg_w * ds), int(tg_h * ds)), interpolation=cv2.INTER_AREA)
            small_a = cv2.resize(alpha,     (int(tg_w * ds), int(tg_h * ds)), interpolation=cv2.INTER_NEAREST)

            res = cv2.matchTemplate(s_cap, small_g, cv2.TM_CCOEFF_NORMED, mask=small_a)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_v:
                best_v = max_v
                best_l = max_l
                best_s = s

        actual_s = rel_base * best_s
        tw = int(gw * actual_s)
        th = int(gh * actual_s)
        x0 = int(best_l[0] / ds)
        y0 = int(best_l[1] / ds)

        y1 = min(H, y0 + th)
        x1 = min(W, x0 + tw)
        y0 = max(0, y0)
        x0 = max(0, x0)
        cap_roi = cap_bgr[y0:y1, x0:x1]

        if cap_roi.shape[0] < 20 or cap_roi.shape[1] < 20:
            cap_roi = cap_bgr

        SIZE = (180, 180)
        g_small = cv2.resize(ghost_bgr, SIZE, interpolation=cv2.INTER_AREA)
        a_raw = cv2.resize(alpha, SIZE, interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode((a_raw > 30).astype(np.uint8), kernel, iterations=2) > 0
        if mask.sum() < 50:
            mask = a_raw > 30

        c_small = cv2.resize(cap_roi, SIZE, interpolation=cv2.INTER_AREA)

        if mask.sum() < 50:
            return None

        g_gray = cv2.cvtColor(g_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        c_gray = cv2.cvtColor(c_small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gv = g_gray[mask]
        cv_ = c_gray[mask]

        # 1. Normalized Cross-Correlation (NCC)
        gm, cm = gv - gv.mean(), cv_ - cv_.mean()
        ncc = float(np.dot(gm, cm) / (np.sqrt((gm**2).sum() * (cm**2).sum()) + 1e-6))
        ncc = max(0.0, min(1.0, ncc))

        # 2. Histogram Correlation
        h1 = cv2.calcHist([gv.astype(np.uint8)], [0], None, [64], [0, 256])
        h2 = cv2.calcHist([cv_.astype(np.uint8)], [0], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        corr = max(0.0, min(1.0, corr))

        # 3. Sobel Edge Correlation
        gx1 = cv2.Sobel(g_gray, cv2.CV_32F, 1, 0)
        gy1 = cv2.Sobel(g_gray, cv2.CV_32F, 0, 1)
        mag1 = np.sqrt(gx1**2 + gy1**2)[mask]

        gx2 = cv2.Sobel(c_gray, cv2.CV_32F, 1, 0)
        gy2 = cv2.Sobel(c_gray, cv2.CV_32F, 0, 1)
        mag2 = np.sqrt(gx2**2 + gy2**2)[mask]

        edge_score = float(np.dot(mag1, mag2) / (np.sqrt((mag1**2).sum() * (mag2**2).sum()) + 1e-6))
        edge_score = max(0.0, min(1.0, edge_score))

        # 4. Color / HSV Hue & Saturation Correlation
        hsv_g = cv2.cvtColor(g_small, cv2.COLOR_BGR2HSV)
        hsv_c = cv2.cvtColor(c_small, cv2.COLOR_BGR2HSV)
        m_u8 = mask.astype(np.uint8)
        ch1 = cv2.calcHist([hsv_g], [0, 1], m_u8, [18, 16], [0, 180, 0, 256])
        ch2 = cv2.calcHist([hsv_c], [0, 1], m_u8, [18, 16], [0, 180, 0, 256])
        cv2.normalize(ch1, ch1)
        cv2.normalize(ch2, ch2)
        color_score = float(cv2.compareHist(ch1, ch2, cv2.HISTCMP_CORREL))
        color_score = max(0.0, min(1.0, color_score))

        # Điểm căn chỉnh quang học & cấu trúc (Contour Silhouette + Glaze Color Tone + Luminance):
        # Tránh phạt oan ảnh điện thoại tự nhiên vì vi sai góc nhỏ của NCC thô; ưu tiên viền và màu sắc men
        geom_tone = 0.45 * edge_score + 0.30 * color_score + 0.25 * corr
        score = round(float(np.clip(0.40 + 0.70 * max(0.0, geom_tone - 0.20), 0.0, 0.96)), 3)
        if color_score < 0.08:
            score *= 0.6
        score = round(float(np.clip(score, 0.0, 1.0)), 3)

        center_offset_x = float((x0 + tw / 2.0) - (W / 2.0)) / float(W)

        return {
            "score": score,
            "ncc": round(ncc, 3),
            "edge": round(edge_score, 3),
            "corr": round(corr, 3),
            "color": round(color_score, 3),
            "scale": round(best_s, 2),
            "center_offset_x": round(center_offset_x, 3),
        }
    except Exception as e:
        print(f"[WARN] Similarity computation error: {e}")
        return None


def validate_upload_match(ghost_path: str, image_bytes: bytes, ref_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Cửa soát chất lượng tức thì tại Tầng 1 (Fast Quality & Orientation Gate, < 120ms):
    1. Kiểm tra rung lắc quang học (Adaptive Sharpness / Anti-Motion Blur).
    2. Kiểm tra lóa sáng chói đèn diện rộng (Anti-Flash Glare Saturation).
    3. Xác thực góc xoay và cấu trúc qua Fast SIFT Alignment (nếu có ảnh tham chiếu).
    4. Kiểm tra cự ly và căn tâm khung hình (Scale & Centering Framing).
    Loại bỏ hoàn toàn bộ lọc màu tĩnh để tránh bắt nhầm màu men gốm trắng.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        cap_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cap_bgr is None:
            return {"valid": False, "message": "Không thể giải mã dữ liệu ảnh camera!"}

        ih, iw = cap_bgr.shape[:2]

        # 1. Kiểm tra độ nét quang học (Adaptive Sharpness cho gốm sứ)
        cw, ch = int(iw * 0.45), int(ih * 0.45)
        cx, cy = (iw - cw) // 2, (ih - ch) // 2
        crop = cap_bgr[cy:cy+ch, cx:cx+cw]
        g = cv2.cvtColor(cv2.resize(crop, (240, 240)), cv2.COLOR_BGR2GRAY).astype(np.float64)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        center_lap = float(np.mean(cv2.filter2D(g, -1, kernel)**2))

        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(gx**2 + gy**2)
        top_edge_grad = float(np.percentile(grad_mag, 90))

        # Nếu rung nhòe nghiêm trọng (Laplacian < 35 hoặc kết hợp mất nét viền)
        if center_lap < 35.0 or (center_lap < 60.0 and top_edge_grad < 28.0):
            return {
                "valid": False,
                "message": f"❌ Ảnh chụp bị rung/nhòe tay (Độ nét: {int(center_lap)}/50)! Vui lòng giữ chắc điện thoại và chạm màn hình lấy nét rõ trước khi chụp.",
            }

        # 2. Kiểm tra lóa sáng chói đèn diện rộng (Extreme Specular Saturation)
        sat_pixels = np.count_nonzero((crop[:, :, 0] >= 252) & (crop[:, :, 1] >= 252) & (crop[:, :, 2] >= 252))
        sat_ratio = float(sat_pixels) / float(max(1, cw * ch))
        if sat_ratio > 0.16:
            return {
                "valid": False,
                "message": f"👉 Ánh đèn chiếu thẳng vào mặt bình gây chói lóa ({round(sat_ratio*100, 1)}%)! Vui lòng nghiêng nhẹ góc máy để tránh phản chiếu đèn.",
            }

        # 3. Xác thực góc xoay mặt sản phẩm qua Fast SIFT Alignment
        if ref_path and os.path.exists(ref_path):
            raw_ref = cv2.imread(ref_path)
            if raw_ref is not None:
                sift_fast = cv2.SIFT_create(nfeatures=500)
                g_cap = cv2.cvtColor(cv2.resize(cap_bgr, (480, 480)), cv2.COLOR_BGR2GRAY)
                g_ref = cv2.cvtColor(cv2.resize(raw_ref, (480, 480)), cv2.COLOR_BGR2GRAY)
                kp1, d1 = sift_fast.detectAndCompute(g_cap, None)
                kp2, d2 = sift_fast.detectAndCompute(g_ref, None)
                if d1 is not None and d2 is not None and len(d1) >= 4 and len(d2) >= 4:
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                    matches = sorted(bf.match(d1, d2), key=lambda x: x.distance)
                    if len(matches) >= 6:
                        src = np.float32([kp1[m.queryIdx].pt for m in matches[:60]]).reshape(-1, 1, 2)
                        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:60]]).reshape(-1, 1, 2)
                        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)
                        nin = int(inliers.sum()) if inliers is not None else 0
                        if M is not None and nin >= 4:
                            theta_deg = abs(np.arctan2(M[1, 0], M[0, 0]) * 180.0 / np.pi)
                            if theta_deg > 28.0:
                                return {
                                    "valid": False,
                                    "message": f"👉 Mặt bình bị xoay lệch góc quá nhiều ({int(theta_deg)}°)! Vui lòng xoay bình hướng thẳng mặt về ống kính.",
                                }

        # 4. Kiểm tra tương đồng cấu trúc và cự ly với ghost template
        if ghost_path and os.path.exists(ghost_path):
            sim_res = compute_similarity_from_bgr(ghost_path, cap_bgr)
            if sim_res is not None:
                score = sim_res.get("score", 0.0)
                edge = sim_res.get("edge", 0.0)
                ncc = sim_res.get("ncc", 0.0)
                scale = sim_res.get("scale", 0.60)
                offset_x = sim_res.get("center_offset_x", 0.0)

                # Chặn ảnh chụp vật thể lạ hoàn toàn (bàn phím, sàn nhà, dây điện, mặt bàn không có bình)
                if ncc < 0.12 or (ncc < 0.16 and edge < 0.12):
                    return {
                        "valid": False,
                        "message": f"Không nhận diện được bình gốm trong khung hình (Cấu trúc: {round(ncc*100, 1)}%, Viền: {round(edge*100, 1)}%). Vui lòng căn chỉnh bình gốm vào giữa khung ngắm!",
                    }

                # Cảnh báo hướng dẫn cự ly và căn tâm
                if score < 0.22:
                    if scale <= 0.30:
                        return {
                            "valid": False,
                            "message": "👉 Đưa điện thoại lại gần bình hơn để bình vừa vặn trong khung hình!",
                        }
                    elif scale >= 0.85:
                        return {
                            "valid": False,
                            "message": "👉 Lùi điện thoại ra xa một chút để thấy trọn vẹn cả miệng và đáy bình!",
                        }
                    elif abs(offset_x) > 0.22:
                        direction = "sang phải" if offset_x > 0 else "sang trái"
                        return {
                            "valid": False,
                            "message": f"👉 Dịch camera {direction} một chút để đưa bình vào chính giữa khung ngắm!",
                        }
                    else:
                        return {
                            "valid": False,
                            "message": "👉 Mặt bình bị xoay lệch góc quá nhiều! Vui lòng xoay bình hướng thẳng mặt này về ống kính.",
                        }

        return {"valid": True}
    except Exception as e:
        print(f"[WARN] Upload validation error: {e}")
        return {"valid": True}
