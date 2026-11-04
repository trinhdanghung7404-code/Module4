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

        ghost_bgr = ghost_bgra[:, :, :3]
        alpha = ghost_bgra[:, :, 3]

        H, W = cap_bgr.shape[:2]
        gh, gw = ghost_bgra.shape[:2]

        rel_base = H / float(gh)
        ds = 0.5
        s_cap = cv2.resize(cap_bgr, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_AREA)

        best_v = -1
        best_l = (0, 0)
        best_s = 0.60

        for s in [0.40, 0.48, 0.54, 0.60, 0.66, 0.72, 0.80]:
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

        # Overall similarity score: Uu tien do tuong quan cau truc (NCC) va duong vien Sobel
        score = 0.45 * ncc + 0.35 * edge_score + 0.10 * corr + 0.10 * color_score
        if color_score < 0.08:
            score *= 0.6
        score = round(float(np.clip(score, 0.0, 1.0)), 3)

        return {
            "score": score,
            "ncc": round(ncc, 3),
            "edge": round(edge_score, 3),
            "corr": round(corr, 3),
            "color": round(color_score, 3),
        }
    except Exception as e:
        print(f"[WARN] Similarity computation error: {e}")
        return None


def validate_upload_match(ghost_path: str, image_bytes: bytes) -> Dict[str, Any]:
    """
    Cửa soát chất lượng tải lên:
    - Ngăn chặn ảnh rung lắc dữ dội, out-of-focus.
    - Ngăn chặn vật thể lạ, bàn phím, sàn nhà, dây điện.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        cap_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cap_bgr is None:
            return {"valid": False, "message": "Không thể giải mã dữ liệu ảnh camera!"}

        # 1. Kiểm tra độ nét quang học (Chống nhòe mờ rung lắc)
        ih, iw = cap_bgr.shape[:2]
        cw, ch = int(iw * 0.45), int(ih * 0.45)
        cx, cy = (iw - cw) // 2, (ih - ch) // 2
        crop = cap_bgr[cy:cy+ch, cx:cx+cw]
        g = cv2.cvtColor(cv2.resize(crop, (240, 240)), cv2.COLOR_BGR2GRAY).astype(np.float64)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        center_lap = float(np.mean(cv2.filter2D(g, -1, kernel)**2))

        if center_lap < 180.0:
            return {
                "valid": False,
                "message": f"Ảnh chụp bị mờ nét (Độ nét: {int(center_lap)}/cần ≥ 180)! Vui lòng giữ chắc tay và chạm màn hình lấy nét rõ bình trước khi bấm chụp.",
            }

        # 2. Kiểm tra tương đồng với ghost template nếu có
        if os.path.exists(ghost_path):
            sim_res = compute_similarity_from_bgr(ghost_path, cap_bgr)
            if sim_res is not None:
                score = sim_res.get("score", 0.0)
                edge = sim_res.get("edge", 0.0)
                ncc = sim_res.get("ncc", 0.0)
                color = sim_res.get("color", 1.0)
                corr = sim_res.get("corr", 0.0)

                # Chặn ảnh chụp vật thể lạ, bàn phím, dây điện, mặt bàn không có bình
                if edge < 0.20 or ncc < 0.18:
                    return {
                        "valid": False,
                        "message": f"Không nhận diện được bình gốm trong khung hình (Đường viền: {round(edge*100, 1)}%, Cấu trúc: {round(ncc*100, 1)}%). Vui lòng căn chỉnh bình gốm vào giữa khung ngắm!",
                    }

                if color < 0.08 and corr < 0.15:
                    return {
                        "valid": False,
                        "message": f"Ảnh không khớp sản phẩm mẫu (Màu sắc chỉ {round(color*100, 1)}%, phát hiện vật thể lạ). Vui lòng đưa đúng bình vào khung hình!",
                    }

                if score < 0.32:
                    return {
                        "valid": False,
                        "message": f"Độ khớp quá thấp ({round(score*100, 1)}% / cần ≥ 32%). Vui lòng căn chỉnh bình gốm khớp với bóng mờ mẫu!",
                    }

        return {"valid": True}
    except Exception as e:
        print(f"[WARN] Upload validation error: {e}")
        return {"valid": True}
