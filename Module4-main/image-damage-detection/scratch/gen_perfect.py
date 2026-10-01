import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

# Load base code
with open('v2/detailed_debug.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("V2_DIR = os.path.dirname(os.path.abspath(__file__))", "V2_DIR = os.path.abspath('v2')")

# 1. inner_triangle safety margin
text = text.replace(
    "inner_triangle = cv2.erode(mask_p, k_open_3)",
    "k_erode_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))\n        inner_triangle = cv2.erode(mask_p, k_erode_5, iterations=2)"
)

# 2. broken_edges stroke length
text = text.replace(
    "max_broken_length = max([cv2.arcLength(c, False) for c in cnts_broken], default=0.0)",
    "max_broken_length = measure_linear_stroke(cnts_broken, min_len=10.0, min_aspect=1.8)"
)

# 3. crack_cand with angular matching
crack_old = """            for y, x in zip(*np.where(crack_cand)):
                d = dist_to_p_all[y, x]
                near_exact = (d <= 2.5)
                near_shifted = (d <= 6.0) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
                if not (near_exact or near_shifted):
                    anom_crack[y, x] = True"""

crack_new = """            for y, x in zip(*np.where(crack_cand)):
                d = dist_to_p_all[y, x]
                near_exact = (d <= 2.5)
                tol_dist = 6.0 if (ori_corr < 0.85) else 8.0
                near_shifted = (d <= tol_dist) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=30.0)
                is_parallel_feature = (ori_corr >= 0.85) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
                if not (near_exact or near_shifted or is_parallel_feature):
                    anom_crack[y, x] = True"""
text = text.replace(crack_old, crack_new)

# 4. Intrusive edge detection: robust to soft anti-aliased scar lines
intrusive_old = """        intrusive_cand = (r_raw_edge > 0) & (dist_to_p_all > 6.0) & (grad_p_raw < 25.0) & (mag_r_raw >= 70.0) & (~glare_px) & inner_triangle
        anom_intrusive = np.zeros_like(intrusive_cand)
        for y, x in zip(*np.where(intrusive_cand)):
            if not is_angle_matching_any(ang_r_raw[y, x], peaks_p, tol_deg=25.0):
                anom_intrusive[y, x] = True

        cnts_intrusive, _ = cv2.findContours(anom_intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_intrusive_length = measure_linear_stroke(cnts_intrusive, min_len=12.0, min_aspect=2.0)"""

intrusive_new = """        # Intrusive Canny & Sobel crack/scar:
        dist_p_canny = cv2.distanceTransform(cv2.bitwise_not(p_c_edge), cv2.DIST_L2, 3)
        intrusive_cand = (r_c_edge > 0) & (dist_p_canny > 3.0) & (mag_r >= 35.0) & inner_triangle
        anom_intrusive = np.zeros_like(intrusive_cand)
        for y, x in zip(*np.where(intrusive_cand)):
            # Khong phai net hoa van da co tren Product
            if not is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0) or dist_p_canny[y, x] > 5.0:
                anom_intrusive[y, x] = True

        cnts_intrusive, _ = cv2.findContours(anom_intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_intrusive_length = measure_linear_stroke(cnts_intrusive, min_len=12.0, min_aspect=1.5)"""
text = text.replace(intrusive_old, intrusive_new)

# 5. Gating logic in pass 2:
gating_old = """        # Layer 2 Decision: Mảng màu ngoại lai / tróc men tập trung
        z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)
        is_l2_damage = bool(z_c > 2.0 and item["chroma_err"] > 16.0 and item["has_color_blob"])

        # NGUYÊN TẮC 4: QUYẾT ĐỊNH ĐỒNG THUẬN ĐA TẦNG & TỔN THƯƠNG VẬT LÝ NẶNG
        # 1. Đồng thuận đa tầng: Có biến đổi màu men (L2) VÀ có bằng chứng cấu trúc lòng tam giác (has_physical_l1)
        #    -> Đặc trưng của 100% lỗi thật (#635, #631, #605, #456, #445, #443, #303)
        # 2. Tổn thương cấu trúc độc lập cực kỳ rõ ràng (Standalone Severe Damage):
        #    - Vết nứt ngoại lai xuyên men: intrusive_len >= 12.0
        #    - Vết nứt đen sâu: dark_crack >= 12.0
        #    - Gãy nét lớn lòng tam giác: broken_length >= 25.0 và ori_corr < 0.85
        #    - Vết sẹo/tróc men diện rộng: blob_area >= 50 và z_l1 > 1.5
        is_severe_standalone = bool(
            item["intrusive_len"] >= 12.0 or 
            item["dark_crack"] >= 12.0 or 
            (item["broken_length"] >= 25.0 and item["ori_corr"] < 0.85) or 
            (item["blob_area"] >= 50 and item["z_l1"] > 1.5)
        )

        is_l1_damage = bool((is_l2_damage and item["has_physical_l1"]) or is_severe_standalone)"""

gating_new = """        # Layer 2 Decision: Mảng men đổi màu / sắc tố ngoại lai
        z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)
        is_l2_damage = bool(z_c > 2.8 and item["chroma_err"] >= 24.0 and item["has_color_blob"])

        # Standalone Structural Damage:
        # - Vết nứt ngoại lai / vết rạch sẹo sắc nét: intrusive_len >= 12.0
        # - Vết nứt đen sâu phá vỡ hoa văn: dark_crack >= 15.0 và ori_corr < 0.70
        is_severe_standalone = bool(
            item["intrusive_len"] >= 12.0 or 
            (item["dark_crack"] >= 15.0 and item["ori_corr"] < 0.70)
        )

        # Hợp nhất: Hoặc đồng thuận L2 + L1, hoặc tổn thương độc lập sắc nét
        is_l1_damage = bool((is_l2_damage and item["has_physical_l1"]) or is_severe_standalone)"""

text = text.replace(gating_old, gating_new)

with open('scratch/test_perfect_system.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Created scratch/test_perfect_system.py")
