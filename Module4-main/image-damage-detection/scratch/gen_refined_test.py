import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

# Load base detailed_debug code
with open('v2/detailed_debug.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace("V2_DIR = os.path.dirname(os.path.abspath(__file__))", "V2_DIR = os.path.abspath('v2')")

# Let's inspect where inner_triangle is created in detailed_debug.py
# In detailed_debug:
# inner_triangle = cv2.erode(mask_p, k_open_3)
# We will change to:
# k_erode_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# inner_triangle = cv2.erode(mask_p, k_erode_5, iterations=2)

code = code.replace(
    "inner_triangle = cv2.erode(mask_p, k_open_3)",
    "k_erode_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))\n        inner_triangle = cv2.erode(mask_p, k_erode_5, iterations=2)"
)

# And in broken_edges: use measure_linear_stroke instead of cv2.arcLength
code = code.replace(
    "max_broken_length = max([cv2.arcLength(c, False) for c in cnts_broken], default=0.0)",
    "max_broken_length = measure_linear_stroke(cnts_broken, min_len=10.0, min_aspect=1.8)"
)

# And for crack_cand: if ori_corr >= 0.85, require not angle matching any peaks_p
crack_loop_old = """            for y, x in zip(*np.where(crack_cand)):
                d = dist_to_p_all[y, x]
                near_exact = (d <= 2.5)
                near_shifted = (d <= 6.0) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
                if not (near_exact or near_shifted):
                    anom_crack[y, x] = True"""

crack_loop_new = """            for y, x in zip(*np.where(crack_cand)):
                d = dist_to_p_all[y, x]
                near_exact = (d <= 2.5)
                # Neu ori_corr >= 0.85 (hoa van khop hoan toan): mo rong dung sai khoang cach len 5.0 neu goc trung khop
                tol_dist = 6.0 if (ori_corr < 0.85) else 8.0
                near_shifted = (d <= tol_dist) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=30.0)
                # Neu hoa van song song va goc khop voi hoa van Product -> khong phai crack bat thuong
                is_parallel_feature = (ori_corr >= 0.85) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
                if not (near_exact or near_shifted or is_parallel_feature):
                    anom_crack[y, x] = True"""

code = code.replace(crack_loop_old, crack_loop_new)

# In diff_loc: exclude pixels near shifted existing edges for standalone blob detection
blob_logic_old = """        diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
        diff_loc[~valid] = 0.0
        diff_loc[glare_px] = 0.0
        bin_diff_loc = (diff_loc > 30.0).astype(np.uint8) * 255"""

blob_logic_new = """        diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
        diff_loc[~valid] = 0.0
        diff_loc[glare_px] = 0.0
        # Loai bo cac pixel nam sat net hoa van cu (dist <= 3.5px) neu ori_corr >= 0.80 de khong bi nham dich chuyen vien thanh blob
        diff_clean = diff_loc.copy()
        if ori_corr >= 0.80 and p_edge_cnt >= 20:
            diff_clean[dist_to_p_all <= 3.5] = 0.0
        bin_diff_loc = (diff_clean > 30.0).astype(np.uint8) * 255"""

code = code.replace(blob_logic_old, blob_logic_new)

# Gating rule:
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

gating_new = """        # Layer 2 Decision: Men bi bien doi mau / soc ngoai lai
        z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)
        # Nguong L2: z_c > 2.8 va chroma_err >= 24.0 va co blob mau ro ret
        is_l2_damage = bool(z_c > 2.8 and item["chroma_err"] >= 24.0 and item["has_color_blob"])

        # Standalone Physical Damage (danh rieng cho loi khong doi mau nhu vet seo / nứt tren nen trang):
        # 1. Vet nut ngoai lai sac net: intrusive_len >= 12.0
        # 2. Vet nut den sau khong trung hoa van cu: dark_crack >= 12.0 va ori_corr < 0.85
        # 3. Gay net lon: broken_length >= 25.0 va ori_corr < 0.80
        # 4. Vet seo lon tren gom su: blob_area >= 60 va z_l1 > 0.4 (dac trung bo dataset seo)
        is_severe_standalone = bool(
            item["intrusive_len"] >= 12.0 or 
            (item["dark_crack"] >= 12.0 and item["ori_corr"] < 0.85) or 
            (item["broken_length"] >= 25.0 and item["ori_corr"] < 0.80) or 
            (item["blob_area"] >= 60 and item["z_l1"] > 0.4)
        )

        is_l1_damage = bool((is_l2_damage and item["has_physical_l1"]) or is_severe_standalone)"""

code = code.replace(gating_old, gating_new)

with open('scratch/test_refined_logic.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Created scratch/test_refined_logic.py successfully.")
