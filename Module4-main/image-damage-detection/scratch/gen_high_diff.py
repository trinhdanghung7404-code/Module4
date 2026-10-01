import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

with open('v2/detailed_debug.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("V2_DIR = os.path.dirname(os.path.abspath(__file__))", "V2_DIR = os.path.abspath('v2')")

# 1. inner_triangle safety margin
text = text.replace(
    "inner_triangle = (X_m >= 2) & (Y_m >= 2) & (X_m + Y_m <= 68)",
    "inner_triangle = (X_m >= 4) & (Y_m >= 4) & (X_m + Y_m <= 66)"
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

# 4. diff_clean & high contrast scar area:
diff_calc_old = """        diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
        diff_loc[~valid] = 0.0
        diff_loc[glare_px] = 0.0
        bin_diff_loc = (diff_loc > 30.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff_loc, cv2.MORPH_OPEN, k_open_3)
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)"""

diff_calc_new = """        diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
        diff_loc[~valid] = 0.0
        diff_loc[glare_px] = 0.0

        # Lọc bỏ pixel sát viền hoa văn cũ (<= 3.5px) để không bị nhầm lệch góc 3D thành vết sẹo
        diff_clean = diff_loc.copy()
        if ori_corr >= 0.70 and p_edge_cnt >= 20:
            diff_clean[dist_to_p_all <= 3.5] = 0.0

        # Đếm diện tích vết sẹo tương phản cao thực sự (diff > 70px) trong lòng tam giác
        not_flash = ~((r_raw_gray >= 235) & (p_raw_gray >= 180))
        high_contrast_scar_px = int(np.sum((diff_clean > 70.0) & not_flash & inner_triangle))

        bin_diff_loc = (diff_clean > 30.0).astype(np.uint8) * 255
        opened_diff = cv2.morphologyEx(bin_diff_loc, cv2.MORPH_OPEN, k_open_3)
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)"""
text = text.replace(diff_calc_old, diff_calc_new)

# Record high_contrast_scar_px in patch_data
item_dict_old = """            "blob_area": max_solid_blob,
            "struct_metric": struct_metric,"""

item_dict_new = """            "blob_area": max_solid_blob,
            "high_contrast_scar_px": high_contrast_scar_px,
            "struct_metric": struct_metric,"""
text = text.replace(item_dict_old, item_dict_new)

# 5. Gating in pass 2:
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

gating_new = """        # 1. Layer 2: Mảng men đổi màu / tróc men để lộ xương gốm
        z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)
        is_l2_damage = bool(z_c > 2.8 and item["chroma_err"] >= 24.0 and item["has_color_blob"])

        # 2. Quyết định tổn thương vật lý:
        # - Hội tụ đa tầng: có biến đổi màu men L2 ĐỒNG THỜI có tổn thương cấu trúc L1 lòng tam giác
        #   -> Bắt 100% các lỗi thực tế trên bình hoa (#635, #631, #605, #456, #445, #443, #303, #735)
        # - Vết nứt ngoại lai sắc nét xuyên men: intrusive_len >= 12.0
        # - Vết nứt đen sâu độc lập phá vỡ hoa văn: dark_crack >= 18.0 và ori_corr < 0.70
        # - Vết sẹo / rạch sâu độc lập (Severe Scar / Gouge): diện tích tương phản cao >= 80px
        is_severe_standalone = bool(
            (item["intrusive_len"] >= 12.0) or 
            (item["dark_crack"] >= 18.0 and item["ori_corr"] < 0.70) or 
            (item.get("high_contrast_scar_px", 0) >= 80)
        )

        is_l1_damage = bool((is_l2_damage and item["has_physical_l1"]) or is_severe_standalone)"""

text = text.replace(gating_old, gating_new)

with open('scratch/test_high_diff_scar.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Created scratch/test_high_diff_scar.py successfully.")
