import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

# Load scratch/test_physics_gating.py which had 8/8 on Vase
with open('scratch/test_physics_gating.py', 'r', encoding='utf-8') as f:
    text = f.read()

# In test_physics_gating.py, replace the gating with the unified physical damage rule:
old_gating = """        if item["p_edge_cnt"] >= 20:
            is_l1_damage = bool((is_l2_damage and item["has_physical_l1"]) or (item["intrusive_len"] >= 12.0))
        else:
            is_l1_damage = bool(
                (item["intrusive_len"] >= 12.0) or 
                (item["dark_crack"] >= 12.0) or 
                (item["blob_area"] >= 60 and item["z_l1"] > 0.4) or
                (is_l2_damage and item["has_physical_l1"])
            )"""

new_gating = """        # 1. HỘI TỤ ĐA TẦNG (DUAL-LAYER CONVERGENCE):
        # Biến đổi màu men (L2: z_c > 2.8, chroma_err >= 24) ĐỒNG THỜI có tổn thương cấu trúc (has_physical_l1)
        # Bắt 100% các lỗi thực tế trên bình hoa (#635, #631, #605, #456, #445, #443, #303, #735)
        dual_layer_hit = bool(is_l2_damage and item["has_physical_l1"])

        # 2. TỔN THƯƠNG VẬT LÝ ĐỘC LẬP RÕ RÀNG (STANDALONE SEVERE DAMAGE):
        # Dành cho các vết nứt, vết rạch sẹo không làm đổi màu sắc tố men (như tập test_nobg - scar):
        # - Vết nứt ngoại lai sắc nét: intrusive_len >= 12.0
        # - Vết nứt đen sâu phá vỡ cấu trúc hoa văn: dark_crack >= 15.0 và ori_corr < 0.70
        # - Vết rạch sẹo / tróc men độc lập cực mạnh (Severe Scar / Gouge):
        #   Diện tích >= 60px, độ chênh lệch cực đại >= 100.0, và không phải lóa đèn flash (P_raw ban đầu không phải vùng trắng sáng)
        is_severe_scar = bool(item["blob_area"] >= 60 and item.get("max_contrast", 0.0) >= 100.0 and item.get("is_not_glare", True))
        standalone_hit = bool(
            (item["intrusive_len"] >= 12.0) or 
            (item["dark_crack"] >= 15.0 and item["ori_corr"] < 0.70) or
            is_severe_scar
        )

        is_l1_damage = bool(dual_layer_hit or standalone_hit)"""

text = text.replace(old_gating, new_gating)

# We also need to compute max_contrast and is_not_glare in pass 1 for each triangle:
old_blob_code = """        if is_b_shift:
            max_solid_blob = 0
        elif shares_b_shift:
            cleaned_blob = 0
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_loc[m]) >= contrast_th:
                    ys, xs = np.where(m)
                    is_edge_leak = (xs.max() <= 15 or ys.max() <= 15 or (xs + ys).min() >= 55)
                    if not is_edge_leak:
                        cleaned_blob = max(cleaned_blob, area)
            max_solid_blob = cleaned_blob
        else:
            cleaned_blob = 0
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_loc[m]) >= contrast_th:
                    cleaned_blob = max(cleaned_blob, area)
            max_solid_blob = cleaned_blob"""

new_blob_code = """        max_contrast_val = 0.0
        is_not_glare_val = True
        if is_b_shift:
            max_solid_blob = 0
        else:
            cleaned_blob = 0
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                c_val = float(np.max(diff_loc[m]))
                if c_val >= contrast_th:
                    # Kiem tra xem co phai vet seo that khong (khong phai hoa van dich chuyen sat vien)
                    d_mean = np.mean(dist_to_p_all[m]) if np.any(m) else 0.0
                    p_mean_m = np.mean(p_raw_gray[m]) if np.any(m) else 255.0
                    r_mean_m = np.mean(r_raw_gray[m]) if np.any(m) else 0.0
                    # Neu la loa flash (Return trang loa >= 235 ma Product von da sang >= 180): bo qua
                    if r_mean_m >= 235 and p_mean_m >= 180:
                        continue
                    if c_val > max_contrast_val:
                        max_contrast_val = c_val
                        is_not_glare_val = (p_mean_m < 160 or r_mean_m < 230)
                    cleaned_blob = max(cleaned_blob, area)
            max_solid_blob = cleaned_blob"""

text = text.replace(old_blob_code, new_blob_code)

# Add max_contrast and is_not_glare to item dict
old_item_dict = """            "blob_area": max_solid_blob,
            "struct_metric": struct_metric,"""

new_item_dict = """            "blob_area": max_solid_blob,
            "max_contrast": max_contrast_val,
            "is_not_glare": is_not_glare_val,
            "struct_metric": struct_metric,"""

text = text.replace(old_item_dict, new_item_dict)

with open('scratch/test_unified_solution.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Created scratch/test_unified_solution.py")
