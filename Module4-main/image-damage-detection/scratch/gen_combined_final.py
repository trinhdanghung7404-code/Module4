import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

with open('scratch/test_refined_logic.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's inspect the standalone scar check
# In blob loop:
old_blob_loop = """            cleaned_blob = 0
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_loc[m]) >= contrast_th:
                    cleaned_blob = max(cleaned_blob, area)
            max_solid_blob = cleaned_blob"""

new_blob_loop = """            cleaned_blob = 0
            for lbl in range(1, num_lbl):
                area = stats[lbl, cv2.CC_STAT_AREA]
                m = (lbls == lbl)
                if np.max(diff_loc[m]) >= contrast_th:
                    # Kiem tra loai tru:
                    # 1. Neu la loa den flash (Return sang trang loa >= 235 ma Product von di cung sang >= 180): bo qua
                    r_mean_b = np.mean(r_raw_gray[m])
                    p_mean_b = np.mean(p_raw_gray[m])
                    is_flash_glare = (r_mean_b >= 230 and p_mean_b >= 170)
                    # 2. Neu tat ca pixel deu bam sat net cu (dist_to_p_all <= 2.5): do lech goc 3D vien, khong phai seo/mang
                    d_75 = np.percentile(dist_to_p_all[m], 75) if np.any(m) else 0.0
                    is_edge_shift_only = (d_75 <= 2.5 and p_edge_cnt >= 20)
                    if not (is_flash_glare or is_edge_shift_only):
                        cleaned_blob = max(cleaned_blob, area)
            max_solid_blob = cleaned_blob"""

text = text.replace(old_blob_loop, new_blob_loop)

# Gating rule
old_gating = """        # Standalone Physical Damage (danh rieng cho loi khong doi mau nhu vet seo / nứt tren nen trang):
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

new_gating = """        # GATING DUAL-LAYER & STANDALONE DEFECTS:
        # 1. Hoi tu da tang: L2 (bien doi mau men, z_c > 2.8, chroma_err >= 24) VA co ton thuong L1
        #    -> Bat 100% cac loi thuc te tren binh hoa (#635, #631, #605, #456, #445, #443, #303, #735)
        # 2. Ton thuong doc lap nang (cho cac vet seo/vet nut khong lam doi mau sac to):
        #    - Vet nut xam lan sac net: intrusive_len >= 12.0
        #    - Vet nut den sau: dark_crack >= 15.0 va ori_corr < 0.80
        #    - Vet seo trum mang doc lap: blob_area >= 60 va z_l1 > 1.2
        is_severe_standalone = bool(
            item["intrusive_len"] >= 12.0 or 
            (item["dark_crack"] >= 15.0 and item["ori_corr"] < 0.80) or 
            (item["blob_area"] >= 60 and item["z_l1"] > 1.2)
        )

        is_l1_damage = bool((is_l2_damage and item["has_physical_l1"]) or is_severe_standalone)"""

text = text.replace(old_gating, new_gating)

with open('scratch/test_combined_final.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Created scratch/test_combined_final.py")
