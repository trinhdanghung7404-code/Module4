import sys, os
sys.path.insert(0, 'v2')
import cv2, numpy as np

with open('scratch/test_refined_logic.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the gating in test_refined_logic.py
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

new_gating = """        # QUY LUẬT VẬT LÝ VỀ ĐỒ GỐM SỨ (PHYSICAL PORCELAIN LAWS):
        # 1. Với đồ gốm có hoa văn vẽ (Painted Porcelain, p_edge_cnt >= 20):
        #    - Tổn thương vật lý (sứt mẻ, tróc men, gãy nét) luôn làm tróc lớp men màu, để lộ xương gốm 
        #      -> Bắt buộc phải có sự hội tụ đa tầng: is_l2_damage AND has_physical_l1.
        #    - Hoặc vết nứt đen đâm xuyên nền cực kỳ sắc nét (Intrusive Crack >= 12px) không trùng bất kỳ nét cũ nào.
        # 2. Với đồ gốm men trơn / không hoa văn (Plain / Monochrome, p_edge_cnt < 20):
        #    - Không có hoa văn để đổi màu sắc tố (như tập dữ liệu vết sẹo test_nobg - scar).
        #    - Tổn thương biểu hiện trực tiếp qua mảng biến đổi độ sáng (blob_area >= 60) hoặc vết nứt/sẹo đâm xuyên (intrusive >= 12px).
        if item["p_edge_cnt"] >= 20:
            is_l1_damage = bool((is_l2_damage and item["has_physical_l1"]) or (item["intrusive_len"] >= 12.0))
        else:
            is_l1_damage = bool(
                (item["intrusive_len"] >= 12.0) or 
                (item["dark_crack"] >= 12.0) or 
                (item["blob_area"] >= 60 and item["z_l1"] > 0.4) or
                (is_l2_damage and item["has_physical_l1"])
            )"""

text = text.replace(old_gating, new_gating)

with open('scratch/test_physics_gating.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Created scratch/test_physics_gating.py successfully.")
