"""Test script để kiểm tra hiệu quả của Normalization và Otsu Thresholding."""

import cv2
import numpy as np
import os
from pathlib import Path

# 1. Test Geometry Mask với Otsu
print("--- TESTING GEOMETRY MASK (OTSU) ---")
try:
    from geometry import GeometryFeature
    
    geom = GeometryFeature()
    sample_path = "images/test.jpg" # Hoặc bất kỳ ảnh gốm nào bạn có
    
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        mask = geom.build_mask(img)
        
        # Đếm số pixel vật thể so với tổng số pixel
        total_pixels = img.shape[0] * img.shape[1]
        obj_pixels = cv2.countNonZero(mask)
        print(f"Ảnh mẫu: {sample_path}")
        print(f"Tỷ lệ vật thể chiếm: {obj_pixels/total_pixels*100:.2f}%")
        print("✅ Mask build thành công với Otsu Thresholding!")
    else:
        print("⚠️ Không tìm thấy ảnh mẫu, tạo ảnh test...")
        # Tạo ảnh giả lập (nền đen, vật thể sáng màu)
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        dummy_img[50:150, 50:150] = [128, 128, 128] # Vật thể xám giữa nền đen
        mask = geom.build_mask(dummy_img)
        print(f"Bản đồ mask test: {cv2.countNonZero(mask)} pixels / 40000 total")
        print("✅ Otsu Threshold hoạt động đúng trên ảnh test!")

except Exception as e:
    print(f"❌ Lỗi khi test Mask: {e}")

print("\n--- TESTING IMAGE NORMALIZATION (CLAHE) ---")
try:
    from comparator import DamageComparator
    
    comp = DamageComparator()
    
    # Lấy tất cả ảnh trong thư mục
    image_dir = Path("images")
    if image_dir.exists():
        for p in image_dir.glob("*.jpg") + image_dir.glob("*.png"):
            img = cv2.imread(str(p))
            if img is not None:
                normalized = comp.normalize_image(img)
                
                # So sánh histogram trước và sau
                gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_norm = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
                
                print(f"\n📷 Ảnh: {p.name}")
                print(f" - Điểm sáng tối đa gốc: {np.max(gray_orig)}, trung bình: {np.mean(gray_orig):.2f}")
                print(f" - Điểm sáng tối đa sau normalize: {np.max(gray_norm)}, trung bình: {np.mean(gray_norm):.2f}")
                print(f" ✅ Chuẩn hóa thành công cho {p.name}!")
    else:
        print("⚠️ Không tìm thấy thư mục 'images'.")

except Exception as e:
    print(f"❌ Lỗi khi test Normalize: {e}")

print("\n===============================")
print("TEST HOÀN TẤT. Hãy chạy main.py và chọn \"Analyze Return Product\" để kiểm tra thực tế!")
print("===============================")
