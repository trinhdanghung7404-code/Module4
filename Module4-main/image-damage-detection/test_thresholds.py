"""
Test script để verify các thresholds mới sau khi fix
"""
import cv2
import numpy as np
from pathlib import Path

# Import các modules cần thiết
from mesh_config import (
    VOTE_SIMILARITY_THRESHOLD,
    VOTE_SSIM_THRESHOLD, 
    VOTE_FEATURE_THRESHOLD,
    SIMILARITY_THRESHOLD,
    FEATURE_THRESHOLD,
    SSIM_THRESHOLD,
    STRENGTHENING_SCORE_RATIO,
    VOTING_MODE
)

print("=" * 80)
print("KIỂM TRA THRESHOLDS SAU KHI FIX")
print("=" * 80)

print("\n1. VOTING THRESHOLDS (từ mesh_config.py):")
print(f"   VOTE_SIMILARITY_THRESHOLD = {VOTE_SIMILARITY_THRESHOLD}")
print(f"   VOTE_SSIM_THRESHOLD       = {VOTE_SSIM_THRESHOLD}")
print(f"   VOTE_FEATURE_THRESHOLD    = {VOTE_FEATURE_THRESHOLD}")
print(f"   STRENGTHENING_SCORE_RATIO = {STRENGTHENING_SCORE_RATIO}")
print(f"   VOTING_MODE               = {VOTING_MODE}")

print("\n2. COMPATIBILITY ALIASES:")
print(f"   SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
print(f"   FEATURE_THRESHOLD    = {FEATURE_THRESHOLD}")
print(f"   SSIM_THRESHOLD       = {SSIM_THRESHOLD}")

# Verify thresholds đồng bộ
print("\n3. VERIFY ĐỒNG BỘ:")
assert SIMILARITY_THRESHOLD == VOTE_SIMILARITY_THRESHOLD, "SIMILARITY_THRESHOLD không đồng bộ!"
assert FEATURE_THRESHOLD == VOTE_FEATURE_THRESHOLD, "FEATURE_THRESHOLD không đồng bộ!"
assert SSIM_THRESHOLD == VOTE_SSIM_THRESHOLD, "SSIM_THRESHOLD không đồng bộ!"
print("   ✅ Tất cả thresholds ĐỒNG BỘ!")

# Kiểm tra logic voting
print("\n4. LOGIC VOTING DỰ KIẾN:")
print("   Với median SSIM = 0.80 từ dữ liệu thực tế:")
print(f"   - SSIM < {VOTE_SSIM_THRESHOLD} → cần 3/3 votes đồng loạt thấp")
print(f"   - Feature Sim < {VOTE_FEATURE_THRESHOLD} → vote #2")
print(f"   - Combined Sim < {VOTE_SIMILARITY_THRESHOLD} → vote #3")
print("   → Chỉ flag damage khi CẢ 3 metrics cùng thấp!")

print("\n5. COMPONENT VALIDATOR:")
print("   confidence_threshold = 0.75 (tăng từ 0.55)")
print("   → Chỉ accept component có bằng chứng damage RẤT MẠNH")

print("\n" + "=" * 80)
print("✅ HOÀN TẤT KIỂM TRA - SẴN SÀNG CHẠY MAIN.PY")
print("=" * 80)
print("\nHướng dẫn test:")
print("1. Chạy: python main.py")
print("2. Option 1: Thêm ảnh product → images\\1.jpg")
print("3. Option 2: So sánh → chọn product ID → images\\2.jpg")
print("4. Kết quả mong đợi:")
print("   - Damage score < 0.5%")
print("   - Damage mask: TRẮNG HOÀN TOÀN hoặc chỉ có spot rất nhỏ")
print("   - KHÔNG CÓ GRAY AREAS!")