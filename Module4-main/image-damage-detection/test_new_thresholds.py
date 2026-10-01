"""Test thresholds mới - BALANCED VOTING SYSTEM."""

from mesh_config import (
    VOTE_SIMILARITY_THRESHOLD,
    VOTE_SSIM_THRESHOLD,
    VOTE_FEATURE_THRESHOLD,
    VOTING_MODE,
    SIMILARITY_THRESHOLD,
    FEATURE_THRESHOLD,
    SSIM_THRESHOLD,
)

print("=" * 70)
print("BALANCED VOTING SYSTEM - THRESHOLDS VERIFICATION")
print("=" * 70)

print(f"\n1. VOTING THRESHOLDS:")
print(f"   VOTE_SIMILARITY_THRESHOLD = {VOTE_SIMILARITY_THRESHOLD}")
print(f"   VOTE_SSIM_THRESHOLD       = {VOTE_SSIM_THRESHOLD}")
print(f"   VOTE_FEATURE_THRESHOLD    = {VOTE_FEATURE_THRESHOLD}")
print(f"   VOTING_MODE               = '{VOTING_MODE}'")

print(f"\n2. COMPATIBILITY ALIASES:")
print(f"   SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
print(f"   FEATURE_THRESHOLD    = {FEATURE_THRESHOLD}")
print(f"   SSIM_THRESHOLD       = {SSIM_THRESHOLD}")

print(f"\n3. VERIFY ĐỒNG BỘ:")
if (SIMILARITY_THRESHOLD == VOTE_SIMILARITY_THRESHOLD and 
    FEATURE_THRESHOLD == VOTE_FEATURE_THRESHOLD and 
    SSIM_THRESHOLD == VOTE_SSIM_THRESHOLD):
    print("   ✅ Tất cả thresholds ĐỒNG BỘ!")
else:
    print("   ❌ CÓ LỖI ĐỒNG BỘ!")

print(f"\n4. LOGIC VOTING (với data thực tế: Median SSIM=0.87, Lowest=0.41):")
print(f"   - SSIM < {VOTE_SSIM_THRESHOLD} → Vote #1 cho damage")
print(f"   - Feature Sim < {VOTE_FEATURE_THRESHOLD} → Vote #2")
print(f"   - Combined Sim < {VOTE_SIMILARITY_THRESHOLD} → Vote #3")
print(f"   - VOTING_MODE = '{VOTING_MODE}' → Cần 2/3 votes để flag damage")

print(f"\n5. DỰ KIẾN KẾT QUẢ:")
print(f"   - Bình thường (SSIM > 0.87):   KHÔNG flag ✅")
print(f"   - Biến dạng (SSIM 0.65-0.87):  0-1 votes → KHÔNG flag ✅")
print(f"   - Damage nhẹ (SSIM 0.50-0.65): 1-2 votes → Có thể flag")
print(f"   - Damage nặng (SSIM < 0.50):   2-3 votes → FLAG ✅")

print("\n" + "=" * 70)
print("HƯỚNG DẪN: Chạy 'python main.py' để test với 1.jpg vs 2.jpg")
print("=" * 70)
