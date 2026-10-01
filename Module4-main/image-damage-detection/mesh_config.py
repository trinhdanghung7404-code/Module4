"""Configuration for Adaptive Mesh Region Matching."""

from __future__ import annotations

# Debug output
DEBUG = True

# Adaptive mesh refinement
MAX_TOTAL_TRIANGLES = 2000
TOP_K = 24
MAX_DEPTH = 3
MIN_TRIANGLE_AREA = 80.0

# ===== STRICT UNANIMOUS VOTING SYSTEM v2 — CHỈ DAMAGE RÕ NÉT MỚI BỊ FLAG ===
# MỤC TIÊU: Giảm thiểu FALSE POSITIVE đến mức tối đa
# NGUYÊN TẮC: Với ảnh chụp bình gốm khác góc/ánh sáng, nếu KHÔNG CÓ DAMAGE THẬT
#              thì hiếm khi cả 3 metrics cùng xuống thấp. Damage thật sẽ kéo
#              SIMILARITY, SSIM, FEATURE CÙNG GIẢM MẠNH tại vùng đó.
#
# DỮ LIỆU THỰC TẾ (1.jpg vs 2.jpg - KHÔNG DAMAGE):
#   Median SSIM = 0.83 → >50% tam giác giống nhau ≥83%
#   Average SSIM = 0.78 → trung bình bị outliers kéo xuống
#   Lowest SSIM = 0.21 → cực ít outlier
#   Highest SSIM = 0.96 → vùng giống hệt nhau
#
# PHÂN TÍCH VẤN ĐỀ HIỆN TẠI:
#   Với threshold = 0.70, các patch có SSIM = 0.75-0.82 sẽ được ACCEPT
#   Nhưng 87% patches bị REJECT! → LOGIC NGƯỢC
#
# GIẢI PHÁP: ĐẶT THRESHOLD CAO HƠN MEDIAN
#   - Threshold > median → chỉ flag những vùng THỰC SỰ khác biệt
#   - Ví dụ: threshold = 0.88 → chỉ reject nếu SSIM < 88%
#   - Vùng 0.83-0.88 sẽ được accept vì là "normal variation"

# NEW STRATEGY: Thresholds ABOVE median để bắt ONLY damage thật sự
#   - Nếu median = 0.83, đặt threshold = 0.88+ (cao hơn ~5 điểm phần trăm)
#   - Damage thật sẽ kéo mạnh cả 3 metrics xuống dưới threshold
#   - Biến dạng ảnh thông thường chỉ ảnh hưởng 1-2 metrics

# CRITICAL INSIGHT: Higher thresholds make MORE patches vote damage!
# Because "similarity < threshold" = vote for damage
# With threshold=0.88 and median SSIM=0.80 → most patches vote damage!
# 
# SOLUTION: Set thresholds at the EXTREME LOW end of normal variation
# Normal patches: SSIM 0.70-0.96 (median 0.80)
# Only REAL damage should have metrics < 0.35 (very different)
# With strict_unanimous (3/3 votes), only patches < threshold on ALL 3 metrics get flagged

# ===== BALANCED VOTING SYSTEM - CÂN BẰNG GIỮA FALSE POSITIVE VÀ FALSE NEGATIVE =====
# Dựa trên dữ liệu thực tế: Median SSIM=0.87, Lowest=0.41, Median Similarity=0.95
# Threshold phải nằm GIỮA median và lowest để bắt damage nhưng bỏ qua biến dạng
#
# NGUYÊN TẮC:
#   - Threshold ~0.65-0.70: Thấp hơn median ~20% nhưng CAO HƠN lowest
#   - Majority voting (2/3): Cần 2 trong 3 metrics thấp mới flag
#   - Damage thật sẽ kéo CẢ 3 metrics xuống, biến dạng thường chỉ kéo 1-2 metrics

VOTE_SIMILARITY_THRESHOLD = 0.70   # Flag khi similarity < 70% (different enough)
VOTE_SSIM_THRESHOLD = 0.65         # Flag khi SSIM < 65% (different enough)
VOTE_FEATURE_THRESHOLD = 0.70      # Flag khi feature < 70% (different enough)

# Strengthening: Auto-flag khi score cực thấp (dưới 50% = rất khác biệt)
STRENGTHENING_MODE = "extreme_only"
STRENGTHENING_SCORE_RATIO = 0.50

# Mode voting: majority = cần 2/3 votes (linh hoạt hơn strict_unanimous)
VOTING_MODE = "majority"

# Compatibility thresholds for old code (ĐỒNG BỘ với voting thresholds)
SIMILARITY_THRESHOLD = VOTE_SIMILARITY_THRESHOLD  # 0.70
FEATURE_THRESHOLD = VOTE_FEATURE_THRESHOLD         # 0.70
SSIM_THRESHOLD = VOTE_SSIM_THRESHOLD               # 0.65
STRONG_EVIDENCE_THRESHOLD = 0.50

# Nearby edge correspondence after local residual alignment.
EDGE_SEARCH_RADIUS = 4
EDGE_MATCH_SCORE_THRESHOLD = 0.68
EDGE_ORIENTATION_TOLERANCE_DEG = 20.0

# Lab a/b color difference. Color is evaluated only in the suspicious mesh
# region plus a small context margin, not across the whole object.
COLOR_DIFF_THRESHOLD = 20.0  # Tăng từ 15.0 → b channel diff mean = 5.4
COLOR_MIN_AREA = 5
COLOR_CONTEXT_RADIUS = 3

# Faint colour changes are evaluated by a separate, stricter component
# branch.  This permits a visibly faded patch to be detected without making
# the normal colour-error threshold sensitive to single-pixel colour noise.
FAINT_COLOR_DIFF_THRESHOLD = 12.0  # Tăng từ 8.0
FAINT_SATURATION_DIFF_THRESHOLD = 10.0  # Tăng từ 6.0
FAINT_LUMINANCE_DIFF_THRESHOLD = 15.0  # Tăng từ 6.0 → L diff mean = 38.3
FAINT_COLOR_MIN_AREA = 36
FAINT_COLOR_MIN_FILL_RATIO = 0.18
FAINT_COLOR_MIN_MEAN_SCORE = 10.0  # Tăng từ 8.0

# Human-visible local defects.
# These thresholds add a luminance-sensitive branch so white, grey, black,
# dull and dark marks can be detected even when Lab a/b barely changes.
# ĐIỀU CHỈNH: L diff trung bình thực tế = 38.3 nên threshold phải cao hơn
PERCEPTUAL_LUMINANCE_ENABLED = False  # TẮT: Gây false positive với L diff mean = 38.3
PERCEPTUAL_L_DIFF_THRESHOLD = 50.0  # Nếu bật lại: threshold rất cao
PERCEPTUAL_SCORE_THRESHOLD = 40.0  # Tighten: Chỉ flag defect mạnh
PERCEPTUAL_STRONG_SCORE_THRESHOLD = 60.0  # Tighten: Chỉ flag defect rất mạnh
PERCEPTUAL_MIN_AREA = 20
PERCEPTUAL_STRONG_MIN_AREA = 10
PERCEPTUAL_REFERENCE_OBJECT_AREA = 300000
PERCEPTUAL_LOCAL_SIGMA = 7.0
PERCEPTUAL_L_WEIGHT = 1.0

# Visible-defect recall outside suspicious mesh triangles.
# This branch is intentionally stricter because exact-position comparison can
# react to small paint/geometry displacement in real photographs. Compact,
# high-contrast defects are kept; large diffuse or edge-following regions are
# rejected. ĐIỀU CHỈNH: Tăng thresholds để tránh false positive từ ánh sáng khác nhau.
PERCEPTUAL_OUTSIDE_MESH_ENABLED = False  # TẮT: Gây false positive với biến dạng hình học
PERCEPTUAL_OUTSIDE_MESH_SCORE_THRESHOLD = 60.0  # Nếu bật lại: threshold rất cao
PERCEPTUAL_OUTSIDE_MESH_MIN_AREA = 14
PERCEPTUAL_OUTSIDE_MESH_MAX_AREA_RATIO = 0.0025
PERCEPTUAL_OUTSIDE_MESH_MIN_FILL_RATIO = 0.12
PERCEPTUAL_OUTSIDE_MESH_MATCHED_EDGE_REJECT_RATIO = 0.65

# Diagnostic-only strong-color recall outside the suspicious mesh.
# It compares each return pixel against nearby product colors and only writes
# debug masks. The recall mask has been verified and is added to final damage.
GLOBAL_STRONG_COLOR_RECALL_ENABLED = False  # TẮT: Đánh giá toàn bộ ảnh, gây false positive
GLOBAL_STRONG_COLOR_THRESHOLD = 60.0  # Nếu bật lại: threshold rất cao
GLOBAL_STRONG_COLOR_SEARCH_RADIUS = 3
GLOBAL_STRONG_COLOR_MIN_AREA = 5
GLOBAL_STRONG_COLOR_ADD_TO_FINAL = False  # KHÔNG thêm vào final mask

# Local residual alignment for grouped suspicious triangles. The center of a
# suspicious region is excluded from scoring; only its surrounding ring is used
# to estimate a small translation.
LOCAL_ALIGNMENT_ENABLED = True
LOCAL_ALIGNMENT_SEARCH_RADIUS = 4
LOCAL_ALIGNMENT_MAX_KEYPOINT_OFFSET = 8.0
LOCAL_ALIGNMENT_RING_WIDTH = 12
LOCAL_ALIGNMENT_MIN_RING_PIXELS = 96
LOCAL_ALIGNMENT_MIN_IMPROVEMENT = 0.12
LOCAL_ALIGNMENT_MAX_TRIANGLES_PER_REGION = 12
LOCAL_ALIGNMENT_MAX_REGION_EXTENT = 180
LOCAL_ALIGNMENT_TRIM_KEEP_RATIO = 0.90
LOCAL_ALIGNMENT_GRADIENT_WEIGHT = 0.70
LOCAL_ALIGNMENT_SMOOTH_SIGMA = 1.5

# Performance
SIMILARITY_WORKERS = 4

# Compatibility aliases for mesh_damage_detector.py
COLOR_SEARCH_RADIUS = COLOR_CONTEXT_RADIUS
STRONG_COLOR_DIFF_THRESHOLD = GLOBAL_STRONG_COLOR_THRESHOLD

# ===== ALIASES CHO COMPATIBILITY (các module cũ vẫn dùng tên này) =====
SSIM_THRESHOLD = VOTE_SSIM_THRESHOLD          # 0.65
FEATURE_THRESHOLD = VOTE_FEATURE_THRESHOLD    # 0.70
DIFF_THRESHOLD = 45                           # Gradient threshold
MIN_AREA_THRESHOLD = 20                       # Min component area

# === MULTI-LAYER DAMAGE DETECTION CONFIGURATION ===
# Edge Detection Layer
EDGE_CANNY_LOW = 50
EDGE_CANNY_HIGH = 150
EDGE_DILATE_KERNEL = 3

# Intensity Difference Layer
INTENSITY_THRESHOLD = 20
INTENSITY_BLUR_KERNEL = 5

# Texture Analysis Layer
TEXTURE_DIFF_THRESHOLD = 0.2
TEXTURE_RESIZE_DIM = (64, 64)

# Combine Layers
EDGE_WEIGHT = 0.4
INTENSITY_WEIGHT = 0.3
TEXTURE_WEIGHT = 0.3
FINAL_BINARY_THRESHOLD = 128
FINAL_CLOSE_ITERATIONS = 2

# Geometric Intensity Layer (ORB + ΔI)
GEOMETRIC_INTENSITY_ENABLED = True
GEOMETRIC_INTENSITY_THRESHOLD = 5
GEOMETRIC_INTENSITY_WEIGHT = 0.5
GEOMETRIC_INTENSITY_DILATE_KERNEL = 3

