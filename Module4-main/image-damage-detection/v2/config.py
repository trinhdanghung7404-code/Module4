"""Centralized configuration for damage detection v2."""

# --- Segmentation ---
MORPH_KERNEL_SIZE = 5
MORPH_CLOSE_ITERATIONS = 2
MORPH_OPEN_ITERATIONS = 1

# --- Normalization ---
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
COLOR_TRANSFER_STD_RATIO_CLIP = (0.5, 2.0)  # min, max for std ratio clipping
RETINEX_SIGMA = 50  # Gaussian blur sigma for illumination estimation
LOCAL_NORM_KERNEL_SIZE = 31  # kernel size for local mean/std

# --- Registration ---
SUPERPOINT_MAX_KEYPOINTS = None  # None = extract ALL keypoints on pattern
SUPERPOINT_CONFIDENCE = 0.005
SUPERPOINT_NMS_RADIUS = 4
RANSAC_REPROJ_THRESHOLD = 10.0  # Suitable for 2560x1920 curved surfaces (350+ inliers)
MIN_INLIERS = 10  # minimum inlier matches to proceed

# --- Mesh ---
MIN_TRIANGLE_AREA = 50.0  # pixels^2, skip tiny triangles
MAX_TRIANGLE_EDGE = 260.0  # pixels, filter out elongated bridge triangles across empty porcelain

# --- Structure Layer ---
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE_RADIUS = 2  # dilate edges before comparison to tolerate small shifts
STRUCTURE_ZSCORE_THRESHOLD = 2.0  # z-score above which a triangle is flagged

# --- Appearance Layer ---
SSIM_WIN_SIZE = 7
APPEARANCE_ZSCORE_THRESHOLD = 2.0
COLOR_DIFF_PERCENTILE = 95  # percentile-based threshold for color diff

# --- Fusion ---
# Both layers flag → HIGH confidence damage
# Only structure flags → MEDIUM confidence → accept
# Only appearance flags → LOW confidence → reject (might be residual noise)
FUSION_REQUIRE_STRUCTURE = True  # structure layer must agree for damage to be confirmed

# --- Validation ---
MIN_DAMAGE_AREA = 100  # minimum connected component area in pixels
MAX_DAMAGE_ASPECT_RATIO = 10.0  # reject very thin components (likely edge artifacts)

# --- Patch Extraction ---
PATCH_SIZE = 64  # size of patches cropped around keypoints for comparison
PATCH_RESIZE = 64  # resize patches to this before comparison
