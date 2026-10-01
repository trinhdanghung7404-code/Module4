"""
Cấu hình cho Capture Server.
Đổi thông tin Cloudinary trước khi chạy.
"""
import os

# --- Server ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# --- Cloudinary ---
# Cách lấy: https://console.cloudinary.com/settings/api-keys
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "xoipkxaa")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY", "264816369139255")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "G7AurecZaSEQqHDVSCTuX8YcUqY")

# --- Session ---
SESSION_EXPIRE_MINUTES = 30

# --- Ghost images ---
# Thư mục chứa ghost images, cấu trúc: {product_id}/truoc_nb.png, sau_nb.png, ...
GHOST_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Test_Camera"
)

# --- Các góc chụp ---
VIEWS = ["truoc", "sau", "trai", "phai"]
VIEW_LABELS = {
    "truoc": "Mặt Trước",
    "sau": "Mặt Sau",
    "trai": "Mặt Trái",
    "phai": "Mặt Phải",
}

# --- Auto-capture (gửi xuống client JS) ---
AUTO_CAPTURE = {
    "stable_duration_sec": 7.0,   # Giữ yên 7 giây (đủ thời gian thong thả căn chỉnh & săn đỉnh cao nhất)
    "quality_threshold": 0.65,    # Ngưỡng chất lượng cao hơn để không nhận nhầm sàn/tường
    "glare_max_ratio": 0.08,      # tối đa 8% pixel cháy sáng
    "duplicate_threshold": 0.97,  # > 97% giống ảnh liền trước → trùng góc (bình xoay 90° sẽ pass)
    "buffer_size": 20,            # buffer chọn best frame
    "min_object_ratio": 0.18,     # vật thể phải chiếm ít nhất 18% khung hình
    "max_object_ratio": 0.80,     # không quá sát camera
}

# --- Cấu hình tương thích Image Damage Detection (v2) ---
MORPH_KERNEL_SIZE = 5
MORPH_CLOSE_ITERATIONS = 2
MORPH_OPEN_ITERATIONS = 1
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
COLOR_TRANSFER_STD_RATIO_CLIP = (0.5, 2.0)
RETINEX_SIGMA = 50
LOCAL_NORM_KERNEL_SIZE = 31
SUPERPOINT_MAX_KEYPOINTS = None
SUPERPOINT_CONFIDENCE = 0.005
SUPERPOINT_NMS_RADIUS = 4
RANSAC_REPROJ_THRESHOLD = 10.0
MIN_INLIERS = 10
MIN_TRIANGLE_AREA = 50.0
MAX_TRIANGLE_EDGE = 260.0
CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATE_RADIUS = 2
STRUCTURE_ZSCORE_THRESHOLD = 2.0
SSIM_WIN_SIZE = 7
APPEARANCE_ZSCORE_THRESHOLD = 2.0
COLOR_DIFF_PERCENTILE = 95
FUSION_REQUIRE_STRUCTURE = True
MIN_DAMAGE_AREA = 100
MAX_DAMAGE_ASPECT_RATIO = 10.0
PATCH_SIZE = 64
PATCH_RESIZE = 64
