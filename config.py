"""
config.py
=========
Global Configuration for Ceramic Defect Detection & Inspection Server (Module4_AIReturn).
"""

import os
import sys
import shutil

# --- Base & Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROFILES_DIR = os.path.join(DATA_DIR, "profiles")
REFERENCE_IMAGES_DIR = os.path.join(DATA_DIR, "reference_images")
GHOST_IMAGES_DIR = os.path.join(DATA_DIR, "ghost_overlays")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
UTILS_DIR = os.path.join(BASE_DIR, "utils")

# Cross-platform cloudflared binary locator
_system_cloudflared = shutil.which("cloudflared")
_local_cloudflared = os.path.join(UTILS_DIR, "cloudflared.exe" if sys.platform == "win32" else "cloudflared")
CLOUDFLARED_BIN = _system_cloudflared if _system_cloudflared else _local_cloudflared

# Ensure required directories exist
for d in [DATA_DIR, PROFILES_DIR, REFERENCE_IMAGES_DIR, GHOST_IMAGES_DIR, MODELS_DIR, TEMPLATES_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Server Settings ---
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", 8000))

# --- Cloudinary Credentials ---
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "velll47x")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY", "832925522629552")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "YZxcLiUx5_RqDHEHG3Ywnv6wXKI")

# --- Session Management ---
SESSION_EXPIRE_MINUTES = 30

# --- Standard Inspection Views ---
VIEWS = ["truoc", "sau", "trai", "phai"]
VIEW_LABELS = {
    "truoc": "Mặt Trước",
    "sau": "Mặt Sau",
    "trai": "Mặt Trái",
    "phai": "Mặt Phải",
}

# --- Client-Side Auto-Capture Parameters ---
AUTO_CAPTURE = {
    "stable_duration_sec": 1.0,   # Giữ yên liên tục 1.0 giây để chụp
    "min_match_threshold": 0.35,  # Ngưỡng khớp mẫu ≥ 35%
    "quality_threshold": 0.35,    # Ngưỡng chất lượng tương ứng ≥ 35%
    "glare_max_ratio": 0.08,      # Tối đa 8% pixel lóa sáng
    "duplicate_threshold": 0.97,  # > 97% tương đồng ảnh liền trước → trùng góc
    "buffer_size": 20,            # Buffer chọn best frame
    "min_object_ratio": 0.18,     # Tối thiểu 18% khung hình
    "max_object_ratio": 0.80,     # Không quá sát camera
}
