"""
tools/sync_cloudinary_data.py
=============================
Clean old local data (reference_images, ghost_overlays, profiles, uploads)
and re-download all product images (B1..B22) from Cloudinary with standardized
English naming (front, back, left, right), generate tight ghost overlays,
and enroll DINOv2 reference profiles.
"""

import os
import sys
import shutil
import urllib.request
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cloudinary
import cloudinary.api

from config import (
    CLOUDINARY_CLOUD_NAME,
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET,
    REFERENCE_IMAGES_DIR,
    GHOST_IMAGES_DIR,
    PROFILES_DIR,
    UPLOADS_DIR,
    VIEWS,
)
from core.pipeline import ReferencePipeline
from core.preprocessor import clean_rembg_image

SLOT_MAP = {
    "front": "01_front",
    "back": "02_back",
    "left": "03_left",
    "right": "04_right",
}


def clean_directory(dir_path: str):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return
    for name in os.listdir(dir_path):
        if name == ".gitkeep":
            continue
        full = os.path.join(dir_path, name)
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def crop_rgba_tight(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return rgba[y:y + h, x:x + w]
    return rgba


def main():
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
    )

    print("[1/4] Cleaning old local data directories...")
    for d in [REFERENCE_IMAGES_DIR, GHOST_IMAGES_DIR, PROFILES_DIR, UPLOADS_DIR]:
        clean_directory(d)

    print("[2/4] Fetching product list from Cloudinary (ceramic_shop/products)...")
    folders_resp = cloudinary.api.subfolders("ceramic_shop/products")
    products = sorted(
        f["name"] for f in folders_resp.get("folders", [])
        if not f["name"].startswith("_")
    )
    print(f"Found {len(products)} products: {products}")

    print("[3/4] Initializing ReferencePipeline (DINOv2 + BiRefNet)...")
    pipeline = ReferencePipeline(profiles_dir=PROFILES_DIR)

    print("[4/4] Downloading & enrolling products...")
    for pid in products:
        res = cloudinary.api.resources(
            type="upload",
            prefix=f"ceramic_shop/products/{pid}/",
            max_results=50,
        )
        by_id = {r["public_id"]: r["secure_url"] for r in res.get("resources", [])}

        ref_dir = os.path.join(REFERENCE_IMAGES_DIR, pid)
        ghost_dir = os.path.join(GHOST_IMAGES_DIR, pid)
        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(ghost_dir, exist_ok=True)

        for view in VIEWS:
            raw_key = f"ceramic_shop/products/{pid}/4_angles/{view}"
            prep_key = f"ceramic_shop/products/{pid}/preprocess/{view}"

            raw_url = by_id.get(raw_key)
            prep_url = by_id.get(prep_key)

            if not raw_url and not prep_url:
                continue

            raw_ref_path = os.path.join(ref_dir, f"{view}.jpg")
            ghost_raw_path = os.path.join(ghost_dir, f"{view}.jpg")
            ghost_nb_path = os.path.join(ghost_dir, f"{view}_nb.png")

            if raw_url:
                raw_bytes = download_bytes(raw_url)
                with open(raw_ref_path, "wb") as f:
                    f.write(raw_bytes)
                with open(ghost_raw_path, "wb") as f:
                    f.write(raw_bytes)

            enroll_src_path = raw_ref_path
            temp_white_path = None

            if prep_url:
                prep_bytes = download_bytes(prep_url)
                arr = np.frombuffer(prep_bytes, np.uint8)
                rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                    # Save tight-cropped RGBA as ghost overlay
                    tight = crop_rgba_tight(rgba)
                    cv2.imwrite(ghost_nb_path, tight)

                    # Composite onto white background for clean 1024x1024 enrollment
                    alpha = (rgba[:, :, 3:4].astype(np.float32) / 255.0)
                    bgr = rgba[:, :, :3].astype(np.float32)
                    white_bgr = (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
                    temp_white_path = os.path.join(ref_dir, f"{view}_clean.png")
                    cv2.imwrite(temp_white_path, white_bgr)
                    enroll_src_path = temp_white_path

            # Fallback ghost generation if preprocess/ wasn't on Cloudinary
            if not os.path.exists(ghost_nb_path) and os.path.exists(raw_ref_path):
                bgr = cv2.imread(raw_ref_path)
                mask = pipeline.extract_vase_mask(bgr)
                bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = mask
                cv2.imwrite(ghost_nb_path, crop_rgba_tight(bgra))

            if os.path.exists(enroll_src_path):
                if temp_white_path is None:
                    std_bgr = clean_rembg_image(enroll_src_path, target_size=1024)
                    if std_bgr is not None:
                        temp_white_path = os.path.join(ref_dir, f"{view}_clean.png")
                        cv2.imwrite(temp_white_path, std_bgr)
                        enroll_src_path = temp_white_path
                profile_id = f"{pid}_{SLOT_MAP[view]}"
                pipeline.enroll_product(enroll_src_path, profile_id)

            if temp_white_path and os.path.exists(temp_white_path):
                os.remove(temp_white_path)

    print("[DONE] All Cloudinary data synced and enrolled cleanly in English!")


if __name__ == "__main__":
    main()
