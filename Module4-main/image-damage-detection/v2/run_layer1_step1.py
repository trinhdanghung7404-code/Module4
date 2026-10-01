"""
Script thực thi và đánh giá Độc lập: LAYER 1 - BƯỚC 1 (Macro Structural Feature Screening)
HOÀN TOÀN ĐEN TRẮNG / CHỈ DÙNG BẢN ĐỒ NÉT (NO COLOR)
"""

import os
import sys
import cv2
import numpy as np

# Ensure v2 and root are in path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from layer1_macro import (
    extract_binary_edge_profile,
    compare_binary_profiles,
    visualize_step1_grayscale,
)


def run_step1_on_pair(product_path: str, return_path: str, out_dir: str):
    print("\n" + "=" * 70)
    print(f"  RUNNING LAYER 1 - STEP 1 [NO COLOR - BINARY EDGE ONLY]")
    print(f"  Product: {os.path.basename(product_path)}")
    print(f"  Return:  {os.path.basename(return_path)}")
    print(f"  Output:  {out_dir}")
    print("=" * 70)

    # 1. Load images
    product_img = cv2.imread(product_path)
    return_img = cv2.imread(return_path)
    if product_img is None or return_img is None:
        raise FileNotFoundError(f"Could not load images: {product_path}, {return_path}")

    h, w = product_img.shape[:2]

    # 2. Segment & Normalization (để đăng ký điểm SuperPoint)
    segmenter = ObjectSegmenter()
    p_seg = segmenter.segment(product_img)
    r_seg = segmenter.segment(return_img)

    normalizer = ImageNormalizer()
    return_normalized = normalizer.normalize(return_img, product_img, r_seg["mask"], p_seg["mask"])

    # 3. Registration
    registrator = ImageRegistration(max_keypoints=None)
    reg_result = registrator.register(product_img, return_normalized, p_seg["mask"], r_seg["mask"])

    inliers_p = reg_result["product_points"]
    inliers_r = reg_result["return_points"]

    # 4. Mesh
    mesh_builder = MeshBuilder()
    vertices, triangles = mesh_builder.build(inliers_p, inliers_r, np.arange(len(inliers_p)))
    print(f"  Constructed Mesh with {len(vertices)} vertices and {len(triangles)} triangles.")

    # 5. TẠO BẢN ĐỒ NÉT ĐEN TRẮNG SẠCH (LAYER 1 GROUND TRUTH)
    p_gray = cv2.cvtColor(product_img, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(return_normalized, cv2.COLOR_BGR2GRAY)

    p_med = cv2.medianBlur(p_gray, 3)
    r_med = cv2.medianBlur(r_gray, 3)

    p_edge_raw = cv2.Canny(p_med, 60, 160)
    r_edge_raw = cv2.Canny(r_med, 60, 160)

    # Mask chỉ trong vùng mesh
    mesh_mask_p = np.zeros((h, w), dtype=np.uint8)
    mesh_mask_r = np.zeros((h, w), dtype=np.uint8)
    for tri in triangles:
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.int32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
        cv2.fillConvexPoly(mesh_mask_p, pts_p, 255)
        cv2.fillConvexPoly(mesh_mask_r, pts_r, 255)

    p_edge_raw[mesh_mask_p == 0] = 0
    r_edge_raw[mesh_mask_r == 0] = 0

    # Lọc mảnh rác li ti
    def filter_noise(edge_mask: np.ndarray, min_area: int = 50, min_diag: float = 35.0) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
        clean_mask = np.zeros_like(edge_mask)
        for lbl in range(1, num_labels):
            w_c = stats[lbl, cv2.CC_STAT_WIDTH]
            h_c = stats[lbl, cv2.CC_STAT_HEIGHT]
            area = stats[lbl, cv2.CC_STAT_AREA]
            diag = np.sqrt(w_c**2 + h_c**2)
            if area >= min_area or diag >= min_diag:
                clean_mask[labels == lbl] = 255
        return clean_mask

    p_edge_clean = filter_noise(p_edge_raw, min_area=50, min_diag=35.0)
    r_edge_clean = filter_noise(r_edge_raw, min_area=50, min_diag=35.0)

    # 6. BƯỚC 1: HỌC & SO SÁNH ĐẶC TRƯNG HÌNH HỌC TRONG TỪNG TAM GIÁC (THUẦN ĐEN TRẮNG)
    profiles_p = []
    profiles_r = []
    comparisons = []

    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

        # Cắt patch từ bản đồ nét đen trắng (p_edge_clean & r_edge_clean)
        patch_p, mask_p = canonical_triangle_patch(p_edge_clean, pts_p, target_size=80)
        patch_r, mask_r = canonical_triangle_patch(r_edge_clean, pts_r, target_size=80)

        # Trích xuất đặc trưng thuần túy trên nét nhị phân
        prof_p = extract_binary_edge_profile(patch_p, mask_p)
        prof_r = extract_binary_edge_profile(patch_r, mask_r)

        # So sánh 2 hồ sơ
        comp = compare_binary_profiles(prof_p, prof_r)

        profiles_p.append(prof_p)
        profiles_r.append(prof_r)
        comparisons.append(comp)

    # 7. XUẤT BỘ ẢNH TRỰC QUAN ĐEN TRẮNG
    res = visualize_step1_grayscale(
        p_edge_full=p_edge_clean,
        r_edge_full=r_edge_clean,
        triangles=triangles,
        vertices=vertices,
        profiles_p=profiles_p,
        profiles_r=profiles_r,
        comparisons=comparisons,
        out_dir=out_dir,
    )
    return res


if __name__ == "__main__":
    # Test case 1: Bình nguyên vẹn (1 vs 2)
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    out1 = os.path.join(SCRIPT_DIR, "debug", "01_undamaged_vase", "03_layer1_step1_macro")
    run_step1_on_pair(p1, p2, out1)

    # Test case 2: Bình có vết rách / sẹo nứt thật
    s1 = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
    s2 = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")
    if os.path.exists(s1) and os.path.exists(s2):
        out2 = os.path.join(SCRIPT_DIR, "debug", "02_scar_defect", "03_layer1_step1_macro")
        run_step1_on_pair(s1, s2, out2)
