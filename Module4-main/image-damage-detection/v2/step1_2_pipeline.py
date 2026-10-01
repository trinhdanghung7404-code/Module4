"""Step 1 & Step 2 Pipeline: SuperPoint Extraction + MNN RANSAC Matching + Delaunay Mesh.

Focus: Patterned ceramic surface (captures fine ornament details, white porcelain background ignored).
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure v2 directory is in path
V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder


def run_step_1_and_2(product_path: str, return_path: str, output_dir: str = "debug_step1_2"):
    """Execute Step 1 (SuperPoint feature extraction) and Step 2 (MNN + RANSAC + Mesh).

    Saves high-quality visual outputs illustrating keypoints, inlier matches, and Delaunay mesh.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("  EXECUTING STEP 1 & 2: SUPERPOINT EXTRACTION & MESH REGISTRATION")
    print("=" * 70)

    start_time = time.time()

    # 1. Load images
    product_img = cv2.imread(product_path)
    return_img = cv2.imread(return_path)
    if product_img is None:
        raise FileNotFoundError(f"Cannot load product image: {product_path}")
    if return_img is None:
        raise FileNotFoundError(f"Cannot load return image: {return_path}")

    print(f"[Load] Product: {product_img.shape} | Return: {return_img.shape}")

    # 2. Segment objects
    segmenter = ObjectSegmenter()
    prod_seg = segmenter.segment(product_img)
    ret_seg = segmenter.segment(return_img)
    print(f"[Segmentation] Product mask: {prod_seg['area']:.0f}px | Return mask: {ret_seg['area']:.0f}px")

    # 3. Preprocessing normalization (Reinhard LAB + CLAHE on L-channel)
    normalizer = ImageNormalizer()
    return_normalized = normalizer.normalize(
        return_img, product_img, ret_seg["mask"], prod_seg["mask"]
    )
    print(f"[Normalization] Color & illumination transfer completed.")

    # 4. STEP 1: SuperPoint Extraction & STEP 2: MNN + RANSAC Matching
    print("\n--- STEP 1: Extracting SuperPoint Keypoints & Descriptors ---")
    registrator = ImageRegistration(max_keypoints=None)
    reg_result = registrator.register(
        product_img, return_normalized, prod_seg["mask"], ret_seg["mask"]
    )

    all_prod_kps = reg_result["all_product_keypoints"]
    all_ret_kps = reg_result["all_return_keypoints"]
    inliers_prod = reg_result["product_points"]
    inliers_ret = reg_result["return_points"]
    inlier_count = reg_result["inlier_count"]
    total_matches = reg_result["total_matches"]

    print(f"[Step 1] Product Keypoints Detected: {len(all_prod_kps)}")
    print(f"[Step 1] Return Keypoints Detected:  {len(all_ret_kps)}")
    print(f"\n--- STEP 2: MNN Matching & RANSAC Inlier Filtering ---")
    print(f"[Step 2] Total MNN Matches:          {total_matches}")
    print(f"[Step 2] RANSAC Inliers Confirmed:   {inlier_count} (Ratio: {inlier_count/max(1, total_matches):.2%})")

    # 5. Build Delaunay Mesh on Inliers
    print("\n--- STEP 2 (cont): Delaunay Triangulation on Pattern Vertices ---")
    mesh_builder = MeshBuilder()
    vertices, triangles = mesh_builder.build(
        inliers_prod, inliers_ret, np.arange(len(inliers_prod))
    )
    print(f"[Step 2] Delaunay Mesh Built:        {len(vertices)} Vertices | {len(triangles)} Triangles")

    elapsed = time.time() - start_time
    print(f"\n[Execution Time] {elapsed:.2f}s")

    # =========================================================================
    # VISUAL OUTPUT GENERATION
    # =========================================================================
    print("\n--- Generating Output Visualizations ---")

    # Output 1 & 2: SuperPoint Keypoints on Product and Return
    vis_kp_prod = product_img.copy()
    for pt in all_prod_kps:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis_kp_prod, (x, y), 3, (0, 255, 255), -1)  # Yellow dots
        cv2.circle(vis_kp_prod, (x, y), 4, (0, 165, 255), 1)

    vis_kp_ret = return_img.copy()
    for pt in all_ret_kps:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis_kp_ret, (x, y), 3, (0, 255, 255), -1)  # Yellow dots
        cv2.circle(vis_kp_ret, (x, y), 4, (0, 165, 255), 1)

    cv2.putText(
        vis_kp_prod,
        f"SuperPoint Keypoints: {len(all_prod_kps)}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3,
    )
    cv2.putText(
        vis_kp_ret,
        f"SuperPoint Keypoints: {len(all_ret_kps)}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3,
    )

    path_kp_prod = os.path.join(output_dir, "step1_superpoint_product.jpg")
    path_kp_ret = os.path.join(output_dir, "step1_superpoint_return.jpg")
    cv2.imwrite(path_kp_prod, vis_kp_prod)
    cv2.imwrite(path_kp_ret, vis_kp_ret)

    # Output 3: RANSAC Inlier Matches (Side-by-side with connection lines)
    # Resize for visualization if images are very large
    max_dim = 1280
    h_p, w_p = product_img.shape[:2]
    scale = min(max_dim / max(h_p, w_p), 1.0)
    resized_p = cv2.resize(product_img, (int(w_p * scale), int(h_p * scale)))
    resized_r = cv2.resize(return_img, (int(w_p * scale), int(h_p * scale)))
    h_r, w_r = resized_p.shape[:2]

    vis_matches = np.hstack([resized_p, resized_r])
    # Draw inlier matches
    np.random.seed(42)
    for p_pt, r_pt in zip(inliers_prod, inliers_ret):
        pt1 = (int(p_pt[0] * scale), int(p_pt[1] * scale))
        pt2 = (int(r_pt[0] * scale) + w_r, int(r_pt[1] * scale))
        color = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(150, 255)),
            int(np.random.randint(50, 255)),
        )
        cv2.line(vis_matches, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(vis_matches, pt1, 3, (0, 0, 255), -1)
        cv2.circle(vis_matches, pt2, 3, (0, 0, 255), -1)

    cv2.putText(
        vis_matches,
        f"Product (Studio)",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis_matches,
        f"Return (Customer Phone)",
        (w_r + 30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        vis_matches,
        f"RANSAC Inlier Matches: {inlier_count}",
        (30, h_r - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    path_matches = os.path.join(output_dir, "step2_ransac_matches.jpg")
    cv2.imwrite(path_matches, vis_matches)

    # Output 4 & 5: Delaunay Mesh on Product and Return
    vis_mesh_prod = product_img.copy()
    vis_mesh_ret = return_img.copy()

    for tri in triangles:
        # Product space triangle
        pts_prod = np.array(
            [list(vertices[idx].product_xy) for idx in tri.vertex_indices],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(vis_mesh_prod, [pts_prod], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Return space triangle
        pts_ret = np.array(
            [list(vertices[idx].return_xy) for idx in tri.vertex_indices],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(vis_mesh_ret, [pts_ret], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # Highlight vertex points
    for v in vertices:
        cv2.circle(vis_mesh_prod, (int(v.product_xy[0]), int(v.product_xy[1])), 3, (0, 0, 255), -1)
        cv2.circle(vis_mesh_ret, (int(v.return_xy[0]), int(v.return_xy[1])), 3, (0, 0, 255), -1)

    cv2.putText(
        vis_mesh_prod,
        f"Delaunay Mesh (Product): {len(triangles)} triangles",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )
    cv2.putText(
        vis_mesh_ret,
        f"Delaunay Mesh (Return): {len(triangles)} triangles",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )

    path_mesh_prod = os.path.join(output_dir, "step2_mesh_product.jpg")
    path_mesh_ret = os.path.join(output_dir, "step2_mesh_return.jpg")
    cv2.imwrite(path_mesh_prod, vis_mesh_prod)
    cv2.imwrite(path_mesh_ret, vis_mesh_ret)

    # Output 6: Combined 3-Panel Overview Summary
    h, w = resized_p.shape[:2]
    mesh_p_small = cv2.resize(vis_mesh_prod, (w, h))
    mesh_r_small = cv2.resize(vis_mesh_ret, (w, h))
    matches_small = cv2.resize(vis_matches, (w * 2, h))

    top_row = np.hstack([mesh_p_small, mesh_r_small])
    summary_banner = np.zeros((70, top_row.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        summary_banner,
        f"STEP 1 & 2 SUMMARY | Keypoints: P={len(all_prod_kps)}, R={len(all_ret_kps)} | Inliers: {inlier_count} | Mesh: {len(triangles)} Triangles",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    combined_summary = np.vstack([summary_banner, top_row, matches_small])
    path_summary = os.path.join(output_dir, "step1_2_combined_summary.jpg")
    cv2.imwrite(path_summary, combined_summary)

    print(f"\n[Visual Outputs Saved to '{output_dir}/']:")
    print(f"  1. {path_kp_prod}")
    print(f"  2. {path_kp_ret}")
    print(f"  3. {path_matches}")
    print(f"  4. {path_mesh_prod}")
    print(f"  5. {path_mesh_ret}")
    print(f"  6. {path_summary}")

    return {
        "keypoints_product": len(all_prod_kps),
        "keypoints_return": len(all_ret_kps),
        "inliers": inlier_count,
        "triangles": len(triangles),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    prod_file = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    ret_file = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    run_step_1_and_2(prod_file, ret_file)
