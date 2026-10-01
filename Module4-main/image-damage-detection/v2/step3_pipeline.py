"""Step 3 Pipeline: 2-Layer Damage Inspection (Không màu & Có màu) per Triangle.

- Layer 1 (Không màu): Quét nét vẽ hoa văn, vết nứt, đứt gãy cấu trúc (bất biến ánh sáng).
- Layer 2 (Có màu): Quét tróc men màu, trầy xước mất màu hoa văn, vết ố bẩn.
"""

import os
import sys
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder, Vertex, Triangle


def canonical_triangle_patch(image: np.ndarray, pts: np.ndarray, target_size: int = 72) -> tuple:
    """Warp a single triangle patch into a canonical right triangle of size target_size x target_size."""
    target_pts = np.array([
        [0.0, 0.0],
        [float(target_size - 1), 0.0],
        [0.0, float(target_size - 1)],
    ], dtype=np.float32)

    src_pts = pts.astype(np.float32)
    M = cv2.getAffineTransform(src_pts, target_pts)
    warped = cv2.warpAffine(image, M, (target_size, target_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Mask for the canonical right triangle
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    triangle_poly = np.array([[0, 0], [target_size - 1, 0], [0, target_size - 1]], dtype=np.int32)
    cv2.fillConvexPoly(mask, triangle_poly, 255)
    return warped, mask


def evaluate_triangle_structure(prod_patch: np.ndarray, ret_patch: np.ndarray, mask: np.ndarray,
                                v_indices: tuple, vertices: list,
                                prod_descs: np.ndarray, ret_descs: np.ndarray) -> dict:
    """LAYER 1: Không màu - So sánh cấu trúc nét vẽ, vết nứt, gãy hoa văn."""
    # 1. Grayscale
    p_gray = cv2.cvtColor(prod_patch, cv2.COLOR_BGR2GRAY) if prod_patch.ndim == 3 else prod_patch
    r_gray = cv2.cvtColor(ret_patch, cv2.COLOR_BGR2GRAY) if ret_patch.ndim == 3 else ret_patch

    # CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    p_gray_clahe = clahe.apply(p_gray)
    r_gray_clahe = clahe.apply(r_gray)

    # 2. Canny edge analysis
    p_edge = cv2.Canny(p_gray_clahe, 40, 120)
    r_edge = cv2.Canny(r_gray_clahe, 40, 120)

    # Dilate product edge slightly (tolerance for 1-2px shift)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    p_edge_dilated = cv2.dilate(p_edge, kernel)

    # New edge = edge in return that has no corresponding edge in product (signs of cracks)
    new_edges = cv2.bitwise_and(r_edge, cv2.bitwise_not(p_edge_dilated))
    new_edges[mask == 0] = 0

    valid_pixels = np.count_nonzero(mask)
    new_edge_ratio = np.count_nonzero(new_edges) / max(1, valid_pixels)

    # 3. SuperPoint descriptor similarity at the 3 vertices
    desc_sims = []
    if prod_descs is not None and ret_descs is not None:
        for idx in v_indices:
            v = vertices[idx]
            if v.descriptor_index is not None and v.descriptor_index < len(prod_descs):
                pd = prod_descs[v.descriptor_index]
                rd = ret_descs[v.descriptor_index]
                sim = float(np.dot(pd, rd) / (np.linalg.norm(pd) * np.linalg.norm(rd) + 1e-8))
                desc_sims.append(sim)
    mean_desc_sim = float(np.mean(desc_sims)) if desc_sims else 1.0

    # 4. SSIM on grayscale
    ssim_val = ssim(p_gray_clahe, r_gray_clahe, data_range=255)

    # Composite structure score (higher = intact, lower = damaged)
    structure_score = 0.4 * ssim_val + 0.3 * mean_desc_sim + 0.3 * (1.0 - min(new_edge_ratio * 5.0, 1.0))

    return {
        "ssim": ssim_val,
        "new_edge_ratio": new_edge_ratio,
        "desc_sim": mean_desc_sim,
        "structure_score": structure_score,
    }


def evaluate_triangle_color(prod_patch: np.ndarray, ret_patch: np.ndarray, mask: np.ndarray,
                            color_offset_ab: tuple = (0.0, 0.0)) -> dict:
    """LAYER 2: Có màu - So sánh màu men, tróc men, mất màu hoa văn."""
    # Convert to Lab
    p_lab = cv2.cvtColor(prod_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(ret_patch, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Remove global color cast on return patch
    r_lab[..., 1] -= color_offset_ab[0]
    r_lab[..., 2] -= color_offset_ab[1]

    # Measure chroma distance delta_E on channels a and b only (ignore L to avoid light/shadow bias)
    delta_a = p_lab[..., 1] - r_lab[..., 1]
    delta_b = p_lab[..., 2] - r_lab[..., 2]
    chroma_diff = np.sqrt(delta_a ** 2 + delta_b ** 2)

    valid_pixels = mask > 0
    mean_chroma_diff = float(np.mean(chroma_diff[valid_pixels])) if np.any(valid_pixels) else 0.0

    # Color score: 1.0 if diff is 0, drops toward 0 as diff grows
    # A difference of 15-20 units in Lab chroma indicates visible discoloration/scratch
    color_score = float(np.exp(-mean_chroma_diff / 15.0))

    return {
        "mean_chroma_diff": mean_chroma_diff,
        "color_score": color_score,
    }


def run_step_3(product_path: str, return_path: str, output_dir: str = "debug_step3"):
    """Run Step 1 & 2 first, then execute Step 3 (2-Layer inspection) and save visual outputs."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("  EXECUTING STEP 3: 2-LAYER TRIANGLE DAMAGE INSPECTION")
    print("=" * 70)

    start_time = time.time()

    # --- SETUP: Run Step 1 & 2 Registration ---
    product_img = cv2.imread(product_path)
    return_img = cv2.imread(return_path)
    if product_img is None or return_img is None:
        raise FileNotFoundError("Could not load input images.")

    segmenter = ObjectSegmenter()
    prod_seg = segmenter.segment(product_img)
    ret_seg = segmenter.segment(return_img)

    normalizer = ImageNormalizer()
    return_normalized = normalizer.normalize(
        return_img, product_img, ret_seg["mask"], prod_seg["mask"]
    )

    registrator = ImageRegistration(max_keypoints=None)
    reg_result = registrator.register(
        product_img, return_normalized, prod_seg["mask"], ret_seg["mask"]
    )

    mesh_builder = MeshBuilder()
    vertices, triangles = mesh_builder.build(
        reg_result["product_points"],
        reg_result["return_points"],
        np.arange(len(reg_result["product_points"])),
    )
    print(f"[Input] Registered {len(triangles)} triangles across the patterned surface.")

    # Calculate global color cast offset between product and return on object mask
    p_lab_full = cv2.cvtColor(product_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab_full = cv2.cvtColor(return_normalized, cv2.COLOR_BGR2LAB).astype(np.float32)
    p_valid = prod_seg["mask"] > 0
    r_valid = ret_seg["mask"] > 0
    offset_a = float(np.median(r_lab_full[..., 1][r_valid]) - np.median(p_lab_full[..., 1][p_valid]))
    offset_b = float(np.median(r_lab_full[..., 2][r_valid]) - np.median(p_lab_full[..., 2][p_valid]))
    print(f"[Color Calibration] Global cast offset: da={offset_a:.2f}, db={offset_b:.2f}")

    # =========================================================================
    # STEP 3: INSPECT EACH TRIANGLE WITH LAYER 1 (KHÔNG MÀU) & LAYER 2 (CÓ MÀU)
    # =========================================================================
    print("\n--- Evaluating Triangles Across 2 Independent Layers ---")
    struct_scores = []
    color_scores = []

    for tri in triangles:
        # Get 3 coordinates in product and return
        pts_prod = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_ret = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

        # Warp both into canonical 72x72 right triangle patch
        patch_p, mask_p = canonical_triangle_patch(product_img, pts_prod, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(return_normalized, pts_ret, target_size=72)

        # Layer 1: Không màu (Structure)
        s_eval = evaluate_triangle_structure(
            patch_p, patch_r, mask_p,
            tri.vertex_indices, vertices,
            reg_result["product_descriptors"], reg_result["return_descriptors"]
        )
        struct_scores.append(s_eval)

        # Layer 2: Có màu (Color / Men)
        c_eval = evaluate_triangle_color(patch_p, patch_r, mask_p, (offset_a, offset_b))
        color_scores.append(c_eval)

    # --- Adaptive Outlier Detection via Z-score ---
    all_s_scores = [s["structure_score"] for s in struct_scores]
    s_mean, s_std = np.mean(all_s_scores), np.std(all_s_scores)

    all_c_diffs = [c["mean_chroma_diff"] for c in color_scores]
    c_mean, c_std = np.mean(all_c_diffs), np.std(all_c_diffs)

    layer1_damaged_indices = set()
    layer2_damaged_indices = set()

    for i in range(len(triangles)):
        # Layer 1 flag: abnormally low structure score (z < -2.2)
        z_s = (all_s_scores[i] - s_mean) / (s_std + 1e-8)
        if z_s < -2.2 and all_s_scores[i] < 0.65:
            layer1_damaged_indices.add(i)

        # Layer 2 flag: abnormally high chroma color difference (z > 2.5 and diff > 15)
        z_c = (all_c_diffs[i] - c_mean) / (c_std + 1e-8)
        if z_c > 2.5 and all_c_diffs[i] > 15.0:
            layer2_damaged_indices.add(i)

    print(f"[Layer 1 - Structure] Flagged {len(layer1_damaged_indices)}/{len(triangles)} triangles (crack / broken lines)")
    print(f"[Layer 2 - Color]     Flagged {len(layer2_damaged_indices)}/{len(triangles)} triangles (glaze scratch / color loss)")

    fused_damaged = layer1_damaged_indices.union(layer2_damaged_indices)
    print(f"[Combined Decision]   Total Damaged Triangles: {len(fused_damaged)}/{len(triangles)}")

    elapsed = time.time() - start_time
    print(f"[Processing Time] {elapsed:.2f}s")

    # =========================================================================
    # VISUAL OUTPUT GENERATION FOR BOTH LAYERS
    # =========================================================================
    print("\n--- Generating Visual Outputs for Both Layers ---")

    # Image 1: LAYER 1 - KHÔNG MÀU (Hiển thị thân bình đen trắng + Lưới nét vẽ)
    gray_prod = cv2.cvtColor(product_img, cv2.COLOR_BGR2GRAY)
    vis_l1 = cv2.cvtColor(gray_prod, cv2.COLOR_GRAY2BGR)

    for i, tri in enumerate(triangles):
        pts = np.array([list(vertices[idx].product_xy) for idx in tri.vertex_indices], dtype=np.int32).reshape((-1, 1, 2))
        if i in layer1_damaged_indices:
            cv2.fillPoly(vis_l1, [pts], (0, 0, 255))  # Solid Red fill for damaged structure
            cv2.polylines(vis_l1, [pts], True, (0, 0, 255), 2)
        else:
            cv2.polylines(vis_l1, [pts], True, (0, 255, 0), 1)  # Green wireframe for intact

    cv2.putText(vis_l1, "LAYER 1: KHONG MAU (Structure / Edge / Nứt vỡ)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    cv2.putText(vis_l1, f"Damaged: {len(layer1_damaged_indices)} / {len(triangles)} triangles", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if layer1_damaged_indices else (0, 255, 0), 2)

    path_l1 = os.path.join(output_dir, "step3_layer1_structure.jpg")
    cv2.imwrite(path_l1, vis_l1)

    # Image 2: LAYER 2 - CÓ MÀU (Hiển thị thân bình màu + Lưới tróc men/mất màu)
    vis_l2 = product_img.copy()

    for i, tri in enumerate(triangles):
        pts = np.array([list(vertices[idx].product_xy) for idx in tri.vertex_indices], dtype=np.int32).reshape((-1, 1, 2))
        if i in layer2_damaged_indices:
            cv2.fillPoly(vis_l2, [pts], (0, 0, 255))  # Solid Red fill for color defect
            cv2.polylines(vis_l2, [pts], True, (0, 0, 255), 2)
        else:
            cv2.polylines(vis_l2, [pts], True, (0, 255, 0), 1)  # Green wireframe for intact

    cv2.putText(vis_l2, "LAYER 2: CO MAU (Color / Men / Tróc màu)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    cv2.putText(vis_l2, f"Damaged: {len(layer2_damaged_indices)} / {len(triangles)} triangles", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if layer2_damaged_indices else (0, 255, 0), 2)

    path_l2 = os.path.join(output_dir, "step3_layer2_color.jpg")
    cv2.imwrite(path_l2, vis_l2)

    # Image 3: Combined Side-by-Side 3-Panel Inspection
    max_dim = 1080
    h_orig, w_orig = product_img.shape[:2]
    sc = min(max_dim / max(h_orig, w_orig), 1.0)
    w_small, h_small = int(w_orig * sc), int(h_orig * sc)

    p_orig_small = cv2.resize(product_img, (w_small, h_small))
    p_l1_small = cv2.resize(vis_l1, (w_small, h_small))
    p_l2_small = cv2.resize(vis_l2, (w_small, h_small))

    # Fourth panel: Fused Decision
    vis_fused = product_img.copy()
    for i, tri in enumerate(triangles):
        pts = np.array([list(vertices[idx].product_xy) for idx in tri.vertex_indices], dtype=np.int32).reshape((-1, 1, 2))
        if i in fused_damaged:
            overlay = vis_fused.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255))
            vis_fused = cv2.addWeighted(vis_fused, 0.6, overlay, 0.4, 0)
            cv2.polylines(vis_fused, [pts], True, (0, 0, 255), 2)
        else:
            cv2.polylines(vis_fused, [pts], True, (0, 255, 0), 1)

    cv2.putText(vis_fused, "KET QUA HOP NHAT (Fused Decision)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    cv2.putText(vis_fused, f"Tong loi: {len(fused_damaged)} / {len(triangles)} tam giac", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if fused_damaged else (0, 255, 0), 2)
    p_fused_small = cv2.resize(vis_fused, (w_small, h_small))

    combined_grid = np.hstack([p_orig_small, p_l1_small, p_l2_small, p_fused_small])
    path_summary = os.path.join(output_dir, "step3_2layers_combined_inspection.jpg")
    cv2.imwrite(path_summary, combined_grid)

    print(f"\n[Visual Outputs Saved to '{output_dir}/']:")
    print(f"  1. {path_l1}")
    print(f"  2. {path_l2}")
    print(f"  3. {path_summary}")

    return {
        "triangles": len(triangles),
        "layer1_damaged": len(layer1_damaged_indices),
        "layer2_damaged": len(layer2_damaged_indices),
        "fused_damaged": len(fused_damaged),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    prod_file = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    ret_file = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    run_step_3(prod_file, ret_file, output_dir="debug_step3_vase_nodamage")

    # Also run on the damaged test pair if available
    scar_prod = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
    scar_ret = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")
    if os.path.exists(scar_prod) and os.path.exists(scar_ret):
        print("\n" + "=" * 70)
        print("  RUNNING STEP 3 ON DAMAGED TEST CASE (SCAR DAMAGE)")
        print("=" * 70)
        run_step_3(scar_prod, scar_ret, output_dir="debug_step3_vase_scar")
