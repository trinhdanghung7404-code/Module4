import sys
sys.path.insert(0, 'v2')
import os, cv2, numpy as np

# Load our earlier extracted patch_data or run the complete test
from detailed_debug import (
    micro_align_patches, compute_orientation_profile, is_angle_matching_any
)
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from skimage.metrics import structural_similarity as ssim

p1 = 'images/1.jpg'
p2 = 'images/2.jpg'
product_img = cv2.imread(p1)
return_img = cv2.imread(p2)
h, w = product_img.shape[:2]

segmenter = ObjectSegmenter()
p_seg = segmenter.segment(product_img)
r_seg = segmenter.segment(return_img)

normalizer = ImageNormalizer()
return_normalized = normalizer.normalize(return_img, product_img, r_seg["mask"], p_seg["mask"])
return_ct = normalizer.color_transfer(return_img, product_img, r_seg["mask"], p_seg["mask"])

registrator = ImageRegistration()
reg_result = registrator.register(product_img, return_normalized, p_seg["mask"], r_seg["mask"])

inliers_p = reg_result["product_points"]
inliers_r = reg_result["return_points"]

mesh_builder = MeshBuilder()
vertices, triangles = mesh_builder.build(inliers_p, inliers_r, np.arange(len(inliers_p)))

p_gray = cv2.cvtColor(product_img, cv2.COLOR_BGR2GRAY)
r_gray = cv2.cvtColor(return_ct, cv2.COLOR_BGR2GRAY)
p_med = cv2.medianBlur(p_gray, 3)
r_med = cv2.medianBlur(r_gray, 3)

clahe_global = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
p_clahe_global = clahe_global.apply(p_med)
r_clahe_global = clahe_global.apply(r_med)

p_edge_raw = cv2.Canny(p_clahe_global, 40, 120)
r_edge_raw = cv2.Canny(r_clahe_global, 40, 120)

mesh_mask_p = np.zeros((h, w), dtype=np.uint8)
mesh_mask_r = np.zeros((h, w), dtype=np.uint8)
for i, tri in enumerate(triangles):
    pts_p = np.array([list(vertices[idx].product_xy) for idx in tri.vertex_indices], dtype=np.int32).reshape((-1, 1, 2))
    pts_r = np.array([list(vertices[idx].return_xy) for idx in tri.vertex_indices], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillConvexPoly(mesh_mask_p, pts_p, 255)
    cv2.fillConvexPoly(mesh_mask_r, pts_r, 255)

p_edge_raw[mesh_mask_p == 0] = 0
r_edge_raw[mesh_mask_r == 0] = 0

def filter_edge_noise(edge_mask, min_stroke_len=12.0, max_round_diag=22.0):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask, connectivity=8)
    clean_mask = np.zeros_like(edge_mask)
    for lbl in range(1, num_labels):
        w_ = stats[lbl, cv2.CC_STAT_WIDTH]
        h_ = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        diag = np.sqrt(w_**2 + h_**2)
        if area < 6: continue
        aspect = max(w_, h_) / max(min(w_, h_), 1)
        if aspect < 2.0 and diag <= max_round_diag: continue
        if diag >= min_stroke_len or aspect >= 2.2 or area >= 30:
            clean_mask[labels == lbl] = 255
    return clean_mask

p_edge_denoised = filter_edge_noise(p_edge_raw)
r_edge_denoised = filter_edge_noise(r_edge_raw)

p_lab_full = cv2.cvtColor(product_img, cv2.COLOR_BGR2LAB)
r_lab_full = cv2.cvtColor(return_ct, cv2.COLOR_BGR2LAB)
offset_a = float(np.median(r_lab_full[mesh_mask_r > 0, 1]) - np.median(p_lab_full[mesh_mask_p > 0, 1]))
offset_b = float(np.median(r_lab_full[mesh_mask_r > 0, 2]) - np.median(p_lab_full[mesh_mask_p > 0, 2]))

# Extract patches
raw_patch_items = []
for i, tri in enumerate(triangles):
    pts_p = np.array([list(vertices[idx].product_xy) for idx in tri.vertex_indices], dtype=np.float32)
    pts_r = np.array([list(vertices[idx].return_xy) for idx in tri.vertex_indices], dtype=np.float32)
    patch_p, mask_p = canonical_triangle_patch(product_img, pts_p, target_size=72)
    patch_r, mask_r = canonical_triangle_patch(return_ct, pts_r, target_size=72)
    patch_r_raw, _ = canonical_triangle_patch(return_img, pts_r, target_size=72)
    patch_pe, _ = canonical_triangle_patch(p_edge_denoised, pts_p, target_size=72)
    patch_re, _ = canonical_triangle_patch(r_edge_denoised, pts_r, target_size=72)
    raw_patch_items.append({
        "id": i, "patch_p": patch_p, "patch_r": patch_r, "patch_r_raw": patch_r_raw,
        "patch_pe": patch_pe, "patch_re": patch_re, "mask_p": mask_p,
        "pts_p": pts_p, "pts_r": pts_r, "tri": tri, "is_b_shift": False
    })

# Now let's implement the 4 PRINCIPLES in test mode!
k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_open_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k_erode_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

patch_data_4p = []
color_diffs = []

for item in raw_patch_items:
    i = item["id"]
    patch_p = item["patch_p"]
    patch_r = item["patch_r"]
    patch_r_raw = item["patch_r_raw"]
    mask_p = item["mask_p"]
    tri = item["tri"]
    valid = mask_p > 0

    # PRINCIPLE 1: Vùng an toàn nội bộ tam giác (cách biên ít nhất 4px)
    inner_triangle = cv2.erode(mask_p, k_erode_5) > 0

    p_c_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)
    r_c_gray = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
    p_raw_gray = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
    r_raw_gray = cv2.cvtColor(patch_r_raw, cv2.COLOR_BGR2GRAY)

    glare_px = (r_c_gray > 220) | (p_c_gray > 220)

    hist_p, peaks_p, coh_p = compute_orientation_profile(p_c_gray, mask_p)
    hist_r, peaks_r, coh_r = compute_orientation_profile(r_c_gray, mask_p)
    ori_corr = float(np.dot(hist_p, hist_r) / (np.linalg.norm(hist_p) * np.linalg.norm(hist_r) + 1e-8)) if (np.sum(hist_p) > 0 and np.sum(hist_r) > 0) else 1.0

    p_c_lab = cv2.cvtColor(patch_p, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_c_lab = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_c_lab[..., 1] -= offset_a
    r_c_lab[..., 2] -= offset_b

    p_what = cv2.morphologyEx(p_c_gray, cv2.MORPH_TOPHAT, k_morph)
    p_bhat = cv2.morphologyEx(p_c_gray, cv2.MORPH_BLACKHAT, k_morph)
    r_what = cv2.morphologyEx(r_c_gray, cv2.MORPH_TOPHAT, k_morph)
    r_bhat = cv2.morphologyEx(r_c_gray, cv2.MORPH_BLACKHAT, k_morph)

    p_c_edge = item["patch_pe"].copy()
    r_c_edge = item["patch_re"].copy()
    p_c_edge[~valid] = 0
    r_c_edge[~valid] = 0

    p_edge_cnt = int(np.sum(p_c_edge > 0))
    r_edge_cnt = int(np.sum(r_c_edge > 0))

    gx_r = cv2.Sobel(r_c_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(r_c_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_r = np.sqrt(gx_r**2 + gy_r**2)
    ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

    p_all_edges = ((p_bhat > 15) | (p_what > 15) | (p_c_edge > 0)).astype(np.uint8) * 255
    dist_to_p_all = cv2.distanceTransform(cv2.bitwise_not(p_all_edges), cv2.DIST_L2, 3)

    # 1. Crack / Scratch detection
    if p_edge_cnt < 20:
        max_broken_length = 0.0
        max_white_scratch = 0.0
        max_dark_crack = 0.0
    else:
        crack_cand = (r_bhat > 30) & valid & inner_triangle & (~glare_px)
        scratch_cand = (r_what > 30) & valid & inner_triangle & (~glare_px)
        anom_crack = np.zeros_like(crack_cand)
        anom_scratch = np.zeros_like(scratch_cand)

        for y, x in zip(*np.where(crack_cand)):
            d = dist_to_p_all[y, x]
            near_exact = (d <= 2.5)
            near_shifted = (d <= 6.0) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
            if not (near_exact or near_shifted):
                anom_crack[y, x] = True

        for y, x in zip(*np.where(scratch_cand)):
            d = dist_to_p_all[y, x]
            near_exact = (d <= 2.5)
            near_shifted = (d <= 6.0) and is_angle_matching_any(ang_r[y, x], peaks_p, tol_deg=25.0)
            if not (near_exact or near_shifted):
                anom_scratch[y, x] = True

        cnts_w, _ = cv2.findContours(anom_scratch.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_b, _ = cv2.findContours(anom_crack.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_white_scratch = max([cv2.arcLength(c, False) for c in cnts_w], default=0.0)
        max_dark_crack = max([cv2.arcLength(c, False) for c in cnts_b], default=0.0)

    # PRINCIPLE 1 & 2: Vùng an toàn nội bộ + Dung sai vi mô cho Broken Edges
    # Chỉ xét nét Product nằm bên trong lòng tam giác (inner_triangle), loại bỏ nét mép (như Tri #86, Tri #166)!
    if p_edge_cnt >= 20:
        if np.any(r_c_edge > 0):
            dist_r = cv2.distanceTransform(cv2.bitwise_not(r_c_edge), cv2.DIST_L2, 3)
            # Dung sai đàn hồi: dist_r > 6.0 và Return hoàn toàn phẳng (mag_r < 25.0) và NẰM TRONG LÒNG TAM GIÁC
            broken_edges = (p_c_edge > 0) & (dist_r > 6.0) & (mag_r < 25.0) & (~glare_px) & inner_triangle
            cnts_broken, _ = cv2.findContours(broken_edges.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_broken_length = max([cv2.arcLength(c, False) for c in cnts_broken], default=0.0)
        else:
            broken_edges = (p_c_edge > 0) & inner_triangle & (~glare_px)
            cnts_broken, _ = cv2.findContours(broken_edges.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            max_broken_length = max([cv2.arcLength(c, False) for c in cnts_broken], default=0.0)
    else:
        max_broken_length = 0.0

    # Intrusive edge on blank porcelain
    grad_p_raw = np.sqrt(cv2.Sobel(p_raw_gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_raw_gray, cv2.CV_32F, 0, 1)**2)
    r_raw_edge = cv2.Canny(r_raw_gray, 40, 120)
    r_raw_edge[~valid] = 0
    gx_r_raw = cv2.Sobel(r_raw_gray, cv2.CV_32F, 1, 0)
    gy_r_raw = cv2.Sobel(r_raw_gray, cv2.CV_32F, 0, 1)
    mag_r_raw = np.sqrt(gx_r_raw**2 + gy_r_raw**2)
    ang_r_raw = (np.arctan2(gy_r_raw, gx_r_raw) * 180.0 / np.pi) % 180.0

    intrusive_cand = (r_raw_edge > 0) & (dist_to_p_all > 6.0) & (grad_p_raw < 25.0) & (mag_r_raw >= 70.0) & (~glare_px) & inner_triangle
    anom_intrusive = np.zeros_like(intrusive_cand)
    for y, x in zip(*np.where(intrusive_cand)):
        if not is_angle_matching_any(ang_r_raw[y, x], peaks_p, tol_deg=25.0):
            anom_intrusive[y, x] = True

    cnts_intrusive, _ = cv2.findContours(anom_intrusive.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_intrusive_length = max([cv2.arcLength(c, False) for c in cnts_intrusive], default=0.0)

    # Blob
    p_vals = p_raw_gray[valid & (~glare_px)].astype(np.float32)
    r_vals = r_raw_gray[valid & (~glare_px)].astype(np.float32)
    if len(p_vals) > 10 and np.std(r_vals) > 1e-3:
        mu_p_loc, std_p_loc = np.mean(p_vals), np.std(p_vals)
        mu_r_loc, std_r_loc = np.mean(r_vals), np.std(r_vals)
        scale_loc = np.clip(std_p_loc / (std_r_loc + 1e-5), 0.7, 1.4)
        r_norm_loc = (r_raw_gray.astype(np.float32) - mu_r_loc) * scale_loc + mu_p_loc
    else:
        r_norm_loc = r_raw_gray.astype(np.float32)

    diff_loc = np.abs(r_norm_loc - p_raw_gray.astype(np.float32))
    diff_loc[~valid] = 0.0
    diff_loc[glare_px] = 0.0
    bin_diff_loc = (diff_loc > 30.0).astype(np.uint8) * 255
    opened_diff = cv2.morphologyEx(bin_diff_loc, cv2.MORPH_OPEN, k_open_3)
    num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(opened_diff, connectivity=8)

    # PRINCIPLE 2 & 3: Ngưỡng tương phản cho mảng hoa văn đã khớp hướng (như Tri #122)
    # Nếu ori_corr >= 0.80 hoặc p_edge_cnt < 20, và không có nứt sắc nét:
    # thì hoa văn khớp hình học -> Đòi hỏi contrast_th = 90.0 để tránh bắt nhầm chênh đổ bóng!
    is_shape_intact = bool((ori_corr >= 0.80 or p_edge_cnt < 20) and max_dark_crack < 10.0 and max_white_scratch < 10.0 and max_intrusive_length < 10.0)
    contrast_th = 90.0 if is_shape_intact else 60.0

    valid_blobs = []
    for lbl in range(1, num_lbl):
        area = stats[lbl, cv2.CC_STAT_AREA]
        m = (lbls == lbl)
        if np.max(diff_loc[m]) >= contrast_th:
            valid_blobs.append(area)
    max_solid_blob = max(valid_blobs, default=0)

    struct_metric = (
        1.5 * max_dark_crack +
        1.5 * max_white_scratch +
        2.0 * max_intrusive_length +
        0.1 * max_solid_blob +
        1.5 * max_broken_length
    )

    # Layer 2 Color
    c_da = p_c_lab[..., 1] - r_c_lab[..., 1]
    c_db = p_c_lab[..., 2] - r_c_lab[..., 2]
    patch_diff = np.sqrt(c_da ** 2 + c_db ** 2)
    non_glare = (mask_p > 0) & (r_c_lab[..., 0] <= 225) & (p_c_lab[..., 0] <= 225)
    mean_chroma_err = float(np.mean(patch_diff[non_glare])) if np.any(non_glare) else 0.0
    color_diffs.append(mean_chroma_err)

    strong_color_diff = (patch_diff > 18.0) & non_glare
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(strong_color_diff.astype(np.uint8) * 255, connectivity=8)
    max_blob_area = max([stats[lbl, cv2.CC_STAT_AREA] for lbl in range(1, num_labels)], default=0)
    has_color_blob = bool(max_blob_area >= 25)

    patch_data_4p.append({
        "id": i, "broken_length": max_broken_length, "dark_crack": max_dark_crack,
        "white_scratch": max_white_scratch, "intrusive_len": max_intrusive_length,
        "blob_area": max_solid_blob, "struct_metric": struct_metric,
        "has_color_blob": has_color_blob, "chroma_err": mean_chroma_err,
        "v_indices": set(tri.vertex_indices), "ori_corr": ori_corr,
        "p_edge_cnt": p_edge_cnt, "r_edge_cnt": r_edge_cnt,
        "pts_p": pts_p, "pts_r": pts_r
    })

l1_metrics = [p["struct_metric"] for p in patch_data_4p]
l1_mean, l1_std = float(np.mean(l1_metrics)), float(np.std(l1_metrics))
c_mean, c_std = float(np.mean(color_diffs)), float(np.std(color_diffs))

# PRINCIPLE 3 & 4: Quyết định dựa trên Hội tụ đa tầng & Không lây lan bừa bãi
tp_ids = {635, 445, 443, 631, 303, 456, 605}

flagged_4p = []
for item in patch_data_4p:
    i = item["id"]
    z_l1 = (item["struct_metric"] - l1_mean) / (l1_std + 1e-8)
    z_c = (item["chroma_err"] - c_mean) / (c_std + 1e-8)
    
    # Layer 2: Biến màu sắc men
    is_l2 = bool(z_c > 2.0 and item["chroma_err"] > 16.0 and item["has_color_blob"])
    
    # Layer 1: Có tổn thương vật lý thực thụ bên trong lòng tam giác
    has_strong_l1 = bool(
        item["broken_length"] >= 12.0 or 
        item["dark_crack"] >= 12.0 or 
        item["white_scratch"] >= 12.0 or 
        item["intrusive_len"] >= 12.0 or 
        (item["blob_area"] >= 50 and z_l1 > 1.2)
    )

    # QUY TẮC HỘI TỤ (DUAL-LAYER CONVERGENCE):
    # 1. Nếu có biến đổi màu men (L2) VÀ có dấu hiệu cấu trúc (L1) -> CHẮC CHẮN LỖI (HIGH CONFIDENCE)
    # 2. Hoặc nếu tổn thương cấu trúc cực lớn trong lòng tam giác (broken >= 20 hoặc crack >= 15) -> LỖI
    # 3. Loại bỏ hoàn toàn lây lan không kiểm soát
    is_defect = bool((is_l2 and has_strong_l1) or (item["broken_length"] >= 25.0) or (item["dark_crack"] >= 15.0) or (item["intrusive_len"] >= 15.0))
    
    if is_defect:
        flagged_4p.append(i)

print("\n" + "="*80)
print("=== EXPERIMENTAL RESULTS OF 4 PRINCIPLES (1.jpg vs 2.jpg) ===")
print("="*80)
print(f"Total flagged BEFORE: 116 (including 109 FALSE POSITIVES)")
print(f"Total flagged AFTER:  {len(flagged_4p)}")

tp_detected = [i for i in tp_ids if i in flagged_4p]
tp_missed = [i for i in tp_ids if i not in flagged_4p]
fp_remaining = [i for i in flagged_4p if i not in tp_ids]

print(f"\n1. TRUE POSITIVE RECALL: {len(tp_detected)} / {len(tp_ids)}")
print(f"   - Detected TPs: {sorted(tp_detected)}")
print(f"   - Missed TPs:   {sorted(tp_missed)}")

print(f"\n2. FALSE POSITIVE REDUCTION:")
print(f"   - Previous FPs:   109")
print(f"   - Remaining FPs:  {len(fp_remaining)}")
print(f"   - FP Drop Rate:   {(109 - len(fp_remaining)) / 109 * 100:.1f}%")
print(f"   - Remaining FP IDs: {sorted(fp_remaining)}")

print(f"\n3. SPECIFIC USER-REPORTED TRIANGLES:")
for tid, name in [(86, "Tri #86 (top border leakage)"), (122, "Tri #122 (dark pattern shading)"), (166, "Tri #166 (corner Z-snippet)"), (131, "Tri #131 (blank porcelain)")]:
    st = "DEFECT (FP)" if tid in flagged_4p else "INTACT (SUCCESSFULLY ELIMINATED FP!)"
    print(f"   - {name}: {st}")
