import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

V2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(V2_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from preprocessing.segmentation import ObjectSegmenter
from preprocessing.registration import ImageRegistration
from preprocessing.normalization import ImageNormalizer
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches


def run_step1_macro(product_path: str, return_path: str, out_dir: str):
    """
    BƯỚC 1: SÀNG LỌC ĐẶC TRƯNG BAO QUÁT (Macro Feature Screening per Triangle)
    
    Ý tưởng:
    1. Trong từng tam giác mesh, chuẩn hóa về ô 72x72 và vi căn chỉnh (micro_align).
    2. Trích xuất các đặc trưng cấu trúc (Layer 1 - thuần thang xám / không màu):
       - Phân bố góc hướng nét (Orientation Histogram 8 bins, [0..180 độ]):
         Phát hiện sự thay đổi cấu trúc, ví dụ ảnh gốc nét thẳng nhưng ảnh trả về có nét cong/xiên lạ.
       - Mật độ nét / Cường độ gradient (Edge Density):
         Đo độ chênh lệch mật độ nét do vết nứt hay nét bị đè mất.
       - Độ tương quan hình dạng (SSIM cấu trúc).
    3. Tính điểm bất thường bao quát (Macro Anomaly Score):
       S_macro = 0.4 * D_orient + 0.3 * D_density + 0.3 * D_ssim
    4. Sàng lọc tam giác nghi ngờ (Suspect Triangles):
       Những tam giác có S_macro cao bất thường (vượt > 2.0 độ lệch chuẩn hoặc > 0.45).
    5. Xuất các ảnh debug trực quan rõ ràng.
    """
    os.makedirs(out_dir, exist_ok=True)
    p_img = cv2.imread(product_path)
    r_img = cv2.imread(return_path)
    if p_img is None or r_img is None:
        raise FileNotFoundError(f"Cannot read {product_path} or {return_path}")
    h, w = p_img.shape[:2]

    # Segmentation & Normalization
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img)
    r_seg = seg.segment(r_img)

    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg["mask"], p_seg["mask"])

    # Registration & Mesh
    reg = ImageRegistration(max_keypoints=None)
    res = reg.register(p_img, r_norm, p_seg["mask"], r_seg["mask"])
    p_pts = res["product_points"]
    r_pts = res["return_points"]

    builder = MeshBuilder()
    vertices, triangles = builder.build(p_pts, r_pts, np.arange(len(p_pts)))

    macro_scores = []
    tri_details = []

    for idx, tri in enumerate(triangles):
        tp = np.array([vertices[i].product_xy for i in tri.vertex_indices], dtype=np.float32)
        tr = np.array([vertices[i].return_xy for i in tri.vertex_indices], dtype=np.float32)

        patch_p, mask_p = canonical_triangle_patch(p_img, tp, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, tr, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        # Grayscale ONLY (Layer 1 - Cấu trúc không dùng màu)
        gp = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)

        valid = mask_p > 0
        if np.count_nonzero(valid) < 50:
            continue

        # 1. Gradient & Orientation Histogram (8 bins từ 0 đến 180 độ)
        gx_p = cv2.Sobel(gp, cv2.CV_32F, 1, 0, ksize=3)
        gy_p = cv2.Sobel(gp, cv2.CV_32F, 0, 1, ksize=3)
        mag_p = np.sqrt(gx_p**2 + gy_p**2)
        ang_p = (np.arctan2(gy_p, gx_p) * 180.0 / np.pi) % 180.0

        gx_r = cv2.Sobel(gr, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(gr, cv2.CV_32F, 0, 1, ksize=3)
        mag_r = np.sqrt(gx_r**2 + gy_r**2)
        ang_r = (np.arctan2(gy_r, gx_r) * 180.0 / np.pi) % 180.0

        edge_p = valid & (mag_p > 25.0)
        edge_r = valid & (mag_r > 25.0)

        if np.any(edge_p):
            hist_p, _ = np.histogram(ang_p[edge_p], bins=8, range=(0, 180), weights=mag_p[edge_p])
            hist_p = hist_p / (np.sum(hist_p) + 1e-6)
        else:
            hist_p = np.zeros(8, dtype=np.float32)

        if np.any(edge_r):
            hist_r, _ = np.histogram(ang_r[edge_r], bins=8, range=(0, 180), weights=mag_r[edge_r])
            hist_r = hist_r / (np.sum(hist_r) + 1e-6)
        else:
            hist_r = np.zeros(8, dtype=np.float32)

        # Chi-Square khoảng cách phân bố hướng cạnh
        d_orient = 0.5 * float(np.sum((hist_p - hist_r)**2 / (hist_p + hist_r + 1e-6)))

        # 2. Độ chênh lệch mật độ nét
        den_p = float(np.mean(mag_p[valid]))
        den_r = float(np.mean(mag_r[valid]))
        d_density = float(abs(den_p - den_r) / max(den_p, den_r, 1.0))

        # 3. Tương đồng cấu trúc (SSIM)
        ssim_val = float(ssim(gp, gr, data_range=255))
        d_ssim = 1.0 - max(0.0, ssim_val)

        # Tổng hợp điểm bất thường cấu trúc bao quát
        s_macro = 0.4 * d_orient + 0.3 * d_density + 0.3 * d_ssim
        macro_scores.append(s_macro)

        tri_details.append({
            "id": idx,
            "pts_p": tp,
            "pts_r": tr,
            "s_macro": s_macro,
            "d_orient": d_orient,
            "d_density": d_density,
            "ssim": ssim_val,
            "hist_p": hist_p,
            "hist_r": hist_r,
            "gp": gp,
            "gr": gr,
            "patch_p": patch_p,
            "patch_r": patch_r_aligned,
        })

    m_mean = float(np.mean(macro_scores))
    m_std = float(np.std(macro_scores))

    suspects = []
    heatmap_vis = np.zeros((h, w), dtype=np.float32)
    overlay_suspect = r_img.copy()

    for item in tri_details:
        z = (item["s_macro"] - m_mean) / (m_std + 1e-8)
        item["z"] = z
        # Sàng lọc tam giác nghi ngờ: z > 2.0 hoặc điểm bất thường vượt 0.45
        is_suspect = bool(z > 2.0 or item["s_macro"] > 0.45)
        item["is_suspect"] = is_suspect
        if is_suspect:
            suspects.append(item)

        pts_r_int = item["pts_r"].astype(np.int32).reshape((-1, 1, 2))
        val_norm = np.clip(item["s_macro"] / 0.5 * 255.0, 0, 255)
        cv2.fillPoly(heatmap_vis, [pts_r_int], val_norm)

        if is_suspect:
            # Tô màu vàng cam nổi bật cho tam giác nghi ngờ
            cv2.fillPoly(overlay_suspect, [pts_r_int], (0, 165, 255))
            cv2.polylines(overlay_suspect, [pts_r_int], True, (0, 0, 255), 2)
            cx = int(np.mean(item["pts_r"][:, 0]))
            cy = int(np.mean(item["pts_r"][:, 1]))
            cv2.putText(overlay_suspect, str(item["id"]), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.polylines(overlay_suspect, [pts_r_int], True, (0, 255, 0), 1)

    heatmap_color = cv2.applyColorMap(heatmap_vis.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color[heatmap_vis == 0] = (0, 0, 0)
    blend_heatmap = cv2.addWeighted(r_img, 0.4, heatmap_color, 0.6, 0)

    # 1. Heatmap bất thường đặc trưng
    cv2.imwrite(os.path.join(out_dir, "01_macro_anomaly_heatmap.jpg"), blend_heatmap)
    # 2. Khung lưới tam giác nghi ngờ
    cv2.imwrite(os.path.join(out_dir, "02_suspect_triangles_overlay.jpg"), overlay_suspect)

    # 3. So sánh trực quan đặc trưng các tam giác nghi ngờ hàng đầu
    if suspects:
        top_suspects = sorted(suspects, key=lambda x: x["s_macro"], reverse=True)[:6]
        panels = []
        for s in top_suspects:
            p_box = cv2.resize(s["gp"], (120, 120))
            r_box = cv2.resize(s["gr"], (120, 120))
            diff_box = cv2.absdiff(p_box, r_box)

            # Vẽ biểu đồ 8-bin so sánh hướng nét (Xanh = Product, Đỏ = Return)
            bar_img = np.zeros((120, 180, 3), dtype=np.uint8)
            hp = (s["hist_p"] * 90).astype(int)
            hr = (s["hist_r"] * 90).astype(int)
            for b in range(8):
                bx = b * 22 + 4
                cv2.rectangle(bar_img, (bx, 60 - hp[b]), (bx + 8, 60), (0, 255, 0), -1)
                cv2.rectangle(bar_img, (bx + 10, 60 - hr[b]), (bx + 18, 60), (0, 0, 255), -1)
            cv2.putText(bar_img, f"Tri #{s['id']}", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(bar_img, f"d_ori:{s['d_orient']:.2f}", (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(bar_img, f"SSIM:{s['ssim']:.2f}", (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            p_box_bgr = cv2.cvtColor(p_box, cv2.COLOR_GRAY2BGR)
            r_box_bgr = cv2.cvtColor(r_box, cv2.COLOR_GRAY2BGR)
            diff_box_bgr = cv2.cvtColor(diff_box, cv2.COLOR_GRAY2BGR)
            cv2.putText(p_box_bgr, "Product", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(r_box_bgr, "Return", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(diff_box_bgr, "Diff", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            row = np.hstack([p_box_bgr, r_box_bgr, diff_box_bgr, bar_img])
            panels.append(row)

        comp_grid = np.vstack(panels)
        cv2.imwrite(os.path.join(out_dir, "03_macro_feature_comparison.jpg"), comp_grid)

    print(f"[{os.path.basename(product_path)} vs {os.path.basename(return_path)}]")
    print(f"  Total Triangles: {len(tri_details)}")
    print(f"  Suspect Triangles (Step 1): {len(suspects)}")
    print(f"  Mean Macro Score: {m_mean:.3f}, Std: {m_std:.3f}")
    return len(suspects)


if __name__ == "__main__":
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    out1 = os.path.join(V2_DIR, "debug", "01_undamaged_vase", "03_layer1_step1_macro")
    print("\n--- TEST CASE 1: UNDAMAGED VASE (1.jpg vs 2.jpg) ---")
    run_step1_macro(p1, p2, out1)

    s1 = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
    s2 = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")
    out2 = os.path.join(V2_DIR, "debug", "02_scar_defect", "03_layer1_step1_macro")
    print("\n--- TEST CASE 2: SCAR DEFECT (test_nobg vs scar) ---")
    run_step1_macro(s1, s2, out2)
