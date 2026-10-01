import os
import sys
import cv2
import numpy as np

# Thiết lập UTF-8 cho Windows stdout
sys.stdout.reconfigure(encoding="utf-8")

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


def run_automatic_matching(product_path: str, return_path: str, out_dir: str, title: str):
    """
    Thuật toán TỰ ĐỘNG 100% duyệt qua toàn bộ tam giác, tự động phân loại:
    1. TRƠN (BỎ QUA): Cả 2 ảnh đều không có nét hoa văn.
    2. GIỐNG NHAU: Hoa văn giữ nguyên hình dạng và vị trí.
    3. KHÁC NHAU: Nét gốc bị cắt đứt hoặc xuất hiện nét nứt/sẹo mới đè lên.
    
    Sau đó tự động xuất toàn bộ ảnh bằng chứng trực quan mà không chọn lọc thủ công.
    """
    os.makedirs(out_dir, exist_ok=True)
    p_img = cv2.imread(product_path)
    r_img = cv2.imread(return_path)
    if p_img is None or r_img is None:
        raise FileNotFoundError(f"Cannot read {product_path} or {return_path}")
    h, w = p_img.shape[:2]

    # 1. Tiền xử lý
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img)
    r_seg = seg.segment(r_img)

    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg["mask"], p_seg["mask"])

    # 2. Đăng ký và dựng lưới tam giác Delaunay
    reg = ImageRegistration(max_keypoints=None)
    res = reg.register(p_img, r_norm, p_seg["mask"], r_seg["mask"])
    builder = MeshBuilder()
    vertices, triangles = builder.build(res["product_points"], res["return_points"], np.arange(len(res["product_points"])))

    total_triangles = len(triangles)
    blank_skipped = []
    matching_triangles = []
    different_triangles = []

    overlay_mesh = r_img.copy()

    for idx, tri in enumerate(triangles):
        tp = np.array([vertices[i].product_xy for i in tri.vertex_indices], dtype=np.float32)
        tr = np.array([vertices[i].return_xy for i in tri.vertex_indices], dtype=np.float32)

        # Cắt patch tam giác chuẩn hóa 72x72 và vi căn chỉnh
        patch_p, mask_p = canonical_triangle_patch(p_img, tp, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, tr, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        gp = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)

        # Canny edge
        ep = cv2.Canny(cv2.medianBlur(gp, 3), 50, 150)
        er = cv2.Canny(cv2.medianBlur(gr, 3), 50, 150)
        ep[mask_p == 0] = 0
        er[mask_p == 0] = 0

        p_edge_cnt = int(np.count_nonzero(ep))
        r_edge_cnt = int(np.count_nonzero(er))
        pts_r_int = tr.astype(np.int32).reshape((-1, 1, 2))

        # --- QUY TẮC 1: NỀN TRƠN Ở CẢ 2 BÊN -> BỎ QUA ---
        if p_edge_cnt < 15 and r_edge_cnt < 15:
            blank_skipped.append(idx)
            cv2.polylines(overlay_mesh, [pts_r_int], True, (100, 100, 100), 1, cv2.LINE_AA)
            continue

        # --- QUY TẮC 2: SO SÁNH ĐẶC TRƯNG CẤU TRÚC NÉT HOA VĂN ---
        # A. Kiểm tra nét gốc Product có bị cắt đứt không (Cut / Severed Edge):
        er_dilated = cv2.dilate(er, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        missing_edges = (ep > 0) & (er_dilated == 0)
        cnts_cut, _ = cv2.findContours(missing_edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_cut_len = max([cv2.arcLength(c, False) for c in cnts_cut], default=0.0)
        is_severed = bool((max_cut_len >= 16.0) and (np.count_nonzero(missing_edges) / max(1, p_edge_cnt) > 0.35))

        # B. Kiểm tra có nét nứt lạ / vết sẹo mới đè lên không (Intrusive Crack / Scar):
        ep_dilated = cv2.dilate(ep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        new_edges = (er > 0) & (ep_dilated == 0)
        cnts_new, _ = cv2.findContours(new_edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_new_len = max([cv2.arcLength(c, False) for c in cnts_new], default=0.0)
        is_intrusive = bool((max_new_len >= 16.0) and (np.count_nonzero(new_edges) / max(1, r_edge_cnt) > 0.35))

        # Quyết định tự động của thuật toán:
        is_different = is_severed or is_intrusive

        item_data = {
            "id": idx,
            "tp": tp,
            "tr": tr,
            "patch_p": patch_p,
            "patch_r": patch_r_aligned,
            "ep": ep,
            "er": er,
            "p_edge_cnt": p_edge_cnt,
            "r_edge_cnt": r_edge_cnt,
            "max_cut_len": max_cut_len,
            "max_new_len": max_new_len,
            "is_severed": is_severed,
            "is_intrusive": is_intrusive,
        }

        if is_different:
            different_triangles.append(item_data)
            # Vẽ tam giác KHÁC NHAU: Tô màu cam và viền đỏ
            cv2.fillPoly(overlay_mesh, [pts_r_int], (0, 165, 255))
            cv2.polylines(overlay_mesh, [pts_r_int], True, (0, 0, 255), 2, cv2.LINE_AA)
            cx = int(np.mean(tr[:, 0]))
            cy = int(np.mean(tr[:, 1]))
            cv2.putText(overlay_mesh, str(idx), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            matching_triangles.append(item_data)
            # Vẽ tam giác GIỐNG NHAU: Viền xanh lá
            cv2.polylines(overlay_mesh, [pts_r_int], True, (0, 255, 0), 1, cv2.LINE_AA)

    pattern_count = len(matching_triangles) + len(different_triangles)
    match_ratio = (len(matching_triangles) / max(1, pattern_count)) * 100.0

    # 3. TỰ ĐỘNG XUẤT ẢNH SHOWCASE TỪ KẾT QUẢ THUẬT TOÁN (KHÔNG CHỌN TAY)
    def export_showcase_grid(item_list, filename, showcase_title, is_diff=False, max_items=8):
        if not item_list:
            return
        # Lấy tối đa max_items tam giác đầu tiên mà thuật toán tự phân loại
        selected = item_list[:max_items]
        rows = []
        for item in selected:
            p_box = cv2.resize(item["patch_p"], (100, 100))
            r_box = cv2.resize(item["patch_r"], (100, 100))
            ep_box = cv2.resize(item["ep"], (100, 100))
            er_box = cv2.resize(item["er"], (100, 100))

            ep_bgr = cv2.cvtColor(ep_box, cv2.COLOR_GRAY2BGR)
            er_bgr = cv2.cvtColor(er_box, cv2.COLOR_GRAY2BGR)

            diff_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
            # Nét cũ bị mất = Xanh lam, Nét mới đè lên = Đỏ, Nét khớp = Xanh lá
            diff_bgr[(ep_box > 0) & (er_box == 0)] = (255, 0, 0)
            diff_bgr[(er_box > 0) & (ep_box == 0)] = (0, 0, 255)
            diff_bgr[(er_box > 0) & (ep_box > 0)] = (0, 255, 0)

            cv2.putText(p_box, f"Tri #{item['id']}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(p_box, "Product", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(r_box, "Return", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(ep_bgr, f"Edge P", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(er_bgr, f"Edge R", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(diff_bgr, "Sai Khac", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            row = np.hstack([p_box, r_box, ep_bgr, er_bgr, diff_bgr])
            rows.append(row)

        grid = np.vstack(rows)
        header = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, f"{showcase_title} (Tu dong phan loai: {len(item_list)} tam giac)",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if not is_diff else (0, 0, 255), 2)
        full_showcase = np.vstack([header, grid])
        cv2.imwrite(os.path.join(out_dir, filename), full_showcase)

    # Xuất ảnh các tam giác thuật toán tự phân loại là GIỐNG NHAU
    export_showcase_grid(matching_triangles, "01_auto_matching_triangles.jpg", "THUAT TOAN TU PHAN LOAI: GIU NGUYEN HOA VAN (GIONG NHAU)", is_diff=False)
    # Xuất ảnh các tam giác thuật toán tự phân loại là KHÁC NHAU
    export_showcase_grid(different_triangles, "02_auto_different_triangles.jpg", "THUAT TOAN TU PHAN LOAI: BI VET SEO / CAT NET (KHAC NHAU)", is_diff=True)

    # 4. Xuất ảnh toàn bộ bình (Overlay Mesh)
    cv2.putText(overlay_mesh, f"{title}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(overlay_mesh, f"Tong so: {total_triangles} | Tron bo qua: {len(blank_skipped)}",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    cv2.putText(overlay_mesh, f"Giong nhau: {len(matching_triangles)}/{pattern_count} ({match_ratio:.1f}%) | Khac nhau: {len(different_triangles)}",
                (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if match_ratio >= 90.0 else (0, 140, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "03_mesh_full_overlay.jpg"), overlay_mesh)

    print(f"\n==================================================================")
    print(f"  KẾT QUẢ TỰ ĐỘNG THUẬT TOÁN: {title}")
    print(f"  Tổng số tam giác:                  {total_triangles}")
    print(f"  Tam giác TRƠN (BỎ QUA):            {len(blank_skipped)}")
    print(f"  Tam giác CÓ HOA VĂN:               {pattern_count}")
    print(f"  Hoa văn GIỐNG NHAU:                {len(matching_triangles)} ({match_ratio:.1f}%)")
    print(f"  Hoa văn KHÁC NHAU:                 {len(different_triangles)} ({100.0 - match_ratio:.1f}%)")
    print(f"  Ảnh lưu tại: {out_dir}")
    print(f"==================================================================")


if __name__ == "__main__":
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    out1 = os.path.join(V2_DIR, "debug", "auto_matcher_undamaged")
    print("\n--- CHẠY TỰ ĐỘNG TEST 1: BÌNH NGUYÊN VẸN (1.jpg vs 2.jpg) ---")
    run_automatic_matching(p1, p2, out1, "BINH NGUYEN VEN (1.jpg vs 2.jpg)")

    s1 = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
    s2 = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")
    out2 = os.path.join(V2_DIR, "debug", "auto_matcher_scar")
    print("\n--- CHẠY TỰ ĐỘNG TEST 2: BÌNH CÓ SẸO NỨT THẬT (test_nobg vs scar) ---")
    run_automatic_matching(s1, s2, out2, "BINH CO SEO NUT THAT (test_nobg vs scar)")
