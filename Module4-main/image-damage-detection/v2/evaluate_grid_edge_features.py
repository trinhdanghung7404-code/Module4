import os
import sys
import cv2
import numpy as np

# Thiết lập encoding UTF-8 cho Windows stdout
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


def extract_cell_grid_features(edge_img: np.ndarray, gray_img: np.ndarray, mask: np.ndarray, grid_n: int = 4) -> list:
    """
    Chia tam giác chuẩn hóa (72x72) thành lưới grid_n x grid_n ô vị trí nhỏ.
    Tại mỗi ô vị trí:
    - Có nét hay không (has_edge).
    - Hướng nghiêng của nét (angle_deg [0..180]).
    - Mật độ nét trong ô (edge_count).
    """
    h, w = edge_img.shape[:2]
    cell_h = h // grid_n
    cell_w = w // grid_n

    # Tính hướng gradient bằng Sobel
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

    cells = []
    for r in range(grid_n):
        for c in range(grid_n):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w

            cell_mask = mask[y0:y1, x0:x1]
            valid_px = np.count_nonzero(cell_mask)

            # Bỏ qua các ô nằm ngoài phạm vi tam giác (diện tích quá nhỏ)
            if valid_px < (cell_h * cell_w * 0.25):
                continue

            cell_edge = edge_img[y0:y1, x0:x1]
            cell_mag = mag[y0:y1, x0:x1]
            cell_ang = ang[y0:y1, x0:x1]

            edge_pixels = (cell_edge > 0) & (cell_mask > 0)
            edge_count = int(np.count_nonzero(edge_pixels))
            has_edge = bool(edge_count >= 3)

            if has_edge:
                weights = cell_mag[edge_pixels]
                angles = cell_ang[edge_pixels]
                if np.sum(weights) > 0:
                    mean_angle = float(np.average(angles, weights=weights))
                else:
                    mean_angle = float(np.mean(angles))
            else:
                mean_angle = -1.0  # Không có nét

            cells.append({
                "row": r,
                "col": c,
                "x_center": (x0 + x1) / 2.0,
                "y_center": (y0 + y1) / 2.0,
                "has_edge": has_edge,
                "edge_count": edge_count,
                "angle": mean_angle,
            })

    return cells


def compare_grid_features(cells_p: list, cells_r: list, angle_tolerance: float = 35.0) -> tuple:
    """
    So sánh đặc trưng edge giữa Product và Return theo từng ô vị trí.
    
    Quy tắc:
    - Cả 2 ô cùng không có nét -> KHỚP.
    - Cả 2 ô cùng có nét và góc lệch <= 35 độ -> KHỚP.
    - Một bên có nét, kiểm tra ô lân cận (bán kính 1 ô) có nét cùng hướng không (tránh lệch biên 1-2px).
    - Trả về: (is_match, match_ratio, cell_results)
    """
    if not cells_p or not cells_r:
        return True, 1.0, []

    matched_cells = 0
    total_cells = len(cells_p)
    cell_results = []

    # Map tọa độ (row, col) cho Return để tra cứu nhanh
    r_map = {(c["row"], c["col"]): c for c in cells_r}

    for cp in cells_p:
        r, c = cp["row"], cp["col"]
        cr = r_map.get((r, c), None)

        if cr is None:
            continue

        cell_matched = False
        reason = ""

        # Trường hợp 1: Cả hai ô đều là nền trơn (không có nét)
        if not cp["has_edge"] and not cr["has_edge"]:
            cell_matched = True
            reason = "Cả 2 đều trơn"

        # Trường hợp 2: Cả hai ô đều có nét
        elif cp["has_edge"] and cr["has_edge"]:
            diff_ang = abs(cp["angle"] - cr["angle"])
            diff_ang = min(diff_ang, 180.0 - diff_ang)
            if diff_ang <= angle_tolerance:
                cell_matched = True
                reason = f"Cùng hướng (lệch {diff_ang:.1f}°)"
            else:
                cell_matched = False
                reason = f"Lệch hướng ({cp['angle']:.0f}° vs {cr['angle']:.0f}°)"

        # Trường hợp 3: Một bên có nét, một bên không -> Kiểm tra ô lân cận (dung sai biên)
        else:
            # Tìm trong các ô lân cận (r±1, c±1) của Return
            target_angle = cp["angle"] if cp["has_edge"] else cr["angle"]
            found_neighbor = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    neighbor = r_map.get((nr, nc), None)
                    if neighbor and neighbor["has_edge"]:
                        diff_ang = abs(target_angle - neighbor["angle"])
                        diff_ang = min(diff_ang, 180.0 - diff_ang)
                        if diff_ang <= angle_tolerance:
                            found_neighbor = True
                            break
                if found_neighbor:
                    break

            if found_neighbor:
                cell_matched = True
                reason = "Khớp với ô lân cận"
            else:
                cell_matched = False
                reason = "Một bên có nét, bên kia không"

        if cell_matched:
            matched_cells += 1

        cell_results.append({
            "row": r,
            "col": c,
            "matched": cell_matched,
            "reason": reason,
            "p_angle": cp["angle"],
            "r_angle": cr["angle"],
        })

    match_ratio = matched_cells / max(1, total_cells)
    # Tam giác được coi là GIỐNG NHAU nếu ít nhất 70% các ô vị trí có đặc trưng tương đồng
    is_triangle_match = bool(match_ratio >= 0.70)

    return is_triangle_match, match_ratio, cell_results


def run_grid_evaluation(product_path: str, return_path: str, out_dir: str, title: str):
    os.makedirs(out_dir, exist_ok=True)
    p_img = cv2.imread(product_path)
    r_img = cv2.imread(return_path)
    if p_img is None or r_img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {product_path} hoặc {return_path}")
    h, w = p_img.shape[:2]

    # Phân đoạn & Chuẩn hóa
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img)
    r_seg = seg.segment(r_img)

    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg["mask"], p_seg["mask"])

    # Đăng ký & Dựng lưới tam giác
    reg = ImageRegistration(max_keypoints=None)
    res = reg.register(p_img, r_norm, p_seg["mask"], r_seg["mask"])
    p_pts = res["product_points"]
    r_pts = res["return_points"]

    builder = MeshBuilder()
    vertices, triangles = builder.build(p_pts, r_pts, np.arange(len(p_pts)))

    total_triangles = len(triangles)
    skipped_blank_count = 0
    pattern_triangles_count = 0
    pattern_matching_count = 0
    pattern_mismatch_count = 0

    triangle_results = []
    overlay = r_img.copy()

    for idx, tri in enumerate(triangles):
        tp = np.array([vertices[i].product_xy for i in tri.vertex_indices], dtype=np.float32)
        tr = np.array([vertices[i].return_xy for i in tri.vertex_indices], dtype=np.float32)

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

        p_edge_total = np.count_nonzero(ep)
        r_edge_total = np.count_nonzero(er)

        pts_r_int = tr.astype(np.int32).reshape((-1, 1, 2))

        # ĐIỀU KIỆN THEO Ý BẠN: NẾU TRONG TAM GIÁC ĐÓ BỊ TRƠN Ở CẢ 2 HÌNH -> BỎ QUA
        is_blank_both = (p_edge_total < 10) and (r_edge_total < 10)

        if is_blank_both:
            skipped_blank_count += 1
            # Tam giác trơn: Vẽ viền xám mờ để phân biệt, không tính vào đánh giá hoa văn
            cv2.polylines(overlay, [pts_r_int], True, (120, 120, 120), 1, cv2.LINE_AA)
            triangle_results.append({
                "id": idx,
                "type": "TRƠN (BỎ QUA)",
                "is_match": None,
                "ratio": 1.0,
                "reason": "Nền men trơn ở cả 2 ảnh"
            })
            continue

        # CÓ HOA VĂN -> BẮT ĐẦU ĐÁNH GIÁ ĐẶC TRƯNG THEO TỪNG VỊ TRÍ
        pattern_triangles_count += 1

        cells_p = extract_cell_grid_features(ep, gp, mask_p, grid_n=4)
        cells_r = extract_cell_grid_features(er, gr, mask_p, grid_n=4)

        is_match, ratio, cell_results = compare_grid_features(cells_p, cells_r, angle_tolerance=35.0)

        if is_match:
            pattern_matching_count += 1
            # Tam giác có hoa văn GIỐNG NHAU: Vẽ viền xanh lá đậm
            cv2.polylines(overlay, [pts_r_int], True, (0, 255, 0), 2, cv2.LINE_AA)
            triangle_results.append({
                "id": idx,
                "type": "HOA VĂN - GIỐNG NHAU",
                "is_match": True,
                "ratio": ratio,
                "reason": f"Khớp {ratio*100:.1f}% các ô vị trí"
            })
        else:
            pattern_mismatch_count += 1
            # Tam giác có hoa văn KHÁC NHAU: Tô màu cam và viền đỏ
            cv2.fillPoly(overlay, [pts_r_int], (0, 165, 255))
            cv2.polylines(overlay, [pts_r_int], True, (0, 0, 255), 2, cv2.LINE_AA)
            cx = int(np.mean(tr[:, 0]))
            cy = int(np.mean(tr[:, 1]))
            cv2.putText(overlay, str(idx), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            triangle_results.append({
                "id": idx,
                "type": "HOA VĂN - KHÁC NHAU",
                "is_match": False,
                "ratio": ratio,
                "reason": f"Lệch đặc trưng vị trí (chỉ khớp {ratio*100:.1f}% các ô)"
            })

    # Tính tỉ lệ giống nhau trên các tam giác CÓ HOA VĂN
    final_ratio = (pattern_matching_count / max(1, pattern_triangles_count)) * 100.0

    # Vẽ thông tin lên ảnh overlay
    cv2.putText(overlay, f"{title}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(overlay, f"Tong so: {total_triangles} | Tron bo qua: {skipped_blank_count}",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    cv2.putText(overlay, f"Hoa van giong nhau: {pattern_matching_count}/{pattern_triangles_count} ({final_ratio:.1f}%)",
                (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if final_ratio >= 85.0 else (0, 140, 255), 2)

    overlay_path = os.path.join(out_dir, "01_triangles_classification_overlay.jpg")
    cv2.imwrite(overlay_path, overlay)

    # Ghi báo cáo chi tiết
    report_path = os.path.join(out_dir, "02_grid_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== BÁO CÁO KIỂM THỬ: {title} ===\n\n")
        f.write(f"Tổng số tam giác lưới Mesh:          {total_triangles}\n")
        f.write(f"Tam giác NỀN TRƠN (Được BỎ QUA):     {skipped_blank_count} ({(skipped_blank_count/total_triangles)*100:.1f}%)\n")
        f.write(f"Tam giác THỰC SỰ CÓ HOA VĂN:         {pattern_triangles_count}\n")
        f.write(f"  -> Hoa văn GIỐNG NHAU (Khớp ô):    {pattern_matching_count} ({final_ratio:.2f}%)\n")
        f.write(f"  -> Hoa văn KHÁC NHAU (Nghi ngờ):   {pattern_mismatch_count} ({100.0 - final_ratio:.2f}%)\n\n")
        f.write("--- DANH SÁCH CHI TIẾT CÁC TAM GIÁC KHÁC NHAU ---\n")
        for r in triangle_results:
            if r["is_match"] is False:
                f.write(f"Tam giác #{r['id']:03d}: {r['reason']}\n")

    print(f"\n==================================================================")
    print(f"  KẾT QUẢ KIỂM THỬ: {title}")
    print(f"  Tổng số tam giác:                  {total_triangles}")
    print(f"  Tam giác TRƠN (BỎ QUA):            {skipped_blank_count}")
    print(f"  Tam giác CÓ HOA VĂN:               {pattern_triangles_count}")
    print(f"  Hoa văn GIỐNG NHAU:                {pattern_matching_count} ({final_ratio:.1f}%)")
    print(f"  Hoa văn KHÁC NHAU:                 {pattern_mismatch_count} ({100.0 - final_ratio:.1f}%)")
    print(f"  Ảnh Overlay lưu tại: {overlay_path}")
    print(f"==================================================================")

    return final_ratio


if __name__ == "__main__":
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    out1 = os.path.join(V2_DIR, "debug", "test_grid_undamaged")
    print("\n--- CHẠY TEST 1: BÌNH NGUYÊN VẸN (1.jpg vs 2.jpg) ---")
    run_grid_evaluation(p1, p2, out1, "BINH NGUYEN VEN (1.jpg vs 2.jpg)")

    s1 = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
    s2 = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")
    out2 = os.path.join(V2_DIR, "debug", "test_grid_scar")
    print("\n--- CHẠY TEST 2: BÌNH CÓ SẸO THẬT (test_nobg vs scar) ---")
    run_grid_evaluation(s1, s2, out2, "BINH CO SEO NỨT THẬT (test_nobg vs scar)")
