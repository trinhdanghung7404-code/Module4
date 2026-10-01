import os
import sys
import cv2
import numpy as np

# Thiết lập encoding UTF-8 cho stdout trên Windows
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


def analyze_edge_shape_and_position(edge_mask: np.ndarray, min_length: float = 10.0) -> list:
    """
    Trích xuất danh sách các nét trong tam giác kèm VỊ TRÍ và HÌNH DẠNG.
    
    Thông số của từng nét:
    - Vị trí: Trọng tâm (cx, cy) trong ô chuẩn hóa 72x72.
    - Chiều dài: Tổng độ dài đường cong (arcLength).
    - Độ thẳng/cong (Linearity): Tỉ lệ = Khoảng cách thẳng 2 đầu mút / Chiều dài đường cong.
      + Linearity >= 0.85: Nét THẲNG.
      + 0.50 <= Linearity < 0.85: Nét CONG NHẸ.
      + Linearity < 0.50: Nét CONG GẬP / VÒNG TRÒN / KHÉP KÍN.
    - Hướng (Orientation): Góc nghiêng theta [0..180 độ] từ điểm đầu tới điểm cuối.
    """
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    features = []

    for cnt in contours:
        length = float(cv2.arcLength(cnt, False))
        if length < min_length:
            continue

        # 1. Vị trí trọng tâm
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float(np.mean(cnt[:, 0, 0]))
            cy = float(np.mean(cnt[:, 0, 1]))

        # 2. Hai điểm đầu mút (Start và End)
        pt_start = cnt[0, 0].astype(np.float32)
        pt_end = cnt[-1, 0].astype(np.float32)
        endpoint_dist = float(np.linalg.norm(pt_end - pt_start))

        # 3. Độ thẳng vs độ cong (Linearity)
        linearity = float(endpoint_dist / max(length, 1.0))

        # 4. Hướng nét (Góc [0..180 độ])
        dx = pt_end[0] - pt_start[0]
        dy = pt_end[1] - pt_start[1]
        angle = float((np.arctan2(dy, dx) * 180.0 / np.pi) % 180.0)

        # Phân loại hình dạng
        if linearity >= 0.85:
            shape_type = "THẲNG"
        elif linearity >= 0.50:
            shape_type = "CONG"
        else:
            shape_type = "TRÒN/GẬP"

        features.append({
            "cx": cx,
            "cy": cy,
            "length": length,
            "linearity": linearity,
            "angle": angle,
            "shape_type": shape_type,
            "contour": cnt,
        })

    return features


def compare_triangles_shape_position(p_feats: list, r_feats: list,
                                     pos_tolerance: float = 12.0,
                                     angle_tolerance: float = 35.0,
                                     lin_tolerance: float = 0.35) -> tuple:
    """
    So sánh các nét giữa Product và Return dựa trên VỊ TRÍ và HÌNH DẠNG.
    Không phụ thuộc vào số lượng nét nhiều hay ít.
    
    Quy tắc:
    - Nếu cả 2 đều không có nét chính: GIỐNG NHAU (Cùng là nền trơn).
    - Với mỗi nét chính của Product: Tìm trong Return xem tại vị trí đó (khoảng cách <= pos_tolerance)
      có nét nào cùng hình dạng (cùng độ thẳng/cong và góc nghiêng) không.
    - Nếu Product có nét nhưng Return không có nét tương ứng tại vị trí đó -> KHÁC NHAU.
    - Nếu Return tự dưng xuất hiện nét nứt mới rất dài (length >= 20px) mà Product hoàn toàn không có -> KHÁC NHAU.
    """
    # 1. Cả 2 đều là nền trơn (không có nét)
    if len(p_feats) == 0 and len(r_feats) == 0:
        return True, "Cả hai đều là nền men trơn (Khớp 100%)"

    # 2. Một bên trơn hoàn toàn, một bên có nét rõ rệt
    if len(p_feats) == 0 and len(r_feats) > 0:
        # Kiểm tra xem nét bên Return có phải nét rõ rệt không (length >= 15px)
        max_r_len = max([f["length"] for f in r_feats])
        if max_r_len >= 15.0:
            return False, f"Ảnh gốc nền trơn, ảnh trả về xuất hiện nét lạ (dài {max_r_len:.1f}px)"
        else:
            return True, "Nét vụn nhỏ không đáng kể trên nền trơn"

    if len(p_feats) > 0 and len(r_feats) == 0:
        max_p_len = max([f["length"] for f in p_feats])
        if max_p_len >= 15.0:
            return False, f"Ảnh gốc có nét rõ rệt (dài {max_p_len:.1f}px), ảnh trả về bị mất nét"
        else:
            return True, "Nét vụn nhỏ ảnh gốc"

    # 3. Cả hai bên đều có nét: So khớp theo VỊ TRÍ và HÌNH DẠNG
    matched_p = []
    for i, pf in enumerate(p_feats):
        found_match = False
        for j, rf in enumerate(r_feats):
            # Kiểm tra khoảng cách vị trí tương đối
            dist_pos = np.sqrt((pf["cx"] - rf["cx"])**2 + (pf["cy"] - rf["cy"])**2)
            if dist_pos > pos_tolerance:
                continue

            # Kiểm tra hình dạng: Độ thẳng/cong
            diff_lin = abs(pf["linearity"] - rf["linearity"])
            if diff_lin > lin_tolerance:
                continue

            # Kiểm tra hướng nét (nếu nét thẳng)
            if pf["shape_type"] == "THẲNG" and rf["shape_type"] == "THẲNG":
                diff_ang = abs(pf["angle"] - rf["angle"])
                diff_ang = min(diff_ang, 180.0 - diff_ang)
                if diff_ang > angle_tolerance:
                    continue

            # Tìm thấy nét khớp về vị trí và hình dạng!
            found_match = True
            break

        if found_match:
            matched_p.append(i)

    # Tỉ lệ các nét Product tìm được nét khớp ở Return
    match_ratio = len(matched_p) / len(p_feats)
    if match_ratio >= 0.70:
        # Kiểm tra thêm: Return có nét lạ đâm ngang rất dài không?
        unmatched_r = []
        for j, rf in enumerate(r_feats):
            is_near_any_p = any(np.sqrt((rf["cx"] - pf["cx"])**2 + (rf["cy"] - pf["cy"])**2) <= pos_tolerance for pf in p_feats)
            if not is_near_any_p and rf["length"] >= 22.0:
                unmatched_r.append(rf)

        if len(unmatched_r) > 0:
            return False, f"Xuất hiện nét lạ xâm lấn mới tại ({unmatched_r[0]['cx']:.0f}, {unmatched_r[0]['cy']:.0f})"

        return True, f"Khớp vị trí & hình dạng ({len(matched_p)}/{len(p_feats)} nét khớp)"
    else:
        return False, f"Lệch hình dạng/vị trí (chỉ khớp {len(matched_p)}/{len(p_feats)} nét)"


def run_evaluation(product_path: str, return_path: str, out_dir: str, title: str):
    os.makedirs(out_dir, exist_ok=True)
    p_img = cv2.imread(product_path)
    r_img = cv2.imread(return_path)
    if p_img is None or r_img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh {product_path} hoặc {return_path}")
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

    matching_count = 0
    mismatch_count = 0
    results = []

    overlay = r_img.copy()

    for idx, tri in enumerate(triangles):
        tp = np.array([vertices[i].product_xy for i in tri.vertex_indices], dtype=np.float32)
        tr = np.array([vertices[i].return_xy for i in tri.vertex_indices], dtype=np.float32)

        patch_p, mask_p = canonical_triangle_patch(p_img, tp, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, tr, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=2)

        # Grayscale
        gp = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)

        # Canny edge
        ep = cv2.Canny(cv2.medianBlur(gp, 3), 50, 150)
        er = cv2.Canny(cv2.medianBlur(gr, 3), 50, 150)
        ep[mask_p == 0] = 0
        er[mask_p == 0] = 0

        # Trích xuất Vị trí & Hình dạng của các nét
        p_feats = analyze_edge_shape_and_position(ep, min_length=10.0)
        r_feats = analyze_edge_shape_and_position(er, min_length=10.0)

        # So sánh dựa trên vị trí và hình dạng
        is_match, reason = compare_triangles_shape_position(p_feats, r_feats)

        pts_r_int = tr.astype(np.int32).reshape((-1, 1, 2))

        if is_match:
            matching_count += 1
            # Tam giác GIỐNG NHAU: Vẽ viền xanh lá mảnh
            cv2.polylines(overlay, [pts_r_int], True, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            mismatch_count += 1
            # Tam giác KHÁC NHAU: Tô màu cam và viền đỏ
            cv2.fillPoly(overlay, [pts_r_int], (0, 165, 255))
            cv2.polylines(overlay, [pts_r_int], True, (0, 0, 255), 2, cv2.LINE_AA)

        results.append({
            "id": idx,
            "is_match": is_match,
            "reason": reason,
            "p_edges": len(p_feats),
            "r_edges": len(r_feats),
        })

    total = len(triangles)
    match_ratio = (matching_count / max(1, total)) * 100.0

    # Lưu ảnh overlay
    cv2.putText(overlay, f"Tỉ lệ giống nhau: {matching_count}/{total} ({match_ratio:.1f}%)",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0) if match_ratio >= 90.0 else (0, 140, 255), 3)
    cv2.imwrite(os.path.join(out_dir, "01_matching_triangles_overlay.jpg"), overlay)

    # Ghi báo cáo chi tiết
    report_path = os.path.join(out_dir, "02_shape_position_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== BÁO CÁO KIỂM THỬ: {title} ===\n")
        f.write(f"Tổng số tam giác kiểm tra: {total}\n")
        f.write(f"Số tam giác GIỐNG NHAU (Khớp hình dạng & vị trí): {matching_count} ({match_ratio:.2f}%)\n")
        f.write(f"Số tam giác KHÁC NHAU (Nghi ngờ biến dạng): {mismatch_count} ({100.0 - match_ratio:.2f}%)\n\n")
        f.write("--- Danh sách các tam giác KHÁC NHAU ---\n")
        for r in results:
            if not r["is_match"]:
                f.write(f"Tam giác #{r['id']:03d}: {r['reason']} (P có {r['p_edges']} nét, R có {r['r_edges']} nét)\n")

    print(f"\n=======================================================")
    print(f"  KẾT QUẢ KIỂM THỬ: {title}")
    print(f"  Tổng số tam giác:            {total}")
    print(f"  Tam giác GIỐNG NHAU:         {matching_count} ({match_ratio:.1f}%)")
    print(f"  Tam giác KHÁC NHAU:          {mismatch_count} ({100.0 - match_ratio:.1f}%)")
    print(f"  Ảnh lưu tại: {os.path.join(out_dir, '01_matching_triangles_overlay.jpg')}")
    print(f"=======================================================")

    return match_ratio


if __name__ == "__main__":
    p1 = os.path.join(PROJECT_ROOT, "images", "1.jpg")
    p2 = os.path.join(PROJECT_ROOT, "images", "2.jpg")
    out1 = os.path.join(V2_DIR, "debug", "test_shape_position_undamaged")
    run_evaluation(p1, p2, out1, "BÌNH NGUYÊN VẸN (1.jpg vs 2.jpg)")

    s1 = os.path.join(PROJECT_ROOT, "images", "test_nobg.png")
    s2 = os.path.join(PROJECT_ROOT, "images", "test_nobg - scar.png")
    out2 = os.path.join(V2_DIR, "debug", "test_shape_position_scar")
    run_evaluation(s1, s2, out2, "BÌNH CÓ SẸO NỨT THẬT (test_nobg vs scar)")
