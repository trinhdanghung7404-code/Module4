"""
LAYER 1 - BƯỚC 1: SÀNG LỌC ĐẶC TRƯNG HÌNH HỌC VĨ MÔ (MACRO STRUCTURAL PROFILE)
CHỈ XỬ LÝ TRÊN BẢN ĐỒ NÉT / ĐEN TRẮNG (GRAYSCALE / BINARY EDGE MAP)
HOÀN TOÀN KHÔNG DÙNG ẢNH MÀU RGB ĐỂ TRÁNH PHỨC TẠP VÀ NHIỄU MÀU.

Nhiệm vụ:
  1. Trong từng tam giác mesh, chỉ nhận đầu vào là bản đồ nét nhị phân của Product và Return.
  2. Học đặc trưng hình học của nét đen-trắng:
     - Hướng nét chủ đạo (qua ma trận mô-men quán tính bậc 2 - Moments).
     - Hình dạng nét: Nét thẳng/mở (open stroke) vs Nét tròn/vòng khép kín (closed loop/blob).
  3. So sánh 2 hồ sơ nét đen-trắng để tìm các tam giác nghi vấn:
     - Product toàn nét thẳng mà Return lại mọc thêm nét tròn / vòng khép kín.
     - Hoặc góc nghiêng hướng nét trong tam giác bị xoay lệch / biến dạng.
  4. Xuất toàn bộ file debug và crop ĐEN TRẮNG trực quan, rõ ràng.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any


def extract_binary_edge_profile(edge_patch: np.ndarray, triangle_mask: np.ndarray) -> Dict[str, Any]:
    """
    Trích xuất đặc trưng hình học thuần túy từ bản đồ nét đen trắng trong tam giác.
    edge_patch: ảnh nhị phân (0 là nền đen, 255 là nét trắng)
    triangle_mask: mặt nạ tam giác (255 là trong tam giác, 0 là ngoài)
    """
    # Chỉ giữ các pixel nét nằm trong tam giác
    edges = np.zeros_like(edge_patch)
    edges[(edge_patch > 127) & (triangle_mask > 0)] = 255

    valid_px = max(1, int(np.count_nonzero(triangle_mask)))
    edge_px = int(np.count_nonzero(edges))
    density = float(edge_px) / float(valid_px)

    # 1. Tính hướng nét chủ đạo qua Moments quán tính bậc 2
    dominant_angle = -1.0
    if edge_px >= 10:
        M = cv2.moments(edges)
        mu20 = M["mu20"]
        mu02 = M["mu02"]
        mu11 = M["mu11"]
        # Góc nét [0, pi)
        dominant_angle = float(0.5 * np.arctan2(2.0 * mu11, mu20 - mu02) % np.pi)

    # 2. Phân loại hình thái nét: Thẳng/mở vs Tròn/khép kín
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    open_strokes = 0
    closed_loops = 0
    max_length = 0.0
    loop_details = []

    for cnt in contours:
        length = cv2.arcLength(cnt, False)
        if length < 8.0:
            continue
        max_length = max(max_length, length)

        area = cv2.contourArea(cnt)
        p_closed = cv2.arcLength(cnt, True)
        circ = 4.0 * np.pi * area / (p_closed**2 + 1e-6)

        # Nét tạo thành vòng tròn hoặc khép kín bao quanh một diện tích
        if circ >= 0.25 and area >= 12.0:
            closed_loops += 1
            loop_details.append({"circ": float(circ), "area": float(area), "len": float(length)})
        else:
            open_strokes += 1

    return {
        "edge_mask": edges,
        "density": density,
        "edge_px": edge_px,
        "valid_px": valid_px,
        "dominant_angle": dominant_angle,
        "open_strokes": open_strokes,
        "closed_loops": closed_loops,
        "max_length": float(max_length),
        "loop_details": loop_details,
    }


def compare_binary_profiles(p_prof: Dict[str, Any], r_prof: Dict[str, Any]) -> Dict[str, Any]:
    """
    So sánh hồ sơ nét đen-trắng giữa Product và Return.
    """
    p_density = p_prof["density"]
    r_density = r_prof["density"]

    # Men trắng không có nét ở cả 2 ảnh -> Hoàn toàn bình thường
    if p_density < 0.015 and r_density < 0.015:
        return {
            "ang_diff_deg": 0.0,
            "new_loops": 0,
            "density_diff": 0.0,
            "macro_score": 0.0,
            "is_suspicious": False,
            "reason": "Khong_Co_Net",
        }

    # 1. Độ lệch hướng nét chính (độ)
    ang_diff_deg = 0.0
    ang_p = p_prof["dominant_angle"]
    ang_r = r_prof["dominant_angle"]
    if ang_p >= 0 and ang_r >= 0:
        diff_rad = abs(ang_p - ang_r)
        if diff_rad > np.pi / 2.0:
            diff_rad = np.pi - diff_rad
        ang_diff_deg = float(np.degrees(diff_rad))
    elif (ang_p >= 0) != (ang_r >= 0):
        ang_diff_deg = 45.0  # Một bên có hướng rõ, một bên mất hẳn hướng

    # 2. Xuất hiện vòng tròn / khép kín lạ trên Return
    new_loops = max(0, r_prof["closed_loops"] - p_prof["closed_loops"])

    # 3. Chênh lệch mật độ nét
    density_diff = abs(p_density - r_density)

    # 4. Điểm dị thường hình học
    # Điểm góc: lệch >= 40 độ được coi là bất thường
    angle_score = min(1.0, ang_diff_deg / 45.0)
    loop_score = min(1.0, float(new_loops) * 0.5)
    density_score = min(1.0, density_diff * 6.0)

    macro_score = angle_score * 0.35 + loop_score * 0.45 + density_score * 0.20

    reasons = []
    if new_loops > 0:
        reasons.append(f"Moc_Net_Tron_La({new_loops})")
    if ang_diff_deg >= 35.0 and p_prof["max_length"] >= 15.0 and r_prof["max_length"] >= 15.0:
        reasons.append(f"Lech_Huong_Net({ang_diff_deg:.0f}deg)")
    if p_density >= 0.05 and r_density <= 0.01:
        reasons.append("Mat_Han_Net_Goc")

    is_suspicious = len(reasons) > 0 or macro_score >= 0.55
    reason_str = "+".join(reasons) if reasons else ("Diem_Di_Thuong_Cao" if is_suspicious else "Binh_Thuong")

    return {
        "ang_diff_deg": ang_diff_deg,
        "new_loops": new_loops,
        "density_diff": density_diff,
        "macro_score": float(macro_score),
        "is_suspicious": is_suspicious,
        "reason": reason_str,
    }


def visualize_step1_grayscale(
    p_edge_full: np.ndarray,
    r_edge_full: np.ndarray,
    triangles: list,
    vertices: list,
    profiles_p: list,
    profiles_r: list,
    comparisons: list,
    out_dir: str,
):
    """
    Xuất bộ ảnh debug Bước 1 THUẦN TÚY ĐEN TRẮNG / KHÔNG MÀU:
      01_edge_orientations_product.jpg: Nền đen, nét trắng Product + vạch hướng nét
      02_edge_orientations_return.jpg: Nền đen, nét trắng Return + vạch hướng nét
      03_shape_features_loops.jpg: Bản đồ nét đen trắng đánh dấu nét tròn lạ
      04_macro_anomaly_mesh.jpg: Lưới mesh đen trắng thể hiện mức độ dị thường
      05_suspicious_mesh_candidates.jpg: Bản đồ nét Return khoanh vùng tam giác nghi vấn
      suspicious_crops/: Folder crop đen trắng từng tam giác nghi vấn
    """
    os.makedirs(out_dir, exist_ok=True)
    dir_crops = os.path.join(out_dir, "suspicious_crops")
    os.makedirs(dir_crops, exist_ok=True)

    h, w = p_edge_full.shape[:2]

    # --- 1. PRODUCT: BẢN ĐỒ NÉT + HƯỚNG NÉT TRONG TỪNG TAM GIÁC ---
    # Nền đen (0), nét vẽ xám sáng (200)
    vis_orient_p = np.zeros((h, w, 3), dtype=np.uint8)
    vis_orient_p[p_edge_full > 127] = (200, 200, 200)

    for i, tri in enumerate(triangles):
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.int32)
        # Lưới tam giác màu xám tối (70, 70, 70) để không lấn át nét
        cv2.polylines(vis_orient_p, [pts_p], True, (60, 60, 60), 1, cv2.LINE_AA)

        ang_p = profiles_p[i]["dominant_angle"]
        if ang_p >= 0:
            cp = np.mean(pts_p, axis=0).astype(int)
            dx = int(16 * np.cos(ang_p))
            dy = int(16 * np.sin(ang_p))
            # Vạch chỉ hướng màu trắng tinh (255, 255, 255)
            cv2.line(vis_orient_p, (cp[0] - dx, cp[1] - dy), (cp[0] + dx, cp[1] + dy), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis_orient_p, (cp[0], cp[1]), 2, (0, 255, 255), -1)

    cv2.imwrite(os.path.join(out_dir, "01_edge_orientations_product.jpg"), vis_orient_p, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # --- 2. RETURN: BẢN ĐỒ NÉT + HƯỚNG NÉT TRONG TỪNG TAM GIÁC ---
    vis_orient_r = np.zeros((h, w, 3), dtype=np.uint8)
    vis_orient_r[r_edge_full > 127] = (200, 200, 200)

    for i, tri in enumerate(triangles):
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
        cv2.polylines(vis_orient_r, [pts_r], True, (60, 60, 60), 1, cv2.LINE_AA)

        ang_r = profiles_r[i]["dominant_angle"]
        if ang_r >= 0:
            cr = np.mean(pts_r, axis=0).astype(int)
            dx = int(16 * np.cos(ang_r))
            dy = int(16 * np.sin(ang_r))
            cv2.line(vis_orient_r, (cr[0] - dx, cr[1] - dy), (cr[0] + dx, cr[1] + dy), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis_orient_r, (cr[0], cr[1]), 2, (0, 255, 255), -1)

    cv2.imwrite(os.path.join(out_dir, "02_edge_orientations_return.jpg"), vis_orient_r, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # --- 3. ĐẶC TRƯNG HÌNH THÁI: PHÁT HIỆN NÉT TRÒN / KHÉP KÍN LẠ TRÊN RETURN ---
    vis_loops = np.zeros((h, w, 3), dtype=np.uint8)
    vis_loops[r_edge_full > 127] = (160, 160, 160)
    for i, tri in enumerate(triangles):
        if comparisons[i]["new_loops"] > 0:
            pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
            # Đánh dấu tam giác có nét tròn lạ bằng viền nổi bật
            cv2.polylines(vis_loops, [pts_r], True, (0, 255, 255), 2, cv2.LINE_AA)
            cr = np.mean(pts_r, axis=0).astype(int)
            cv2.putText(vis_loops, f"+{comparisons[i]['new_loops']}", (cr[0] - 10, cr[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.imwrite(os.path.join(out_dir, "03_shape_features_loops.jpg"), vis_loops, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # --- 4. BẢN ĐỒ DỊ THƯỜNG MESH (Thang độ xám dị thường trên bản đồ nét) ---
    vis_mesh_diff = np.zeros((h, w, 3), dtype=np.uint8)
    vis_mesh_diff[r_edge_full > 127] = (120, 120, 120)

    for i, tri in enumerate(triangles):
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
        score = comparisons[i]["macro_score"]
        # Độ sáng tương ứng với độ dị thường
        brightness = int(np.clip(score * 200, 0, 200))
        cv2.fillConvexPoly(vis_mesh_diff, pts_r, (brightness, brightness, brightness))
        cv2.polylines(vis_mesh_diff, [pts_r], True, (40, 40, 40), 1, cv2.LINE_AA)

    # Đè lại nét Return lên trên để dễ quan sát
    vis_mesh_diff[r_edge_full > 127] = (255, 255, 255)
    cv2.imwrite(os.path.join(out_dir, "04_macro_anomaly_mesh.jpg"), vis_mesh_diff, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # --- 5. DANH SÁCH MESH NGHI VẤN TRÊN BẢN ĐỒ NÉT RETURN ---
    vis_candidates = np.zeros((h, w, 3), dtype=np.uint8)
    vis_candidates[r_edge_full > 127] = (180, 180, 180)
    suspicious_indices = []

    for i, tri in enumerate(triangles):
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.int32)
        cv2.polylines(vis_candidates, [pts_r], True, (50, 50, 50), 1, cv2.LINE_AA)

        if comparisons[i]["is_suspicious"]:
            suspicious_indices.append(i)
            # Khoanh vùng tam giác nghi vấn bằng màu vàng cam trên nền nét đen trắng
            cv2.polylines(vis_candidates, [pts_r], True, (0, 140, 255), 2, cv2.LINE_AA)
            cr = np.mean(pts_r, axis=0).astype(int)
            cv2.putText(vis_candidates, f"#{i}", (cr[0] - 12, cr[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.imwrite(os.path.join(out_dir, "05_suspicious_mesh_candidates.jpg"), vis_candidates, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # --- 6. XUẤT CÁC CROP ĐEN TRẮNG ĐỐI CHIẾU NÉT CHO TỪNG TAM GIÁC NGHI VẤN ---
    from step3_pipeline import canonical_triangle_patch

    for s_idx in suspicious_indices:
        tri = triangles[s_idx]
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

        # Cắt patch đen trắng
        patch_p, _ = canonical_triangle_patch(p_edge_full, pts_p, target_size=120)
        patch_r, _ = canonical_triangle_patch(r_edge_full, pts_r, target_size=120)

        # Chuyển sang 3 kênh để vẽ chú thích
        patch_p_bgr = cv2.cvtColor(patch_p, cv2.COLOR_GRAY2BGR)
        patch_r_bgr = cv2.cvtColor(patch_r, cv2.COLOR_GRAY2BGR)

        # Đặt 2 patch đen trắng cạnh nhau
        panel = np.hstack([patch_p_bgr, patch_r_bgr])
        panel_h, panel_w = panel.shape[:2]

        canvas = np.zeros((panel_h + 85, panel_w, 3), dtype=np.uint8)
        canvas[:panel_h] = panel

        # Phân cách giữa 2 ảnh
        cv2.line(canvas, (120, 0), (120, panel_h), (100, 100, 100), 1)

        comp = comparisons[s_idx]
        prof_p = profiles_p[s_idx]
        prof_r = profiles_r[s_idx]

        ang_p_deg = np.degrees(prof_p["dominant_angle"]) if prof_p["dominant_angle"] >= 0 else 0
        ang_r_deg = np.degrees(prof_r["dominant_angle"]) if prof_r["dominant_angle"] >= 0 else 0

        cv2.putText(canvas, f"Tri #{s_idx} [NGHI VAN] - Score: {comp['macro_score']:.2f}", (6, panel_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
        cv2.putText(canvas, f"Ly do: {comp['reason']}", (6, panel_h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(canvas, f"Left(P): Ang={ang_p_deg:.0f} deg | Loops={prof_p['closed_loops']}", (6, panel_h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(canvas, f"Right(R): Ang={ang_r_deg:.0f} deg | Loops={prof_r['closed_loops']}", (6, panel_h + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        fname = f"suspicious_tri_{s_idx:03d}_{comp['reason']}.jpg"
        cv2.imwrite(os.path.join(dir_crops, fname), canvas, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print(f"  [Step 1 Grayscale Complete]:")
    print(f"    Total Triangles: {len(triangles)}")
    print(f"    Suspicious Candidates flagged: {len(suspicious_indices)} ({len(suspicious_indices)/len(triangles):.1%})")
    print(f"    Outputs saved to: {out_dir}")

    return {
        "total_triangles": len(triangles),
        "suspicious_count": len(suspicious_indices),
        "suspicious_indices": suspicious_indices,
    }
