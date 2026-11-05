"""
training/dataset_generator.py
=============================
Sinh dữ liệu khuyết tật gốm sứ giả lập có chiều sâu quang học (Physics-guided Defect Augmentation)
kết hợp ghi chép "Sổ địa chỉ khuyết tật" (Ground-Truth Address Logging).

Nguyên lý:
1. Đọc các ảnh chuẩn B1 -> B21 từ data/reference_images/
2. Cấy khuyết tật:
   - Crack (Nhãn 1): Khe nứt vỡ rạn tóc có rãnh bóng đổ 3D và mép phản xạ men đón sáng.
   - Chip (Nhãn 2): Vết sứt mẻ đa giác lộ xương gốm thô nhám.
   - Stain (Nhãn 3): Vết ố loang màu men (trà, cà phê, lem mực).
3. Tự động lưu "Sổ địa chỉ" [x, y, w, h] vào ground_truth_defects.json (Zero manual labeling).
"""

import os
import sys
import cv2
import json
import random
import numpy as np
from typing import List, Dict, Any, Tuple

# Đảm bảo in tiếng Việt không bị lỗi cp1252 trên Windows console
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



def generate_crack_contour(start_pt: Tuple[int, int], length: int = 80, num_steps: int = 15) -> List[Tuple[int, int]]:
    """Tạo đường đi ngẫu nhiên ziczac mô phỏng vết nứt gốm sứ"""
    pts = [start_pt]
    curr_x, curr_y = start_pt
    angle = random.uniform(0, 2 * np.pi)
    
    for _ in range(num_steps):
        step_len = length / num_steps
        angle += random.uniform(-0.5, 0.5)
        curr_x += int(step_len * np.cos(angle))
        curr_y += int(step_len * np.sin(angle))
        pts.append((curr_x, curr_y))
    return pts


def inject_crack(img_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Cấy vết nứt có hiệu ứng rãnh bóng đổ và mép men đón sáng 3D
    """
    h, w = img_bgr.shape[:2]
    out_img = img_bgr.copy()
    defect_mask = np.zeros((h, w), dtype=np.uint8)

    # Chọn điểm xuất phát ngẫu nhiên bên trong thân bình
    valid_ys, valid_xs = np.where(mask > 200)
    if len(valid_ys) == 0:
        return out_img, defect_mask, [0, 0, 0, 0]

    idx = random.randint(0, len(valid_ys) - 1)
    start_pt = (valid_xs[idx], valid_ys[idx])
    
    crack_pts = generate_crack_contour(start_pt, length=random.randint(40, 110), num_steps=random.randint(10, 20))
    pts_arr = np.array(crack_pts, dtype=np.int32)

    # Giới hạn trong kích thước ảnh
    pts_arr[:, 0] = np.clip(pts_arr[:, 0], 5, w - 6)
    pts_arr[:, 1] = np.clip(pts_arr[:, 1], 5, h - 6)

    # 1. Rãnh bóng đổ tối màu (Shadow Fissure)
    for i in range(len(pts_arr) - 1):
        pt1 = tuple(pts_arr[i])
        pt2 = tuple(pts_arr[i + 1])
        cv2.line(out_img, pt1, pt2, (25, 25, 25), thickness=random.randint(2, 3), lineType=cv2.LINE_AA)
        cv2.line(defect_mask, pt1, pt2, 255, thickness=4, lineType=cv2.LINE_AA)

    # 2. Mép phản xạ men đón sáng song song (Specular Highlight Rim)
    offset_x = 1
    offset_y = -1
    for i in range(len(pts_arr) - 1):
        pt1 = (pts_arr[i][0] + offset_x, pts_arr[i][1] + offset_y)
        pt2 = (pts_arr[i + 1][0] + offset_x, pts_arr[i + 1][1] + offset_y)
        cv2.line(out_img, pt1, pt2, (230, 230, 230), thickness=1, lineType=cv2.LINE_AA)

    # Tính bounding box
    xs = pts_arr[:, 0]
    ys = pts_arr[:, 1]
    bx, by = int(np.min(xs)), int(np.min(ys))
    bw, bh = int(np.max(xs) - bx + 1), int(np.max(ys) - by + 1)
    
    # Padding nhẹ cho bbox
    bx = max(0, bx - 4)
    by = max(0, by - 4)
    bw = min(w - bx, bw + 8)
    bh = min(h - by, bh + 8)

    return out_img, defect_mask, [bx, by, bw, bh]


def inject_chip(img_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Cấy vết sứt mẻ đa giác lộ xương gốm thô nhám
    """
    h, w = img_bgr.shape[:2]
    out_img = img_bgr.copy()
    defect_mask = np.zeros((h, w), dtype=np.uint8)

    valid_ys, valid_xs = np.where(mask > 200)
    if len(valid_ys) == 0:
        return out_img, defect_mask, [0, 0, 0, 0]

    idx = random.randint(0, len(valid_ys) - 1)
    cx, cy = valid_xs[idx], valid_ys[idx]
    radius = random.randint(12, 30)

    # Sinh đa giác lồi ngẫu nhiên
    angles = np.sort(np.random.uniform(0, 2 * np.pi, random.randint(5, 8)))
    rads = np.random.uniform(radius * 0.6, radius, len(angles))
    poly_pts = []
    for a, r in zip(angles, rads):
        px = int(np.clip(cx + r * np.cos(a), 0, w - 1))
        py = int(np.clip(cy + r * np.sin(a), 0, h - 1))
        poly_pts.append([px, py])
    poly_arr = np.array(poly_pts, dtype=np.int32)

    # Điền màu xương gốm thô nhám (màu đất nung / xám ngà có hạt cát)
    chip_color = np.array([random.randint(130, 160), random.randint(145, 175), random.randint(160, 190)], dtype=np.uint8)
    noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)

    chip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(chip_mask, [poly_arr], 255)

    # Viền sứt mẻ có cạnh bóng tối
    cv2.polylines(out_img, [poly_arr], isClosed=True, color=(35, 35, 35), thickness=2, lineType=cv2.LINE_AA)

    # Gán chất liệu xương gốm
    for c in range(3):
        blended = np.clip(chip_color[c] + noise[..., c], 0, 255).astype(np.uint8)
        out_img[chip_mask > 0, c] = blended[chip_mask > 0]

    defect_mask = cv2.bitwise_or(defect_mask, chip_mask)

    bx, by, bw, bh = cv2.boundingRect(poly_arr)
    return out_img, defect_mask, [int(bx), int(by), int(bw), int(bh)]


def inject_stain(img_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Cấy vết ố loang màu men (trà, cà phê, lem mực)
    """
    h, w = img_bgr.shape[:2]
    out_img = img_bgr.copy()
    defect_mask = np.zeros((h, w), dtype=np.uint8)

    valid_ys, valid_xs = np.where(mask > 200)
    if len(valid_ys) == 0:
        return out_img, defect_mask, [0, 0, 0, 0]

    idx = random.randint(0, len(valid_ys) - 1)
    cx, cy = valid_xs[idx], valid_ys[idx]
    rx = random.randint(15, 35)
    ry = random.randint(15, 35)

    stain_type = random.choice(["tea_brown", "ink_blue", "yellow_grease"])
    if stain_type == "tea_brown":
        target_color = np.array([40, 75, 130], dtype=np.float32)  # Nâu trà BGR
    elif stain_type == "ink_blue":
        target_color = np.array([140, 60, 30], dtype=np.float32)  # Mực xanh BGR
    else:
        target_color = np.array([50, 140, 160], dtype=np.float32)  # Vàng ố BGR

    # Tạo màng loang mềm mượt bằng Gaussian Blur
    blob = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(blob, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, 1.0, -1)
    blob = cv2.GaussianBlur(blob, (35, 35), 0)
    blob[mask < 100] = 0.0

    alpha = np.clip(blob * random.uniform(0.55, 0.85), 0.0, 1.0)[..., np.newaxis]
    out_img = (out_img * (1.0 - alpha) + target_color * alpha).astype(np.uint8)

    defect_mask = (blob > 0.15).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    else:
        bx, by, bw, bh = cx - rx, cy - ry, 2 * rx, 2 * ry

    return out_img, defect_mask, [int(bx), int(by), int(bw), int(bh)]


def build_synthetic_dataset(
    ref_images_dir: str,
    output_dir: str,
    samples_per_vase: int = 2
) -> str:
    """
    Duyệt qua các bình chuẩn, cấy 3 loại lỗi và ghi lại Sổ địa chỉ metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    images_out_dir = os.path.join(output_dir, "synthetic_images")
    os.makedirs(images_out_dir, exist_ok=True)

    catalog = []
    defect_funcs = [
        ("crack", 1, inject_crack),
        ("chip", 2, inject_chip),
        ("stain", 3, inject_stain)
    ]

    print(f"[GEN] Bắt đầu sinh dữ liệu khuyết tật từ: {ref_images_dir}")
    vase_folders = sorted([d for d in os.listdir(ref_images_dir) if os.path.isdir(os.path.join(ref_images_dir, d))])

    total_generated = 0
    for vase_id in vase_folders:
        vase_path = os.path.join(ref_images_dir, vase_id)
        views = sorted([v for v in os.listdir(vase_path) if os.path.isdir(os.path.join(vase_path, v))])

        for view_name in views:
            img_path = os.path.join(vase_path, view_name, "anh_chuan.jpg")
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Tạo mask đơn giản cho thân bình (vùng khác màu trắng nền)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = ((img[..., 0] < 240) | (img[..., 1] < 240) | (img[..., 2] < 240)).astype(np.uint8) * 255
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

            for defect_name, class_id, defect_func in defect_funcs:
                for rep in range(samples_per_vase):
                    syn_img, d_mask, bbox = defect_func(img, mask)
                    bx, by, bw, bh = bbox
                    if bw < 5 or bh < 5:
                        continue

                    item_id = f"{vase_id}_{view_name}_{defect_name}_{rep:02d}"
                    syn_filename = f"{item_id}.jpg"
                    syn_filepath = os.path.join(images_out_dir, syn_filename)
                    cv2.imwrite(syn_filepath, syn_img)

                    record = {
                        "defect_id": total_generated + 1,
                        "item_id": item_id,
                        "vase_id": vase_id,
                        "view_name": view_name,
                        "reference_image_path": img_path,
                        "synthetic_image_path": syn_filepath,
                        "defect_type": defect_name,
                        "class_id": class_id,
                        "bbox_xywh": [bx, by, bw, bh],
                        "image_size": [img.shape[1], img.shape[0]]  # [w, h]
                    }
                    catalog.append(record)
                    total_generated += 1

    catalog_path = os.path.join(output_dir, "ground_truth_defects.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"[GEN] Đã tạo thành công {total_generated} mẫu khuyết tật!")
    print(f"[GEN] Sổ địa chỉ đã được lưu tại: {catalog_path}")
    return catalog_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_dir = os.path.join(base_dir, "data", "reference_images")
    out_dir = os.path.join(base_dir, "training")
    build_synthetic_dataset(ref_dir, out_dir, samples_per_vase=2)
