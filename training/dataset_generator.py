"""
training/dataset_generator.py
=============================
Sinh dữ liệu khuyết tật gốm sứ giả lập sử dụng trực tiếp các file ảnh đã tách nền _nb.png
(từ data/ghost_overlays/) để đảm bảo:
1. Mặt nạ thân bình chuẩn 100% từ kênh Alpha (loại bỏ hoàn toàn nền trắng và móc sắt).
2. Khuyết tật chỉ được cấy bên trong lòng thân bình (inner_mask).
3. Tự động ghi chép "Sổ địa chỉ khuyết tật" [x, y, w, h] vào ground_truth_defects.json.
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

    # Giữ vết nứt chỉ nằm trong mask thân bình
    out_img[mask == 0] = img_bgr[mask == 0]
    defect_mask[mask == 0] = 0

    cnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    else:
        bx, by, bw, bh = 0, 0, 0, 0

    return out_img, defect_mask, [int(bx), int(by), int(bw), int(bh)]


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

    chip_color = np.array([random.randint(130, 160), random.randint(145, 175), random.randint(160, 190)], dtype=np.uint8)
    noise = np.random.normal(0, 12, (h, w, 3)).astype(np.int16)

    chip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(chip_mask, [poly_arr], 255)
    chip_mask = cv2.bitwise_and(chip_mask, mask)

    # Viền sứt mẻ có cạnh bóng tối
    cv2.polylines(out_img, [poly_arr], isClosed=True, color=(35, 35, 35), thickness=2, lineType=cv2.LINE_AA)

    # Gán chất liệu xương gốm
    for c in range(3):
        blended = np.clip(chip_color[c] + noise[..., c], 0, 255).astype(np.uint8)
        out_img[chip_mask > 0, c] = blended[chip_mask > 0]

    out_img[mask == 0] = img_bgr[mask == 0]
    defect_mask = chip_mask

    cnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    else:
        bx, by, bw, bh = 0, 0, 0, 0

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

    alpha_blend = np.clip(blob * random.uniform(0.55, 0.85), 0.0, 1.0)[..., np.newaxis]
    out_img = (out_img * (1.0 - alpha_blend) + target_color * alpha_blend).astype(np.uint8)
    out_img[mask == 0] = img_bgr[mask == 0]

    defect_mask = (blob > 0.15).astype(np.uint8) * 255
    defect_mask[mask == 0] = 0

    cnts, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    else:
        bx, by, bw, bh = 0, 0, 0, 0

    return out_img, defect_mask, [int(bx), int(by), int(bw), int(bh)]


def build_synthetic_dataset(
    ghost_overlays_dir: str,
    output_dir: str,
    samples_per_vase: int = 2
) -> str:
    """
    Đọc trực tiếp các file ảnh đã tách nền _nb.png từ data/ghost_overlays/
    Lấy kênh Alpha làm mask chuẩn 100%, cấy lỗi hoàn toàn bên trong thân bình.
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

    view_specs = [
        ("01_front", "truoc_nb.png"),
        ("02_back", "sau_nb.png"),
        ("03_left", "trai_nb.png"),
        ("04_right", "phai_nb.png")
    ]

    print(f"[GEN] Bắt đầu sinh dữ liệu khuyết tật từ ảnh tách nền (_nb.png) tại: {ghost_overlays_dir}")
    vase_folders = sorted([d for d in os.listdir(ghost_overlays_dir) if os.path.isdir(os.path.join(ghost_overlays_dir, d))])

    total_generated = 0
    for vase_id in vase_folders:
        vase_path = os.path.join(ghost_overlays_dir, vase_id)

        for view_name, nb_filename in view_specs:
            nb_path = os.path.join(vase_path, nb_filename)
            if not os.path.exists(nb_path):
                continue

            # Đọc ảnh 4 kênh RGBA
            im_rgba = cv2.imread(nb_path, cv2.IMREAD_UNCHANGED)
            if im_rgba is None or len(im_rgba.shape) != 3:
                continue

            if im_rgba.shape[2] == 4:
                bgr = im_rgba[:, :, :3]
                alpha = im_rgba[:, :, 3]
            else:
                bgr = im_rgba
                alpha = np.ones((bgr.shape[0], bgr.shape[1]), dtype=np.uint8) * 255

            # Mặt nạ thân bình chuẩn 100% từ kênh Alpha (loại bỏ móc sắt và nền ngoài)
            mask = (alpha > 200).astype(np.uint8) * 255
            # Thu hẹp mặt nạ 15px để đảm bảo không chạm viền ngoài
            inner_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

            for defect_name, class_id, defect_func in defect_funcs:
                for rep in range(samples_per_vase):
                    syn_bgr, d_mask, bbox = defect_func(bgr, inner_mask)
                    bx, by, bw, bh = bbox
                    if bw < 5 or bh < 5:
                        continue

                    # Đóng gói ảnh PNG có kênh Alpha trong suốt
                    syn_rgba = np.dstack([syn_bgr, alpha])

                    item_id = f"{vase_id}_{view_name}_{defect_name}_{rep:02d}"
                    syn_filename = f"{item_id}.png"
                    syn_filepath = os.path.join(images_out_dir, syn_filename)
                    cv2.imwrite(syn_filepath, syn_rgba)

                    record = {
                        "defect_id": total_generated + 1,
                        "item_id": item_id,
                        "vase_id": vase_id,
                        "view_name": view_name,
                        "reference_image_path": nb_path,
                        "synthetic_image_path": syn_filepath,
                        "defect_type": defect_name,
                        "class_id": class_id,
                        "bbox_xywh": [bx, by, bw, bh],
                        "image_size": [bgr.shape[1], bgr.shape[0]]  # [w, h]
                    }
                    catalog.append(record)
                    total_generated += 1

    catalog_path = os.path.join(output_dir, "ground_truth_defects.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"[GEN] Đã tạo thành công {total_generated} mẫu khuyết tật chuẩn xác bên trong thân bình!")
    print(f"[GEN] Sổ địa chỉ đã được lưu tại: {catalog_path}")
    return catalog_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ghost_dir = os.path.join(base_dir, "data", "ghost_overlays")
    out_dir = os.path.join(base_dir, "training")
    build_synthetic_dataset(ghost_dir, out_dir, samples_per_vase=2)
