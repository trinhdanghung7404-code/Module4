"""
training/extract_features.py
============================
Trích xuất vector đặc trưng DINOv2 (384-dim) từ Sổ địa chỉ khuyết tật (ground_truth_defects.json)
kèm lấy mẫu âm đối chứng (Nhãn 0: Men lành / Normal) từ ảnh tách nền _nb.png.

Đặc tính:
1. Đọc ảnh RGBA tách nền, đặt trên nền xám trung tính 128 (loại bỏ nhầm lẫn nền trắng).
2. Lấy mẫu âm (Men lành) 100% bên trong thân bình dựa theo kênh Alpha (alpha > 0.8).
3. Lưu toàn bộ dữ liệu vào training/dataset_features.pt.
"""

import os
import sys
import cv2
import json
import torch
import random
import numpy as np
from torchvision import transforms
from typing import Tuple, List, Dict, Any

# Đảm bảo in tiếng Việt không bị lỗi cp1252 trên Windows console
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_dinov2_backbone(device: torch.device):
    """Nạp DINOv2 ViT-S/14 từ PyTorch Hub hoặc cache local"""
    print("[EXTRACT] Đang nạp backbone DINOv2 (dinov2_vits14)...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def bbox_to_patch_indices(
    bbox: List[int],
    orig_size: Tuple[int, int],
    target_size: int = 448,
    patch_size: int = 14
) -> List[int]:
    """
    Chuyển đổi tọa độ bbox [x, y, w, h] từ ảnh gốc sang danh sách index các patch 14x14 trên lưới 32x32
    """
    orig_w, orig_h = orig_size
    bx, by, bw, bh = bbox
    grid_size = target_size // patch_size  # 32

    scale_x = target_size / float(orig_w)
    scale_y = target_size / float(orig_h)

    x1 = int(bx * scale_x)
    y1 = int(by * scale_y)
    x2 = int((bx + bw) * scale_x)
    y2 = int((by + bh) * scale_y)

    c1 = max(0, min(grid_size - 1, x1 // patch_size))
    c2 = max(0, min(grid_size - 1, x2 // patch_size))
    r1 = max(0, min(grid_size - 1, y1 // patch_size))
    r2 = max(0, min(grid_size - 1, y2 // patch_size))

    patch_indices = []
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            patch_indices.append(r * grid_size + c)

    return patch_indices


def prepare_rgb_image(im_path: str, target_size: int = 448) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc ảnh RGBA và đặt lên nền xám trung tính 128"""
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {im_path}")

    h, w = im.shape[:2]
    if len(im.shape) == 3 and im.shape[2] == 4:
        alpha = im[:, :, 3].astype(np.float32) / 255.0
        bgr = im[:, :, :3]
    else:
        bgr = im
        alpha = np.ones((h, w), dtype=np.float32)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Ghép lên nền xám trung tính 128 (chống lẫn màu nền trắng)
    composited = (rgb * alpha[:, :, None] + 128.0 * (1.0 - alpha[:, :, None])).astype(np.uint8)
    resized_rgb = cv2.resize(composited, (target_size, target_size), interpolation=cv2.INTER_AREA)
    resized_alpha = cv2.resize(alpha, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return resized_rgb, resized_alpha


def extract_features_dataset(
    catalog_path: str,
    output_pt_path: str,
    max_samples_per_defect: int = 4,
    negative_samples_per_image: int = 3
) -> str:
    """
    Đọc sổ địa chỉ và trích xuất vector huấn luyện từ ảnh _nb.png
    """
    device = torch.device("cpu")
    backbone = get_dinov2_backbone(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(catalog_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    all_features = []
    all_labels = []

    print(f"[EXTRACT] Bắt đầu trích xuất vector từ ảnh tách nền cho {len(records)} mục khuyết tật...")

    for i, item in enumerate(records):
        syn_path = item["synthetic_image_path"]
        ref_path = item["reference_image_path"]
        bbox = item["bbox_xywh"]
        class_id = item["class_id"]
        img_w, img_h = item["image_size"]

        if not (os.path.exists(syn_path) and os.path.exists(ref_path)):
            continue

        try:
            syn_rgb, syn_alpha = prepare_rgb_image(syn_path, target_size=448)
            ref_rgb, ref_alpha = prepare_rgb_image(ref_path, target_size=448)
        except Exception:
            continue

        t_syn = transform(syn_rgb).unsqueeze(0).to(device)
        t_ref = transform(ref_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            f_syn = backbone.get_intermediate_layers(t_syn, n=1)[0].squeeze(0)  # (1024, 384)
            f_ref = backbone.get_intermediate_layers(t_ref, n=1)[0].squeeze(0)  # (1024, 384)

            f_syn = f_syn / f_syn.norm(dim=-1, keepdim=True)
            f_ref = f_ref / f_ref.norm(dim=-1, keepdim=True)

            delta_f = torch.abs(f_syn - f_ref).cpu()

        # 1. Trích xuất vector vùng khuyết tật (Nhãn 1, 2, 3)
        defect_patch_idxs = bbox_to_patch_indices(bbox, (img_w, img_h), target_size=448, patch_size=14)
        if len(defect_patch_idxs) > max_samples_per_defect:
            defect_patch_idxs = random.sample(defect_patch_idxs, max_samples_per_defect)

        for p_idx in defect_patch_idxs:
            all_features.append(delta_f[p_idx])
            all_labels.append(class_id)

        # 2. Lấy mẫu âm (Men lành - Nhãn 0): 100% bên trong thân bình theo kênh Alpha
        grid_alpha = cv2.resize(syn_alpha, (32, 32), interpolation=cv2.INTER_AREA).flatten()
        defect_set = set(defect_patch_idxs)
        candidate_normals = [idx for idx in range(1024) if (grid_alpha[idx] > 0.85 and idx not in defect_set)]

        if candidate_normals:
            sampled_negatives = random.sample(candidate_normals, min(negative_samples_per_image, len(candidate_normals)))
            for neg_idx in sampled_negatives:
                all_features.append(delta_f[neg_idx])
                all_labels.append(0)  # Nhãn 0: Normal

        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"  -> Đã xử lý {i + 1}/{len(records)} ảnh... (Tổng vector: {len(all_features)})")

    features_tensor = torch.stack(all_features, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save({
        "features": features_tensor,
        "labels": labels_tensor,
        "num_samples": len(all_labels),
        "in_features": features_tensor.shape[1]
    }, output_pt_path)

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for l in all_labels:
        counts[l] = counts.get(l, 0) + 1

    print("\n[EXTRACT] Hoàn tất trích xuất dữ liệu sạch từ ảnh _nb.png!")
    print(f"  • Tổng số mẫu: {len(all_labels)} vector")
    print(f"  • Kích thước tensor: {features_tensor.shape}")
    print(f"  • Phân phối nhãn: 0 (Normal): {counts[0]} | 1 (Crack): {counts[1]} | 2 (Chip): {counts[2]} | 3 (Stain): {counts[3]}")
    print(f"  • Tệp lưu trữ: {output_pt_path}")

    return output_pt_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cat_path = os.path.join(base_dir, "training", "ground_truth_defects.json")
    out_pt = os.path.join(base_dir, "training", "dataset_features.pt")
    if os.path.exists(cat_path):
        extract_features_dataset(cat_path, out_pt)
    else:
        print("[ERROR] Không tìm thấy ground_truth_defects.json. Vui lòng chạy dataset_generator.py trước!")
