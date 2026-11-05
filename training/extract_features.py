"""
training/extract_features.py
============================
Trích xuất vector đặc trưng DINOv2 (384-dim) từ Sổ địa chỉ khuyết tật (ground_truth_defects.json)
kèm lấy mẫu âm đối chứng (Nhãn 0: Men lành / Normal).

Nguyên lý:
1. Nạp mô hình nền tảng DINOv2 (dinov2_vits14) ở chế độ đóng băng (Frozen).
2. Chiếu tọa độ Bounding Box [x, y, w, h] lên lưới 32x32 patch (kích thước 14x14 pixel trên ảnh 448x448).
3. Trích xuất vector vi sai Δf = |f_syn - f_ref| tại các ô patch khuyết tật (Nhãn 1, 2, 3).
4. Lấy mẫu âm: Trích xuất các ô patch men lành nằm xa vùng lỗi (Nhãn 0).
5. Lưu toàn bộ dữ liệu vào training/dataset_features.pt.
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

    # Scale tọa độ sang target_size (448)
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


def extract_features_dataset(
    catalog_path: str,
    output_pt_path: str,
    max_samples_per_defect: int = 4,
    negative_samples_per_image: int = 3
) -> str:
    """
    Đọc sổ địa chỉ và trích xuất vector huấn luyện
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

    print(f"[EXTRACT] Bắt đầu trích xuất vector cho {len(records)} mục khuyết tật...")

    for i, item in enumerate(records):
        syn_path = item["synthetic_image_path"]
        ref_path = item["reference_image_path"]
        bbox = item["bbox_xywh"]
        class_id = item["class_id"]
        img_w, img_h = item["image_size"]

        if not (os.path.exists(syn_path) and os.path.exists(ref_path)):
            continue

        syn_bgr = cv2.imread(syn_path)
        ref_bgr = cv2.imread(ref_path)
        if syn_bgr is None or ref_bgr is None:
            continue

        # Resize ảnh về 448x448 cho DINOv2
        syn_rgb = cv2.cvtColor(cv2.resize(syn_bgr, (448, 448)), cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(cv2.resize(ref_bgr, (448, 448)), cv2.COLOR_BGR2RGB)

        t_syn = transform(syn_rgb).unsqueeze(0).to(device)
        t_ref = transform(ref_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            f_syn = backbone.get_intermediate_layers(t_syn, n=1)[0].squeeze(0)  # (1024, 384)
            f_ref = backbone.get_intermediate_layers(t_ref, n=1)[0].squeeze(0)  # (1024, 384)

            # Chuẩn hóa L2 từng patch vector
            f_syn = f_syn / f_syn.norm(dim=-1, keepdim=True)
            f_ref = f_ref / f_ref.norm(dim=-1, keepdim=True)

            # Vector vi sai |f_syn - f_ref|
            delta_f = torch.abs(f_syn - f_ref).cpu()

        # 1. Trích xuất vector vùng khuyết tật (Nhãn 1, 2, 3)
        defect_patch_idxs = bbox_to_patch_indices(bbox, (img_w, img_h), target_size=448, patch_size=14)
        if len(defect_patch_idxs) > max_samples_per_defect:
            defect_patch_idxs = random.sample(defect_patch_idxs, max_samples_per_defect)

        for p_idx in defect_patch_idxs:
            all_features.append(delta_f[p_idx])
            all_labels.append(class_id)

        # 2. Lấy mẫu âm: Vùng men lành cách xa vết lỗi (Nhãn 0)
        # Chỉ lấy trong khu vực thân bình (giữa ảnh, r: 8->24, c: 8->24)
        defect_set = set(defect_patch_idxs)
        candidate_normals = []
        for r in range(8, 25):
            for c in range(8, 25):
                idx = r * 32 + c
                if idx not in defect_set:
                    candidate_normals.append(idx)

        if candidate_normals:
            sampled_negatives = random.sample(candidate_normals, min(negative_samples_per_image, len(candidate_normals)))
            for neg_idx in sampled_negatives:
                all_features.append(delta_f[neg_idx])
                all_labels.append(0)  # Nhãn 0: Normal

        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"  -> Đã xử lý {i + 1}/{len(records)} ảnh... (Tổng vector hiện tại: {len(all_features)})")

    features_tensor = torch.stack(all_features, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save({
        "features": features_tensor,
        "labels": labels_tensor,
        "num_samples": len(all_labels),
        "in_features": features_tensor.shape[1]
    }, output_pt_path)

    # Thống kê phân phối nhãn
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for l in all_labels:
        counts[l] = counts.get(l, 0) + 1

    print("\n[EXTRACT] Hoàn tất trích xuất dữ liệu huấn luyện!")
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
