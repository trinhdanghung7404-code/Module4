"""
core/defect_simulator.py
========================
Module giả lập khuyết tật nhân tạo (Crack, Stain, Chip)
phục vụ kiểm thử độ nhạy AI (Stress Test) và Demo cho Tester / Developer.
Hoàn toàn phi phá hủy (Non-destructive) bằng cơ chế tự động sao lưu ảnh gốc.
"""

import os
import cv2
import shutil
import random
import json
import numpy as np
from typing import Dict, Any, List, Optional

from training.dataset_generator import (
    inject_crack,
    inject_stain,
    inject_chip,
    Vase3DSurfaceModel,
)


def _get_vase_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Tạo mask vật thể gốm từ ảnh đã tách nền 1024x1024."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = ((gray > 40) & (gray < 252)).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def simulate_defect_on_session(
    session_dir: str,
    product_id: str,
    pipeline: Any,
    view: str = "random",
    defect_type: str = "random",
    views_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Cấy khuyết tật nhân tạo lên ảnh đã tiền xử lý của một view trong session:
    - Lưu bản sao an toàn customer_{view}_processed_clean.png nếu chưa có.
    - Cấy vết nứt (crack), vết ố men (stain) hoặc sứt mẻ (chip) trực tiếp lên customer_{view}_processed.png.
    - Tự động chạy lại pipeline.inspect_return để cập nhật kết quả kiểm định.
    """
    if views_list is None:
        views_list = ["front", "back", "left", "right"]

    # 1. Xác định view mục tiêu
    target_views = []
    for v in views_list:
        p_path = os.path.join(session_dir, f"customer_{v}_processed.png")
        if os.path.exists(p_path):
            target_views.append(v)

    if not target_views:
        raise ValueError(f"Chưa có ảnh tiền xử lý nào trong session {os.path.basename(session_dir)}")

    chosen_view = view.lower()
    if chosen_view == "random" or chosen_view not in target_views:
        chosen_view = random.choice(target_views)

    # 2. Xác định loại lỗi
    chosen_type = defect_type.lower()
    if chosen_type == "random":
        chosen_type = random.choice(["crack", "stain", "chip"])

    proc_path = os.path.join(session_dir, f"customer_{chosen_view}_processed.png")
    clean_path = os.path.join(session_dir, f"customer_{chosen_view}_processed_clean.png")

    # 3. Sao lưu ảnh gốc nếu chưa sao lưu
    if not os.path.exists(clean_path):
        shutil.copy2(proc_path, clean_path)

    # 4. Đọc ảnh sạch từ bản sao lưu để đảm bảo không bị tích lũy lỗi
    img_bgr = cv2.imread(clean_path)
    if img_bgr is None:
        raise ValueError(f"Không thể đọc ảnh sạch: {clean_path}")

    mask = _get_vase_mask(img_bgr)
    geo = Vase3DSurfaceModel(img_bgr, mask)

    # 5. Cấy lỗi theo loại đã chọn
    if chosen_type == "crack":
        defect_img, defect_mask, bbox = inject_crack(img_bgr, mask, geo)
    elif chosen_type == "stain":
        defect_img, defect_mask, bbox = inject_stain(img_bgr, mask, geo)
    elif chosen_type == "chip":
        defect_img, defect_mask, bbox = inject_chip(img_bgr, mask, geo)
    else:
        defect_img, defect_mask, bbox = inject_crack(img_bgr, mask, geo)

    # 6. Ghi đè vào file processed để pipeline nhận diện
    cv2.imwrite(proc_path, defect_img)

    # 7. Chạy lại inspect_return cho view này
    from core.pipeline import resolve_profile_id
    profile_id = resolve_profile_id(product_id, chosen_view, pipeline.profiles_dir)
    if not profile_id:
        raise ValueError(f"Không tìm thấy reference profile cho {product_id} [{chosen_view}]")

    report = pipeline.inspect_return(
        return_img_path=proc_path,
        product_id=profile_id,
        output_dir=session_dir,
    )

    # 8. Cập nhật detect_meta.json
    detect_meta_path = os.path.join(session_dir, "detect_meta.json")
    detect_meta = {}
    if os.path.exists(detect_meta_path):
        try:
            with open(detect_meta_path, "r", encoding="utf-8") as f:
                detect_meta = json.load(f)
        except Exception:
            detect_meta = {}

    res_img_name = os.path.basename(report.get("result_image", ""))
    res_img_url = f"/uploads/{os.path.basename(session_dir)}/{res_img_name}" if res_img_name else None

    detect_meta[chosen_view] = {
        "status": "ok",
        "product_id": profile_id,
        "is_damaged": report.get("is_damaged", False),
        "defect_count": report.get("defect_count", 0),
        "total_defects": report.get("defect_count", 0),
        "damage_percentage": report.get("damage_percentage", 0.0),
        "severity": report.get("severity", "DEFECTIVE"),
        "result_image_url": res_img_url,
        "side_by_side_url": res_img_url,
        "defects": report.get("defects", []),
        "is_simulated": True,
        "simulated_type": chosen_type,
    }

    with open(detect_meta_path, "w", encoding="utf-8") as f:
        json.dump(detect_meta, f, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "view": chosen_view,
        "defect_type": chosen_type,
        "bbox": bbox,
        "is_damaged": report.get("is_damaged", False),
        "defect_count": report.get("defect_count", 0),
        "severity": report.get("severity", "DEFECTIVE"),
    }


def reset_simulation_on_session(
    session_dir: str,
    product_id: str,
    pipeline: Any,
    views_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Khôi phục ảnh gốc sạch 100% và chạy lại thẩm định để đưa hệ thống về trạng thái nguyên bản.
    """
    if views_list is None:
        views_list = ["front", "back", "left", "right"]

    from core.pipeline import resolve_profile_id
    restored_views = []

    detect_meta_path = os.path.join(session_dir, "detect_meta.json")
    detect_meta = {}
    if os.path.exists(detect_meta_path):
        try:
            with open(detect_meta_path, "r", encoding="utf-8") as f:
                detect_meta = json.load(f)
        except Exception:
            detect_meta = {}

    for v in views_list:
        clean_path = os.path.join(session_dir, f"customer_{v}_processed_clean.png")
        proc_path = os.path.join(session_dir, f"customer_{v}_processed.png")

        if os.path.exists(clean_path):
            shutil.copy2(clean_path, proc_path)
            try:
                os.remove(clean_path)
            except Exception:
                pass
            restored_views.append(v)

            # Chạy lại inspect_return cho view đã khôi phục
            profile_id = resolve_profile_id(product_id, v, pipeline.profiles_dir)
            if profile_id:
                report = pipeline.inspect_return(
                    return_img_path=proc_path,
                    product_id=profile_id,
                    output_dir=session_dir,
                )
                res_img_name = os.path.basename(report.get("result_image", ""))
                res_img_url = f"/uploads/{os.path.basename(session_dir)}/{res_img_name}" if res_img_name else None

                detect_meta[v] = {
                    "status": "ok",
                    "product_id": profile_id,
                    "is_damaged": report.get("is_damaged", False),
                    "defect_count": report.get("defect_count", 0),
                    "total_defects": report.get("defect_count", 0),
                    "damage_percentage": report.get("damage_percentage", 0.0),
                    "severity": report.get("severity", "INTACT"),
                    "result_image_url": res_img_url,
                    "side_by_side_url": res_img_url,
                    "defects": report.get("defects", []),
                    "is_simulated": False,
                }

    if detect_meta:
        with open(detect_meta_path, "w", encoding="utf-8") as f:
            json.dump(detect_meta, f, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "restored_views": restored_views,
    }
