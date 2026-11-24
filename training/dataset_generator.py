"""
training/dataset_generator.py
=============================
Sinh dữ liệu khuyết tật gốm sứ giả lập sử dụng trực tiếp các file ảnh đã tách nền _nb.png
(từ data/ghost_overlays/) với thuật toán chiếu cong 3D (3D Surface Conforming Projection):
1. Tự động trích xuất mô hình hình học 3D mặt cong tròn xoay (Xc(y), R(y), Normal Map N) từ kênh Alpha.
2. Vết lỗi ăn theo độ cong 3D của thân bình (uốn lượn theo vĩ tuyến cầu, co dẹp theo cos_theta).
3. Đúng bản chất vật lý quang học men thủy tinh và xương gốm đất nung.
4. Tự động ghi chép Sổ địa chỉ khuyết tật [x, y, w, h] vào ground_truth_defects.json.

Nhóm CRACK (1 Nhỏ, 2 To - Đã thống nhất & phê duyệt):
- [Nứt Nhỏ]: Rạn tóc men vi mô 1.5px (Hairline Crack).
- [Nứt To 1]: Nứt toác xẻ từ miệng bình kèm vệt dập xước men trắng ở đầu và mép rãnh (Rim Macro Crack).
- [Nứt To 2]: Nứt toác thân bình phân nhánh lớn kèm vệt cào xước men trắng khởi đầu (Body Macro Crack).
"""

import os
import sys
import cv2
import json
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class Vase3DSurfaceModel:
    """Tự động dựng mô hình hình học 3D mặt cong tròn xoay từ ảnh tách nền"""
    def __init__(self, img_bgr: np.ndarray, mask: np.ndarray):
        self.h, self.w = img_bgr.shape[:2]
        self.bgr = img_bgr
        self.mask = mask.copy()
        self.mask[int(self.h * 0.90):, :] = 0
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(self.mask)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            self.mask = (labels == largest).astype(np.uint8) * 255
            
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        row_indices = np.where(self.mask.any(axis=1))[0]
        self.top_y = row_indices[0] if len(row_indices) > 0 else 0
        self.bottom_y = row_indices[-1] if len(row_indices) > 0 else self.h
        vase_h_px = self.bottom_y - self.top_y
        self.px_per_mm = float(vase_h_px) / 350.0  # ~2.62 px/mm
        
        # 1. Trích xuất tâm đối xứng Xc(y) và bán kính R(y)
        self.xc = np.full(self.h, np.nan, dtype=np.float32)
        self.radius = np.full(self.h, 0.0, dtype=np.float32)
        for y in range(self.h):
            row = np.where(self.mask[y] > 0)[0]
            if len(row) > 10:
                xl, xr = float(row[0]), float(row[-1])
                self.xc[y] = (xl + xr) / 2.0
                self.radius[y] = (xr - xl) / 2.0
                
        valid_y = np.where(~np.isnan(self.xc))[0]
        if len(valid_y) > 20:
            k = np.ones(15) / 15.0
            pad_w = 7
            self.radius[valid_y] = np.convolve(np.pad(self.radius[valid_y], pad_w, mode='edge'), k, mode='valid')
            self.xc[valid_y] = np.convolve(np.pad(self.xc[valid_y], pad_w, mode='edge'), k, mode='valid')
            
        # 2. Xây dựng bản đồ pháp tuyến Normal Map
        self.normal_map = np.zeros((self.h, self.w, 3), dtype=np.float32)
        self.cos_map = np.zeros((self.h, self.w), dtype=np.float32)
        
        for y in range(self.h):
            r = self.radius[y]
            c = self.xc[y]
            if np.isnan(c) or r <= 5.0:
                continue
            dr_dy = 0.0
            if 0 < y < self.h - 1 and not np.isnan(self.radius[y-1]) and not np.isnan(self.radius[y+1]):
                dr_dy = (self.radius[y+1] - self.radius[y-1]) / 2.0
                
            xl = max(0, int(c - r))
            xr = min(self.w, int(c + r) + 1)
            for x in range(xl, xr):
                dx = x - c
                if abs(dx) < r and self.mask[y, x] > 0:
                    sin_th = np.clip(dx / r, -0.999, 0.999)
                    cos_th = np.sqrt(1.0 - sin_th**2)
                    self.cos_map[y, x] = cos_th
                    nx = sin_th
                    ny = -dr_dy / max(r, 10.0)
                    nz = cos_th
                    norm = np.hypot(np.hypot(nx, ny), nz)
                    self.normal_map[y, x] = [nx / norm, ny / norm, nz / norm]

    def surface_to_camera(self, theta: float, y: float):
        iy = int(np.clip(y, 0, self.h - 1))
        r = self.radius[iy]
        c = self.xc[iy]
        if np.isnan(c) or r <= 5.0:
            return None, None, 0.0
        if abs(theta) >= np.pi / 2.02:
            return None, None, 0.0
        x_cam = c + r * np.sin(theta)
        cos_th = np.cos(theta)
        return x_cam, y, max(0.0, float(cos_th))


def _inject_small_hairline_crack(geo: Vase3DSurfaceModel, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Dạng nứt nhỏ: Rạn tóc men vi mô 1.5px uốn lượn 3D"""
    valid_ys, valid_xs = np.where(geo.mask > 200)
    y_min, y_max = int(geo.h * 0.20), int(geo.h * 0.75)
    candidates = np.where((valid_ys >= y_min) & (valid_ys <= y_max))[0]
    idx = candidates[random.randint(0, len(candidates) - 1)] if len(candidates) > 0 else random.randint(0, len(valid_ys) - 1)
    
    start_x, start_y = valid_xs[idx], valid_ys[idx]
    r_start = max(10.0, geo.radius[start_y])
    c_start = geo.xc[start_y]
    start_theta = float(np.arcsin(np.clip((start_x - c_start) / r_start, -0.85, 0.85)))
    length_mm = float(random.uniform(40.0, 65.0))
    base_angle_deg = float(random.choice([random.uniform(35.0, 65.0), random.uniform(115.0, 145.0)]))
    
    branches = [(start_theta, float(start_y), base_angle_deg, length_mm, length_mm, True)]
    segments = []
    all_cam_pts = []
    
    while branches:
        c_th, c_y, cur_ang, rem_l, tot_l, is_main = branches.pop(0)
        pts = []
        step_mm = 0.8
        while rem_l > 0:
            prog = 1.0 - (rem_l / max(0.1, tot_l))
            rem_l -= step_mm
            iy = int(np.clip(c_y, 0, geo.h - 1))
            r = geo.radius[iy]
            if r <= 10 or abs(c_th) >= np.pi / 2.05:
                break
            cur_ang += np.random.choice([-16, -10, -5, 0, 5, 10, 16]) + np.random.uniform(-3, 3)
            cur_ang = cur_ang * 0.88 + base_angle_deg * 0.12
            rad = np.radians(cur_ang)
            step_px = step_mm * geo.px_per_mm
            c_th += (step_px * np.sin(rad)) / r
            c_y += step_px * np.cos(rad)
            xc, yc, cos_th = geo.surface_to_camera(c_th, c_y)
            if xc is not None and 0 <= xc < geo.w and 0 <= yc < geo.h:
                if geo.mask[int(yc), int(xc)] > 0:
                    pts.append((float(xc), float(yc), float(cos_th), float(prog), is_main))
                    all_cam_pts.append((int(xc), int(yc)))
            if is_main and 0.35 < prog < 0.55 and np.random.rand() < 0.08 and len(branches) == 0:
                fork_ang = cur_ang + np.random.choice([-45, 48])
                branches.append((c_th, c_y, fork_ang, float(np.random.uniform(10.0, 18.0)), 18.0, False))
        if len(pts) > 3:
            segments.append(pts)
            
    if not segments:
        return img_bgr.copy(), np.zeros(img_bgr.shape[:2], dtype=np.uint8), [0, 0, 0, 0]

    out = img_bgr.copy().astype(np.float32)
    defect_mask = np.zeros((geo.h, geo.w), dtype=np.uint8)
    light_dir = np.array([-0.5, -0.8], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    
    for pts in segments:
        for i in range(len(pts) - 1):
            p1, p2 = np.array([pts[i][0], pts[i][1]]), np.array([pts[i+1][0], pts[i+1][1]])
            cos_th, prog, is_main = pts[i][2], pts[i][3], pts[i][4]
            fade = np.sin(prog * np.pi) * (1.0 if is_main else (1.0 - prog))
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            if seg_len < 0.3:
                continue
            tangent = seg_vec / seg_len
            normal_2d = np.array([-tangent[1], tangent[0]])
            if np.dot(normal_2d, light_dir) < 0:
                normal_2d = -normal_2d
            n_micro = max(2, int(seg_len / 0.5))
            for st in range(n_micro):
                mid_pt = p1 + tangent * (seg_len * (st / float(n_micro)))
                offset_px = max(0.35, 0.65 * cos_th)
                hx, hy = int(round(mid_pt[0] + normal_2d[0] * offset_px)), int(round(mid_pt[1] + normal_2d[1] * offset_px))
                sx, sy = int(round(mid_pt[0] - normal_2d[0] * offset_px)), int(round(mid_pt[1] - normal_2d[1] * offset_px))
                if 0 <= hx < geo.w and 0 <= hy < geo.h and geo.mask[hy, hx] > 0:
                    out[hy, hx] = np.clip(out[hy, hx] + 24.0 * fade * max(0.25, cos_th), 0, 255)
                    defect_mask[hy, hx] = 255
                if 0 <= sx < geo.w and 0 <= sy < geo.h and geo.mask[sy, sx] > 0:
                    out[sy, sx] = np.clip(out[sy, sx] * (1.0 - 0.22 * fade * max(0.35, cos_th)), 0, 255)
                    defect_mask[sy, sx] = 255

    arr = np.array(all_cam_pts)
    bx, by = np.min(arr, axis=0)
    bx2, by2 = np.max(arr, axis=0)
    return out.astype(np.uint8), defect_mask, [int(bx), int(by), int(bx2 - bx), int(by2 - by)]


def _inject_macro_rim_crack(geo: Vase3DSurfaceModel, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Dạng nứt to 1: Xẻ từ vành miệng (chuẩn vật lý: lòng rãnh tối sâu đồng nhất, vệt cào xước ngoài mặt men, không chấm trắng trong lòng)"""
    start_y = geo.top_y + 4
    start_theta = float(np.random.uniform(-0.25, 0.25))
    xc0, yc0, cos0 = geo.surface_to_camera(start_theta, start_y)
    if xc0 is None:
        return _inject_small_hairline_crack(geo, img_bgr)
        
    out = img_bgr.copy().astype(np.float32)
    defect_mask = np.zeros((geo.h, geo.w), dtype=np.uint8)
    all_cam_pts = []
    
    base_angle_deg = float(np.random.uniform(12.0, 24.0) * np.random.choice([-1, 1]))
    
    # 1. Vệt cào ma sát men trắng liên tục trên mặt ngoài miệng (Glaze scratch streak)
    sc_len_px = int(8.0 * geo.px_per_mm)
    sc_angle_rad = np.radians(abs(base_angle_deg) - 25.0)
    for st in range(sc_len_px):
        sx = int(xc0 + st * np.sin(sc_angle_rad) * cos0)
        sy = int(yc0 + st * np.cos(sc_angle_rad))
        if 0 <= sx < geo.w and 0 <= sy < geo.h and geo.mask[sy, sx] > 0:
            out[sy, sx] = np.clip(out[sy, sx] * 0.4 + np.array([245.0, 248.0, 252.0]) * 0.6, 0, 255)
            defect_mask[sy, sx] = 255
            all_cam_pts.append((sx, sy))

    # 2. Rãnh nứt toác xẻ dọc sâu xuống cổ
    length_mm = float(np.random.uniform(70.0, 95.0))
    branches = [(start_theta, float(start_y), abs(base_angle_deg), length_mm, length_mm, 3.8, True)]
    branch_draw_data = []
    
    while branches:
        c_th, c_y, cur_ang, rem_l, tot_l, cur_thick, is_main = branches.pop(0)
        pts = []
        step_mm = 1.0
        while rem_l > 0:
            prog = 1.0 - (rem_l / max(0.1, tot_l))
            rem_l -= step_mm
            iy = int(np.clip(c_y, 0, geo.h - 1))
            r = geo.radius[iy]
            if r <= 10 or abs(c_th) >= np.pi / 2.05:
                break
            cur_ang += np.random.choice([-15, -8, 0, 8, 15]) + np.random.uniform(-4, 4)
            cur_ang = cur_ang * 0.86 + abs(base_angle_deg) * 0.14
            rad = np.radians(cur_ang)
            step_px = step_mm * geo.px_per_mm
            c_th += (step_px * np.sin(rad)) / r
            c_y += step_px * np.cos(rad)
            xc, yc, cos_th = geo.surface_to_camera(c_th, c_y)
            if xc is not None and 0 <= xc < geo.w and 0 <= yc < geo.h:
                if geo.mask[int(yc), int(xc)] > 0:
                    w_prog = cur_thick * (1.0 - prog * 0.55)
                    pts.append((float(xc), float(yc), float(cos_th), float(prog), float(w_prog)))
                    all_cam_pts.append((int(xc), int(yc)))
            if is_main and (0.35 < prog < 0.45 or 0.60 < prog < 0.70) and np.random.rand() < 0.12 and len(branches) < 2:
                fork_ang = cur_ang + np.random.choice([-46, 50])
                branches.append((c_th, c_y, fork_ang, float(np.random.uniform(16.0, 26.0)), 26.0, max(1.5, cur_thick * 0.6), False))
        if len(pts) > 3:
            branch_draw_data.append(pts)

    if not branch_draw_data:
        return _inject_small_hairline_crack(geo, img_bgr)

    light_dir = np.array([-0.6, -0.8], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    for pts in branch_draw_data:
        for i in range(len(pts) - 1):
            p1 = np.array([pts[i][0], pts[i][1]])
            p2 = np.array([pts[i+1][0], pts[i+1][1]])
            cos_th = pts[i][2]
            w_px = max(1, int(round(pts[i][4] * max(cos_th, 0.4))))
            
            p1_i = (int(round(p1[0])), int(round(p1[1])))
            p2_i = (int(round(p2[0])), int(round(p2[1])))
            
            # 1. Rãnh tối hốc sâu liên tục (Ambient Occlusion - tuyệt đối không có chấm trắng trong lòng)
            cv2.line(out, p1_i, p2_i, (28.0, 30.0, 32.0), thickness=w_px, lineType=cv2.LINE_AA)
            cv2.line(defect_mask, p1_i, p2_i, 255, thickness=w_px + 2, lineType=cv2.LINE_AA)
            
            # 2. Gờ men bắt sáng nằm hoàn toàn ở mép ngoài rãnh nứt
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            if seg_len > 0.2:
                tangent = seg_vec / seg_len
                normal = np.array([-tangent[1], tangent[0]])
                if np.dot(normal, light_dir) < 0:
                    normal = -normal
                outer_offset = max(1.5, (w_px / 2.0) + 0.8)
                hl_p1 = (int(round(p1[0] + normal[0] * outer_offset)), int(round(p1[1] + normal[1] * outer_offset)))
                hl_p2 = (int(round(p2[0] + normal[0] * outer_offset)), int(round(p2[1] + normal[1] * outer_offset)))
                if 0 <= hl_p1[0] < geo.w and 0 <= hl_p1[1] < geo.h and geo.mask[hl_p1[1], hl_p1[0]] > 0:
                    cv2.line(out, hl_p1, hl_p2, (250.0, 252.0, 245.0), thickness=1, lineType=cv2.LINE_AA)
                    defect_mask[hl_p1[1], hl_p1[0]] = 255

    arr = np.array(all_cam_pts)
    bx, by = np.min(arr, axis=0)
    bx2, by2 = np.max(arr, axis=0)
    return np.clip(out, 0, 255).astype(np.uint8), defect_mask, [int(bx), int(by), int(bx2 - bx), int(by2 - by)]


def _inject_macro_body_crack(geo: Vase3DSurfaceModel, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Dạng nứt to 2: Nứt toác thân bình (chuẩn vật lý: lòng rãnh tối sâu đồng nhất, vệt cào xước ngoài mặt men, không chấm trắng trong lòng)"""
    start_y = int(np.random.uniform(geo.h * 0.30, geo.h * 0.50))
    start_theta = float(np.random.uniform(-0.35, 0.35))
    xc0, yc0, cos0 = geo.surface_to_camera(start_theta, start_y)
    if xc0 is None:
        return _inject_small_hairline_crack(geo, img_bgr)
        
    out = img_bgr.copy().astype(np.float32)
    defect_mask = np.zeros((geo.h, geo.w), dtype=np.uint8)
    all_cam_pts = []
    
    # 1. Vệt cào ma sát men trắng khởi đầu liên tục (Glaze scratch streak)
    base_angle_deg = float(np.random.uniform(45.0, 65.0))
    sc_len = int(12.0 * geo.px_per_mm)
    sc_rad = np.radians(base_angle_deg - 15)
    for st in range(sc_len):
        sx = int(xc0 + st * np.sin(sc_rad) * cos0)
        sy = int(yc0 + st * np.cos(sc_rad))
        if 0 <= sx < geo.w and 0 <= sy < geo.h and geo.mask[sy, sx] > 0:
            out[sy, sx] = np.clip(out[sy, sx] * 0.35 + np.array([245.0, 248.0, 250.0]) * 0.65, 0, 255)
            defect_mask[sy, sx] = 255
            all_cam_pts.append((sx, sy))

    # 2. Rãnh nứt toác thân uốn theo 3D
    length_mm = float(np.random.uniform(80.0, 110.0))
    branches = [(start_theta, float(start_y), base_angle_deg, length_mm, length_mm, 3.5, True)]
    branch_draw_data = []
    
    while branches:
        c_th, c_y, cur_ang, rem_l, tot_l, cur_thick, is_main = branches.pop(0)
        pts = []
        step_mm = 1.0
        while rem_l > 0:
            prog = 1.0 - (rem_l / max(0.1, tot_l))
            rem_l -= step_mm
            iy = int(np.clip(c_y, 0, geo.h - 1))
            r = geo.radius[iy]
            if r <= 10 or abs(c_th) >= np.pi / 2.05:
                break
            cur_ang += np.random.choice([-16, -8, 0, 8, 16]) + np.random.uniform(-4, 4)
            cur_ang = cur_ang * 0.88 + base_angle_deg * 0.12
            rad = np.radians(cur_ang)
            step_px = step_mm * geo.px_per_mm
            c_th += (step_px * np.sin(rad)) / r
            c_y += step_px * np.cos(rad)
            xc, yc, cos_th = geo.surface_to_camera(c_th, c_y)
            if xc is not None and 0 <= xc < geo.w and 0 <= yc < geo.h:
                if geo.mask[int(yc), int(xc)] > 0:
                    w_prog = cur_thick * (1.0 - prog * 0.5)
                    pts.append((float(xc), float(yc), float(cos_th), float(prog), float(w_prog)))
                    all_cam_pts.append((int(xc), int(yc)))
            if is_main and (0.30 < prog < 0.40 or 0.60 < prog < 0.70) and np.random.rand() < 0.14 and len(branches) < 2:
                fork_ang = cur_ang + np.random.choice([-52, 48])
                branches.append((c_th, c_y, fork_ang, float(np.random.uniform(20.0, 35.0)), 35.0, max(1.5, cur_thick * 0.6), False))
        if len(pts) > 3:
            branch_draw_data.append(pts)

    if not branch_draw_data:
        return _inject_small_hairline_crack(geo, img_bgr)

    light_dir = np.array([-0.6, -0.8], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    for pts in branch_draw_data:
        for i in range(len(pts) - 1):
            p1 = np.array([pts[i][0], pts[i][1]])
            p2 = np.array([pts[i+1][0], pts[i+1][1]])
            cos_th = pts[i][2]
            w_px = max(1, int(round(pts[i][4] * max(cos_th, 0.4))))
            
            p1_i = (int(round(p1[0])), int(round(p1[1])))
            p2_i = (int(round(p2[0])), int(round(p2[1])))
            
            # 1. Rãnh tối hốc sâu đồng nhất (Ambient Occlusion - tuyệt đối không có chấm trắng trong lòng)
            cv2.line(out, p1_i, p2_i, (28.0, 30.0, 32.0), thickness=w_px, lineType=cv2.LINE_AA)
            cv2.line(defect_mask, p1_i, p2_i, 255, thickness=w_px + 2, lineType=cv2.LINE_AA)
            
            # 2. Gờ men bắt sáng nằm ở mép ngoài rãnh nứt
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            if seg_len > 0.2:
                tangent = seg_vec / seg_len
                normal = np.array([-tangent[1], tangent[0]])
                if np.dot(normal, light_dir) < 0:
                    normal = -normal
                outer_offset = max(1.5, (w_px / 2.0) + 0.8)
                hl_p1 = (int(round(p1[0] + normal[0] * outer_offset)), int(round(p1[1] + normal[1] * outer_offset)))
                hl_p2 = (int(round(p2[0] + normal[0] * outer_offset)), int(round(p2[1] + normal[1] * outer_offset)))
                if 0 <= hl_p1[0] < geo.w and 0 <= hl_p1[1] < geo.h and geo.mask[hl_p1[1], hl_p1[0]] > 0:
                    cv2.line(out, hl_p1, hl_p2, (250.0, 252.0, 245.0), thickness=1, lineType=cv2.LINE_AA)
                    defect_mask[hl_p1[1], hl_p1[0]] = 255

    arr = np.array(all_cam_pts)
    bx, by = np.min(arr, axis=0)
    bx2, by2 = np.max(arr, axis=0)
    return np.clip(out, 0, 255).astype(np.uint8), defect_mask, [int(bx), int(by), int(bx2 - bx), int(by2 - by)]


def inject_crack(img_bgr: np.ndarray, mask: np.ndarray, geo: Vase3DSurfaceModel = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Cấy vết nứt Nhóm 1 (Đã chuẩn hóa gồm 1 Nhỏ và 2 To):
    - 33% xác suất: Nứt Nhỏ (Hairline Crack 1.5px)
    - 33% xác suất: Nứt To 1 (Rim Macro Crack xẻ miệng)
    - 34% xác suất: Nứt To 2 (Body Macro Crack xẻ thân)
    """
    if geo is None:
        geo = Vase3DSurfaceModel(img_bgr, mask)
        
    choice = random.choice(["small", "macro_rim", "macro_body"])
    if choice == "small":
        return _inject_small_hairline_crack(geo, img_bgr)
    elif choice == "macro_rim":
        return _inject_macro_rim_crack(geo, img_bgr)
    else:
        return _inject_macro_body_crack(geo, img_bgr)


def _inject_edge_corner_chip(geo: Vase3DSurfaceModel, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Dạng 1: Mẻ góc mép viền ngoài (Edge Corner Chip with 3D Wall Thickness)"""
    out = img_bgr.copy().astype(np.float32)
    defect_mask = np.zeros((geo.h, geo.w), dtype=np.uint8)
    
    px_per_mm = geo.px_per_mm
    notch_r_mm = float(np.random.uniform(5.5, 7.5))
    wall_thick_mm = float(np.random.uniform(2.8, 3.5))
    r_px = int(notch_r_mm * px_per_mm)
    thick_px = int(wall_thick_mm * px_per_mm)
    
    loc = random.choice(["rim_left", "rim_right", "foot_left", "foot_right"])
    if loc == "rim_left":
        corner_y = geo.top_y + random.randint(16, 22)
        corner_x = int(geo.xc[corner_y] - geo.radius[corner_y] + 2)
        outward_dir = np.array([-1.0, -0.3])
    elif loc == "rim_right":
        corner_y = geo.top_y + random.randint(16, 22)
        corner_x = int(geo.xc[corner_y] + geo.radius[corner_y] - 2)
        outward_dir = np.array([1.0, -0.3])
    elif loc == "foot_left":
        corner_y = geo.bottom_y - random.randint(20, 28)
        corner_x = int(geo.xc[corner_y] - geo.radius[corner_y] + 3)
        outward_dir = np.array([-1.0, 0.4])
    else: # foot_right
        corner_y = geo.bottom_y - random.randint(20, 28)
        corner_x = int(geo.xc[corner_y] + geo.radius[corner_y] - 3)
        outward_dir = np.array([1.0, 0.4])
        
    outward_dir /= np.linalg.norm(outward_dir)
    
    # 1. Tạo hình học vết mẻ bất quy tắc
    n_pts = 16
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = []
    for a in angles:
        r_noise = 1.0 + np.random.uniform(-0.16, 0.16)
        px = int(corner_x + r_px * np.cos(a) * r_noise)
        py = int(corner_y + r_px * np.sin(a) * r_noise)
        pts.append([px, py])
        
    notch_mask = np.zeros((geo.h, geo.w), dtype=np.uint8)
    cv2.fillPoly(notch_mask, [np.array(pts, dtype=np.int32)], 255)
    
    # 2. Vùng khuyết vỡ văng hẳn ra ngoài không gian
    void_center_x = int(corner_x + outward_dir[0] * (thick_px * 0.7))
    void_center_y = int(corner_y + outward_dir[1] * (thick_px * 0.7))
    void_mask = np.zeros((geo.h, geo.w), dtype=np.uint8)
    cv2.circle(void_mask, (void_center_x, void_center_y), int(r_px * 0.85), 255, -1)
    
    outer_void = (void_mask > 0) & (geo.mask > 0)
    out[outer_void] = [128.0, 128.0, 128.0]
    defect_mask[outer_void] = 255
    
    # 3. Mặt cắt gãy xương gốm 3D
    facet_mask = (notch_mask > 0) & (geo.mask > 0) & (~outer_void)
    light_dir = np.array([-0.6, -0.8], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    
    for py, px in zip(*np.where(facet_mask)):
        dx = (px - corner_x) / max(1.0, r_px)
        dy = (py - corner_y) / max(1.0, r_px)
        dist = np.sqrt(dx*dx + dy*dy)
        
        ripple = np.sin(dist * 16.0) * 5.5
        roughness = np.random.uniform(-3.5, 3.5)
        
        b_val = np.clip(210.0 + ripple + roughness, 185, 235)
        g_val = np.clip(216.0 + ripple + roughness, 190, 240)
        r_val = np.clip(220.0 + ripple + roughness, 195, 245)
        
        pt_vec = np.array([dx, dy], dtype=np.float32)
        dot_l = np.dot(pt_vec, light_dir)
        shadow_fac = max(0.65, 0.88 + dot_l * 0.25)
        b_val *= shadow_fac
        g_val *= shadow_fac
        r_val *= shadow_fac
        
        out[py, px] = [b_val, g_val, r_val]
        defect_mask[py, px] = 255
        
    # 4. Viền men dập tán xạ trắng dọc theo mép men lành
    dilated = cv2.dilate(notch_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    fringe = (dilated > 0) & (notch_mask == 0) & (geo.mask > 0)
    out[fringe] = np.clip(out[fringe] * 0.4 + np.array([245.0, 248.0, 252.0]) * 0.6, 0, 255)
    defect_mask[fringe] = 255
    
    arr_pts = np.where(defect_mask > 0)
    if len(arr_pts[0]) > 0:
        bx = int(arr_pts[1].min() - 4)
        by = int(arr_pts[0].min() - 4)
        bw = int(arr_pts[1].max() - arr_pts[1].min() + 8)
        bh = int(arr_pts[0].max() - arr_pts[0].min() + 8)
        bbox = [max(0, bx), max(0, by), bw, bh]
    else:
        bbox = [corner_x - r_px, corner_y - r_px, r_px * 2, r_px * 2]
        
    return np.clip(out, 0, 255).astype(np.uint8), defect_mask, bbox


def _inject_surface_spall(geo: Vase3DSurfaceModel, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Dạng 2: Tróc vảy men mặt cong thân bình (uốn 3D, bóc hoa văn, sạch viền đen)"""
    out = img_bgr.copy()
    defect_mask = np.zeros((geo.h, geo.w), dtype=np.uint8)
    
    center_y_ratio = float(np.random.uniform(0.32, 0.62))
    center_theta = float(np.random.uniform(-0.35, 0.35))
    spall_w_mm = float(np.random.uniform(13.0, 18.0))
    spall_h_mm = float(np.random.uniform(9.0, 14.0))
    
    cy = int(geo.h * center_y_ratio)
    xc, yc, cos_th = geo.surface_to_camera(center_theta, cy)
    if xc is None or cos_th < 0.2:
        return _inject_edge_corner_chip(geo, img_bgr)
        
    rx_px = int(spall_w_mm * 0.5 * geo.px_per_mm * cos_th)
    ry_px = int(spall_h_mm * 0.5 * geo.px_per_mm)
    
    spall_mask = np.zeros((geo.h, geo.w), dtype=np.uint8)
    n_verts = 14
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    pts = []
    for a in angles:
        r_noise = 1.0 + np.random.uniform(-0.24, 0.24)
        px = int(xc + rx_px * np.cos(a) * r_noise)
        py = int(yc + ry_px * np.sin(a) * r_noise)
        pts.append([px, py])
    cv2.fillPoly(spall_mask, [np.array(pts, dtype=np.int32)], 255)
    spall_mask = cv2.bitwise_and(spall_mask, geo.mask)
    
    y_coords, x_coords = np.where(spall_mask > 0)
    if len(y_coords) == 0:
        return _inject_edge_corner_chip(geo, img_bgr)
        
    for py, px in zip(y_coords, x_coords):
        dx = (px - xc) / max(1.0, rx_px)
        dy = (py - yc) / max(1.0, ry_px)
        dist = np.sqrt(dx*dx + dy*dy)
        
        ripple = np.sin(dist * 14.0 + dx * 3.0) * 5.5
        roughness = np.random.uniform(-4.0, 4.0)
        
        b_val = np.clip(210.0 + ripple + roughness, 195, 230)
        g_val = np.clip(216.0 + ripple + roughness, 200, 235)
        r_val = np.clip(220.0 + ripple + roughness, 205, 240)
        
        out[py, px] = [int(b_val), int(g_val), int(r_val)]
        defect_mask[py, px] = 255
        
    # Viền bột men trắng tán xạ vi mô 1px ở mép ngoài (sạch hoàn toàn viền đen bóng đổ)
    eroded = cv2.erode(spall_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    rim = (spall_mask > 0) & (eroded == 0)
    out[rim] = np.clip(out[rim].astype(np.float32) * 0.6 + np.array([245.0, 248.0, 252.0]) * 0.4, 0, 255).astype(np.uint8)
    defect_mask[rim] = 255

    bx = int(xc - rx_px * 1.25)
    by = int(yc - ry_px * 1.25)
    bw = int(rx_px * 2.5)
    bh = int(ry_px * 2.5)
    bbox = [max(0, bx), max(0, by), bw, bh]
    return out, defect_mask, bbox


def inject_chip(img_bgr: np.ndarray, mask: np.ndarray, geo: Vase3DSurfaceModel = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Cấy vết sứt mẻ Nhóm 2 (Đã chuẩn hóa gồm 2 dạng vật lý chuẩn xác):
    - 50% xác suất: Mẻ góc mép viền ngoài (Edge Corner Chip with 3D Wall Thickness)
    - 50% xác suất: Tróc vảy men mặt cong thân bình (Body Surface Spall)
    """
    if geo is None:
        geo = Vase3DSurfaceModel(img_bgr, mask)
        
    if random.random() < 0.5:
        return _inject_edge_corner_chip(geo, img_bgr)
    else:
        return _inject_surface_spall(geo, img_bgr)


def _inject_tea_coffee_stain(geo: Vase3DSurfaceModel, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Màu 1: Ố nâu cánh gián (Trà / Cà phê) - Hiệu ứng Coffee-ring viền đậm lòng nhạt"""
    out = img_bgr.copy().astype(np.float32)
    center_y_ratio = float(np.random.uniform(0.35, 0.75))
    center_theta = float(np.random.uniform(-0.35, 0.35))
    size_mm = float(np.random.uniform(18.0, 26.0))
    
    cy = int(geo.h * center_y_ratio)
    xc, yc, cos_th = geo.surface_to_camera(center_theta, cy)
    if xc is None:
        return img_bgr.copy(), np.zeros((geo.h, geo.w), dtype=np.uint8), [0, 0, 0, 0]
        
    rx_px = int(size_mm * 0.5 * geo.px_per_mm * cos_th)
    ry_px = int(size_mm * 0.45 * geo.px_per_mm)
    
    blob = np.zeros((geo.h, geo.w), dtype=np.float32)
    n_pts = 16
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = []
    for a in angles:
        r_noise = 1.0 + np.random.uniform(-0.25, 0.25)
        px = int(xc + rx_px * np.cos(a) * r_noise)
        py = int(yc + ry_px * np.sin(a) * r_noise)
        pts.append([px, py])
    poly_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(blob, [poly_arr], 1.0)
    blob = cv2.GaussianBlur(blob, (15, 15), 0)
    blob[geo.mask == 0] = 0.0
    
    eroded = cv2.erode(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ring = np.clip(blob - eroded * 0.65, 0.0, 1.0)
    alpha_blend = np.clip(ring * 0.72 + blob * 0.25, 0.0, 0.85)
    
    tea_color = np.array([35.0, 75.0, 130.0], dtype=np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - alpha_blend) + tea_color[c] * alpha_blend
        
    defect_mask = (alpha_blend > 0.12).astype(np.uint8) * 255
    defect_mask[geo.mask == 0] = 0
    
    arr_pts = np.where(defect_mask > 0)
    if len(arr_pts[0]) > 0:
        bx = int(arr_pts[1].min() - 4)
        by = int(arr_pts[0].min() - 4)
        bw = int(arr_pts[1].max() - arr_pts[1].min() + 8)
        bh = int(arr_pts[0].max() - arr_pts[0].min() + 8)
        bbox = [max(0, bx), max(0, by), bw, bh]
    else:
        bbox = [int(xc - rx_px), int(yc - ry_px), int(rx_px * 2), int(ry_px * 2)]
        
    return np.clip(out, 0, 255).astype(np.uint8), defect_mask, bbox


def _inject_marker_stroke(geo: Vase3DSurfaceModel, img_bgr: np.ndarray, color_type: str = "blue") -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Màu 2 & 3: Vết bút lông dầu bán trong suốt (Xanh dương hoặc Đỏ)"""
    out = img_bgr.copy().astype(np.float32)
    center_y_ratio = float(np.random.uniform(0.30, 0.65))
    center_theta = float(np.random.uniform(-0.30, 0.30))
    
    cy = int(geo.h * center_y_ratio)
    xc, yc, cos_th = geo.surface_to_camera(center_theta, cy)
    if xc is None:
        return img_bgr.copy(), np.zeros((geo.h, geo.w), dtype=np.uint8), [0, 0, 0, 0]
        
    if color_type == "blue":
        ink_color = np.array([170.0, 60.0, 25.0], dtype=np.float32)
        max_alpha = 0.80
    else: # red
        ink_color = np.array([25.0, 30.0, 205.0], dtype=np.float32)
        max_alpha = 0.82
        
    path_mask = np.zeros((geo.h, geo.w), dtype=np.float32)
    pts = []
    n_steps = 25
    r_px = int(float(np.random.uniform(12.0, 16.0)) * geo.px_per_mm)
    start_ang = float(np.random.uniform(-np.pi, 0))
    
    for t in range(n_steps):
        frac = t / float(n_steps)
        ang = start_ang + frac * 2.2 * np.pi
        rad = r_px * (0.8 + 0.3 * np.sin(frac * np.pi))
        px = int(xc + rad * np.cos(ang) * cos_th)
        py = int(yc + rad * np.sin(ang) * 0.75 + frac * 15.0)
        pts.append((px, py))
        
    pen_w_px = max(3, int(float(np.random.uniform(2.5, 3.5)) * geo.px_per_mm))
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i+1]
        prog = i / float(len(pts) - 1)
        w_cur = max(2, int(pen_w_px * (1.1 - 0.3 * prog)))
        cv2.line(path_mask, p1, p2, 1.0, thickness=w_cur, lineType=cv2.LINE_AA)
        
    path_mask = cv2.GaussianBlur(path_mask, (3, 3), 0)
    path_mask[geo.mask == 0] = 0.0
    
    alpha_blend = np.clip(path_mask * max_alpha, 0.0, 1.0)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - alpha_blend) + ink_color[c] * alpha_blend
        
    defect_mask = (alpha_blend > 0.15).astype(np.uint8) * 255
    defect_mask[geo.mask == 0] = 0
    
    arr_pts = np.where(defect_mask > 0)
    if len(arr_pts[0]) > 0:
        bx = int(arr_pts[1].min() - 4)
        by = int(arr_pts[0].min() - 4)
        bw = int(arr_pts[1].max() - arr_pts[1].min() + 8)
        bh = int(arr_pts[0].max() - arr_pts[0].min() + 8)
        bbox = [max(0, bx), max(0, by), bw, bh]
    else:
        bbox = [int(xc - 20), int(yc - 20), 40, 40]
        
    return np.clip(out, 0, 255).astype(np.uint8), defect_mask, bbox


def _inject_grime_smudge(geo: Vase3DSurfaceModel, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Màu 4: Vệt quẹt xám đen cặn bẩn bìa carton / dầu máy có sọc ma sát"""
    out = img_bgr.copy().astype(np.float32)
    center_y_ratio = float(np.random.uniform(0.45, 0.75))
    center_theta = float(np.random.uniform(-0.30, 0.30))
    size_mm = float(np.random.uniform(20.0, 30.0))
    
    cy = int(geo.h * center_y_ratio)
    xc, yc, cos_th = geo.surface_to_camera(center_theta, cy)
    if xc is None:
        return img_bgr.copy(), np.zeros((geo.h, geo.w), dtype=np.uint8), [0, 0, 0, 0]
        
    rx_px = int(size_mm * 0.5 * geo.px_per_mm * cos_th)
    ry_px = int(size_mm * 0.35 * geo.px_per_mm)
    
    smudge = np.zeros((geo.h, geo.w), dtype=np.float32)
    angle = float(np.random.uniform(0, 30))
    cv2.ellipse(smudge, (int(xc), int(yc)), (rx_px, ry_px), angle, 0, 360, 1.0, -1)
    
    noise = np.random.uniform(0.5, 1.0, (geo.h, geo.w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (1, 15), 0)
    smudge = smudge * noise
    smudge = cv2.GaussianBlur(smudge, (11, 7), 0)
    smudge[geo.mask == 0] = 0.0
    
    alpha_blend = np.clip(smudge * 0.60, 0.0, 0.70)
    grime_color = np.array([55.0, 60.0, 65.0], dtype=np.float32)
    
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - alpha_blend) + grime_color[c] * alpha_blend
        
    defect_mask = (alpha_blend > 0.12).astype(np.uint8) * 255
    defect_mask[geo.mask == 0] = 0
    
    arr_pts = np.where(defect_mask > 0)
    if len(arr_pts[0]) > 0:
        bx = int(arr_pts[1].min() - 4)
        by = int(arr_pts[0].min() - 4)
        bw = int(arr_pts[1].max() - arr_pts[1].min() + 8)
        bh = int(arr_pts[0].max() - arr_pts[0].min() + 8)
        bbox = [max(0, bx), max(0, by), bw, bh]
    else:
        bbox = [int(xc - rx_px), int(yc - ry_px), int(rx_px * 2), int(ry_px * 2)]
        
    return np.clip(out, 0, 255).astype(np.uint8), defect_mask, bbox


def inject_stain(img_bgr: np.ndarray, mask: np.ndarray, geo: Vase3DSurfaceModel = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Cấy vết ố bẩn Nhóm 3 (Đã chuẩn hóa gồm 4 dải màu quang học chuẩn xác):
    - 25% xác suất: Ố nâu cánh gián (Trà / Cà phê với Coffee-ring effect)
    - 25% xác suất: Mực xanh dương (Bút dạ kiểm kho)
    - 25% xác suất: Mực đỏ tươi (Bút lông đánh dấu lỗi)
    - 25% xác suất: Vệt quẹt xám đen (Bụi bẩn carton / Dầu máy)
    """
    if geo is None:
        geo = Vase3DSurfaceModel(img_bgr, mask)
        
    choice = random.choice(["tea", "blue", "red", "grime"])
    if choice == "tea":
        return _inject_tea_coffee_stain(geo, img_bgr)
    elif choice == "blue":
        return _inject_marker_stroke(geo, img_bgr, color_type="blue")
    elif choice == "red":
        return _inject_marker_stroke(geo, img_bgr, color_type="red")
    else:
        return _inject_grime_smudge(geo, img_bgr)


def build_synthetic_dataset(
    ghost_overlays_dir: str,
    output_dir: str,
    samples_per_vase: int = 2,
    target_vase_ids: Optional[List[str]] = None,
    clean_output: bool = False
) -> str:
    """
    Đọc trực tiếp các file ảnh đã tách nền _nb.png từ data/ghost_overlays/
    Lấy kênh Alpha làm mask chuẩn 100%, cấy lỗi hoàn toàn bên trong thân bình.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_out_dir = os.path.join(output_dir, "synthetic_images")
    os.makedirs(images_out_dir, exist_ok=True)

    if clean_output:
        print(f"[CLEAN] Đang dọn sạch toàn bộ ảnh cũ trong: {images_out_dir}")
        for fname in os.listdir(images_out_dir):
            fpath = os.path.join(images_out_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        print("[CLEAN] Đã dọn sạch thư mục synthetic_images.")

    catalog = []
    view_specs = [
        ("01_front", "truoc_nb.png"),
        ("02_back", "sau_nb.png"),
        ("03_left", "trai_nb.png"),
        ("04_right", "phai_nb.png")
    ]

    print(f"[GEN] Bắt đầu sinh dữ liệu khuyết tật từ ảnh tách nền (_nb.png) tại: {ghost_overlays_dir}")
    vase_folders = sorted([d for d in os.listdir(ghost_overlays_dir) if os.path.isdir(os.path.join(ghost_overlays_dir, d))])
    if target_vase_ids is not None:
        vase_folders = [d for d in vase_folders if d in target_vase_ids]
        print(f"[GEN] Chỉ định lọc mã bình: {vase_folders}")

    total_generated = 0
    for vase_id in vase_folders:
        vase_path = os.path.join(ghost_overlays_dir, vase_id)

        for view_name, nb_filename in view_specs:
            nb_path = os.path.join(vase_path, nb_filename)
            if not os.path.exists(nb_path):
                continue

            im_rgba = cv2.imread(nb_path, cv2.IMREAD_UNCHANGED)
            if im_rgba is None or len(im_rgba.shape) != 3:
                continue

            if im_rgba.shape[2] == 4:
                bgr = im_rgba[:, :, :3]
                alpha = im_rgba[:, :, 3]
            else:
                bgr = im_rgba
                alpha = np.ones((bgr.shape[0], bgr.shape[1]), dtype=np.uint8) * 255

            mask = (alpha > 200).astype(np.uint8) * 255
            inner_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

            geo = Vase3DSurfaceModel(bgr, mask)

            defect_configs = [
                ("crack", 1, lambda b, m: inject_crack(b, m, geo)),
                ("chip", 2, lambda b, m: inject_chip(b, m, geo)),
                ("stain", 3, lambda b, m: inject_stain(b, m, geo))
            ]

            for defect_name, class_id, defect_func in defect_configs:
                for rep in range(samples_per_vase):
                    syn_bgr, d_mask, bbox = defect_func(bgr, inner_mask)
                    bx, by, bw, bh = bbox
                    if bw < 5 or bh < 5:
                        continue

                    alpha_out = alpha.copy()
                    if defect_name == "chip":
                        alpha_out[(syn_bgr == 128).all(axis=2) & (d_mask > 0)] = 0

                    syn_rgba = np.dstack([syn_bgr, alpha_out])

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
                        "image_size": [bgr.shape[1], bgr.shape[0]]
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
    import argparse
    parser = argparse.ArgumentParser(description="Sinh dữ liệu khuyết tật 3D cho bình gốm")
    parser.add_argument("--vase", type=str, default=None, help="Mã bình cần sinh (ví dụ: B1). Mặc định sinh tất cả.")
    parser.add_argument("--samples", type=int, default=2, help="Số lượng mẫu cho mỗi loại lỗi trên mỗi góc chụp")
    parser.add_argument("--clean", action="store_true", help="Xóa sạch toàn bộ ảnh cũ trong synthetic_images trước khi sinh mới")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ghost_dir = os.path.join(base_dir, "data", "ghost_overlays")
    out_dir = os.path.join(base_dir, "training")

    target_vases = [args.vase] if args.vase else None
    build_synthetic_dataset(ghost_dir, out_dir, samples_per_vase=args.samples, target_vase_ids=target_vases, clean_output=args.clean)
