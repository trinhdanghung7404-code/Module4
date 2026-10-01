"""DamageComparator: Main orchestrator for ceramic damage detection pipeline v2.
Integrates Step 1 (SuperPoint ROI), Step 2 (MNN+RANSAC Mesh), and Step 3 (2-Layer Canonical Inspection).
"""

import time
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import config
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from utils.scoring import calculate_damage_score
from utils.visualization import DebugVisualizer
from step3_pipeline import (
    canonical_triangle_patch,
    evaluate_triangle_structure,
    evaluate_triangle_color,
)


class DamageComparator:
    """10-step damage comparison pipeline with 2-layer inspection."""

    def __init__(self):
        self.segmenter = ObjectSegmenter()
        self.normalizer = ImageNormalizer()
        self.registrator = ImageRegistration()
        self.mesh_builder = MeshBuilder()
        self.visualizer = DebugVisualizer()

    def compare(self, product_path: str, return_path: str) -> dict:
        """Run full 2-layer damage comparison pipeline using detailed_debug engine."""
        import shutil
        from detailed_debug import run_detailed_debug

        start_time = time.time()
        p_name = os.path.splitext(os.path.basename(product_path))[0]
        r_name = os.path.splitext(os.path.basename(return_path))[0]
        v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(v2_dir, "debug", f"compare_{p_name}_vs_{r_name}")
        latest_dir = os.path.join(v2_dir, "debug", "latest_run")

        res = run_detailed_debug(product_path, return_path, base_debug_dir=out_dir)

        try:
            if os.path.exists(latest_dir):
                shutil.rmtree(latest_dir)
            shutil.copytree(out_dir, latest_dir)
        except Exception:
            pass

        tri_cnt = max(res["triangles"], 1)
        dmg_cnt = res["total_defects"]
        damage_score = (dmg_cnt / tri_cnt) * 100.0
        is_damaged = dmg_cnt > 0

        debug_paths = {
            "fusion": os.path.join(out_dir, "05_fusion", "03_side_by_side_marked.jpg"),
            "overlay_product": os.path.join(out_dir, "05_fusion", "01_product_damage_marked.jpg"),
            "overlay_return": os.path.join(out_dir, "05_fusion", "02_return_damage_marked.jpg"),
            "crops": os.path.join(out_dir, "06_defect_crops"),
            "debug_dir": out_dir,
            "latest_dir": latest_dir,
        }

        pipeline_summary = {
            "Total Triangles": res["triangles"],
            "Layer 1 Defects (Crack/Broken)": res["l1_defects"],
            "Layer 2 Defects (Color/Enamel)": res["l2_defects"],
            "Total Defect Triangles": dmg_cnt,
            "Output Directory": out_dir,
            "Latest Run Alias": latest_dir,
        }

        return {
            "damage_score": damage_score,
            "damage_area": dmg_cnt * 100,
            "largest_damage_area": 100 if is_damaged else 0,
            "damage_mask": np.zeros((10, 10), dtype=np.uint8),
            "is_damaged": is_damaged,
            "processing_time": time.time() - start_time,
            "debug_paths": debug_paths,
            "debug_image_paths": debug_paths,
            "pipeline_summary": pipeline_summary,
            "triangles_count": res["triangles"],
            "damaged_triangles_count": dmg_cnt,
            "l1_defects": res["l1_defects"],
            "l2_defects": res["l2_defects"],
        }

        # Step 1: Load images
        product_img = cv2.imread(product_path)
        return_img = cv2.imread(return_path)
        if product_img is None:
            raise FileNotFoundError(f"Cannot load product image: {product_path}")
        if return_img is None:
            raise FileNotFoundError(f"Cannot load return image: {return_path}")

        print(f"[1/10] Loaded images: {product_img.shape} vs {return_img.shape}")

        # Step 2: Segment
        product_seg = self.segmenter.segment(product_img)
        return_seg = self.segmenter.segment(return_img)
        print(f"[2/10] Segmented: product={product_seg['area']:.1f}px, return={return_seg['area']:.1f}px")

        # Step 3: Normalize
        return_normalized = self.normalizer.normalize(
            return_img, product_img, return_seg["mask"], product_seg["mask"]
        )
        print("[3/10] Normalized return image")

        # Step 4: Register (SuperPoint + MNN + RANSAC)
        reg_result = self.registrator.register(
            product_img, return_normalized,
            product_seg["mask"], return_seg["mask"]
        )
        print(f"[4/10] Registration: {reg_result['inlier_count']} inliers / {reg_result['total_matches']} matches")

        # Step 5: Mesh
        vertices, triangles = self.mesh_builder.build(
            reg_result["product_points"],
            reg_result["return_points"],
            np.arange(len(reg_result["product_points"]))
        )
        print(f"[5/10] Mesh: {len(vertices)} vertices, {len(triangles)} triangles")

        # Calculate global color cast offset between product and return on object mask
        p_lab_full = cv2.cvtColor(product_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_lab_full = cv2.cvtColor(return_normalized, cv2.COLOR_BGR2LAB).astype(np.float32)
        p_valid = product_seg["mask"] > 0
        r_valid = return_seg["mask"] > 0
        offset_a = float(np.median(r_lab_full[..., 1][r_valid]) - np.median(p_lab_full[..., 1][p_valid]))
        offset_b = float(np.median(r_lab_full[..., 2][r_valid]) - np.median(p_lab_full[..., 2][p_valid]))

        # Step 6 & 7: 2-Layer Canonical Triangle Inspection
        struct_scores = []
        color_scores = []

        for tri in triangles:
            pts_prod = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
            pts_ret = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)

            patch_p, mask_p = canonical_triangle_patch(product_img, pts_prod, target_size=72)
            patch_r, mask_r = canonical_triangle_patch(return_normalized, pts_ret, target_size=72)

            s_eval = evaluate_triangle_structure(
                patch_p, patch_r, mask_p,
                tri.vertex_indices, vertices,
                reg_result["product_descriptors"], reg_result["return_descriptors"]
            )
            struct_scores.append(s_eval)

            c_eval = evaluate_triangle_color(patch_p, patch_r, mask_p, (offset_a, offset_b))
            color_scores.append(c_eval)

        # Statistical Outlier Detection with Absolute Thresholds
        all_s_scores = [s["structure_score"] for s in struct_scores]
        s_mean, s_std = np.mean(all_s_scores), np.std(all_s_scores)

        all_c_diffs = [c["mean_chroma_diff"] for c in color_scores]
        c_mean, c_std = np.mean(all_c_diffs), np.std(all_c_diffs)

        layer1_damaged = set()
        layer2_damaged = set()

        for i in range(len(triangles)):
            z_s = (all_s_scores[i] - s_mean) / (s_std + 1e-8)
            # Require both z-score outlier AND low absolute score (< 0.65)
            if z_s < -2.2 and all_s_scores[i] < 0.65:
                layer1_damaged.add(i)

            z_c = (all_c_diffs[i] - c_mean) / (c_std + 1e-8)
            # Require both z-score outlier AND real chroma diff (> 18.0)
            if z_c > 2.5 and all_c_diffs[i] > 18.0:
                layer2_damaged.add(i)

        print(f"[6/10] Structure Layer: {len(layer1_damaged)}/{len(triangles)} triangles flagged")
        print(f"[7/10] Appearance Layer: {len(layer2_damaged)}/{len(triangles)} triangles flagged")

        # Step 8: Fusion
        fused_damaged = layer1_damaged.union(layer2_damaged)
        print(f"[8/10] Fusion: {len(fused_damaged)}/{len(triangles)} triangles confirmed")

        # Step 9: Build pixel-level damage mask
        h, w = product_img.shape[:2]
        damage_mask = np.zeros((h, w), dtype=np.uint8)
        total_damage_area = 0
        largest_damage_area = 0

        for idx in fused_damaged:
            tri = triangles[idx]
            pts = np.array([list(vertices[v_idx].product_xy) for v_idx in tri.vertex_indices], dtype=np.int32)
            tri_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, pts, 255)
            area = int(np.count_nonzero(tri_mask))
            total_damage_area += area
            largest_damage_area = max(largest_damage_area, area)
            damage_mask = cv2.bitwise_or(damage_mask, tri_mask)

        # Step 10: Score
        pattern_mesh_area = 0
        for tri in triangles:
            pts = np.array([list(vertices[v_idx].product_xy) for v_idx in tri.vertex_indices], dtype=np.int32)
            pattern_mesh_area += int(cv2.contourArea(pts))

        base_area = pattern_mesh_area if pattern_mesh_area > 0 else product_seg["area"]
        damage_score = (total_damage_area / max(base_area, 1)) * 100.0
        # If no damaged triangles, score is strictly 0.0%
        if len(fused_damaged) == 0:
            damage_score = 0.0

        is_damaged = damage_score >= 1.5 or len(fused_damaged) >= 5
        print(f"[9/10] Validation: {len(fused_damaged)} damaged triangles, {total_damage_area}px total")
        print(f"[10/10] Damage Score: {damage_score:.2f}% | Is Damaged: {'YES' if is_damaged else 'NO'}")

        processing_time = time.time() - start_time

        # Save Visual Outputs to v2/debug_outputs/
        try:
            debug_paths = self.visualizer.generate_all(
                product_img=product_img,
                return_img=return_img,
                return_normalized=return_normalized,
                product_mask=product_seg["mask"],
                return_mask=return_seg["mask"],
                product_points=reg_result["product_points"],
                return_points=reg_result["return_points"],
                vertices=vertices,
                triangles=triangles,
                structure_scores=[{"is_damaged": i in layer1_damaged} for i in range(len(triangles))],
                appearance_scores=[{"is_damaged": i in layer2_damaged} for i in range(len(triangles))],
                damage_mask=damage_mask,
                damage_score=damage_score
            )
        except Exception as e:
            print(f"Warning: Debug visualization failed: {e}")
            debug_paths = {}

        return {
            "damage_score": damage_score,
            "damage_area": total_damage_area,
            "largest_damage_area": largest_damage_area,
            "damage_mask": damage_mask,
            "is_damaged": is_damaged,
            "processing_time": processing_time,
            "debug_paths": debug_paths,
            "triangles_count": len(triangles),
            "damaged_triangles_count": len(fused_damaged),
        }
