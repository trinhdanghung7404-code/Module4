"""Adaptive Mesh Region Matching — replaces Local Window Matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mesh_builder import MeshBuilder, MeshTriangle, MeshVertex
from mesh_config import (
    DEBUG,
    MAX_TOTAL_TRIANGLES,
    TOP_K,
    MAX_DEPTH,
    MIN_TRIANGLE_AREA,
    SIMILARITY_THRESHOLD,
    FEATURE_THRESHOLD,
    SSIM_THRESHOLD,
    MIN_AREA_THRESHOLD,
    DIFF_THRESHOLD,
    SIMILARITY_WORKERS,
)
from mesh_refiner import MeshRefiner
from priority_scheduler import PriorityMeshScheduler
from region_extractor import RegionExtractor
from region_similarity import RegionSimilarity
from similarity_cache import ScoredTriangle
from triangle_similarity import TriangleSimilarity
from triangle_refiner import TriangleRefiner

MESH_DAMAGE_DETECTOR_VERSION = "product_texture_suppression_v6"
@dataclass
class TriangleResult:
    triangle: MeshTriangle
    similarity: float
    area: float
    metric_scores: Dict[str, float]
    feature_similarity: float
    is_damaged: bool


class MeshDamageDetector:
    """Detect damage by comparing Delaunay mesh regions between aligned images."""
    
    SSIM_WEIGHT_HIGH = 0.90
    SSIM_WEIGHT_MEDIUM = 0.80
    SSIM_WEIGHT_LOW = 0.70

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_depth: int = MAX_DEPTH,
        min_triangle_area: float = MIN_TRIANGLE_AREA,
        diff_threshold: int = DIFF_THRESHOLD,
        refinement_strategy: str = MeshRefiner.STRATEGY_MIDPOINT,
        max_total_triangles: int = MAX_TOTAL_TRIANGLES,
        top_k: int = TOP_K,
        workers: int = SIMILARITY_WORKERS,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_depth = max_depth
        self.min_triangle_area = min_triangle_area
        self.diff_threshold = diff_threshold
        self.max_total_triangles = max_total_triangles
        self.top_k = top_k
        self.workers = workers

        self.feature_threshold = FEATURE_THRESHOLD
        self.ssim_threshold = SSIM_THRESHOLD
        self.min_area_threshold = MIN_AREA_THRESHOLD

        # Compatibility fields consumed by comparator debug paths.
        self.patch_size = 31
        self.window_size = 41

        # Lazy initialization
        self._scheduler = None
        self._triangle_similarity = None

    def _get_scheduler(self) -> PriorityMeshScheduler:
        """Lazy-initialize the scheduler."""
        if self._scheduler is None:
            mesh_builder = MeshBuilder()
            triangle_refiner = TriangleRefiner()
            triangle_similarity = TriangleSimilarity(
                similarity_threshold=self.similarity_threshold,
                feature_threshold=self.feature_threshold,
                ssim_threshold=self.ssim_threshold,
                workers=self.workers,
            )
            self._scheduler = PriorityMeshScheduler(
                mesh_builder=mesh_builder,
                triangle_similarity=triangle_similarity,
                triangle_refiner=triangle_refiner,
                similarity_threshold=self.similarity_threshold,
                min_triangle_area=self.min_triangle_area,
                max_depth=self.max_depth,
                max_total_triangles=self.max_total_triangles,
                top_k=self.top_k,
            )
        return self._scheduler

    def _log_step(self, message: str) -> None:
        print(f"[MeshDamageDetector] {message}")

    def _score_ssim_weight(self, similarity: float) -> Tuple[float, bool, str]:
        if similarity >= self.SSIM_WEIGHT_HIGH:
            return 1.0, True, "weight=1.0"
        if similarity >= self.SSIM_WEIGHT_MEDIUM:
            return 0.7, True, "weight=0.7"
        if similarity >= self.SSIM_WEIGHT_LOW:
            return 0.4, True, "weight=0.4"
        return 0.0, False, "rejected"

    def _build_paired_keypoints(
        self,
        product_keypoints: Sequence[Sequence[float]],
        return_keypoints: Sequence[Sequence[float]],
        inlier_matches: Sequence,
        product_image: np.ndarray,
    ) -> Tuple[List[List[float]], List[List[float]], List[Optional[Tuple[int, int]]]]:
        """Convert SuperPoint inlier matches into paired mesh vertices.

        Important: do NOT add identity grid points here. A grid point has no
        measured correspondence, so mapping product (x, y) to return (x, y)
        can be wrong when the return image is shifted by even a few pixels.
        Only real SuperPoint inlier correspondences are used as mesh vertices.
        """
        product_points: List[List[float]] = []
        return_points: List[List[float]] = []
        keypoint_indices: List[Optional[Tuple[int, int]]] = []

        seen_pairs = set()
        for match in inlier_matches:
            if match.queryIdx >= len(product_keypoints) or match.trainIdx >= len(return_keypoints):
                continue

            pair = (int(match.queryIdx), int(match.trainIdx))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            product_points.append([
                float(product_keypoints[match.queryIdx][0]),
                float(product_keypoints[match.queryIdx][1]),
            ])
            return_points.append([
                float(return_keypoints[match.trainIdx][0]),
                float(return_keypoints[match.trainIdx][1]),
            ])
            keypoint_indices.append(pair)

        return product_points, return_points, keypoint_indices



    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale uint8."""
        if image.ndim == 2:
            return image.astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _gradient_magnitude_u8(self, gray: np.ndarray) -> np.ndarray:
        """Scharr gradient magnitude normalized to uint8."""
        src = gray.astype(np.float32)
        grad_x = cv2.Scharr(src, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(src, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(grad_x, grad_y)
        max_val = float(np.max(mag)) if mag.size else 0.0
        if max_val <= 1e-6:
            return np.zeros(gray.shape[:2], dtype=np.uint8)
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _remove_small_components(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """Remove tiny connected components from a binary mask."""
        binary = (mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            return binary

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= int(min_area):
                cleaned[labels == label] = 255
        return cleaned

    def _local_illumination_normalize(
        self,
        product_gray: np.ndarray,
        return_gray: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """Normalize return brightness/contrast to product inside one triangle."""
        product_values = product_gray[active].astype(np.float32)
        return_values = return_gray[active].astype(np.float32)
        if product_values.size < 8 or return_values.size < 8:
            return return_gray.copy()

        p_mean = float(product_values.mean())
        p_std = float(product_values.std())
        r_mean = float(return_values.mean())
        r_std = float(return_values.std())

        ret = return_gray.astype(np.float32)
        if r_std <= 1e-6 or p_std <= 1e-6:
            norm = ret + (p_mean - r_mean)
        else:
            scale = float(np.clip(p_std / r_std, 0.65, 1.55))
            norm = (ret - r_mean) * scale + p_mean
        return np.clip(norm, 0, 255).astype(np.uint8)

    def _odd_kernel(self, value: int, limit: int) -> int:
        """Return an odd Gaussian kernel size not larger than image extent."""
        k = max(3, int(value))
        if k % 2 == 0:
            k += 1
        max_k = max(3, int(limit))
        if max_k % 2 == 0:
            max_k -= 1
        return max(3, min(k, max_k))

    def _dilate_binary(self, mask: np.ndarray, ksize: int = 5, iterations: int = 1) -> np.ndarray:
        """Dilate a binary mask."""
        binary = (mask > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.dilate(binary, kernel, iterations=iterations)

    def _compute_triangle_pixel_damage(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        triangle_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Compute product-texture-suppressed pixel damage inside one candidate triangle.

        v6 idea:
        - Brightness difference alone is not damage.
        - Difference on product's existing ornament/texture is suspicious but weak.
        - Strong evidence is a new edge/color in return that product does not have nearby.
        """
        active = triangle_mask > 0
        shape = triangle_mask.shape
        empty = np.zeros(shape, dtype=np.uint8)
        if not np.any(active):
            return empty, {
                'raw_residual': empty,
                'texture': empty,
                'edge': empty,
                'chroma': empty,
                'illumination_reject': empty,
                'product_texture': empty,
                'new_edge': empty,
                'chroma_new': empty,
                'texture_suppressed': empty,
            }

        # 1) Local illumination normalization.
        product_gray = cv2.GaussianBlur(self._to_gray(product_image), (3, 3), 0)
        return_gray = cv2.GaussianBlur(self._to_gray(return_image), (3, 3), 0)
        return_norm = self._local_illumination_normalize(product_gray, return_gray, active)
        raw_residual = cv2.absdiff(product_gray, return_norm)

        # 2) High-pass texture after removing slow illumination.
        min_extent = min(product_gray.shape[:2])
        hp_kernel = self._odd_kernel(31, min_extent)
        product_base = cv2.GaussianBlur(product_gray, (hp_kernel, hp_kernel), 0).astype(np.float32)
        return_base = cv2.GaussianBlur(return_norm, (hp_kernel, hp_kernel), 0).astype(np.float32)
        product_hp = product_gray.astype(np.float32) - product_base
        return_hp = return_norm.astype(np.float32) - return_base
        texture_diff = np.clip(np.abs(product_hp - return_hp), 0, 255).astype(np.uint8)

        product_edge = self._gradient_magnitude_u8(product_hp)
        return_edge = self._gradient_magnitude_u8(return_hp)
        edge_diff = cv2.absdiff(product_edge, return_edge)

        # 3) Chroma difference in Lab a/b. Ignore L brightness.
        if product_image.ndim == 3 and return_image.ndim == 3:
            product_lab = cv2.cvtColor(product_image, cv2.COLOR_BGR2LAB)
            return_lab = cv2.cvtColor(return_image, cv2.COLOR_BGR2LAB)
            chroma_diff = (
                np.abs(product_lab[:, :, 1].astype(np.int16) - return_lab[:, :, 1].astype(np.int16))
                + np.abs(product_lab[:, :, 2].astype(np.int16) - return_lab[:, :, 2].astype(np.int16))
            )
            chroma_diff = np.clip(chroma_diff, 0, 255).astype(np.uint8)

            product_color_strength = (
                np.abs(product_lab[:, :, 1].astype(np.int16) - 128)
                + np.abs(product_lab[:, :, 2].astype(np.int16) - 128)
            )
            product_color_strength = np.clip(product_color_strength, 0, 255).astype(np.uint8)
        else:
            chroma_diff = empty.copy()
            product_color_strength = empty.copy()

        # 4) Product texture mask: existing product ornament, edge, and strong color.
        # Residual on this mask is often caused by slight misalignment of printed patterns.
        product_edge_binary = (product_edge >= 42)
        product_color_binary = (product_color_strength >= 36)
        product_texture = ((product_edge_binary | product_color_binary) & active).astype(np.uint8) * 255
        product_texture_dilated = self._dilate_binary(product_texture, ksize=7, iterations=1) > 0

        # 5) New evidence in return.
        # New edge = return has an edge, product has no corresponding product texture nearby.
        return_edge_binary = (return_edge >= 48)
        new_edge = (return_edge_binary & (~product_texture_dilated) & active)

        # New chroma = strong color shift. Strong chroma may override product texture overlap.
        chroma_new = (
            ((chroma_diff >= 42) & active)
            | ((chroma_diff >= 32) & (~product_texture_dilated) & active)
        )

        # 6) Illumination-like = brightness residual without texture/edge/chroma evidence.
        raw_low = max(16, int(round(self.diff_threshold * 0.65)))
        texture_mid = max(20, int(round(self.diff_threshold * 0.80)))
        edge_mid = max(28, int(round(self.diff_threshold * 1.10)))
        chroma_mid = 24
        illumination_like = (
            (raw_residual >= raw_low)
            & (texture_diff < texture_mid)
            & (edge_diff < edge_mid)
            & (chroma_diff < chroma_mid)
            & active
        )

        # 7) Candidate evidence. Texture/edge only count strongly outside product texture.
        texture_high = max(32, int(round(self.diff_threshold * 1.25)))
        edge_high = max(44, int(round(self.diff_threshold * 1.70)))
        raw_high = max(48, int(self.diff_threshold) + 22)

        outside_product_texture = (~product_texture_dilated) & active
        texture_outside = (texture_diff >= texture_high) & outside_product_texture
        edge_outside = (edge_diff >= edge_high) & outside_product_texture
        gated_raw = (
            (raw_residual >= raw_high)
            & outside_product_texture
            & ((texture_diff >= texture_mid) | (edge_diff >= edge_mid) | (chroma_diff >= chroma_mid))
        )

        # Product-texture overlap is not immediately rejected; it is filtered by component score later.
        weak_inside_texture = (
            ((texture_diff >= texture_high) | (edge_diff >= edge_high) | (chroma_diff >= 32))
            & product_texture_dilated
            & active
        )

        damage_pixels = (
            new_edge
            | chroma_new
            | texture_outside
            | edge_outside
            | gated_raw
            | weak_inside_texture
        ) & (~illumination_like) & active

        damage_mask = damage_pixels.astype(np.uint8) * 255
        damage_mask = self._remove_small_components(damage_mask, max(2, self.min_area_threshold // 3))

        maps = {
            'raw_residual': np.where(active, raw_residual, 0).astype(np.uint8),
            'texture': np.where(active, texture_diff, 0).astype(np.uint8),
            'edge': np.where(active, edge_diff, 0).astype(np.uint8),
            'chroma': np.where(active, chroma_diff, 0).astype(np.uint8),
            'illumination_reject': (illumination_like & active).astype(np.uint8) * 255,
            'product_texture': product_texture,
            'new_edge': (new_edge & active).astype(np.uint8) * 255,
            'chroma_new': (chroma_new & active).astype(np.uint8) * 255,
            'texture_suppressed': (weak_inside_texture & active).astype(np.uint8) * 255,
        }
        return damage_mask, maps

    def _filter_product_texture_components(
        self,
        damage_mask: np.ndarray,
        texture_score: np.ndarray,
        edge_score: np.ndarray,
        chroma_score: np.ndarray,
        product_texture_mask: np.ndarray,
        new_edge_mask: np.ndarray,
        chroma_new_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reject components that are mostly old product texture misalignment."""
        binary = (damage_mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            z = np.zeros_like(binary)
            return binary, z, z

        product_texture = product_texture_mask > 0
        new_edge = new_edge_mask > 0
        chroma_new = chroma_new_mask > 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        component_reject = np.zeros_like(binary)
        texture_suppressed = np.zeros_like(binary)
        image_area = float(binary.shape[0] * binary.shape[1])
        min_area = max(3, int(self.min_area_threshold))

        for label in range(1, num_labels):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                component_reject[component] = 255
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = max(1, w * h)
            fill_ratio = area / float(bbox_area)
            area_ratio = area / image_area

            texture_overlap = float(np.count_nonzero(product_texture & component)) / float(area)
            new_edge_ratio = float(np.count_nonzero(new_edge & component)) / float(area)
            chroma_new_ratio = float(np.count_nonzero(chroma_new & component)) / float(area)
            outside_texture_ratio = 1.0 - texture_overlap

            tex_mean = float(texture_score[component].mean()) if area else 0.0
            edge_mean = float(edge_score[component].mean()) if area else 0.0
            chroma_mean = float(chroma_score[component].mean()) if area else 0.0
            strong_chroma_ratio = float(np.count_nonzero((chroma_score >= 55) & component)) / float(area)

            # Positive evidence: a new return edge or a strong chroma change.
            has_new_evidence = (
                (new_edge_ratio >= 0.035 and edge_mean >= 30.0)
                or chroma_new_ratio >= 0.080
                or strong_chroma_ratio >= 0.025
                or (outside_texture_ratio >= 0.45 and (tex_mean >= 34.0 or edge_mean >= 36.0 or chroma_mean >= 34.0))
            )

            # Negative evidence: mostly overlaps product's old ornament and no new edge/color.
            old_texture_like = (
                texture_overlap >= 0.60
                and new_edge_ratio < 0.025
                and chroma_new_ratio < 0.055
                and strong_chroma_ratio < 0.018
            )

            # Negative evidence: big/dense/diffuse blobs are usually lighting or misalignment.
            diffuse_blob = (
                (area > 1800 or area_ratio > 0.004)
                and fill_ratio > 0.22
                and new_edge_ratio < 0.030
                and chroma_new_ratio < 0.070
            )

            if old_texture_like or diffuse_blob or not has_new_evidence:
                component_reject[component] = 255
                if old_texture_like:
                    texture_suppressed[component] = 255
            else:
                cleaned[component] = 255

        return cleaned, component_reject, texture_suppressed

    def _build_damage_map(
        self,
        triangle_results: Sequence[ScoredTriangle],
        vertices: Sequence[MeshVertex],
        product_image: np.ndarray,
        return_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build pixel-level damage map from scored mesh triangles.

        suspected_mask shows full candidate triangles for debug only.
        damage_map keeps only new-edge/new-chroma evidence after suppressing
        residual that lies on product's existing texture/ornament.
        """
        height, width = product_image.shape[:2]
        damage_map = np.zeros((height, width), dtype=np.uint8)
        suspected_mask = np.zeros((height, width), dtype=np.uint8)
        refined_overlay = np.zeros((height, width), dtype=np.uint8)

        debug_scores = {
            'raw_residual': np.zeros((height, width), dtype=np.uint8),
            'texture': np.zeros((height, width), dtype=np.uint8),
            'edge': np.zeros((height, width), dtype=np.uint8),
            'chroma': np.zeros((height, width), dtype=np.uint8),
            'illumination_reject': np.zeros((height, width), dtype=np.uint8),
            'component_reject': np.zeros((height, width), dtype=np.uint8),
            'product_texture': np.zeros((height, width), dtype=np.uint8),
            'new_edge': np.zeros((height, width), dtype=np.uint8),
            'chroma_new': np.zeros((height, width), dtype=np.uint8),
            'texture_suppressed': np.zeros((height, width), dtype=np.uint8),
        }

        region_extractor = RegionExtractor()

        for result in triangle_results:
            mask = region_extractor.create_triangle_mask(
                (height, width),
                vertices,
                result.triangle.vertex_indices,
                space='product',
            )

            if result.triangle.refined:
                refined_overlay = np.maximum(refined_overlay, (mask > 0).astype(np.uint8) * 255)

            if not result.is_damaged:
                continue

            suspected_mask = np.maximum(suspected_mask, (mask > 0).astype(np.uint8) * 255)
            pixel_mask, maps = self._compute_triangle_pixel_damage(product_image, return_image, mask)

            for key, value in maps.items():
                debug_scores[key] = np.maximum(debug_scores[key], value)

            if np.count_nonzero(pixel_mask) == 0:
                continue

            severity = int(np.clip((1.0 - result.similarity) * 255.0, 1, 255))
            active = pixel_mask > 0
            damage_map[active] = np.maximum(damage_map[active], severity)

        if np.count_nonzero(damage_map) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            damage_map = cv2.morphologyEx(damage_map, cv2.MORPH_OPEN, kernel, iterations=1)
            damage_map = cv2.morphologyEx(damage_map, cv2.MORPH_CLOSE, kernel, iterations=1)
            cleaned_binary, component_reject, texture_suppressed = self._filter_product_texture_components(
                damage_map,
                debug_scores['texture'],
                debug_scores['edge'],
                debug_scores['chroma'],
                debug_scores['product_texture'],
                debug_scores['new_edge'],
                debug_scores['chroma_new'],
            )
            debug_scores['component_reject'] = np.maximum(debug_scores['component_reject'], component_reject)
            debug_scores['texture_suppressed'] = np.maximum(debug_scores['texture_suppressed'], texture_suppressed)
            damage_map[cleaned_binary == 0] = 0
            damage_map = self._remove_small_components(damage_map, max(3, self.min_area_threshold))

        self._last_pixel_debug_masks = {
            'raw_residual_mask': debug_scores['raw_residual'],
            'texture_mask': debug_scores['texture'],
            'edge_mask': debug_scores['edge'],
            'chroma_mask': debug_scores['chroma'],
            'illumination_reject_mask': debug_scores['illumination_reject'],
            'component_reject_mask': debug_scores['component_reject'],
            'product_texture_mask': debug_scores['product_texture'],
            'new_edge_mask': debug_scores['new_edge'],
            'chroma_new_mask': debug_scores['chroma_new'],
            'texture_suppressed_mask': debug_scores['texture_suppressed'],
        }
        return damage_map, suspected_mask, refined_overlay

    def _build_mesh_debug_info(
        self,
        damage_map: np.ndarray,
        suspected_mask: np.ndarray,
        refined_overlay: np.ndarray,
        triangle_results: Sequence[ScoredTriangle],
    ) -> dict:
        final_mask = (damage_map > 0).astype(np.uint8) * 255
        suspected_binary = (suspected_mask > 0).astype(np.uint8) * 255
        pixel_debug = getattr(self, '_last_pixel_debug_masks', {}) or {}

        raw_residual_mask = pixel_debug.get('raw_residual_mask', final_mask)
        texture_mask = pixel_debug.get('texture_mask', final_mask)
        edge_mask = pixel_debug.get('edge_mask', final_mask)
        chroma_mask = pixel_debug.get('chroma_mask', np.zeros_like(final_mask))
        illumination_reject_mask = pixel_debug.get('illumination_reject_mask', np.zeros_like(final_mask))
        component_reject_mask = pixel_debug.get('component_reject_mask', np.zeros_like(final_mask))
        product_texture_mask = pixel_debug.get('product_texture_mask', np.zeros_like(final_mask))
        new_edge_mask = pixel_debug.get('new_edge_mask', np.zeros_like(final_mask))
        chroma_new_mask = pixel_debug.get('chroma_new_mask', np.zeros_like(final_mask))
        texture_suppressed_mask = pixel_debug.get('texture_suppressed_mask', np.zeros_like(final_mask))

        return {
            'intensity_pixels': int(np.count_nonzero(raw_residual_mask)),
            'gradient_pixels': int(np.count_nonzero(edge_mask)),
            'rejected_by_feature_patches': sum(
                1 for item in triangle_results
                if item.feature_similarity >= self.feature_threshold
            ),
            'rejected_by_ssim_patches': sum(
                1 for item in triangle_results
                if item.similarity >= self.ssim_threshold
            ),
            'accepted_patches': sum(1 for item in triangle_results if item.is_damaged),
            'rejected_patches': sum(1 for item in triangle_results if not item.is_damaged),
            'diff_area_before_feature': int(np.count_nonzero(suspected_binary)),
            'diff_area_after_feature': int(np.count_nonzero(suspected_binary)),
            'diff_area_after_ssim': int(np.count_nonzero(suspected_binary)),
            'diff_area_after_morphology': int(np.count_nonzero(final_mask)),
            'final_difference_area': int(np.count_nonzero(final_mask)),
            'components_before_validation': 0,
            'components_after_validation': 0,
            'pixels_removed_by_validation': 0,
            'pixels_remaining': int(np.count_nonzero(final_mask)),
            'diff_area_before_validation': int(np.count_nonzero(final_mask)),
            'diff_area_after_validation': int(np.count_nonzero(final_mask)),
            'accepted_components': [],
            'rejected_components': [],
            'all_components': [],
            'triangle_count': len(triangle_results),
            'damaged_triangle_count': sum(1 for item in triangle_results if item.is_damaged),
            'masks': {
                'raw_intensity_mask': raw_residual_mask,
                'raw_gradient_mask': edge_mask,
                'raw_feature_mask': suspected_binary,
                'raw_ssim_mask': suspected_binary,
                'combined_mask_before_morphology': suspected_binary,
                'combined_mask_after_morphology': final_mask,
                'final_damage_mask': final_mask,
                'feature_reject_mask': np.zeros(damage_map.shape[:2], dtype=np.uint8),
                'ssim_reject_mask': np.zeros(damage_map.shape[:2], dtype=np.uint8),
                'raw_residual_mask': raw_residual_mask,
                'texture_mask': texture_mask,
                'edge_mask': edge_mask,
                'chroma_mask': chroma_mask,
                'illumination_reject_mask': illumination_reject_mask,
                'component_reject_mask': component_reject_mask,
                'product_texture_mask': product_texture_mask,
                'new_edge_mask': new_edge_mask,
                'chroma_new_mask': chroma_new_mask,
                'texture_suppressed_mask': texture_suppressed_mask,
            },
        }


    def _scored_to_local_matches(self, triangle_results: Sequence[ScoredTriangle]) -> List[dict]:
        """Convert scored triangles to local match format for compatibility."""
        local_matches: List[dict] = []
        dummy_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)   # thêm dòng này
        for result in triangle_results:
            weight, accepted, weight_label = self._score_ssim_weight(result.similarity)
            local_matches.append({
                "triangle_id": result.triangle.triangle_id,
                "depth": result.triangle.depth,
                "ssim": result.similarity,
                "best_similarity": result.similarity,
                "feature_similarity": result.feature_similarity,
                "weight": weight,
                "accepted": accepted,
                "weight_label": weight_label,
                "area": result.area,
                "metric_scores": result.metric_scores,
                "refined": result.triangle.refined,
                "is_damaged": result.is_damaged,
                "best_top_left": (0, 0),
                "product_patch": dummy_patch,   # ✅ thêm key này
            })
        return local_matches

    def _build_summary(self, triangle_results: Sequence[ScoredTriangle], damage_map: np.ndarray) -> dict:
        """Build summary statistics."""
        if triangle_results:
            similarities = np.asarray([item.similarity for item in triangle_results], dtype=np.float32)
            weights = np.asarray([
                self._score_ssim_weight(item.similarity)[0] for item in triangle_results
            ], dtype=np.float32)
        else:
            similarities = np.asarray([0.0], dtype=np.float32)
            weights = np.asarray([0.0], dtype=np.float32)

        weight_sum = float(np.sum(weights))
        weighted_similarity = float(np.sum(weights * similarities) / weight_sum) if weight_sum > 0 else 0.0
        accepted_matches = int(np.count_nonzero(weights > 0.0))

        return {
            "total_matches": int(len(triangle_results)),
            "filtered_matches": accepted_matches,
            "accepted_matches": accepted_matches,
            "rejected_matches": int(len(triangle_results) - accepted_matches),
            "average_ssim": float(np.mean(similarities)),
            "median_ssim": float(np.median(similarities)),
            "lowest_ssim": float(np.min(similarities)),
            "highest_ssim": float(np.max(similarities)),
            "weighted_average_ssim": weighted_similarity,
            "average_weight": float(np.mean(weights)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "weight_distribution": {
                "weight=1.0": int(np.count_nonzero(np.isclose(weights, 1.0))),
                "weight=0.7": int(np.count_nonzero(np.isclose(weights, 0.7))),
                "weight=0.4": int(np.count_nonzero(np.isclose(weights, 0.4))),
            },
            "average_offset": 0.0,
            "maximum_offset": 0.0,
            "patch_size": int(self.patch_size),
            "search_window_size": int(self.window_size),
            "similarity_threshold": float(self.similarity_threshold),
            "damage_pixels": int(np.count_nonzero(damage_map)),
            "mesh_triangle_count": int(len(triangle_results)),
            "damaged_triangle_count": int(sum(1 for item in triangle_results if item.is_damaged)),
        }

    def _draw_mesh_overlay(
        self,
        image: np.ndarray,
        vertices: Sequence[MeshVertex],
        triangle_results: Sequence[ScoredTriangle],
        point_space: str,
    ) -> np.ndarray:
        """Draw mesh overlay for debugging."""
        overlay = image.copy()
        region_extractor = RegionExtractor()

        for result in triangle_results:
            triangle = result.triangle
            if point_space == "product":
                polygon = region_extractor.triangle_product_points(vertices, triangle.vertex_indices)
            else:
                polygon = region_extractor.triangle_return_points(vertices, triangle.vertex_indices)

            if result.is_damaged:
                color = (0, 0, 255)
            elif triangle.refined:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=1)

        for result in triangle_results[:80]:
            triangle = result.triangle
            if point_space == "product":
                points = [vertices[index].product_point for index in triangle.vertex_indices]
            else:
                points = [vertices[index].return_point for index in triangle.vertex_indices]
            center_x = int(round(sum(point[0] for point in points) / 3.0))
            center_y = int(round(sum(point[1] for point in points) / 3.0))
            label = f"{triangle.triangle_id}:{result.similarity:.2f}"
            cv2.putText(
                overlay,
                label,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return overlay

    def _save_mesh_debug_images(
        self,
        debug_dir: Path,
        product_image: np.ndarray,
        return_image: np.ndarray,
        product_points: Sequence[Sequence[float]],
        return_points: Sequence[Sequence[float]],
        vertices: Sequence[MeshVertex],
        triangle_results: Sequence[ScoredTriangle],
        damage_map: np.ndarray,
        suspected_mask: np.ndarray,
        refined_overlay: np.ndarray,
        stats,
    ) -> Dict[str, str]:
        """Save debug images (only if DEBUG=True)."""
        if not DEBUG:
            return {}
            
        debug_dir.mkdir(parents=True, exist_ok=True)

        delaunay_overlay = self._draw_mesh_overlay(product_image, vertices, triangle_results, "product")
        product_overlay = self._draw_mesh_overlay(product_image, vertices, triangle_results, "product")
        return_overlay = self._draw_mesh_overlay(return_image, vertices, triangle_results, "return")

        suspected_visual = product_image.copy()
        suspected_visual[suspected_mask > 0] = (0, 0, 255)
        suspected_visual = cv2.addWeighted(product_image, 0.7, suspected_visual, 0.3, 0)

        refined_visual = product_image.copy()
        refined_visual[refined_overlay > 0] = (0, 165, 255)
        refined_visual = cv2.addWeighted(product_image, 0.7, refined_visual, 0.3, 0)

        # Similarity histogram
        if triangle_results:
            similarities = np.asarray([item.similarity for item in triangle_results], dtype=np.float32)
            hist_width, hist_height = 640, 240
            canvas = np.full((hist_height, hist_width, 3), 255, dtype=np.uint8)
            hist, _ = np.histogram(similarities, bins=20, range=(0.0, 1.0))
            max_count = max(int(hist.max()) if hist.size else 1, 1)
            
            margin_left, margin_right = 50, 20
            margin_top, margin_bottom = 30, 40
            plot_width = hist_width - margin_left - margin_right
            plot_height = hist_height - margin_top - margin_bottom
            bar_width = plot_width / len(hist)

            for idx, count in enumerate(hist):
                bar_height = int((count / max_count) * plot_height)
                x0 = margin_left + int(idx * bar_width)
                x1 = margin_left + int((idx + 1) * bar_width) - 2
                y1 = hist_height - margin_bottom
                y0 = y1 - bar_height
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 130, 255), -1)

            cv2.putText(
                canvas,
                "Triangle similarity distribution",
                (margin_left, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )
            similarity_histogram = canvas
        else:
            similarity_histogram = np.full((240, 640, 3), 255, dtype=np.uint8)

        files = {
            "01_keypoints.png": self._draw_keypoints(product_image, product_points, return_points),
            "02_delaunay_mesh.png": delaunay_overlay,
            "03_mesh_overlay_product.png": product_overlay,
            "04_mesh_overlay_return.png": return_overlay,
            "05_suspected_triangles.png": suspected_visual,
            "06_refined_mesh.png": refined_visual,
            "07_damage_mask.png": cv2.cvtColor(damage_map, cv2.COLOR_GRAY2BGR),
            "08_similarity_histogram.png": similarity_histogram,
        }

        pixel_debug = getattr(self, '_last_pixel_debug_masks', {}) or {}
        for extra_name, extra_image in {
            '09_raw_residual_mask.png': pixel_debug.get('raw_residual_mask'),
            '10_texture_mask.png': pixel_debug.get('texture_mask'),
            '11_edge_mask.png': pixel_debug.get('edge_mask'),
            '12_chroma_mask.png': pixel_debug.get('chroma_mask'),
            '13_illumination_reject_mask.png': pixel_debug.get('illumination_reject_mask'),
            '14_component_reject_mask.png': pixel_debug.get('component_reject_mask'),
            '15_product_texture_mask.png': pixel_debug.get('product_texture_mask'),
            '16_new_edge_mask.png': pixel_debug.get('new_edge_mask'),
            '17_chroma_new_mask.png': pixel_debug.get('chroma_new_mask'),
            '18_texture_suppressed_mask.png': pixel_debug.get('texture_suppressed_mask'),
        }.items():
            if extra_image is not None:
                files[extra_name] = cv2.cvtColor(extra_image, cv2.COLOR_GRAY2BGR) if extra_image.ndim == 2 else extra_image

        saved_paths: Dict[str, str] = {}
        for filename, image in files.items():
            path = debug_dir / filename
            cv2.imwrite(str(path), image)
            saved_paths[filename] = str(path)

        # Save statistics to text file
        stats_path = debug_dir / "stats.txt"
        with open(stats_path, "w") as f:
            f.write("=== Adaptive Mesh Refinement Statistics ===\n\n")
            f.write(f"Original Triangles   : {stats.original_triangles}\n")
            f.write(f"Refined Triangles    : {stats.refined_triangles}\n")
            f.write(f"Triangles Processed  : {stats.triangles_processed}\n")
            f.write(f"Triangles Skipped    : {stats.triangles_skipped}\n")
            f.write(f"Total Refinements    : {stats.total_refinements}\n")
            f.write(f"Average Similarity   : {stats.average_similarity:.4f}\n")
            f.write(f"Similarity Time      : {stats.similarity_time:.4f}s\n")
            f.write(f"Mesh Time            : {stats.mesh_time:.4f}s\n")

        return saved_paths

    @staticmethod
    def _draw_keypoints(
        image: np.ndarray,
        product_points: Sequence[Sequence[float]],
        return_points: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Draw keypoints on side-by-side images."""
        canvas = np.hstack([image.copy(), image.copy()])
        width_offset = image.shape[1]

        for idx, point in enumerate(product_points):
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(canvas, center, 4, (0, 255, 0), -1)
            cv2.putText(
                canvas, str(idx), (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )

        for idx, point in enumerate(return_points):
            center = (int(round(point[0])) + width_offset, int(round(point[1])))
            cv2.circle(canvas, center, 4, (255, 0, 0), -1)
            cv2.putText(
                canvas, str(idx), (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1,
            )

        return canvas

    def _build_similarity_histogram(self, triangle_results: Sequence[ScoredTriangle]) -> np.ndarray:
        """Draw histogram of similarity scores."""
        width, height = 640, 240
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        if not triangle_results:
            return canvas

        similarities = np.asarray([item.similarity for item in triangle_results], dtype=np.float32)
        hist, _ = np.histogram(similarities, bins=20, range=(0.0, 1.0))
        max_count = max(int(hist.max()) if hist.size else 1, 1)

        margin_left, margin_right = 50, 20
        margin_top, margin_bottom = 30, 40
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        bar_width = plot_width / len(hist)

        for idx, count in enumerate(hist):
            bar_height = int((count / max_count) * plot_height)
            x0 = margin_left + int(idx * bar_width)
            x1 = margin_left + int((idx + 1) * bar_width) - 2
            y1 = height - margin_bottom
            y0 = y1 - bar_height
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 130, 255), -1)

        cv2.putText(
            canvas,
            "Triangle similarity distribution",
            (margin_left, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        return canvas

    def _build_matching_overlay(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        damage_map: np.ndarray,
    ) -> np.ndarray:
        """Overlay product (red) and aligned return (green) with damage highlighted."""
        product_gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY) if product_image.ndim == 3 else product_image
        return_gray = cv2.cvtColor(return_image, cv2.COLOR_BGR2GRAY) if return_image.ndim == 3 else return_image

        product_red = np.zeros_like(product_image)
        aligned_green = np.zeros_like(return_image)
        product_red[:, :, 2] = product_gray
        aligned_green[:, :, 1] = return_gray

        base_overlay = cv2.addWeighted(product_red, 0.5, aligned_green, 0.5, 0)
        damage_highlight = base_overlay.copy()
        damage_highlight[damage_map > 0] = (0, 0, 255)
        return cv2.addWeighted(base_overlay, 0.75, damage_highlight, 0.25, 0)

    def run(
        self,
        product_image: np.ndarray,
        aligned_return_image: np.ndarray,
        product_keypoints: Sequence[Sequence[float]],
        return_keypoints: Sequence[Sequence[float]],
        inlier_matches: Sequence,
        product_descriptors: Optional[Sequence] = None,
        return_descriptors: Optional[Sequence] = None,
        debug_mesh_dir: Optional[Path] = None,
    ) -> dict:
        """Run the adaptive mesh damage detection pipeline."""
        # Build paired keypoints
        product_points, return_points, keypoint_indices = self._build_paired_keypoints(
            product_keypoints,
            return_keypoints,
            inlier_matches,
            product_image,
        )

        # Convert descriptors to numpy
        product_descriptors_np = (
            np.asarray(product_descriptors, dtype=np.float32)
            if product_descriptors is not None and len(product_descriptors) > 0
            else None
        )
        return_descriptors_np = (
            np.asarray(return_descriptors, dtype=np.float32)
            if return_descriptors is not None and len(return_descriptors) > 0
            else None
        )

        # Get scheduler and run refinement
        scheduler = self._get_scheduler()
        result = scheduler.run(
            product_points,
            return_points,
            keypoint_indices,
            product_image,
            aligned_return_image,
            product_descriptors_np,
            return_descriptors_np,
        )

        vertices = result.vertices
        triangle_results = result.leaf_triangles
        stats = result.stats

        # Build illumination-robust pixel-level damage map
        damage_map, suspected_mask, refined_overlay = self._build_damage_map(
            triangle_results,
            vertices,
            product_image,
            aligned_return_image,
        )

        # Convert to local matches format
        local_matches = self._scored_to_local_matches(triangle_results)
        # Build debug info without old component validator
        debug_info = self._build_mesh_debug_info(
            damage_map,
            suspected_mask,
            refined_overlay,
            triangle_results,
        )

        # Build summary
        summary = self._build_summary(triangle_results, damage_map)

        # Build overlay
        overlay = self._build_matching_overlay(product_image, aligned_return_image, damage_map)

        # Debug images
        debug_images = {
            "patch_product": product_image[:self.patch_size, :self.patch_size].copy()
            if product_image.shape[0] >= self.patch_size and product_image.shape[1] >= self.patch_size
            else np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8),
            "search_window": self._draw_mesh_overlay(product_image, vertices, triangle_results, "product"),
            "best_patch": self._draw_mesh_overlay(aligned_return_image, vertices, triangle_results, "return"),
            "similarity_heatmap": self._build_similarity_histogram(triangle_results),
            "best_patch_heatmap": self._build_similarity_histogram(triangle_results),
            "patch_difference": cv2.cvtColor(suspected_mask, cv2.COLOR_GRAY2BGR),
            "damage_map": cv2.cvtColor(damage_map, cv2.COLOR_GRAY2BGR),
            "local_matching_overlay": overlay,
            "ssim_histogram": self._build_similarity_histogram(triangle_results),
        }

        # Save debug images if requested (only if DEBUG=True)
        mesh_debug_paths = {}
        if debug_mesh_dir is not None:
            mesh_debug_paths = self._save_mesh_debug_images(
                debug_mesh_dir,
                product_image,
                aligned_return_image,
                product_points,
                return_points,
                vertices,
                triangle_results,
                damage_map,
                suspected_mask,
                refined_overlay,
                stats,
            )

        # Representative match (lowest similarity)
        representative = min(triangle_results, key=lambda item: item.similarity) if triangle_results else None

        self._log_step(
            "Mesh Matching: "
            f"input_inliers={len(inlier_matches)}, "
            f"mesh_vertices={len(vertices)}, "
            f"base_triangles={stats.original_triangles}, "
            f"leaf_triangles={len(triangle_results)}, "
            f"damaged_triangles={summary['damaged_triangle_count']}, "
            f"similarity_min={summary['lowest_ssim']:.3f}, "
            f"similarity_mean={summary['average_ssim']:.3f}, "
            f"damage_pixels={summary['damage_pixels']}"
        )

        return {
            "local_matches": local_matches,
            "damage_map": damage_map,
            "summary": summary,
            "debug_images": debug_images,
            "representative_match": representative,
            "debug_info": debug_info,
            "mesh_debug_paths": mesh_debug_paths,
            "triangle_results": triangle_results,
            "vertices": vertices,
        }