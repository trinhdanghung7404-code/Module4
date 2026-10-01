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
    EDGE_SEARCH_RADIUS,
    EDGE_MATCH_SCORE_THRESHOLD,
    EDGE_ORIENTATION_TOLERANCE_DEG,
    COLOR_SEARCH_RADIUS,
    COLOR_DIFF_THRESHOLD,
    FAINT_COLOR_DIFF_THRESHOLD,
    FAINT_SATURATION_DIFF_THRESHOLD,
    FAINT_LUMINANCE_DIFF_THRESHOLD,
    FAINT_COLOR_MIN_AREA,
    FAINT_COLOR_MIN_FILL_RATIO,
    FAINT_COLOR_MIN_MEAN_SCORE,
    STRONG_COLOR_DIFF_THRESHOLD,
    COLOR_MIN_AREA,
    PERCEPTUAL_LUMINANCE_ENABLED,
    PERCEPTUAL_L_DIFF_THRESHOLD,
    PERCEPTUAL_SCORE_THRESHOLD,
    PERCEPTUAL_STRONG_SCORE_THRESHOLD,
    PERCEPTUAL_MIN_AREA,
    PERCEPTUAL_STRONG_MIN_AREA,
    PERCEPTUAL_REFERENCE_OBJECT_AREA,
    PERCEPTUAL_LOCAL_SIGMA,
    PERCEPTUAL_L_WEIGHT,
    PERCEPTUAL_OUTSIDE_MESH_ENABLED,
    PERCEPTUAL_OUTSIDE_MESH_SCORE_THRESHOLD,
    PERCEPTUAL_OUTSIDE_MESH_MIN_AREA,
    PERCEPTUAL_OUTSIDE_MESH_MAX_AREA_RATIO,
    PERCEPTUAL_OUTSIDE_MESH_MIN_FILL_RATIO,
    PERCEPTUAL_OUTSIDE_MESH_MATCHED_EDGE_REJECT_RATIO,
)
from mesh_refiner import MeshRefiner
from priority_scheduler import PriorityMeshScheduler
from region_extractor import RegionExtractor
from region_similarity import RegionSimilarity
from similarity_cache import ScoredTriangle
from triangle_similarity import TriangleSimilarity
from triangle_refiner import TriangleRefiner

MESH_DAMAGE_DETECTOR_VERSION = "robust_region_gated_perceptual_v13"
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
        feature_threshold: float = FEATURE_THRESHOLD,       # ← THÊM THAM SỐ NÀY!
        ssim_threshold: float = SSIM_THRESHOLD,              # ← THÊM THAM SỐ NÀY!
        max_depth: int = MAX_DEPTH,
        min_triangle_area: float = MIN_TRIANGLE_AREA,
        diff_threshold: int = DIFF_THRESHOLD,
        refinement_strategy: str = MeshRefiner.STRATEGY_MIDPOINT,
        max_total_triangles: int = MAX_TOTAL_TRIANGLES,
        top_k: int = TOP_K,
        workers: int = SIMILARITY_WORKERS,
        edge_search_radius: int = EDGE_SEARCH_RADIUS,
        edge_match_score_threshold: float = EDGE_MATCH_SCORE_THRESHOLD,
        edge_orientation_tolerance_deg: float = EDGE_ORIENTATION_TOLERANCE_DEG,
        color_search_radius: int = COLOR_SEARCH_RADIUS,
        color_diff_threshold: float = COLOR_DIFF_THRESHOLD,
        faint_color_diff_threshold: float = FAINT_COLOR_DIFF_THRESHOLD,
        faint_saturation_diff_threshold: float = FAINT_SATURATION_DIFF_THRESHOLD,
        faint_luminance_diff_threshold: float = FAINT_LUMINANCE_DIFF_THRESHOLD,
        faint_color_min_area: int = FAINT_COLOR_MIN_AREA,
        faint_color_min_fill_ratio: float = FAINT_COLOR_MIN_FILL_RATIO,
        faint_color_min_mean_score: float = FAINT_COLOR_MIN_MEAN_SCORE,
        strong_color_diff_threshold: float = STRONG_COLOR_DIFF_THRESHOLD,
        color_min_area: int = COLOR_MIN_AREA,
        perceptual_luminance_enabled: bool = PERCEPTUAL_LUMINANCE_ENABLED,
        perceptual_l_diff_threshold: float = PERCEPTUAL_L_DIFF_THRESHOLD,
        perceptual_score_threshold: float = PERCEPTUAL_SCORE_THRESHOLD,
        perceptual_strong_score_threshold: float = PERCEPTUAL_STRONG_SCORE_THRESHOLD,
        perceptual_min_area: int = PERCEPTUAL_MIN_AREA,
        perceptual_strong_min_area: int = PERCEPTUAL_STRONG_MIN_AREA,
        perceptual_reference_object_area: int = PERCEPTUAL_REFERENCE_OBJECT_AREA,
        perceptual_local_sigma: float = PERCEPTUAL_LOCAL_SIGMA,
        perceptual_l_weight: float = PERCEPTUAL_L_WEIGHT,
        perceptual_outside_mesh_enabled: bool = PERCEPTUAL_OUTSIDE_MESH_ENABLED,
        perceptual_outside_mesh_score_threshold: float = PERCEPTUAL_OUTSIDE_MESH_SCORE_THRESHOLD,
        perceptual_outside_mesh_min_area: int = PERCEPTUAL_OUTSIDE_MESH_MIN_AREA,
        perceptual_outside_mesh_max_area_ratio: float = PERCEPTUAL_OUTSIDE_MESH_MAX_AREA_RATIO,
        perceptual_outside_mesh_min_fill_ratio: float = PERCEPTUAL_OUTSIDE_MESH_MIN_FILL_RATIO,
        perceptual_outside_mesh_matched_edge_reject_ratio: float = PERCEPTUAL_OUTSIDE_MESH_MATCHED_EDGE_REJECT_RATIO,
    ):
        self.similarity_threshold = similarity_threshold
        self.feature_threshold = feature_threshold            # ← LƯU VÀO ATTRIBUTE
        self.ssim_threshold = ssim_threshold                  # ← LƯU VÀO ATTRIBUTE
        self.max_depth = max_depth
        self.min_triangle_area = min_triangle_area
        self.diff_threshold = diff_threshold
        self.max_total_triangles = max_total_triangles
        self.top_k = top_k
        self.workers = workers
        self.edge_search_radius = max(0, int(edge_search_radius))
        self.edge_match_score_threshold = float(np.clip(edge_match_score_threshold, 0.0, 1.0))
        self.edge_orientation_tolerance_deg = float(np.clip(edge_orientation_tolerance_deg, 0.0, 90.0))
        self.color_search_radius = max(0, int(color_search_radius))
        self.color_diff_threshold = max(0.0, float(color_diff_threshold))
        self.faint_color_diff_threshold = max(
            0.0, min(float(faint_color_diff_threshold), self.color_diff_threshold)
        )
        self.faint_saturation_diff_threshold = max(
            0.0, float(faint_saturation_diff_threshold)
        )
        self.faint_luminance_diff_threshold = max(
            0.0, float(faint_luminance_diff_threshold)
        )
        self.faint_color_min_area = max(1, int(faint_color_min_area))
        self.faint_color_min_fill_ratio = float(np.clip(
            faint_color_min_fill_ratio, 0.01, 1.0
        ))
        self.faint_color_min_mean_score = max(0.0, float(faint_color_min_mean_score))
        self.strong_color_diff_threshold = max(
            self.color_diff_threshold,
            float(strong_color_diff_threshold),
        )
        self.color_min_area = max(1, int(color_min_area))
        self.perceptual_luminance_enabled = bool(perceptual_luminance_enabled)
        self.perceptual_l_diff_threshold = max(
            0.0,
            float(perceptual_l_diff_threshold),
        )
        self.perceptual_score_threshold = max(
            0.0,
            float(perceptual_score_threshold),
        )
        self.perceptual_strong_score_threshold = max(
            self.perceptual_score_threshold,
            float(perceptual_strong_score_threshold),
        )
        self.perceptual_min_area = max(1, int(perceptual_min_area))
        self.perceptual_strong_min_area = max(
            1,
            min(int(perceptual_strong_min_area), self.perceptual_min_area),
        )
        self.perceptual_reference_object_area = max(
            1,
            int(perceptual_reference_object_area),
        )
        self.perceptual_local_sigma = max(0.5, float(perceptual_local_sigma))
        self.perceptual_l_weight = max(0.0, float(perceptual_l_weight))
        self.perceptual_outside_mesh_enabled = bool(
            perceptual_outside_mesh_enabled
        )
        self.perceptual_outside_mesh_score_threshold = max(
            self.perceptual_score_threshold,
            float(perceptual_outside_mesh_score_threshold),
        )
        self.perceptual_outside_mesh_min_area = max(
            1, int(perceptual_outside_mesh_min_area)
        )
        self.perceptual_outside_mesh_max_area_ratio = float(np.clip(
            perceptual_outside_mesh_max_area_ratio, 0.0001, 0.25
        ))
        self.perceptual_outside_mesh_min_fill_ratio = float(np.clip(
            perceptual_outside_mesh_min_fill_ratio, 0.01, 1.0
        ))
        self.perceptual_outside_mesh_matched_edge_reject_ratio = float(np.clip(
            perceptual_outside_mesh_matched_edge_reject_ratio, 0.0, 1.0
        ))

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
        magnitude, _ = self._gradient_magnitude_and_orientation(gray)
        return magnitude

    def _gradient_magnitude_and_orientation(
        self,
        gray: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Scharr magnitude (uint8) and unsigned edge direction (0..180)."""
        src = gray.astype(np.float32)
        grad_x = cv2.Scharr(src, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(src, cv2.CV_32F, 0, 1)
        magnitude_float = cv2.magnitude(grad_x, grad_y)
        max_val = float(np.max(magnitude_float)) if magnitude_float.size else 0.0
        if max_val <= 1e-6:
            magnitude = np.zeros(gray.shape[:2], dtype=np.uint8)
        else:
            magnitude = cv2.normalize(
                magnitude_float,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)

        orientation = cv2.phase(
            grad_x,
            grad_y,
            angleInDegrees=True,
        )
        orientation = np.mod(orientation, 180.0).astype(np.float32)
        return magnitude, orientation

    @staticmethod
    def _shift_map(array: np.ndarray, dx: int, dy: int, fill_value=0) -> np.ndarray:
        """Shift a 2-D map without wrap-around."""
        height, width = array.shape[:2]
        shifted = np.full_like(array, fill_value)

        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        if src_x0 >= src_x1 or src_y0 >= src_y1:
            return shifted

        dst_x0 = src_x0 + dx
        dst_x1 = src_x1 + dx
        dst_y0 = src_y0 + dy
        dst_y1 = src_y1 + dy
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
        return shifted

    def _build_edge_match_maps(
        self,
        product_edge: np.ndarray,
        product_orientation: np.ndarray,
        return_edge: np.ndarray,
        return_orientation: np.ndarray,
    ) -> dict:
        """Find the best nearby product edge for every return-edge pixel.

        The mesh is not moved. For each return edge, product edges inside a
        small +/-k neighborhood are tested and only the best score is kept.
        """
        product_edge_binary = product_edge >= 42
        return_edge_binary = return_edge >= 48
        best_score = np.zeros(product_edge.shape, dtype=np.float32)

        radius = self.edge_search_radius
        tolerance = max(self.edge_orientation_tolerance_deg, 1.0)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                distance = float(np.hypot(dx, dy))
                if distance > radius:
                    continue

                shifted_edge = self._shift_map(product_edge, dx, dy, 0)
                shifted_binary = self._shift_map(product_edge_binary, dx, dy, False)
                if not np.any(shifted_binary):
                    continue

                shifted_orientation = self._shift_map(
                    product_orientation,
                    dx,
                    dy,
                    0.0,
                )
                angle_diff = np.abs(return_orientation - shifted_orientation)
                angle_diff = np.minimum(angle_diff, 180.0 - angle_diff)
                orientation_valid = angle_diff <= tolerance

                orientation_score = np.clip(1.0 - (angle_diff / 90.0), 0.0, 1.0)
                strength_score = 1.0 - (
                    np.abs(return_edge.astype(np.float32) - shifted_edge.astype(np.float32))
                    / 255.0
                )
                if radius > 0:
                    distance_score = max(0.0, 1.0 - distance / float(radius))
                else:
                    distance_score = 1.0

                score = (
                    0.55 * orientation_score
                    + 0.35 * strength_score
                    + 0.10 * distance_score
                )
                valid = return_edge_binary & shifted_binary & orientation_valid
                score = np.where(valid, score, 0.0).astype(np.float32)
                best_score = np.maximum(best_score, score)

        matched = return_edge_binary & (best_score >= self.edge_match_score_threshold)
        unmatched = return_edge_binary & (~matched)
        return {
            'product_edge': product_edge,
            'return_edge': return_edge,
            'matched_edge': matched.astype(np.uint8) * 255,
            'unmatched_edge': unmatched.astype(np.uint8) * 255,
            'edge_match_score': np.clip(best_score * 255.0, 0, 255).astype(np.uint8),
        }

    def _masked_gaussian_mean(
        self,
        values: np.ndarray,
        valid: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Return a smooth local mean without leaking the black background."""
        weights = valid.astype(np.float32)
        numerator = cv2.GaussianBlur(
            values.astype(np.float32) * weights,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
        )
        denominator = cv2.GaussianBlur(
            weights,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
        )
        return numerator / np.maximum(denominator, 1e-4)

    def _filter_human_visible_components(
        self,
        candidate_mask: np.ndarray,
        perceptual_score: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """Keep visible components and discard specks too small to notice.

        The minimum area scales with object resolution. A moderately sized
        component is kept normally. A smaller component is retained only when
        its perceptual contrast is very strong.
        """
        binary = (candidate_mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            self._last_visible_component_stats = {
                'normal_min_area': 0,
                'strong_min_area': 0,
                'kept_components': 0,
                'rejected_components': 0,
            }
            return binary

        object_area = max(1, int(np.count_nonzero(active)))
        area_scale = float(np.clip(
            object_area / float(self.perceptual_reference_object_area),
            0.50,
            6.00,
        ))
        normal_min_area = max(
            1,
            int(round(self.perceptual_min_area * area_scale)),
        )
        strong_min_area = max(
            1,
            int(round(self.perceptual_strong_min_area * area_scale)),
        )
        strong_min_area = min(strong_min_area, normal_min_area)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        cleaned = np.zeros_like(binary)
        kept_components = 0
        rejected_components = 0

        for label in range(1, num_labels):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            component_scores = perceptual_score[component]
            if component_scores.size == 0:
                rejected_components += 1
                continue

            p90_score = float(np.percentile(component_scores, 90))
            keep_normal = area >= normal_min_area
            keep_small_strong = (
                area >= strong_min_area
                and p90_score >= self.perceptual_strong_score_threshold
            )

            if keep_normal or keep_small_strong:
                cleaned[component] = 255
                kept_components += 1
            else:
                rejected_components += 1

        self._last_visible_component_stats = {
            'normal_min_area': int(normal_min_area),
            'strong_min_area': int(strong_min_area),
            'kept_components': int(kept_components),
            'rejected_components': int(rejected_components),
        }
        return cleaned

    def _filter_faint_color_components(
        self,
        candidate_mask: np.ndarray,
        faint_score: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """Keep coherent faint colour changes while rejecting colour noise.

        A low pixel threshold alone is unsafe on textured products.  Faint
        changes must instead form a reasonably filled, sufficiently large
        component whose average evidence remains above the faint threshold.
        The area is resolution-aware just like the normal visible-error path.
        """
        binary = (candidate_mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            self._last_faint_color_component_stats = {
                'minimum_area': 0,
                'kept_components': 0,
                'rejected_components': 0,
            }
            return binary

        object_area = max(1, int(np.count_nonzero(active)))
        area_scale = float(np.clip(
            object_area / float(self.perceptual_reference_object_area),
            0.50,
            6.00,
        ))
        minimum_area = max(1, int(round(self.faint_color_min_area * area_scale)))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cleaned = np.zeros_like(binary)
        kept = 0
        rejected = 0

        for label in range(1, num_labels):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            fill_ratio = area / float(max(1, width * height))
            scores = faint_score[component]
            mean_score = float(scores.mean()) if scores.size else 0.0
            p75_score = float(np.percentile(scores, 75)) if scores.size else 0.0

            if (
                area >= minimum_area
                and fill_ratio >= self.faint_color_min_fill_ratio
                and mean_score >= self.faint_color_min_mean_score
                and p75_score >= self.faint_color_diff_threshold
            ):
                cleaned[component] = 255
                kept += 1
            else:
                rejected += 1

        self._last_faint_color_component_stats = {
            'minimum_area': int(minimum_area),
            'kept_components': int(kept),
            'rejected_components': int(rejected),
        }
        return cleaned

    def _filter_visible_outside_mesh(
        self,
        candidate_mask: np.ndarray,
        perceptual_score: np.ndarray,
        matched_edge_neighborhood: np.ndarray,
        interior: np.ndarray,
    ) -> np.ndarray:
        """Keep compact, strong visible defects outside the mesh gate.

        Exact-position comparison in real photographs often creates thin
        contours along existing ornament. Outside suspicious triangles, use a
        stricter component test so those contours do not enter the final mask.
        """
        binary = (candidate_mask > 0).astype(np.uint8) * 255
        if not self.perceptual_outside_mesh_enabled:
            return binary
        if np.count_nonzero(binary) == 0:
            self._last_outside_visible_stats = {
                'kept_components': 0,
                'rejected_components': 0,
                'minimum_area': 0,
                'maximum_area': 0,
            }
            return binary

        object_area = max(1, int(np.count_nonzero(interior)))
        area_scale = float(np.clip(
            object_area / float(self.perceptual_reference_object_area),
            0.50,
            6.00,
        ))
        minimum_area = max(
            1,
            int(round(self.perceptual_outside_mesh_min_area * area_scale)),
        )
        maximum_area = max(
            minimum_area * 4,
            int(round(object_area * self.perceptual_outside_mesh_max_area_ratio)),
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cleaned = np.zeros_like(binary)
        matched = matched_edge_neighborhood > 0
        kept = 0
        rejected = 0

        for label in range(1, num_labels):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = max(1, width * height)
            fill_ratio = area / float(bbox_area)

            values = perceptual_score[component]
            p90 = float(np.percentile(values, 90)) if values.size else 0.0
            matched_ratio = float(np.count_nonzero(matched & component)) / float(max(1, area))
            thickness = float(cv2.distanceTransform(
                component.astype(np.uint8), cv2.DIST_L2, 3
            ).max())

            extremely_strong = (
                p90 >= self.perceptual_outside_mesh_score_threshold + 12.0
            )
            shape_visible = (
                fill_ratio >= self.perceptual_outside_mesh_min_fill_ratio
                or thickness >= 1.25
                or area >= minimum_area * 2
                or max(width, height) >= 12
            )
            follows_existing_edge = (
                matched_ratio >= self.perceptual_outside_mesh_matched_edge_reject_ratio
            )

            keep = (
                area >= minimum_area
                and p90 >= self.perceptual_outside_mesh_score_threshold
                and shape_visible
                and (area <= maximum_area or extremely_strong)
                and (not follows_existing_edge or extremely_strong)
            )
            if keep:
                cleaned[component] = 255
                kept += 1
            else:
                rejected += 1

        self._last_outside_visible_stats = {
            'kept_components': int(kept),
            'rejected_components': int(rejected),
            'minimum_area': int(minimum_area),
            'maximum_area': int(maximum_area),
        }
        return cleaned

    def _build_color_match_maps(
        self,
        product_lab: np.ndarray,
        return_lab: np.ndarray,
        active: np.ndarray,
    ) -> dict:
        """Build human-visible colour and local luminance difference maps.

        The a/b branch detects hue and saturation changes. The L branch removes
        slowly varying illumination before looking for local white, grey, dark
        or dull marks. Connected components that are too small to be noticed
        are discarded, unless their local contrast is exceptionally strong.
        """
        product_l = product_lab[:, :, 0].astype(np.float32)
        product_a = product_lab[:, :, 1].astype(np.float32)
        product_b = product_lab[:, :, 2].astype(np.float32)
        return_l = return_lab[:, :, 0].astype(np.float32)
        return_a = return_lab[:, :, 1].astype(np.float32)
        return_b = return_lab[:, :, 2].astype(np.float32)

        # The safe interior mask already excludes the black background. Keep
        # a pixel valid when either image contains object content so a new
        # black/dark defect is not discarded merely because return L is low.
        valid = active & ((product_l > 8.0) | (return_l > 8.0))
        if np.count_nonzero(valid) >= 16:
            a_offset = float(np.median((return_a - product_a)[valid]))
            b_offset = float(np.median((return_b - product_b)[valid]))
            l_offset = float(np.median((return_l - product_l)[valid]))
        else:
            a_offset = 0.0
            b_offset = 0.0
            l_offset = 0.0

        # Correct only the global colour cast. Local defects remain visible.
        a_offset = float(np.clip(a_offset, -20.0, 20.0))
        b_offset = float(np.clip(b_offset, -20.0, 20.0))
        corrected_return_a = np.clip(return_a - a_offset, 0.0, 255.0)
        corrected_return_b = np.clip(return_b - b_offset, 0.0, 255.0)

        delta_a = corrected_return_a - product_a
        delta_b = corrected_return_b - product_b
        color_distance = cv2.magnitude(delta_a, delta_b)
        color_distance[~valid] = 0.0

        # A faded mark can retain nearly the same hue while losing chroma.
        # Track the change in distance from neutral Lab (128, 128) separately
        # so this type of defect is not hidden by the vector-difference score.
        product_saturation = cv2.magnitude(product_a - 128.0, product_b - 128.0)
        return_saturation = cv2.magnitude(
            corrected_return_a - 128.0,
            corrected_return_b - 128.0,
        )
        saturation_difference = np.abs(return_saturation - product_saturation)
        saturation_difference[~valid] = 0.0

        luminance_local_difference = np.zeros_like(product_l, dtype=np.float32)
        if self.perceptual_luminance_enabled:
            delta_l = return_l - product_l
            smooth_l_shift = self._masked_gaussian_mean(
                delta_l,
                valid,
                self.perceptual_local_sigma,
            )
            luminance_local_difference = np.abs(delta_l - smooth_l_shift)
            luminance_local_difference[~valid] = 0.0

        weighted_luminance = (
            luminance_local_difference * self.perceptual_l_weight
        ).astype(np.float32)
        perceptual_difference = cv2.magnitude(
            color_distance.astype(np.float32),
            weighted_luminance,
        )
        perceptual_difference[~valid] = 0.0
        # Faint defects can be a loss of colour *or* a soft grey/white/dark
        # streak.  Use the strongest locally-normalised cue, rather than only
        # Lab a/b, for their component-level validation.
        faint_color_score = np.maximum.reduce([
            color_distance,
            saturation_difference,
            luminance_local_difference,
        ])

        # For recall outside suspicious mesh triangles, tolerate a small local
        # displacement. Printed ornaments on a curved object can move by a few
        # pixels between real photographs even after global normalisation.
        # The exact map above is still used inside suspicious triangles.
        product_l_mean = self._masked_gaussian_mean(
            product_l, valid, self.perceptual_local_sigma
        )
        return_l_mean = self._masked_gaussian_mean(
            return_l, valid, self.perceptual_local_sigma
        )
        product_l_detail = product_l - product_l_mean
        return_l_detail = return_l - return_l_mean

        nearby_perceptual_difference = perceptual_difference.copy()
        if self.color_search_radius > 0:
            nearby_perceptual_difference = np.full_like(
                perceptual_difference, 1.0e6, dtype=np.float32
            )
            radius = int(self.color_search_radius)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    distance = float(np.hypot(dx, dy))
                    if distance > radius:
                        continue

                    shifted_a = self._shift_map(product_a, dx, dy, 0.0)
                    shifted_b = self._shift_map(product_b, dx, dy, 0.0)
                    shifted_l_detail = self._shift_map(
                        product_l_detail, dx, dy, 0.0
                    )
                    shifted_valid = self._shift_map(
                        valid, dx, dy, False
                    )
                    pair_valid = valid & shifted_valid
                    if not np.any(pair_valid):
                        continue

                    shifted_color = cv2.magnitude(
                        corrected_return_a - shifted_a,
                        corrected_return_b - shifted_b,
                    )
                    shifted_luminance = np.abs(
                        return_l_detail - shifted_l_detail
                    ) * self.perceptual_l_weight
                    shifted_perceptual = cv2.magnitude(
                        shifted_color.astype(np.float32),
                        shifted_luminance.astype(np.float32),
                    )
                    # Prefer exact or near-exact correspondence when scores tie.
                    shifted_perceptual += distance * 1.25
                    shifted_perceptual[~pair_valid] = 1.0e6
                    nearby_perceptual_difference = np.minimum(
                        nearby_perceptual_difference,
                        shifted_perceptual,
                    )

            nearby_perceptual_difference[
                nearby_perceptual_difference >= 1.0e5
            ] = 0.0
            nearby_perceptual_difference[~valid] = 0.0

        visible_candidate = valid & (
            (color_distance >= self.color_diff_threshold)
            | (
                self.perceptual_luminance_enabled
                & (
                    luminance_local_difference
                    >= self.perceptual_l_diff_threshold
                )
            )
            | (
                perceptual_difference
                >= self.perceptual_score_threshold
            )
        )
        visible_error = self._filter_human_visible_components(
            visible_candidate.astype(np.uint8) * 255,
            perceptual_difference,
            valid,
        )

        # Faint errors use lower Lab thresholds but must be a coherent patch.
        # This branch is intentionally separate from visible_candidate so
        # lowering recall for faded colours does not weaken the normal noise
        # rejection rules for strong defects.
        faint_candidate = valid & (
            (color_distance >= self.faint_color_diff_threshold)
            | (
                saturation_difference
                >= self.faint_saturation_diff_threshold
            )
            | (
                self.perceptual_luminance_enabled
                & (
                    luminance_local_difference
                    >= self.faint_luminance_diff_threshold
                )
            )
        )
        faint_color_error = self._filter_faint_color_components(
            faint_candidate.astype(np.uint8) * 255,
            faint_color_score,
            valid,
        )
        visible_error = cv2.bitwise_or(visible_error, faint_color_error)

        self._last_color_offsets = {
            'a_offset': a_offset,
            'b_offset': b_offset,
            'l_median_offset': l_offset,
        }

        return {
            'color_match_distance': np.clip(
                color_distance,
                0,
                255,
            ).astype(np.uint8),
            'saturation_difference': np.clip(
                saturation_difference,
                0,
                255,
            ).astype(np.uint8),
            'faint_color_error': faint_color_error,
            'luminance_local_difference': np.clip(
                luminance_local_difference,
                0,
                255,
            ).astype(np.uint8),
            'perceptual_difference': np.clip(
                perceptual_difference,
                0,
                255,
            ).astype(np.uint8),
            'nearby_perceptual_difference': np.clip(
                nearby_perceptual_difference,
                0,
                255,
            ).astype(np.uint8),
            'visible_error': visible_error,
            # Compatibility name used by the rest of the detector.
            'color_error': visible_error,
        }

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

    def _process_edges(self, product_gray: np.ndarray, return_gray: np.ndarray) -> np.ndarray:
        """Edge difference layer: detects cracks, deformations"""
        product_edges = cv2.Canny(product_gray, self.edge_canny_low, self.edge_canny_high)
        return_edges = cv2.Canny(return_gray, self.edge_canny_low, self.edge_canny_high)
        
        edge_diff = cv2.bitwise_xor(product_edges, return_edges)
        
        kernel = np.ones((self.edge_dilate_kernel, self.edge_dilate_kernel), np.uint8)
        edge_mask = cv2.dilate(edge_diff, kernel, iterations=1)
        
        return edge_mask

    def _process_intensity(self, product_gray: np.ndarray, return_gray: np.ndarray) -> np.ndarray:
        """Intensity difference layer: detects lighting/shadow changes"""
        intensity_diff = cv2.absdiff(product_gray, return_gray)
        
        _, intensity_mask = cv2.threshold(
            intensity_diff, 
            self.intensity_threshold, 
            255, 
            cv2.THRESH_BINARY
        )
        
        intensity_mask = cv2.medianBlur(intensity_mask, self.intensity_blur_kernel)
        
        return intensity_mask

    def _process_texture(self, product_gray: np.ndarray, return_gray: np.ndarray) -> np.ndarray:
        """Texture difference layer: detects scratches, abrasions"""
        def compute_glcm(image):
            # Resize để tăng tốc
            resized = cv2.resize(image, self.texture_resize_dim)
            # Tính GLCM dissimilarity
            from skimage.feature import greycomatrix, greycoprops
            glcm = greycomatrix(resized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            return greycoprops(glcm, 'dissimilarity')[0, 0]
        
        product_glcm = compute_glcm(product_gray)
        return_glcm = compute_glcm(return_gray)
        
        texture_diff = abs(product_glcm - return_glcm)
        
        if texture_diff > self.texture_diff_threshold:
            texture_mask = np.full_like(product_gray, 255, dtype=np.uint8)
        else:
            texture_mask = np.zeros_like(product_gray, dtype=np.uint8)
        
        return texture_mask
    def _process_geometric_intensity(self, product_image: np.ndarray, return_image: np.ndarray) -> np.ndarray:
        """Compute binary mask from ORB-matched points where |ΔI| >= THRESHOLD.
        Returns a sparse mask (white dots at matched damaged locations), dilated for visibility.
        """
        from mesh_config import (
            GEOMETRIC_INTENSITY_ENABLED,
            GEOMETRIC_INTENSITY_THRESHOLD,
            GEOMETRIC_INTENSITY_DILATE_KERNEL,
        )
        if not GEOMETRIC_INTENSITY_ENABLED:
            return np.zeros(product_image.shape[:2], dtype=np.uint8)

        # Convert to grayscale
        product_gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
        return_gray = cv2.cvtColor(return_image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints & compute descriptors
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        kp1, des1 = orb.detectAndCompute(product_gray, None)
        kp2, des2 = orb.detectAndCompute(return_gray, None)

        if len(kp1) == 0 or len(kp2) == 0 or des1 is None or des2 is None:
            return np.zeros(product_image.shape[:2], dtype=np.uint8)

        # Match
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) == 0:
            return np.zeros(product_image.shape[:2], dtype=np.uint8)

        # Extract matched point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

        # Clamp coordinates to image bounds
        h1, w1 = product_gray.shape
        h2, w2 = return_gray.shape
        pts1[:, 0] = np.clip(pts1[:, 0], 0, w1 - 1)
        pts1[:, 1] = np.clip(pts1[:, 1], 0, h1 - 1)
        pts2[:, 0] = np.clip(pts2[:, 0], 0, w2 - 1)
        pts2[:, 1] = np.clip(pts2[:, 1], 0, h2 - 1)

        # Optional RANSAC filtering for robustness
        if len(pts1) >= 10:
            try:
                H, mask = cv2.findHomography(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if mask is not None:
                    inliers = mask.ravel().astype(bool)
                    pts1 = pts1[inliers]
                    pts2 = pts2[inliers]
            except Exception:
                pass  # fallback to all matches if RANSAC fails

        # Compute intensity deltas
        deltas = []
        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            i1 = int(product_gray[int(y1), int(x1)])
            i2 = int(return_gray[int(y2), int(x2)])
            deltas.append(abs(i1 - i2))

        deltas = np.array(deltas)
        damaged_mask = np.zeros(product_image.shape[:2], dtype=np.uint8)

        # Draw white dots at damaged matches
        for (x, y), delta in zip(pts1, deltas):
            if delta >= GEOMETRIC_INTENSITY_THRESHOLD:
                cv2.circle(damaged_mask, (int(x), int(y)), radius=1, color=255, thickness=-1)

        # Dilate for robustness
        if GEOMETRIC_INTENSITY_DILATE_KERNEL > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (GEOMETRIC_INTENSITY_DILATE_KERNEL, GEOMETRIC_INTENSITY_DILATE_KERNEL)
            )
            damaged_mask = cv2.dilate(damaged_mask, kernel)

        return damaged_mask


    def _combine_layers(self, edge_mask: np.ndarray, intensity_mask: np.ndarray, texture_mask: np.ndarray) -> np.ndarray:
        """Combine all layers with weighted voting"""
        # Apply weights
        weighted_edge = (edge_mask.astype(np.float32) * self.edge_weight).astype(np.uint8)
        weighted_intensity = (intensity_mask.astype(np.float32) * self.intensity_weight).astype(np.uint8)
        weighted_texture = (texture_mask.astype(np.float32) * self.texture_weight).astype(np.uint8)
        
        # Sum and threshold
        combined = cv2.addWeighted(weighted_edge, 1.0, weighted_intensity, 1.0, 0.0)
        combined = cv2.addWeighted(combined, 1.0, weighted_texture, 1.0, 0.0)
        
        _, final_mask = cv2.threshold(combined, self.final_binary_threshold, 255, cv2.THRESH_BINARY)
        
        # Final morphological closing
        kernel = np.ones((3,3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=self.final_close_iterations)
        
        return final_mask

    def _compute_damage_map(self, product_image: np.ndarray, return_image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Compute damage map using multi-layer approach"""
        # Convert to grayscale
        product_gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
        return_gray = cv2.cvtColor(return_image, cv2.COLOR_BGR2GRAY)
        
        # Process each layer
        edge_mask = self._process_edges(product_gray, return_gray)
        intensity_mask = self._process_intensity(product_gray, return_gray)
        texture_mask = self._process_texture(product_gray, return_gray)
        
        geometric_mask = self._process_geometric_intensity(product_image, return_image)

        # Combine layers
        damage_mask = self._combine_layers(edge_mask, intensity_mask, texture_mask, geometric_mask)
        
        # Store debug maps
        maps = {
            'edge_mask': edge_mask,
            'intensity_mask': intensity_mask,
            'texture_mask': texture_mask,
            'combined_mask': damage_mask,
        }
        
        return damage_mask, maps

    def _build_object_interior_masks(
        self,
        product_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build object, safe-interior, and boundary-band masks.

        The damage detector is intentionally conservative near the product
        silhouette because small alignment errors against a dark background
        create strong false edges there.
        """
        gray = self._to_gray(product_image)
        binary = (gray > 8).astype(np.uint8) * 255

        cleanup_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cleanup_kernel,
            iterations=2,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cleanup_kernel,
            iterations=1,
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        object_mask = np.zeros_like(binary)
        if num_labels > 1:
            largest_label = 1 + int(
                np.argmax(stats[1:, cv2.CC_STAT_AREA])
            )
            object_mask[labels == largest_label] = 255

        contours, _ = cv2.findContours(
            object_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            object_mask.fill(0)
            cv2.drawContours(
                object_mask,
                contours,
                contourIdx=-1,
                color=255,
                thickness=cv2.FILLED,
            )

        height, width = object_mask.shape[:2]
        margin = int(
            np.clip(
                round(min(height, width) * 0.008),
                4,
                12,
            )
        )
        erode_size = margin * 2 + 1
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erode_size, erode_size),
        )
        interior_mask = cv2.erode(
            object_mask,
            erode_kernel,
            iterations=1,
        )
        boundary_band = cv2.subtract(
            object_mask,
            interior_mask,
        )

        return object_mask, interior_mask, boundary_band

    def _compute_global_pixel_damage(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        candidate_mask: np.ndarray,
        interior_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Compute mesh-gated structural evidence plus raw interior color evidence.

        Edge, texture, and brightness remain restricted to suspicious mesh
        triangles. Exact-position Lab a/b color defects are evaluated over the
        full safe interior and can bypass the mesh gate.
        """
        active = candidate_mask > 0
        color_active = interior_mask > 0
        shape = candidate_mask.shape
        empty = np.zeros(shape, dtype=np.uint8)

        if not np.any(color_active):
            return empty, {
                'raw_residual': empty,
                'texture': empty,
                'edge': empty,
                'chroma': empty,
                'illumination_reject': empty,
                'product_texture': empty,
                'product_edge': empty,
                'return_edge': empty,
                'matched_edge': empty,
                'new_edge': empty,
                'edge_match_score': empty,
                'chroma_new': empty,
                'color_match_distance': empty,
                'saturation_difference': empty,
                'faint_color_error': empty,
                'luminance_local_difference': empty,
                'perceptual_difference': empty,
                'nearby_perceptual_difference': empty,
                'visible_error': empty,
                'color_error': empty,
                'color_error_inside': empty,
                'strong_color_error_outside': empty,
                'raw_color_damage': empty,
                'texture_suppressed': empty,
            }

        product_gray = cv2.GaussianBlur(self._to_gray(product_image), (3, 3), 0)
        return_gray = cv2.GaussianBlur(self._to_gray(return_image), (3, 3), 0)
        normalization_active = active if np.any(active) else color_active
        return_norm = self._local_illumination_normalize(
            product_gray,
            return_gray,
            normalization_active,
        )
        raw_residual = cv2.absdiff(product_gray, return_norm)

        min_extent = min(product_gray.shape[:2])
        hp_kernel = self._odd_kernel(31, min_extent)
        product_base = cv2.GaussianBlur(
            product_gray,
            (hp_kernel, hp_kernel),
            0,
        ).astype(np.float32)
        return_base = cv2.GaussianBlur(
            return_norm,
            (hp_kernel, hp_kernel),
            0,
        ).astype(np.float32)
        product_hp = product_gray.astype(np.float32) - product_base
        return_hp = return_norm.astype(np.float32) - return_base
        texture_diff = np.clip(
            np.abs(product_hp - return_hp),
            0,
            255,
        ).astype(np.uint8)

        product_edge, product_orientation = self._gradient_magnitude_and_orientation(
            product_hp,
        )
        return_edge, return_orientation = self._gradient_magnitude_and_orientation(
            return_hp,
        )
        edge_diff = cv2.absdiff(product_edge, return_edge)
        edge_maps = self._build_edge_match_maps(
            product_edge,
            product_orientation,
            return_edge,
            return_orientation,
        )
        matched_edge = edge_maps['matched_edge'] > 0
        unmatched_edge = edge_maps['unmatched_edge'] > 0
        matched_neighborhood = self._dilate_binary(
            edge_maps['matched_edge'],
            ksize=3,
            iterations=1,
        ) > 0

        if product_image.ndim == 3 and return_image.ndim == 3:
            product_lab = cv2.cvtColor(product_image, cv2.COLOR_BGR2LAB)
            return_lab = cv2.cvtColor(return_image, cv2.COLOR_BGR2LAB)
            color_maps = self._build_color_match_maps(
                product_lab,
                return_lab,
                color_active,
            )
            chroma_diff = color_maps['color_match_distance']
            saturation_difference = color_maps['saturation_difference']
            faint_color_error = color_maps['faint_color_error']
            luminance_local_difference = color_maps[
                'luminance_local_difference'
            ]
            perceptual_difference = color_maps['perceptual_difference']
            nearby_perceptual_difference = color_maps[
                'nearby_perceptual_difference'
            ]
            visible_error = color_maps['visible_error'] > 0
            color_error_inside = visible_error & active
            outside_candidate = (
                (~active)
                & color_active
                & (
                    nearby_perceptual_difference
                    >= self.perceptual_outside_mesh_score_threshold
                )
            )
            strict_outside = self._filter_visible_outside_mesh(
                outside_candidate.astype(np.uint8) * 255,
                nearby_perceptual_difference,
                matched_neighborhood.astype(np.uint8) * 255,
                color_active,
            ) > 0
            # Faint errors have already passed the stricter coherent-component
            # filter.  They can be genuine soft grey/faded marks outside a
            # low-similarity triangle, so do not send them through the
            # strong-colour (>=30) outside-mesh gate.  Existing matched edges
            # remain rejected to avoid re-labelling painted outlines as damage.
            faint_outside = (
                (faint_color_error > 0)
                & (~active)
                & color_active
                & (~matched_neighborhood)
            )
            color_error = color_error_inside | strict_outside | faint_outside
            # Compatibility key: strict_outside remains the strong branch;
            # faint_outside is visible in faint_color_error diagnostics.
            strong_color_error_outside = strict_outside
            product_color_strength = (
                np.abs(product_lab[:, :, 1].astype(np.int16) - 128)
                + np.abs(product_lab[:, :, 2].astype(np.int16) - 128)
            )
            product_color_strength = np.clip(
                product_color_strength,
                0,
                255,
            ).astype(np.uint8)
        else:
            chroma_diff = empty.copy()
            saturation_difference = empty.copy()
            faint_color_error = empty.copy()
            luminance_local_difference = empty.copy()
            perceptual_difference = empty.copy()
            nearby_perceptual_difference = empty.copy()
            visible_error = np.zeros(shape, dtype=bool)
            color_error = np.zeros(shape, dtype=bool)
            color_error_inside = np.zeros(shape, dtype=bool)
            strong_color_error_outside = np.zeros(shape, dtype=bool)
            product_color_strength = empty.copy()

        product_edge_binary = product_edge >= 42
        product_color_binary = product_color_strength >= 36
        product_texture = (
            product_edge_binary | product_color_binary
        ).astype(np.uint8) * 255
        product_texture_dilated = self._dilate_binary(
            product_texture,
            ksize=7,
            iterations=1,
        ) > 0

        new_edge = unmatched_edge & active
        outside_product_texture = (~product_texture_dilated) & active
        chroma_new = color_error

        raw_low = max(16, int(round(self.diff_threshold * 0.65)))
        texture_mid = max(20, int(round(self.diff_threshold * 0.80)))
        edge_mid = max(28, int(round(self.diff_threshold * 1.10)))
        chroma_mid = 24
        illumination_like = (
            (raw_residual >= raw_low)
            & (texture_diff < texture_mid)
            & (edge_diff < edge_mid)
            & (chroma_diff < chroma_mid)
            & (~color_error)
            & active
        )

        texture_high = max(32, int(round(self.diff_threshold * 1.25)))
        edge_high = max(44, int(round(self.diff_threshold * 1.70)))
        raw_high = max(48, int(self.diff_threshold) + 22)

        texture_outside = (
            (texture_diff >= texture_high)
            & outside_product_texture
            & (~matched_neighborhood)
        )
        edge_outside = (
            (edge_diff >= edge_high)
            & outside_product_texture
            & (~matched_neighborhood)
        )
        gated_raw = (
            (raw_residual >= raw_high)
            & outside_product_texture
            & (~matched_neighborhood)
            & (
                (texture_diff >= texture_mid)
                | (edge_diff >= edge_mid)
                | (chroma_diff >= chroma_mid)
            )
        )

        weak_inside_texture = (
            (
                (texture_diff >= texture_high)
                | (edge_diff >= edge_high)
                | color_error_inside
            )
            & product_texture_dilated
            & active
        )
        supported_inside_texture = weak_inside_texture & (
            new_edge | color_error_inside
        )

        # =====================================================================
        # CRITICAL FIX: Use triangle voting result as PRIMARY damage indicator
        # Pixel-level differences are ONLY used for strong confirmation
        # =====================================================================
        # The 'active' mask comes from triangle voting - this is our most reliable
        # damage indicator. Pixel differences create false positives from lighting
        # and geometric distortion on curved ceramic surfaces.
        
        # Only use pixel-level checks for VERY STRONG evidence outside triangles
        texture_outside_strong = (
            (texture_diff >= max(60, int(self.diff_threshold * 2.0)))
            & outside_product_texture
            & (~matched_neighborhood)
            & (~active)  # Only outside suspected triangles
        )
        edge_outside_strong = (
            (edge_diff >= max(80, int(self.diff_threshold * 2.5)))
            & outside_product_texture
            & (~matched_neighborhood)
            & (~active)  # Only outside suspected triangles
        )
        raw_residual_strong = (
            (raw_residual >= max(80, int(self.diff_threshold * 2.5)))
            & outside_product_texture
            & (~matched_neighborhood)
            & (~active)  # Only outside suspected triangles
            & (texture_diff >= 40)  # Must have texture confirmation
        )
        
        # Final damage = triangle voting result OR very strong pixel evidence
        mesh_damage_pixels = (
            active  # Triangle voting result is PRIMARY
            | texture_outside_strong
            | edge_outside_strong
            | raw_residual_strong
        ) & (~illumination_like)
        mesh_damage_mask = self._remove_small_components(
            mesh_damage_pixels.astype(np.uint8) * 255,
            max(2, self.min_area_threshold // 3),
        )

        # Detection-first: color defects are already cleaned at COLOR_MIN_AREA
        # and bypass the mesh gate without morphology opening.
        raw_color_damage = color_error.astype(np.uint8) * 255
        damage_mask = cv2.bitwise_or(mesh_damage_mask, raw_color_damage)

        maps = {
            'raw_residual': np.where(active, raw_residual, 0).astype(np.uint8),
            'texture': np.where(active, texture_diff, 0).astype(np.uint8),
            'edge': np.where(active, edge_diff, 0).astype(np.uint8),
            'chroma': np.where(color_active, chroma_diff, 0).astype(np.uint8),
            'illumination_reject': (
                illumination_like & active
            ).astype(np.uint8) * 255,
            'product_texture': product_texture,
            'product_edge': edge_maps['product_edge'],
            'return_edge': edge_maps['return_edge'],
            'matched_edge': (
                matched_edge & active
            ).astype(np.uint8) * 255,
            'new_edge': (
                new_edge & active
            ).astype(np.uint8) * 255,
            'edge_match_score': np.where(
                active,
                edge_maps['edge_match_score'],
                0,
            ).astype(np.uint8),
            'chroma_new': raw_color_damage,
            'color_match_distance': np.where(
                color_active,
                chroma_diff,
                0,
            ).astype(np.uint8),
            'saturation_difference': np.where(
                color_active,
                saturation_difference,
                0,
            ).astype(np.uint8),
            'faint_color_error': faint_color_error.astype(np.uint8),
            'luminance_local_difference': np.where(
                color_active,
                luminance_local_difference,
                0,
            ).astype(np.uint8),
            'perceptual_difference': np.where(
                color_active,
                perceptual_difference,
                0,
            ).astype(np.uint8),
            'nearby_perceptual_difference': np.where(
                color_active,
                nearby_perceptual_difference,
                0,
            ).astype(np.uint8),
            'visible_error': visible_error.astype(np.uint8) * 255,
            'color_error': raw_color_damage,
            'color_error_inside': color_error_inside.astype(np.uint8) * 255,
            'strong_color_error_outside': (
                strong_color_error_outside.astype(np.uint8) * 255
            ),
            'raw_color_damage': raw_color_damage,
            'texture_suppressed': (
                weak_inside_texture & (~supported_inside_texture)
            ).astype(np.uint8) * 255,
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
            strong_chroma_threshold = max(28.0, self.color_diff_threshold + 8.0)
            strong_chroma_ratio = float(
                np.count_nonzero((chroma_score >= strong_chroma_threshold) & component)
            ) / float(area)

            # Positive evidence: a new return edge or a strong chroma change.
            has_new_evidence = (
                (new_edge_ratio >= 0.035 and edge_mean >= 30.0)
                or chroma_new_ratio >= 0.050
                or strong_chroma_ratio >= 0.020
                or (outside_texture_ratio >= 0.45 and (tex_mean >= 34.0 or edge_mean >= 36.0 or chroma_mean >= 34.0))
            )

            # Negative evidence: mostly overlaps product's old ornament and no new edge/color.
            old_texture_like = (
                texture_overlap >= 0.60
                and new_edge_ratio < 0.025
                and chroma_new_ratio < 0.040
                and strong_chroma_ratio < 0.018
            )

            # Negative evidence: big/dense/diffuse blobs are usually lighting or misalignment.
            diffuse_blob = (
                (area > 1800 or area_ratio > 0.004)
                and fill_ratio > 0.22
                and new_edge_ratio < 0.030
                and chroma_new_ratio < 0.050
            )

            if old_texture_like or diffuse_blob or not has_new_evidence:
                component_reject[component] = 255
                if old_texture_like:
                    texture_suppressed[component] = 255
            else:
                cleaned[component] = 255

        return cleaned, component_reject, texture_suppressed

    def _warp_return_by_mesh(
        self,
        return_image: np.ndarray,
        vertices: Sequence[MeshVertex],
        triangle_results: Sequence[ScoredTriangle],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map the return image into product coordinates one mesh triangle at a time.

        A single homography cannot align artwork painted on a curved vessel.
        Each triangle instead supplies a local affine correspondence from its
        three matched return points to its product points.  Only pixels with
        such a measured local mapping are allowed into the later Lab/edge
        comparison; unmapped pixels cannot create false damage merely because
        the two photographs have different perspective.
        """
        height, width = return_image.shape[:2]
        warped = np.zeros_like(return_image)
        coverage = np.zeros((height, width), dtype=np.uint8)

        for result in triangle_results:
            triangle = result.triangle
            try:
                source = np.asarray(
                    [vertices[index].return_point for index in triangle.vertex_indices],
                    dtype=np.float32,
                )
                destination = np.asarray(
                    [vertices[index].product_point for index in triangle.vertex_indices],
                    dtype=np.float32,
                )
            except (IndexError, TypeError):
                continue

            if abs(float(cv2.contourArea(destination))) < 1.0:
                continue

            x, y, w, h = cv2.boundingRect(destination.astype(np.int32))
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            if x0 >= x1 or y0 >= y1:
                continue

            # cv2.warpAffine writes destination pixels.  Shift the affine
            # output into the triangle's small bounding box to avoid warping
            # the full image separately for every mesh triangle.
            matrix = cv2.getAffineTransform(source, destination)
            local_matrix = matrix.copy()
            local_matrix[0, 2] -= x0
            local_matrix[1, 2] -= y0
            local_warp = cv2.warpAffine(
                return_image,
                local_matrix,
                (x1 - x0, y1 - y0),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            polygon = np.rint(destination - np.asarray([x0, y0])).astype(np.int32)
            local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillConvexPoly(local_mask, polygon, 255)
            target = warped[y0:y1, x0:x1]
            target_coverage = coverage[y0:y1, x0:x1]
            target[local_mask > 0] = local_warp[local_mask > 0]
            target_coverage[local_mask > 0] = 255

        return warped, coverage

    def _build_damage_map(
        self,
        triangle_results: Sequence[ScoredTriangle],
        vertices: Sequence[MeshVertex],
        product_image: np.ndarray,
        return_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build final damage while protecting raw interior color defects."""
        height, width = product_image.shape[:2]
        damage_map = np.zeros((height, width), dtype=np.uint8)
        suspected_mask = np.zeros((height, width), dtype=np.uint8)
        refined_overlay = np.zeros((height, width), dtype=np.uint8)
        severity_map = np.zeros((height, width), dtype=np.uint8)

        debug_scores = {
            'raw_residual': np.zeros((height, width), dtype=np.uint8),
            'texture': np.zeros((height, width), dtype=np.uint8),
            'edge': np.zeros((height, width), dtype=np.uint8),
            'chroma': np.zeros((height, width), dtype=np.uint8),
            'illumination_reject': np.zeros((height, width), dtype=np.uint8),
            'component_reject': np.zeros((height, width), dtype=np.uint8),
            'product_texture': np.zeros((height, width), dtype=np.uint8),
            'product_edge': np.zeros((height, width), dtype=np.uint8),
            'return_edge': np.zeros((height, width), dtype=np.uint8),
            'matched_edge': np.zeros((height, width), dtype=np.uint8),
            'new_edge': np.zeros((height, width), dtype=np.uint8),
            'edge_match_score': np.zeros((height, width), dtype=np.uint8),
            'chroma_new': np.zeros((height, width), dtype=np.uint8),
            'color_match_distance': np.zeros((height, width), dtype=np.uint8),
            'luminance_local_difference': np.zeros((height, width), dtype=np.uint8),
            'perceptual_difference': np.zeros((height, width), dtype=np.uint8),
            'nearby_perceptual_difference': np.zeros((height, width), dtype=np.uint8),
            'visible_error': np.zeros((height, width), dtype=np.uint8),
            'color_error': np.zeros((height, width), dtype=np.uint8),
            'color_error_inside': np.zeros((height, width), dtype=np.uint8),
            'strong_color_error_outside': np.zeros((height, width), dtype=np.uint8),
            'raw_color_damage': np.zeros((height, width), dtype=np.uint8),
            'texture_suppressed': np.zeros((height, width), dtype=np.uint8),
            'boundary_reject': np.zeros((height, width), dtype=np.uint8),
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
                refined_overlay = np.maximum(
                    refined_overlay,
                    (mask > 0).astype(np.uint8) * 255,
                )

            if not result.is_damaged:
                continue

            active = mask > 0
            suspected_mask[active] = 255
            # Use binary severity only - no interpolation to avoid gray areas
            # Damage triangles must be clearly flagged by voting logic
            severity = 255
            severity_map[active] = severity

        object_mask, interior_mask, boundary_band = self._build_object_interior_masks(
            product_image,
        )

        locally_warped_return, local_warp_coverage = self._warp_return_by_mesh(
            return_image,
            vertices,
            triangle_results,
        )
        # The mesh has no measured correspondence when there are fewer than
        # three usable inliers.  Preserve the previous fallback in that case,
        # but otherwise never compare unregistered image regions.
        if np.count_nonzero(local_warp_coverage) >= 128:
            comparison_interior = cv2.bitwise_and(
                interior_mask,
                local_warp_coverage,
            )
            comparison_return = locally_warped_return
        else:
            comparison_interior = interior_mask
            comparison_return = return_image
        self._last_local_warp_coverage = local_warp_coverage
        self._last_locally_warped_return = comparison_return

        # CRITICAL FIX: Use ONLY triangle voting result for damage detection
        # Do NOT compute pixel-level differences which cause false positive gray areas
        # The triangle voting logic already combines SSIM + Feature + Similarity scores
        
        # Build final damage map DIRECTLY from suspected_mask (triangle voting result)
        damage_map = np.zeros((height, width), dtype=np.uint8)
        
        # Only mark pixels where triangle voting flagged as damaged
        # This eliminates gray areas from pixel-level differences
        damage_map[suspected_mask > 0] = 255
        
        # Store debug masks (all zeros since we bypassed pixel-level computation)
        self._last_pixel_debug_masks = {
            'raw_residual_mask': np.zeros((height, width), dtype=np.uint8),
            'texture_mask': np.zeros((height, width), dtype=np.uint8),
            'edge_mask': np.zeros((height, width), dtype=np.uint8),
            'chroma_mask': np.zeros((height, width), dtype=np.uint8),
            'illumination_reject_mask': np.zeros((height, width), dtype=np.uint8),
            'component_reject_mask': np.zeros((height, width), dtype=np.uint8),
            'product_texture_mask': np.zeros((height, width), dtype=np.uint8),
            'product_edge_mask': np.zeros((height, width), dtype=np.uint8),
            'return_edge_mask': np.zeros((height, width), dtype=np.uint8),
            'matched_edge_mask': np.zeros((height, width), dtype=np.uint8),
            'new_edge_mask': np.zeros((height, width), dtype=np.uint8),
            'edge_match_score_mask': np.zeros((height, width), dtype=np.uint8),
            'chroma_new_mask': np.zeros((height, width), dtype=np.uint8),
            'color_match_distance_mask': np.zeros((height, width), dtype=np.uint8),
            'saturation_difference_mask': np.zeros((height, width), dtype=np.uint8),
            'faint_color_error_mask': np.zeros((height, width), dtype=np.uint8),
            'luminance_local_difference_mask': np.zeros((height, width), dtype=np.uint8),
            'perceptual_difference_mask': np.zeros((height, width), dtype=np.uint8),
            'nearby_perceptual_difference_mask': np.zeros((height, width), dtype=np.uint8),
            'visible_error_mask': np.zeros((height, width), dtype=np.uint8),
            'color_error_mask': np.zeros((height, width), dtype=np.uint8),
            'color_error_inside_mask': np.zeros((height, width), dtype=np.uint8),
            'strong_color_error_outside_mask': np.zeros((height, width), dtype=np.uint8),
            'raw_color_damage_mask': np.zeros((height, width), dtype=np.uint8),
            'texture_suppressed_mask': np.zeros((height, width), dtype=np.uint8),
            'boundary_reject_mask': np.zeros((height, width), dtype=np.uint8),
            'local_warp_coverage_mask': local_warp_coverage,
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
        product_edge_mask = pixel_debug.get('product_edge_mask', np.zeros_like(final_mask))
        return_edge_mask = pixel_debug.get('return_edge_mask', np.zeros_like(final_mask))
        matched_edge_mask = pixel_debug.get('matched_edge_mask', np.zeros_like(final_mask))
        edge_match_score_mask = pixel_debug.get('edge_match_score_mask', np.zeros_like(final_mask))
        new_edge_mask = pixel_debug.get('new_edge_mask', np.zeros_like(final_mask))
        chroma_new_mask = pixel_debug.get('chroma_new_mask', np.zeros_like(final_mask))
        color_match_distance_mask = pixel_debug.get(
            'color_match_distance_mask',
            np.zeros_like(final_mask),
        )
        saturation_difference_mask = pixel_debug.get(
            'saturation_difference_mask',
            np.zeros_like(final_mask),
        )
        faint_color_error_mask = pixel_debug.get(
            'faint_color_error_mask',
            np.zeros_like(final_mask),
        )
        luminance_local_difference_mask = pixel_debug.get(
            'luminance_local_difference_mask',
            np.zeros_like(final_mask),
        )
        perceptual_difference_mask = pixel_debug.get(
            'perceptual_difference_mask',
            np.zeros_like(final_mask),
        )
        nearby_perceptual_difference_mask = pixel_debug.get(
            'nearby_perceptual_difference_mask',
            np.zeros_like(final_mask),
        )
        visible_error_mask = pixel_debug.get(
            'visible_error_mask',
            np.zeros_like(final_mask),
        )
        color_error_mask = pixel_debug.get('color_error_mask', np.zeros_like(final_mask))
        color_error_inside_mask = pixel_debug.get(
            'color_error_inside_mask',
            np.zeros_like(final_mask),
        )
        strong_color_error_outside_mask = pixel_debug.get(
            'strong_color_error_outside_mask',
            np.zeros_like(final_mask),
        )
        raw_color_damage_mask = pixel_debug.get(
            'raw_color_damage_mask',
            color_error_mask,
        )
        texture_suppressed_mask = pixel_debug.get('texture_suppressed_mask', np.zeros_like(final_mask))
        boundary_reject_mask = pixel_debug.get('boundary_reject_mask', np.zeros_like(final_mask))
        local_warp_coverage_mask = pixel_debug.get(
            'local_warp_coverage_mask', np.zeros_like(final_mask)
        )

        return {
            'intensity_pixels': int(np.count_nonzero(raw_residual_mask)),
            'gradient_pixels': int(np.count_nonzero(edge_mask)),
            'boundary_reject_pixels': int(np.count_nonzero(boundary_reject_mask)),
            'local_warp_coverage_pixels': int(
                np.count_nonzero(local_warp_coverage_mask)
            ),
            'matched_edge_pixels': int(np.count_nonzero(matched_edge_mask)),
            'unmatched_edge_pixels': int(np.count_nonzero(new_edge_mask)),
            'visible_error_pixels': int(np.count_nonzero(visible_error_mask)),
            'luminance_local_difference_pixels': int(
                np.count_nonzero(luminance_local_difference_mask)
            ),
            'color_error_pixels': int(np.count_nonzero(color_error_mask)),
            'color_error_inside_pixels': int(np.count_nonzero(color_error_inside_mask)),
            'strong_color_error_outside_pixels': int(
                np.count_nonzero(strong_color_error_outside_mask)
            ),
            'raw_color_damage_pixels': int(
                np.count_nonzero(raw_color_damage_mask)
            ),
            'edge_search_radius': int(self.edge_search_radius),
            'edge_match_score_threshold': float(self.edge_match_score_threshold),
            'color_search_radius': int(self.color_search_radius),
            'color_diff_threshold': float(self.color_diff_threshold),
            'strong_color_diff_threshold': float(
                self.strong_color_diff_threshold
            ),
            'color_min_area': int(self.color_min_area),
            'perceptual_luminance_enabled': bool(
                self.perceptual_luminance_enabled
            ),
            'perceptual_l_diff_threshold': float(
                self.perceptual_l_diff_threshold
            ),
            'perceptual_score_threshold': float(
                self.perceptual_score_threshold
            ),
            'perceptual_strong_score_threshold': float(
                self.perceptual_strong_score_threshold
            ),
            'perceptual_component_filter': dict(
                getattr(self, '_last_visible_component_stats', {}) or {}
            ),
            'outside_mesh_visible_filter': dict(
                getattr(self, '_last_outside_visible_stats', {}) or {}
            ),
            'color_offsets': dict(getattr(self, '_last_color_offsets', {}) or {}),
            'faint_color_component_filter': dict(
                getattr(self, '_last_faint_color_component_stats', {}) or {}
            ),
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
                'product_edge_mask': product_edge_mask,
                'return_edge_mask': return_edge_mask,
                'matched_edge_mask': matched_edge_mask,
                'edge_match_score_mask': edge_match_score_mask,
                'new_edge_mask': new_edge_mask,
                'chroma_new_mask': chroma_new_mask,
                'color_match_distance_mask': color_match_distance_mask,
                'saturation_difference_mask': saturation_difference_mask,
                'faint_color_error_mask': faint_color_error_mask,
                'luminance_local_difference_mask': luminance_local_difference_mask,
                'perceptual_difference_mask': perceptual_difference_mask,
                'nearby_perceptual_difference_mask': nearby_perceptual_difference_mask,
                'visible_error_mask': visible_error_mask,
                'color_error_mask': color_error_mask,
                'color_error_inside_mask': color_error_inside_mask,
                'strong_color_error_outside_mask': strong_color_error_outside_mask,
                'raw_color_damage_mask': raw_color_damage_mask,
                'texture_suppressed_mask': texture_suppressed_mask,
                'boundary_reject_mask': boundary_reject_mask,
                'local_warp_coverage_mask': local_warp_coverage_mask,
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

        pixel_debug = getattr(self, '_last_pixel_debug_masks', {}) or {}
        matched_edge_pixels = int(np.count_nonzero(pixel_debug.get('matched_edge_mask', 0)))
        unmatched_edge_pixels = int(np.count_nonzero(pixel_debug.get('new_edge_mask', 0)))
        color_error_pixels = int(np.count_nonzero(pixel_debug.get('color_error_mask', 0)))
        color_error_inside_pixels = int(
            np.count_nonzero(pixel_debug.get('color_error_inside_mask', 0))
        )
        strong_color_error_outside_pixels = int(
            np.count_nonzero(pixel_debug.get('strong_color_error_outside_mask', 0))
        )

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
            "matched_edge_pixels": matched_edge_pixels,
            "unmatched_edge_pixels": unmatched_edge_pixels,
            "edge_search_radius": int(self.edge_search_radius),
            "edge_match_score_threshold": float(self.edge_match_score_threshold),
            "color_error_pixels": color_error_pixels,
            "color_error_inside_pixels": color_error_inside_pixels,
            "strong_color_error_outside_pixels": strong_color_error_outside_pixels,
            "color_search_radius": int(self.color_search_radius),
            "color_diff_threshold": float(self.color_diff_threshold),
            "faint_color_diff_threshold": float(self.faint_color_diff_threshold),
            "faint_saturation_diff_threshold": float(
                self.faint_saturation_diff_threshold
            ),
            "faint_color_component_filter": dict(
                getattr(self, '_last_faint_color_component_stats', {}) or {}
            ),
            "strong_color_diff_threshold": float(
                self.strong_color_diff_threshold
            ),
            "color_min_area": int(self.color_min_area),
            "perceptual_luminance_enabled": bool(
                self.perceptual_luminance_enabled
            ),
            "perceptual_l_diff_threshold": float(
                self.perceptual_l_diff_threshold
            ),
            "perceptual_score_threshold": float(
                self.perceptual_score_threshold
            ),
            "perceptual_strong_score_threshold": float(
                self.perceptual_strong_score_threshold
            ),
            "perceptual_component_filter": dict(
                getattr(self, '_last_visible_component_stats', {}) or {}
            ),
            "outside_mesh_visible_filter": dict(
                getattr(self, '_last_outside_visible_stats', {}) or {}
            ),
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
            "08b_locally_warped_return.png": getattr(
                self, '_last_locally_warped_return', return_image
            ),
        }

        pixel_debug = getattr(self, '_last_pixel_debug_masks', {}) or {}
        for extra_name, extra_image in {
            '09_raw_residual_mask.png': pixel_debug.get('raw_residual_mask'),
            '09b_local_warp_coverage.png': pixel_debug.get(
                'local_warp_coverage_mask'
            ),
            '10_texture_mask.png': pixel_debug.get('texture_mask'),
            '11_edge_mask.png': pixel_debug.get('edge_mask'),
            '12_chroma_mask.png': pixel_debug.get('chroma_mask'),
            '13_illumination_reject_mask.png': pixel_debug.get('illumination_reject_mask'),
            '14_component_reject_mask.png': pixel_debug.get('component_reject_mask'),
            '15_product_texture_mask.png': pixel_debug.get('product_texture_mask'),
            '16_new_edge_mask.png': pixel_debug.get('new_edge_mask'),
            '17_chroma_new_mask.png': pixel_debug.get('chroma_new_mask'),
            '18_texture_suppressed_mask.png': pixel_debug.get('texture_suppressed_mask'),
            '19_boundary_reject_mask.png': pixel_debug.get('boundary_reject_mask'),
            '20_product_edge.png': pixel_debug.get('product_edge_mask'),
            '21_return_edge.png': pixel_debug.get('return_edge_mask'),
            '22_matched_edge.png': pixel_debug.get('matched_edge_mask'),
            '23_unmatched_edge.png': pixel_debug.get('new_edge_mask'),
            '24_edge_match_score.png': pixel_debug.get('edge_match_score_mask'),
            '25_color_match_distance.png': pixel_debug.get('color_match_distance_mask'),
            '25b_saturation_difference.png': pixel_debug.get(
                'saturation_difference_mask'
            ),
            '25c_faint_color_error.png': pixel_debug.get(
                'faint_color_error_mask'
            ),
            '26_color_error_mask.png': pixel_debug.get('color_error_mask'),
            '27_color_error_inside_mesh.png': pixel_debug.get('color_error_inside_mask'),
            '28_strong_color_error_outside_mesh.png': pixel_debug.get(
                'strong_color_error_outside_mask'
            ),
            '29_raw_color_damage_mask.png': pixel_debug.get(
                'raw_color_damage_mask'
            ),
            '30_luminance_local_difference.png': pixel_debug.get(
                'luminance_local_difference_mask'
            ),
            '31_perceptual_difference.png': pixel_debug.get(
                'perceptual_difference_mask'
            ),
            '32_visible_error_mask.png': pixel_debug.get(
                'visible_error_mask'
            ),
            '33_nearby_perceptual_difference.png': pixel_debug.get(
                'nearby_perceptual_difference_mask'
            ),
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
