"""Region similarity evaluation for individual mesh triangles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mesh_builder import MeshTriangle, MeshVertex
from region_extractor import RegionExtractor
from region_similarity import RegionSimilarity
from similarity_cache import ScoredTriangle, SimilarityCache


class TriangleSimilarity:
    """Compute and cache per-triangle similarity scores."""

    def __init__(
        self,
        similarity_threshold: float,
        feature_threshold: float,
        ssim_threshold: float,
        workers: int = 4,
    ):
        self.similarity_threshold = similarity_threshold
        self.feature_threshold = feature_threshold
        self.ssim_threshold = ssim_threshold
        self.workers = max(1, workers)
        self.region_extractor = RegionExtractor()
        self.region_similarity = RegionSimilarity()

    def _descriptor_similarity(
        self,
        vertices: Sequence[MeshVertex],
        vertex_indices: Tuple[int, int, int],
        product_descriptors: Optional[np.ndarray],
        return_descriptors: Optional[np.ndarray],
    ) -> float:
        if product_descriptors is None or return_descriptors is None:
            return 0.0

        scores: List[float] = []
        for vertex_index in vertex_indices:
            vertex = vertices[vertex_index]

            product_index = getattr(vertex, "product_keypoint_index", None)
            return_index = getattr(vertex, "return_keypoint_index", None)

            # Backward compatibility for older vertices. Synthetic vertices
            # must keep all indices as None and are skipped here.
            if product_index is None or return_index is None:
                legacy_index = getattr(vertex, "keypoint_index", None)
                if legacy_index is None:
                    continue
                product_index = legacy_index
                return_index = legacy_index

            if product_index >= len(product_descriptors) or return_index >= len(return_descriptors):
                continue

            desc_p = product_descriptors[product_index].astype(np.float32)
            desc_r = return_descriptors[return_index].astype(np.float32)
            norm_p = float(np.linalg.norm(desc_p))
            norm_r = float(np.linalg.norm(desc_r))
            if norm_p <= 1e-8 or norm_r <= 1e-8:
                continue
            scores.append(float(np.dot(desc_p, desc_r) / (norm_p * norm_r)))

        if not scores:
            return 0.0
        return float(np.mean(scores))

    def evaluate_one(
        self,
        triangle: MeshTriangle,
        vertices: Sequence[MeshVertex],
        product_image: np.ndarray,
        return_image: np.ndarray,
        product_descriptors: Optional[np.ndarray],
        return_descriptors: Optional[np.ndarray],
    ) -> ScoredTriangle:
        area = self.region_extractor.polygon_area(vertices, triangle.vertex_indices)
        product_crop, product_mask, return_crop, return_mask = self.region_extractor.extract_triangle_pair(
            product_image,
            return_image,
            vertices,
            triangle.vertex_indices,
        )
        similarity, metric_scores = self.region_similarity.compute(
            product_crop,
            return_crop,
            product_mask,
            return_mask,
        )
        feature_similarity = self._descriptor_similarity(
            vertices,
            triangle.vertex_indices,
            product_descriptors,
            return_descriptors,
        )
        # The old condition compared the combined score with SSIM_THRESHOLD,
        # so SSIM_THRESHOLD was effectively redundant whenever
        # SIMILARITY_THRESHOLD was lower. Use the real SSIM metric here and
        # reject unreliable low-overlap comparisons instead.
        ssim_score = float(metric_scores.get("ssim", similarity))
        overlap_ratio = float(metric_scores.get("overlap_ratio", 1.0))
        comparison_reliable = overlap_ratio >= 0.55

        is_damaged = (
            comparison_reliable
            and similarity < self.similarity_threshold
            and ssim_score < self.ssim_threshold
            and feature_similarity < self.feature_threshold
        )
        return ScoredTriangle(
            triangle=triangle,
            area=area,
            similarity=similarity,
            feature_similarity=feature_similarity,
            metric_scores=metric_scores,
            is_damaged=is_damaged,
        )

    def evaluate_cached(
        self,
        triangle: MeshTriangle,
        vertices: Sequence[MeshVertex],
        product_image: np.ndarray,
        return_image: np.ndarray,
        product_descriptors: Optional[np.ndarray],
        return_descriptors: Optional[np.ndarray],
        cache: SimilarityCache,
    ) -> ScoredTriangle:
        cached = cache.get(triangle.triangle_id)
        if cached is not None:
            return cached

        scored = self.evaluate_one(
            triangle,
            vertices,
            product_image,
            return_image,
            product_descriptors,
            return_descriptors,
        )
        cache.set(scored)
        return scored

    def evaluate_batch(
        self,
        triangles: Sequence[MeshTriangle],
        vertices: Sequence[MeshVertex],
        product_image: np.ndarray,
        return_image: np.ndarray,
        product_descriptors: Optional[np.ndarray],
        return_descriptors: Optional[np.ndarray],
        cache: SimilarityCache,
    ) -> List[ScoredTriangle]:
        """Evaluate similarity for many triangles in parallel."""
        pending = [triangle for triangle in triangles if not cache.has(triangle.triangle_id)]
        if not pending:
            return [cache.get(triangle.triangle_id) for triangle in triangles]

        def _worker(triangle: MeshTriangle) -> ScoredTriangle:
            return self.evaluate_one(
                triangle,
                vertices,
                product_image,
                return_image,
                product_descriptors,
                return_descriptors,
            )

        if len(pending) == 1 or self.workers == 1:
            for triangle in pending:
                cache.set(_worker(triangle))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                for scored in executor.map(_worker, pending):
                    cache.set(scored)

        return [cache.get(triangle.triangle_id) for triangle in triangles]
