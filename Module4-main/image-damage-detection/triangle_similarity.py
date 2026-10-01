"""Region similarity evaluation for individual mesh triangles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mesh_builder import MeshTriangle, MeshVertex
from mesh_config import STRENGTHENING_SCORE_RATIO, VOTING_MODE
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
        strengthening_score_ratio: float = 0.40,
        workers: int = 4,
    ):
        self.similarity_threshold = similarity_threshold
        self.feature_threshold = feature_threshold
        self.ssim_threshold = ssim_threshold
        self.strengthening_score_ratio = strengthening_score_ratio
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

        # ===== STRICT UNANIMOUS VOTING SYSTEM =====
        # Với ảnh chụp sản phẩm khác góc/ánh sáng:
        # - Biến dạng ảnh KHÓ kéo CẢ 3 metrics xuống thấp ĐỒNG THỜI
        # - Damage thật sẽ làm giảm mạnh cả similarity + SSIM + feature_sim
        # Logic mới (strict_unanimous): Cần 3/3 votes → Flag damage
        # Hoặc có thể config "weighted_2of3" trong mesh_config.py
        
        vote_count = 0
        min_score = float('inf')
        
        # Vote 1: Combined similarity thấp
        if similarity < self.similarity_threshold:
            vote_count += 1
            min_score = min(min_score, similarity)
            
        # Vote 2: SSIM thấp  
        if ssim_score < self.ssim_threshold:
            vote_count += 1
            min_score = min(min_score, ssim_score)
            
        # Vote 3: Feature similarity thấp
        if feature_similarity < self.feature_threshold:
            vote_count += 1
            min_score = min(min_score, feature_similarity)
        
        # Decision logic theo VOTING_MODE:
        # Với STRICT_UNANIMOUS: cần CẢ 3 VOTES đồng loạt thấp → damage thật sự
        # Nguyên tắc: Mỗi metric phải THẤP hơn threshold → 1 vote cho damage
        #             Nếu đạt ngưỡng thì NOT VOTE → component不会被 flag
        
        needs_votes = 3 if VOTING_MODE == "strict_unanimous" else 2
        
        # Chỉ flag damage KHI CÓ ĐỦ VOTES
        # Vote = metric THẤP hơn threshold (ít giống nhau)
        is_damaged = (
            comparison_reliable
            and vote_count >= needs_votes
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
