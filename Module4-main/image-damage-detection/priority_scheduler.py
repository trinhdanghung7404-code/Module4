"""Priority-queue scheduler for bounded adaptive mesh refinement."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesh_builder import MeshBuilder, MeshVertex, MeshTriangle
from similarity_cache import ScoredTriangle, SimilarityCache
from triangle_refiner import TriangleRefiner
from triangle_similarity import TriangleSimilarity


@dataclass
class SchedulerStats:
    """Statistics for refinement run."""
    original_triangles: int = 0
    refined_triangles: int = 0
    triangles_processed: int = 0
    triangles_skipped: int = 0
    average_similarity: float = 0.0
    total_refinements: int = 0
    similarity_time: float = 0.0
    mesh_time: float = 0.0


@dataclass
class SchedulerResult:
    """Result of mesh refinement."""
    vertices: List[MeshVertex]
    leaf_triangles: List[ScoredTriangle]
    stats: SchedulerStats
    cache: SimilarityCache = field(default_factory=SimilarityCache)


class PriorityMeshScheduler:
    """Refine only the most suspicious triangles using a min-heap (no recursion)."""
    
    def __init__(
        self,
        mesh_builder: Optional[MeshBuilder] = None,
        triangle_similarity: Optional[TriangleSimilarity] = None,
        triangle_refiner: Optional[TriangleRefiner] = None,
        *,
        similarity_threshold: float,
        min_triangle_area: float,
        max_depth: int,
        max_total_triangles: int,
        top_k: int,
    ):
        self.mesh_builder = mesh_builder or MeshBuilder()
        self.triangle_similarity = triangle_similarity
        self.triangle_refiner = triangle_refiner or TriangleRefiner()
        self.similarity_threshold = similarity_threshold
        self.min_triangle_area = min_triangle_area
        self.max_depth = max_depth
        self.max_total_triangles = max_total_triangles
        self.top_k = top_k

    def _should_refine(self, scored: ScoredTriangle) -> bool:
        """Check if triangle meets refinement criteria."""
        return (
            scored.similarity < self.similarity_threshold
            and scored.similarity < 0.78
            and scored.area > self.min_triangle_area
            and scored.level < self.max_depth
        )

    def _push(self, heap: List, scored: ScoredTriangle) -> None:
        """Push scored triangle onto heap (min-heap by similarity)."""
        heapq.heappush(heap, (scored.similarity, scored.triangle_id, scored))

    def _pop(self, heap: List) -> Optional[ScoredTriangle]:
        """Pop triangle with lowest similarity from heap."""
        if not heap:
            return None
        return heapq.heappop(heap)[2]

    def run(
        self,
        product_points: Sequence[Sequence[float]],
        return_points: Sequence[Sequence[float]],
        keypoint_indices: Optional[Sequence[Optional[Tuple[int, int]]]],
        product_image: np.ndarray,
        return_image: np.ndarray,
        product_descriptors: Optional[np.ndarray],
        return_descriptors: Optional[np.ndarray],
    ) -> SchedulerResult:
        """Run priority-queue based mesh refinement."""
        
        # ====================================================================
        # STEP 1: Build initial Delaunay mesh
        # ====================================================================
        mesh_start = time.perf_counter()
        vertices, base_triangles = self.mesh_builder.build(product_points, return_points, keypoint_indices)
        mesh_build_time = time.perf_counter() - mesh_start

        cache = SimilarityCache()
        stats = SchedulerStats(original_triangles=len(base_triangles))

        if not base_triangles or self.triangle_similarity is None:
            stats.mesh_time = mesh_build_time
            return SchedulerResult(
                vertices=list(vertices), 
                leaf_triangles=[], 
                stats=stats, 
                cache=cache
            )

        # ====================================================================
        # STEP 2: Compute similarity for ALL base triangles
        # ====================================================================
        similarity_start = time.perf_counter()
        
        base_scored = self.triangle_similarity.evaluate_batch(
            base_triangles,
            vertices,
            product_image,
            return_image,
            product_descriptors,
            return_descriptors,
            cache,
        )
        
        if base_scored:
            stats.average_similarity = float(np.mean([item.similarity for item in base_scored]))

        # ====================================================================
        # STEP 3: Track leaves and build initial heap
        # ====================================================================
        leaves: Dict[int, ScoredTriangle] = {item.triangle_id: item for item in base_scored}
        
        # Sort by similarity, take only top-K candidates for refinement
        sorted_base = sorted(base_scored, key=lambda item: (item.similarity, item.triangle_id))
        top_k_candidates = sorted_base[:self.top_k]
        
        heap: List = []
        for scored in top_k_candidates:
            if self._should_refine(scored):
                self._push(heap, scored)

        # ====================================================================
        # STEP 4: Priority-queue refinement loop (NO recursion)
        # ====================================================================
        mutable_vertices = list(vertices)
        midpoint_cache: Dict[Tuple[int, int], int] = {}
        next_triangle_id = len(base_triangles)
        total_triangle_count = len(base_triangles)

        refine_start = time.perf_counter()
        
        while heap and total_triangle_count < self.max_total_triangles:
            scored = self._pop(heap)
            if scored is None:
                break

            stats.triangles_processed += 1

            # Triangle might have been replaced by refined children
            if scored.triangle_id not in leaves:
                stats.triangles_skipped += 1
                continue

            # Re-check refinement conditions (area may have changed)
            if not self._should_refine(scored):
                stats.triangles_skipped += 1
                continue

            # Subdivide the triangle
            children, next_triangle_id = self.triangle_refiner.subdivide(
                scored.triangle,
                mutable_vertices,
                midpoint_cache,
                next_triangle_id,
            )

            if not children:
                stats.triangles_skipped += 1
                continue

            # Remove parent from leaves
            del leaves[scored.triangle_id]
            stats.total_refinements += 1

            # Compute similarity for children
            child_scored = self.triangle_similarity.evaluate_batch(
                children,
                mutable_vertices,
                product_image,
                return_image,
                product_descriptors,
                return_descriptors,
                cache,
            )

            # Add children to leaves and push qualifying ones to heap
            for child in child_scored:
                total_triangle_count += 1
                leaves[child.triangle_id] = child
                
                # Check limit after each addition
                if total_triangle_count >= self.max_total_triangles:
                    break
                    
                if self._should_refine(child):
                    self._push(heap, child)

            # Exit early if limit reached
            if total_triangle_count >= self.max_total_triangles:
                break

        # ====================================================================
        # STEP 5: Finalize stats
        # ====================================================================
        stats.mesh_time = mesh_build_time + (time.perf_counter() - refine_start)
        stats.similarity_time = time.perf_counter() - similarity_start
        stats.refined_triangles = len(leaves)

        return SchedulerResult(
            vertices=mutable_vertices,
            leaf_triangles=list(leaves.values()),
            stats=stats,
            cache=cache,
        )