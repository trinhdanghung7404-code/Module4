"""Triangle subdivision strategies for adaptive mesh refinement."""

from __future__ import annotations

from typing import List, Tuple, Dict

from mesh_builder import MeshTriangle, MeshVertex
from mesh_refiner import MeshRefiner


class TriangleRefiner:
    """Split mesh triangles when region similarity is below threshold.
    
    Wraps MeshRefiner so the subdivision backend can be swapped later
    (e.g. Voronoi cells, quad-tree patches).
    """
    
    STRATEGY_MIDPOINT = "midpoint"
    STRATEGY_LONGEST_EDGE = "longest_edge"

    def __init__(self, strategy: str = STRATEGY_MIDPOINT):
        self._backend = MeshRefiner(strategy=strategy)

    def subdivide(
        self,
        triangle: MeshTriangle,
        vertices: List[MeshVertex],
        midpoint_cache: Dict[Tuple[int, int], int],
        next_triangle_id: int,
    ) -> Tuple[List[MeshTriangle], int]:
        """Subdivide a single triangle into children."""
        return self._backend.subdivide(triangle, vertices, midpoint_cache, next_triangle_id)