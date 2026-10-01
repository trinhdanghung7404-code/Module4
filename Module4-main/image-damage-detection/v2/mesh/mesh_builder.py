from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
from scipy.spatial import Delaunay
import config

@dataclass(frozen=True)
class Vertex:
    """A paired vertex with positions in both product and return image space."""
    product_xy: Tuple[float, float]
    return_xy: Tuple[float, float]
    descriptor_index: Optional[int] = None

@dataclass
class Triangle:
    """A mesh triangle defined by three vertex indices."""
    vertex_indices: Tuple[int, int, int]
    triangle_id: int

def triangle_area(vertices: List[Vertex], tri: Triangle) -> float:
    """Compute area of triangle in product image space using cross product."""
    v0 = np.array(vertices[tri.vertex_indices[0]].product_xy)
    v1 = np.array(vertices[tri.vertex_indices[1]].product_xy)
    v2 = np.array(vertices[tri.vertex_indices[2]].product_xy)
    d1 = v1 - v0
    d2 = v2 - v0
    return 0.5 * abs(float(d1[0] * d2[1] - d1[1] * d2[0]))

def triangle_max_edge(vertices: List[Vertex], tri: Triangle) -> float:
    """Compute maximum edge length of triangle."""
    v0 = np.array(vertices[tri.vertex_indices[0]].product_xy)
    v1 = np.array(vertices[tri.vertex_indices[1]].product_xy)
    v2 = np.array(vertices[tri.vertex_indices[2]].product_xy)
    return float(max(np.linalg.norm(v1 - v0), np.linalg.norm(v2 - v1), np.linalg.norm(v0 - v2)))

class MeshBuilder:
    """Builds a Delaunay triangulation from matched keypoint pairs.
    Triangulation is computed in product image space.
    The mesh is used ONLY for spatial partitioning, NOT for warping."""
    
    def build(self, product_points: np.ndarray, return_points: np.ndarray,
              descriptor_indices: np.ndarray = None) -> Tuple[List[Vertex], List[Triangle]]:
        # 1. Build Vertex objects
        vertices = []
        for i in range(len(product_points)):
            desc_idx = descriptor_indices[i] if descriptor_indices is not None else None
            vertices.append(Vertex(
                product_xy=tuple(product_points[i]),
                return_xy=tuple(return_points[i]),
                descriptor_index=desc_idx
            ))
            
        # 2. scipy.spatial.Delaunay on product_points
        delaunay = Delaunay(product_points)
        
        # 3. Create Triangle objects & filter
        triangles = []
        max_edge_limit = getattr(config, 'MAX_TRIANGLE_EDGE', 220.0)
        for simplex in delaunay.simplices:
            tri = Triangle(vertex_indices=tuple(simplex), triangle_id=len(triangles))
            if triangle_area(vertices, tri) >= config.MIN_TRIANGLE_AREA and triangle_max_edge(vertices, tri) <= max_edge_limit:
                triangles.append(tri)
                
        return vertices, triangles
