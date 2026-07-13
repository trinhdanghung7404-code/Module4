"""Debug visualizations for adaptive mesh region matching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from mesh_builder import MeshVertex
from region_extractor import RegionExtractor
from similarity_cache import ScoredTriangle


class MeshDebugger:
    """Save mesh overlay and diagnostic images when DEBUG mode is enabled."""

    def __init__(self):
        self.region_extractor = RegionExtractor()

    def draw_mesh_overlay(
        self,
        image: np.ndarray,
        vertices: Sequence[MeshVertex],
        triangle_results: Sequence[ScoredTriangle],
        point_space: str,
    ) -> np.ndarray:
        overlay = image.copy()

        for result in triangle_results:
            triangle = result.triangle
            if point_space == "product":
                polygon = self.region_extractor.triangle_product_points(vertices, triangle.vertex_indices)
            else:
                polygon = self.region_extractor.triangle_return_points(vertices, triangle.vertex_indices)

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

    def draw_keypoints(
        self,
        image: np.ndarray,
        product_points: Sequence[Sequence[float]],
        return_points: Sequence[Sequence[float]],
    ) -> np.ndarray:
        canvas = np.hstack([image.copy(), image.copy()])
        width_offset = image.shape[1]

        for index, point in enumerate(product_points):
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(canvas, center, 4, (0, 255, 0), -1)
            cv2.putText(
                canvas, str(index), (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )

        for index, point in enumerate(return_points):
            center = (int(round(point[0])) + width_offset, int(round(point[1])))
            cv2.circle(canvas, center, 4, (255, 0, 0), -1)
            cv2.putText(
                canvas, str(index), (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1,
            )

        return canvas

    def build_similarity_histogram(self, triangle_results: Sequence[ScoredTriangle]) -> np.ndarray:
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

        for index, count in enumerate(hist):
            bar_height = int((count / max_count) * plot_height)
            x0 = margin_left + int(index * bar_width)
            x1 = margin_left + int((index + 1) * bar_width) - 2
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

    def save_mesh_debug_images(
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
    ) -> Dict[str, str]:
        debug_dir.mkdir(parents=True, exist_ok=True)

        delaunay_overlay = self.draw_mesh_overlay(product_image, vertices, triangle_results, "product")
        product_overlay = self.draw_mesh_overlay(product_image, vertices, triangle_results, "product")
        return_overlay = self.draw_mesh_overlay(return_image, vertices, triangle_results, "return")

        suspected_visual = product_image.copy()
        suspected_visual[suspected_mask > 0] = (0, 0, 255)
        suspected_visual = cv2.addWeighted(product_image, 0.7, suspected_visual, 0.3, 0)

        refined_visual = product_image.copy()
        refined_visual[refined_overlay > 0] = (0, 165, 255)
        refined_visual = cv2.addWeighted(product_image, 0.7, refined_visual, 0.3, 0)

        files = {
            "01_keypoints.png": self.draw_keypoints(product_image, product_points, return_points),
            "02_delaunay_mesh.png": delaunay_overlay,
            "03_mesh_overlay_product.png": product_overlay,
            "04_mesh_overlay_return.png": return_overlay,
            "05_suspected_triangles.png": suspected_visual,
            "06_refined_mesh.png": refined_visual,
            "07_damage_mask.png": cv2.cvtColor(damage_map, cv2.COLOR_GRAY2BGR),
        }

        saved_paths: Dict[str, str] = {}
        for filename, image in files.items():
            path = debug_dir / filename
            cv2.imwrite(str(path), image)
            saved_paths[filename] = str(path)

        return saved_paths

    def build_debug_images(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        vertices: Sequence[MeshVertex],
        triangle_results: Sequence[ScoredTriangle],
        damage_map: np.ndarray,
        suspected_mask: np.ndarray,
        patch_size: int,
    ) -> Dict[str, np.ndarray]:
        placeholder = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        similarity_histogram = self.build_similarity_histogram(triangle_results)
        overlay = self._build_matching_overlay(product_image, return_image, damage_map)

        return {
            "patch_product": product_image[:patch_size, :patch_size].copy()
            if product_image.shape[0] >= patch_size and product_image.shape[1] >= patch_size
            else placeholder,
            "search_window": self.draw_mesh_overlay(product_image, vertices, triangle_results, "product"),
            "best_patch": self.draw_mesh_overlay(return_image, vertices, triangle_results, "return"),
            "similarity_heatmap": similarity_histogram,
            "best_patch_heatmap": similarity_histogram,
            "patch_difference": cv2.cvtColor(suspected_mask, cv2.COLOR_GRAY2BGR),
            "damage_map": cv2.cvtColor(damage_map, cv2.COLOR_GRAY2BGR),
            "local_matching_overlay": overlay,
            "ssim_histogram": similarity_histogram,
        }

    @staticmethod
    def _build_matching_overlay(
        product_image: np.ndarray,
        return_image: np.ndarray,
        damage_map: np.ndarray,
    ) -> np.ndarray:
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
