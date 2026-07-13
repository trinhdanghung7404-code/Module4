from pathlib import Path
import re
import shutil

path = Path("mesh_damage_detector.py")
text = path.read_text(encoding="utf-8")
backup = Path("mesh_damage_detector.py.pixel_fix.bak")
shutil.copy2(path, backup)

# Version marker
if "MESH_DAMAGE_DETECTOR_VERSION" in text:
    text = re.sub(
        r'MESH_DAMAGE_DETECTOR_VERSION\s*=\s*"[^"]+"',
        'MESH_DAMAGE_DETECTOR_VERSION = "pixel_level_damage_map_v4"',
        text,
        count=1,
    )
else:
    text = text.replace(
        "from triangle_refiner import TriangleRefiner\n",
        "from triangle_refiner import TriangleRefiner\n\nMESH_DAMAGE_DETECTOR_VERSION = \"pixel_level_damage_map_v4\"\n",
        1,
    )

pixel_methods = r'''
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale uint8."""
        if image.ndim == 2:
            return image.astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _gradient_magnitude(self, gray: np.ndarray) -> np.ndarray:
        """Build normalized Scharr gradient magnitude."""
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(grad_x, grad_y)
        if float(magnitude.max()) <= 1e-6:
            return np.zeros_like(gray, dtype=np.uint8)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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

    def _compute_triangle_pixel_damage_mask(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        triangle_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute pixel-level damage only inside one suspicious triangle.

        The mesh triangle is only a candidate region. The final damage mask
        must not fill the whole triangle; it keeps only pixels that differ.
        """
        active = triangle_mask > 0
        if not np.any(active):
            return np.zeros(triangle_mask.shape, dtype=np.uint8)

        product_gray = cv2.GaussianBlur(self._to_gray(product_image), (3, 3), 0)
        return_gray = cv2.GaussianBlur(self._to_gray(return_image), (3, 3), 0)

        product_values = product_gray[active].astype(np.float32)
        return_values = return_gray[active].astype(np.float32)
        product_mean = float(product_values.mean()) if product_values.size else 0.0
        product_std = float(product_values.std()) if product_values.size else 1.0
        return_mean = float(return_values.mean()) if return_values.size else 0.0
        return_std = float(return_values.std()) if return_values.size else 1.0

        return_float = return_gray.astype(np.float32)
        if return_std <= 1e-6:
            return_norm = return_float + (product_mean - return_mean)
        else:
            return_norm = (return_float - return_mean) * (product_std / return_std) + product_mean
        return_norm = np.clip(return_norm, 0, 255).astype(np.uint8)

        intensity_diff = cv2.absdiff(product_gray, return_norm)
        product_grad = self._gradient_magnitude(product_gray)
        return_grad = self._gradient_magnitude(return_norm)
        gradient_diff = cv2.absdiff(product_grad, return_grad)

        # Start conservative enough to avoid painting whole ceramic decorations.
        # Increase sensitivity later by lowering these numbers.
        intensity_high = max(35, int(self.diff_threshold) + 10)
        intensity_low = max(12, int(round(self.diff_threshold * 0.55)))
        gradient_threshold = max(15, int(round(self.diff_threshold * 0.80)))

        damage_pixels = (
            (intensity_diff >= intensity_high)
            | ((intensity_diff >= intensity_low) & (gradient_diff >= gradient_threshold))
        ) & active

        damage_mask = damage_pixels.astype(np.uint8) * 255
        return self._remove_small_components(damage_mask, max(2, self.min_area_threshold // 3))

    def _build_damage_map(
        self,
        triangle_results: Sequence[ScoredTriangle],
        vertices: Sequence[MeshVertex],
        product_image: np.ndarray,
        return_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build pixel-level damage map from scored mesh triangles.

        suspected_mask still shows full suspicious triangles for debug.
        damage_map contains only pixel-level differences inside those triangles.
        """
        height, width = product_image.shape[:2]
        damage_map = np.zeros((height, width), dtype=np.uint8)
        suspected_mask = np.zeros((height, width), dtype=np.uint8)
        refined_overlay = np.zeros((height, width), dtype=np.uint8)

        region_extractor = RegionExtractor()

        for result in triangle_results:
            mask = region_extractor.create_triangle_mask(
                (height, width),
                vertices,
                result.triangle.vertex_indices,
                space="product",
            )

            if result.triangle.refined:
                refined_overlay = np.maximum(refined_overlay, (mask > 0).astype(np.uint8) * 255)

            if not result.is_damaged:
                continue

            # Debug only: full candidate triangle.
            suspected_mask = np.maximum(suspected_mask, (mask > 0).astype(np.uint8) * 255)

            # Final output: only changed pixels inside candidate triangle.
            pixel_mask = self._compute_triangle_pixel_damage_mask(
                product_image,
                return_image,
                mask,
            )
            if np.count_nonzero(pixel_mask) == 0:
                continue

            severity = int(np.clip((1.0 - result.similarity) * 255.0, 1, 255))
            active = pixel_mask > 0
            damage_map[active] = np.maximum(damage_map[active], severity)

        if np.count_nonzero(damage_map) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            damage_map = cv2.morphologyEx(damage_map, cv2.MORPH_OPEN, kernel, iterations=1)
            damage_map = cv2.morphologyEx(damage_map, cv2.MORPH_CLOSE, kernel, iterations=1)
            damage_map = self._remove_small_components(damage_map, max(2, self.min_area_threshold))

        return damage_map, suspected_mask, refined_overlay
'''

# Replace helpers + _build_damage_map if helpers already exist.
text, n_helpers = re.subn(
    r"\n    def _to_gray\([\s\S]*?\n    def _build_mesh_debug_info",
    "\n" + pixel_methods + "\n\n    def _build_mesh_debug_info",
    text,
    count=1,
)

# Replace only old _build_damage_map if helper block was not present.
if n_helpers == 0:
    text, n_damage = re.subn(
        r"\n    def _build_damage_map\([\s\S]*?\n    def _build_mesh_debug_info",
        "\n" + pixel_methods + "\n\n    def _build_mesh_debug_info",
        text,
        count=1,
    )
    if n_damage == 0:
        text, n_damage = re.subn(
            r"\n    def _build_damage_map\([\s\S]*?\n    def _scored_to_local_matches",
            "\n" + pixel_methods + "\n\n    def _scored_to_local_matches",
            text,
            count=1,
        )
else:
    n_damage = 1

# Update run() call: pass actual images, not product_image.shape.
text = text.replace(
'''        # Build damage map
        damage_map, suspected_mask, refined_overlay = self._build_damage_map(
            triangle_results,
            vertices,
            product_image.shape,
        )
''',
'''        # Build pixel-level damage map
        damage_map, suspected_mask, refined_overlay = self._build_damage_map(
            triangle_results,
            vertices,
            product_image,
            aligned_return_image,
        )
'''
)

# Also normalize any existing product_image/aligned_return_image call comment.
text = text.replace("# Build damage map\n        damage_map, suspected_mask, refined_overlay = self._build_damage_map(\n            triangle_results,\n            vertices,\n            product_image,\n            aligned_return_image,\n        )", "# Build pixel-level damage map\n        damage_map, suspected_mask, refined_overlay = self._build_damage_map(\n            triangle_results,\n            vertices,\n            product_image,\n            aligned_return_image,\n        )")

path.write_text(text, encoding="utf-8")

print("Backup:", backup)
print("Replaced helper/damage_map block:", bool(n_helpers or n_damage))
print("Contains full-triangle assignment:", "damage_map[active] = np.maximum(damage_map[active], severity)" in text and "pixel_mask > 0" not in text)
print("Version:", "pixel_level_damage_map_v4" in text)
print("Now run: python -m py_compile mesh_damage_detector.py")
