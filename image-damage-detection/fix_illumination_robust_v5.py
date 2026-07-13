from pathlib import Path
import re
import shutil

path = Path("mesh_damage_detector.py")
text = path.read_text(encoding="utf-8")
backup = Path("mesh_damage_detector.py.illumination_v5.bak")
shutil.copy2(path, backup)

# Version marker
if "MESH_DAMAGE_DETECTOR_VERSION" in text:
    text = re.sub(
        r'MESH_DAMAGE_DETECTOR_VERSION\s*=\s*"[^"]+"',
        'MESH_DAMAGE_DETECTOR_VERSION = "illumination_robust_pixel_v5"',
        text,
        count=1,
    )
else:
    text = text.replace(
        "from triangle_refiner import TriangleRefiner\n",
        "from triangle_refiner import TriangleRefiner\n\nMESH_DAMAGE_DETECTOR_VERSION = \"illumination_robust_pixel_v5\"\n",
        1,
    )

replacement = r'''
    def _illumination_removed_gray(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split grayscale image into high-frequency texture and low-frequency light.

        Broad shading / darker lighting should mostly stay in the low-frequency
        image. Scratches, cracks, and chipped edges should remain in the
        high-frequency residual.
        """
        gray_f = gray.astype(np.float32)
        low = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=11.0, sigmaY=11.0)
        high = gray_f - low
        high_u8 = np.clip(high + 128.0, 0, 255).astype(np.uint8)
        low_u8 = np.clip(low, 0, 255).astype(np.uint8)
        return high_u8, low_u8

    def _lab_chroma_diff(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """Compare Lab a/b chroma after local mean compensation.

        L channel is brightness, so it is intentionally ignored here. This
        reduces false positives caused by one image being darker/brighter.
        """
        if product_image.ndim != 3 or return_image.ndim != 3:
            return np.zeros(active.shape, dtype=np.uint8)

        p_lab = cv2.cvtColor(product_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        r_lab = cv2.cvtColor(return_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        p_ab = p_lab[:, :, 1:3]
        r_ab = r_lab[:, :, 1:3].copy()

        if np.any(active):
            for ch in range(2):
                p_mean = float(p_ab[:, :, ch][active].mean())
                r_mean = float(r_ab[:, :, ch][active].mean())
                r_ab[:, :, ch] += (p_mean - r_mean)

        da = p_ab[:, :, 0] - r_ab[:, :, 0]
        db = p_ab[:, :, 1] - r_ab[:, :, 1]
        chroma = np.sqrt(da * da + db * db)
        return np.clip(chroma, 0, 255).astype(np.uint8)

    def _filter_pixel_damage_components(
        self,
        binary_mask: np.ndarray,
        texture_diff: np.ndarray,
        edge_diff: np.ndarray,
        chroma_diff: np.ndarray,
    ) -> np.ndarray:
        """Reject low-frequency lighting blobs but keep thin/structured defects."""
        binary = (binary_mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            return binary

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)

        min_area = max(3, int(self.min_area_threshold // 3))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            component = labels == label
            t_vals = texture_diff[component].astype(np.float32)
            e_vals = edge_diff[component].astype(np.float32)
            c_vals = chroma_diff[component].astype(np.float32)

            t_p80 = float(np.percentile(t_vals, 80)) if t_vals.size else 0.0
            t_mean = float(t_vals.mean()) if t_vals.size else 0.0
            e_p80 = float(np.percentile(e_vals, 80)) if e_vals.size else 0.0
            e_ratio = float(np.mean(e_vals >= 22.0)) if e_vals.size else 0.0
            c_p80 = float(np.percentile(c_vals, 80)) if c_vals.size else 0.0

            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = max(1, w * h)
            fill_ratio = area / bbox_area
            long_thin = max(w, h) >= 3 * max(1, min(w, h)) and fill_ratio <= 0.45

            # Keep if it has texture/edge/chroma evidence. Broad lighting blobs
            # usually have weak high-pass texture and weak edge evidence.
            strong_texture = t_p80 >= 34.0
            structured_texture = (t_mean >= 16.0 and e_ratio >= 0.12) or (t_p80 >= 24.0 and e_p80 >= 26.0)
            color_damage = c_p80 >= 24.0 and e_ratio >= 0.08
            scratch_like = long_thin and t_p80 >= 20.0

            # Large filled blobs with weak texture are usually shadow/illumination.
            broad_lighting_blob = area >= 2500 and fill_ratio >= 0.35 and t_p80 < 28.0 and e_ratio < 0.18

            if not broad_lighting_blob and (strong_texture or structured_texture or color_damage or scratch_like):
                cleaned[component] = 255

        return cleaned

    def _compute_triangle_pixel_damage_mask(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        triangle_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute illumination-robust pixel-level damage inside one triangle.

        The previous v4 still used too much raw brightness difference, so
        darker lighting could be classified as damage. This version compares:
        1. high-pass texture after removing slow lighting changes;
        2. edge/gradient difference on that high-pass texture;
        3. Lab chroma difference without the L brightness channel.
        """
        active = triangle_mask > 0
        if not np.any(active):
            return np.zeros(triangle_mask.shape, dtype=np.uint8)

        product_gray = cv2.GaussianBlur(self._to_gray(product_image), (3, 3), 0)
        return_gray = cv2.GaussianBlur(self._to_gray(return_image), (3, 3), 0)

        p_high, p_low = self._illumination_removed_gray(product_gray)
        r_high, r_low = self._illumination_removed_gray(return_gray)

        # Normalize high-frequency residual locally, not raw brightness.
        p_vals = p_high[active].astype(np.float32)
        r_vals = r_high[active].astype(np.float32)
        p_mean = float(p_vals.mean()) if p_vals.size else 128.0
        r_mean = float(r_vals.mean()) if r_vals.size else 128.0
        p_std = float(p_vals.std()) if p_vals.size else 1.0
        r_std = float(r_vals.std()) if r_vals.size else 1.0

        r_high_f = r_high.astype(np.float32)
        if r_std > 1e-6:
            r_high_norm = (r_high_f - r_mean) * (p_std / r_std) + p_mean
        else:
            r_high_norm = r_high_f + (p_mean - r_mean)
        r_high_norm = np.clip(r_high_norm, 0, 255).astype(np.uint8)

        texture_diff = cv2.absdiff(p_high, r_high_norm)

        p_edge = self._gradient_magnitude(p_high)
        r_edge = self._gradient_magnitude(r_high_norm)
        edge_diff = cv2.absdiff(p_edge, r_edge)

        chroma_diff = self._lab_chroma_diff(product_image, return_image, active)

        # Raw brightness difference is only used as a veto/debug signal, not as
        # the main detector. If only the low-frequency light changed, do not mark it.
        low_light_diff = cv2.absdiff(p_low, r_low)

        damage_pixels = (
            (texture_diff >= 38)
            | ((texture_diff >= 20) & (edge_diff >= 24))
            | ((chroma_diff >= 28) & (edge_diff >= 18))
        ) & active

        # Veto likely illumination-only pixels: high low-frequency light change
        # but weak high-frequency/edge/chroma evidence.
        illumination_only = (
            (low_light_diff >= 20)
            & (texture_diff < 26)
            & (edge_diff < 28)
            & (chroma_diff < 24)
        )
        damage_pixels = damage_pixels & (~illumination_only)

        damage_mask = damage_pixels.astype(np.uint8) * 255

        if np.count_nonzero(damage_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return self._filter_pixel_damage_components(
            damage_mask,
            texture_diff,
            edge_diff,
            chroma_diff,
        )

    def _build_damage_map(
'''

# Replace the v4 _compute_triangle_pixel_damage_mask block and keep existing _build_damage_map signature/body.
text, n_compute = re.subn(
    r"\n    def _compute_triangle_pixel_damage_mask\([\s\S]*?\n    def _build_damage_map\(",
    "\n" + replacement,
    text,
    count=1,
)

# If the local file does not have v4, patch the older full-triangle block too.
if n_compute == 0:
    text, n_compute = re.subn(
        r"\n    def _build_damage_map\([\s\S]*?\n    def _build_mesh_debug_info",
        "\n" + replacement + "\n        self,\n        triangle_results: Sequence[ScoredTriangle],\n        vertices: Sequence[MeshVertex],\n        product_image: np.ndarray,\n        return_image: np.ndarray,\n    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:\n        \"\"\"Build pixel-level damage map from scored mesh triangles.\"\"\"\n        height, width = product_image.shape[:2]\n        damage_map = np.zeros((height, width), dtype=np.uint8)\n        suspected_mask = np.zeros((height, width), dtype=np.uint8)\n        refined_overlay = np.zeros((height, width), dtype=np.uint8)\n\n        region_extractor = RegionExtractor()\n\n        for result in triangle_results:\n            mask = region_extractor.create_triangle_mask(\n                (height, width),\n                vertices,\n                result.triangle.vertex_indices,\n                space=\"product\",\n            )\n\n            if result.triangle.refined:\n                refined_overlay = np.maximum(refined_overlay, (mask > 0).astype(np.uint8) * 255)\n\n            if not result.is_damaged:\n                continue\n\n            suspected_mask = np.maximum(suspected_mask, (mask > 0).astype(np.uint8) * 255)\n            pixel_mask = self._compute_triangle_pixel_damage_mask(product_image, return_image, mask)\n            if np.count_nonzero(pixel_mask) == 0:\n                continue\n\n            severity = int(np.clip((1.0 - result.similarity) * 255.0, 1, 255))\n            active = pixel_mask > 0\n            damage_map[active] = np.maximum(damage_map[active], severity)\n\n        if np.count_nonzero(damage_map) > 0:\n            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))\n            damage_map = cv2.morphologyEx(damage_map, cv2.MORPH_CLOSE, kernel, iterations=1)\n            damage_map = self._remove_small_components(damage_map, max(3, self.min_area_threshold))\n\n        return damage_map, suspected_mask, refined_overlay\n\n    def _build_mesh_debug_info",
        text,
        count=1,
    )

# Make sure run() passes images into _build_damage_map.
text = text.replace(
'''        # Build damage map
        damage_map, suspected_mask, refined_overlay = self._build_damage_map(
            triangle_results,
            vertices,
            product_image.shape,
        )
''',
'''        # Build illumination-robust pixel-level damage map
        damage_map, suspected_mask, refined_overlay = self._build_damage_map(
            triangle_results,
            vertices,
            product_image,
            aligned_return_image,
        )
'''
)
text = text.replace(
'''        # Build pixel-level damage map
        damage_map, suspected_mask, refined_overlay = self._build_damage_map(
            triangle_results,
            vertices,
            product_image,
            aligned_return_image,
        )
''',
'''        # Build illumination-robust pixel-level damage map
        damage_map, suspected_mask, refined_overlay = self._build_damage_map(
            triangle_results,
            vertices,
            product_image,
            aligned_return_image,
        )
'''
)

path.write_text(text, encoding="utf-8")

print("Backup:", backup)
print("Patched illumination-robust pixel detector:", bool(n_compute))
print("Version marker:", "illumination_robust_pixel_v5" in text)
print("Now run: python -m py_compile mesh_damage_detector.py")
