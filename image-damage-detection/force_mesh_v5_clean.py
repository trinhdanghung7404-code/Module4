from pathlib import Path
import re
import shutil

path = Path('mesh_damage_detector.py')
if not path.exists():
    raise FileNotFoundError('mesh_damage_detector.py not found in current directory')

text = path.read_text(encoding='utf-8')
backup = Path('mesh_damage_detector.py.before_force_v5.bak')
shutil.copy2(path, backup)

version_line = 'MESH_DAMAGE_DETECTOR_VERSION = "illumination_robust_pixel_v5_clean"'
if re.search(r'^MESH_DAMAGE_DETECTOR_VERSION\s*=\s*".*"\s*$', text, flags=re.M):
    text = re.sub(r'^MESH_DAMAGE_DETECTOR_VERSION\s*=\s*".*"\s*$', version_line, text, flags=re.M)
else:
    anchor = 'from triangle_refiner import TriangleRefiner\n'
    if anchor not in text:
        raise RuntimeError('Could not find import anchor: from triangle_refiner import TriangleRefiner')
    text = text.replace(anchor, anchor + '\n' + version_line + '\n', 1)

new_methods = r'''
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
            return np.zeros_like(gray, dtype=np.uint8)
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
        """Normalize return brightness/contrast to product inside one triangle.

        This removes simple local lighting changes before computing residuals.
        """
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
            # Clamp scale to avoid amplifying noise in very flat ceramic regions.
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

    def _compute_triangle_pixel_damage(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        triangle_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Compute illumination-robust pixel damage inside one candidate triangle.

        A pixel is not damage merely because it is darker/brighter. It needs
        texture, edge, or chroma evidence after local illumination correction.
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
            }

        product_gray = cv2.GaussianBlur(self._to_gray(product_image), (3, 3), 0)
        return_gray = cv2.GaussianBlur(self._to_gray(return_image), (3, 3), 0)
        return_norm = self._local_illumination_normalize(product_gray, return_gray, active)

        raw_residual = cv2.absdiff(product_gray, return_norm)

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

        if product_image.ndim == 3 and return_image.ndim == 3:
            product_lab = cv2.cvtColor(product_image, cv2.COLOR_BGR2LAB)
            return_lab = cv2.cvtColor(return_image, cv2.COLOR_BGR2LAB)
            chroma_diff = (
                np.abs(product_lab[:, :, 1].astype(np.int16) - return_lab[:, :, 1].astype(np.int16))
                + np.abs(product_lab[:, :, 2].astype(np.int16) - return_lab[:, :, 2].astype(np.int16))
            )
            chroma_diff = np.clip(chroma_diff, 0, 255).astype(np.uint8)
        else:
            chroma_diff = empty.copy()

        # Conservative thresholds: raw brightness alone is not enough.
        raw_high = max(45, int(self.diff_threshold) + 18)
        raw_low = max(16, int(round(self.diff_threshold * 0.65)))
        texture_high = max(26, int(round(self.diff_threshold * 1.00)))
        texture_mid = max(18, int(round(self.diff_threshold * 0.70)))
        edge_high = max(38, int(round(self.diff_threshold * 1.45)))
        edge_mid = max(26, int(round(self.diff_threshold * 1.00)))
        chroma_high = 34
        chroma_mid = 22

        texture_mask = texture_diff >= texture_high
        edge_mask = edge_diff >= edge_high
        chroma_mask = chroma_diff >= chroma_high
        gated_raw_mask = (
            (raw_residual >= raw_high)
            & (
                (texture_diff >= texture_mid)
                | (edge_diff >= edge_mid)
                | (chroma_diff >= chroma_mid)
            )
        )

        # Smooth brightness-only changes are likely illumination, not damage.
        illumination_like = (
            (raw_residual >= raw_low)
            & (texture_diff < texture_mid)
            & (edge_diff < edge_mid)
            & (chroma_diff < chroma_mid)
        )

        damage_pixels = (texture_mask | edge_mask | chroma_mask | gated_raw_mask) & active & (~illumination_like)
        damage_mask = damage_pixels.astype(np.uint8) * 255
        damage_mask = self._remove_small_components(damage_mask, max(2, self.min_area_threshold // 3))

        maps = {
            'raw_residual': np.where(active, raw_residual, 0).astype(np.uint8),
            'texture': np.where(active, texture_diff, 0).astype(np.uint8),
            'edge': np.where(active, edge_diff, 0).astype(np.uint8),
            'chroma': np.where(active, chroma_diff, 0).astype(np.uint8),
            'illumination_reject': (illumination_like & active).astype(np.uint8) * 255,
        }
        return damage_mask, maps

    def _filter_shadow_like_components(
        self,
        damage_mask: np.ndarray,
        texture_score: np.ndarray,
        edge_score: np.ndarray,
        chroma_score: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reject big, smooth components that look like lighting variation."""
        binary = (damage_mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            return binary, np.zeros_like(binary)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)
        rejected = np.zeros_like(binary)
        image_area = float(binary.shape[0] * binary.shape[1])
        min_area = max(2, int(self.min_area_threshold))

        for label in range(1, num_labels):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                rejected[component] = 255
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = max(1, w * h)
            fill_ratio = area / float(bbox_area)
            area_ratio = area / image_area

            tex_mean = float(texture_score[component].mean()) if area else 0.0
            edge_mean = float(edge_score[component].mean()) if area else 0.0
            chroma_mean = float(chroma_score[component].mean()) if area else 0.0

            # Large, dense and weak texture/edge/chroma blobs usually come from
            # smooth illumination differences on the curved ceramic surface.
            shadow_like = (
                (area > 1200 or area_ratio > 0.003)
                and fill_ratio > 0.25
                and tex_mean < 34.0
                and edge_mean < 34.0
                and chroma_mean < 28.0
            )
            if shadow_like:
                rejected[component] = 255
            else:
                cleaned[component] = 255

        return cleaned, rejected

    def _build_damage_map(
        self,
        triangle_results: Sequence[ScoredTriangle],
        vertices: Sequence[MeshVertex],
        product_image: np.ndarray,
        return_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build pixel-level damage map from scored mesh triangles.

        suspected_mask shows full candidate triangles for debug only.
        damage_map keeps only illumination-robust pixel differences.
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
            cleaned_binary, component_reject = self._filter_shadow_like_components(
                damage_map,
                debug_scores['texture'],
                debug_scores['edge'],
                debug_scores['chroma'],
            )
            debug_scores['component_reject'] = component_reject
            damage_map[cleaned_binary == 0] = 0
            damage_map = self._remove_small_components(damage_map, max(2, self.min_area_threshold))

        self._last_pixel_debug_masks = {
            'raw_residual_mask': debug_scores['raw_residual'],
            'texture_mask': debug_scores['texture'],
            'edge_mask': debug_scores['edge'],
            'chroma_mask': debug_scores['chroma'],
            'illumination_reject_mask': debug_scores['illumination_reject'],
            'component_reject_mask': debug_scores['component_reject'],
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
            },
        }

'''

# Replace all old pixel/damage/validation methods before _scored_to_local_matches.
start_candidates = []
for marker in ['\n    def _to_gray(', '\n    def _build_damage_map(']:
    idx = text.find(marker)
    if idx != -1:
        start_candidates.append(idx)
if not start_candidates:
    raise RuntimeError('Could not find _to_gray or _build_damage_map in MeshDamageDetector')
start = min(start_candidates)
end = text.find('\n    def _scored_to_local_matches', start)
if end == -1:
    raise RuntimeError('Could not find _scored_to_local_matches after damage-map block')
text = text[:start] + '\n' + new_methods + text[end:]

# Force the run() damage-map call to use product_image + aligned_return_image.
text = re.sub(
    r"damage_map,\s*suspected_mask,\s*refined_overlay\s*=\s*self\._build_damage_map\(\s*triangle_results,\s*vertices,\s*product_image\.shape,\s*\)",
    "damage_map, suspected_mask, refined_overlay = self._build_damage_map(\n            triangle_results,\n            vertices,\n            product_image,\n            aligned_return_image,\n        )",
    text,
    flags=re.S,
)

# Remove any old validation call left in run().
text = re.sub(
    r"\n\s*#\s*Apply morphology and validation\s*\n\s*damage_map,\s*debug_info\s*=\s*self\._apply_morphology_and_validation\(\s*damage_map,\s*triangle_results,\s*local_matches,\s*\)\s*\n",
    "\n",
    text,
    flags=re.S,
)

# Ensure debug_info is built before summary.
if 'debug_info = self._build_mesh_debug_info' not in text:
    marker = "        # Build summary\n        summary = self._build_summary(triangle_results, damage_map)"
    insert = "        # Build debug info without old component validator\n        debug_info = self._build_mesh_debug_info(\n            damage_map,\n            suspected_mask,\n            refined_overlay,\n            triangle_results,\n        )\n\n"
    if marker not in text:
        raise RuntimeError('Could not find summary marker to insert debug_info')
    text = text.replace(marker, insert + marker, 1)

# Save additional mesh debug images if the method has the expected files dict.
extra_debug = """
        pixel_debug = getattr(self, '_last_pixel_debug_masks', {}) or {}
        for extra_name, extra_image in {
            '09_raw_residual_mask.png': pixel_debug.get('raw_residual_mask'),
            '10_texture_mask.png': pixel_debug.get('texture_mask'),
            '11_edge_mask.png': pixel_debug.get('edge_mask'),
            '12_chroma_mask.png': pixel_debug.get('chroma_mask'),
            '13_illumination_reject_mask.png': pixel_debug.get('illumination_reject_mask'),
            '14_component_reject_mask.png': pixel_debug.get('component_reject_mask'),
        }.items():
            if extra_image is not None:
                files[extra_name] = cv2.cvtColor(extra_image, cv2.COLOR_GRAY2BGR) if extra_image.ndim == 2 else extra_image
"""
if '09_raw_residual_mask.png' not in text and '        saved_paths: Dict[str, str] = {}' in text:
    text = text.replace('        saved_paths: Dict[str, str] = {}', extra_debug + '\n        saved_paths: Dict[str, str] = {}', 1)

path.write_text(text, encoding='utf-8')

print('Backup:', backup)
print('Version marker:', version_line)
print('Contains ComponentValidator:', 'ComponentValidator' in text)
print('Contains _apply_morphology_and_validation:', '_apply_morphology_and_validation' in text)
print('Contains _compute_triangle_pixel_damage:', '_compute_triangle_pixel_damage' in text)
print('Contains _local_illumination_normalize:', '_local_illumination_normalize' in text)
