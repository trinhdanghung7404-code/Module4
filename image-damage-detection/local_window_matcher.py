from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from component_validator import ComponentValidator


@dataclass
class LocalMatchResult:
    product_point: Tuple[int, int]
    matched_point: Tuple[int, int]
    best_point: Tuple[int, int]
    best_similarity: float
    pixel_offset: float
    offset_x: float
    offset_y: float
    best_top_left: Tuple[int, int]
    search_origin: Tuple[int, int]
    product_patch: np.ndarray
    search_window: np.ndarray
    best_patch: np.ndarray
    patch_difference: np.ndarray
    similarity_scores: np.ndarray
    similarity_method: str


class LocalWindowMatcher:

    SSIM_WEIGHT_HIGH = 0.90
    SSIM_WEIGHT_MEDIUM = 0.80
    SSIM_WEIGHT_LOW = 0.70

    def __init__(self, patch_size: int = 31, window_size: int = 41, similarity_threshold: float = 0.85, diff_threshold: int = 25):
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        if window_size < patch_size:
            raise ValueError("window_size must be greater than or equal to patch_size")

        self.patch_size = patch_size
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.diff_threshold = diff_threshold
        self._half_patch = patch_size // 2
        self._half_window = window_size // 2
        self._refine_candidates = 9

        # Diagnostic threshold parameters
        self.feature_threshold = 0.95
        self.ssim_threshold = 0.90
        self.min_area_threshold = 15
        self.aspect_ratio_threshold = 5.0
        self.thinness_threshold = 2

    def _log_step(self, message: str) -> None:

        print(f"[LocalWindowMatcher] {message}")

    def _score_ssim_weight(self, ssim: float) -> Tuple[float, bool, str]:

        if ssim >= self.SSIM_WEIGHT_HIGH:
            return 1.0, True, "weight=1.0"

        if ssim >= self.SSIM_WEIGHT_MEDIUM:
            return 0.7, True, "weight=0.7"

        if ssim >= self.SSIM_WEIGHT_LOW:
            return 0.4, True, "weight=0.4"

        return 0.0, False, "rejected"

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _extract_region(self, image: np.ndarray, center: Sequence[float], size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        height, width = image.shape[:2]
        half = size // 2
        center_x = int(round(float(center[0])))
        center_y = int(round(float(center[1])))

        pad_left = max(0, half - center_x)
        pad_top = max(0, half - center_y)
        pad_right = max(0, center_x + half + 1 - width)
        pad_bottom = max(0, center_y + half + 1 - height)

        padded = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT101
        )

        padded_center_x = center_x + pad_left
        padded_center_y = center_y + pad_top
        region = padded[
            padded_center_y - half : padded_center_y + half + 1,
            padded_center_x - half : padded_center_x + half + 1
        ]

        origin_x = center_x - half
        origin_y = center_y - half
        return region, (origin_x, origin_y)

    def extract_patch(self, image: np.ndarray, center: Sequence[float], size: Optional[int] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        region_size = size or self.patch_size
        patch, origin = self._extract_region(image, center, region_size)
        return patch, origin

    def compute_similarity(self, product_patch: np.ndarray, candidate_patch: np.ndarray) -> Tuple[float, np.ndarray, str]:
        product_gray = self._to_gray(product_patch)
        candidate_gray = self._to_gray(candidate_patch)

        try:
            score, similarity_map = structural_similarity(
                product_gray,
                candidate_gray,
                data_range=255,
                full=True
            )

            if not np.isfinite(score):
                raise ValueError("Non-finite SSIM score")

            return float(score), similarity_map.astype(np.float32), "ssim"
        except Exception:
            score = float(cv2.matchTemplate(candidate_gray.astype(np.float32), product_gray.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0])
            similarity_map = np.full(product_gray.shape, score, dtype=np.float32)
            return score, similarity_map, "ncc"

    def _compute_ncc_map(self, product_patch: np.ndarray, search_window: np.ndarray) -> np.ndarray:

        product_gray = self._to_gray(product_patch).astype(np.float32)
        window_gray = self._to_gray(search_window).astype(np.float32)

        if window_gray.shape[0] < product_gray.shape[0] or window_gray.shape[1] < product_gray.shape[1]:
            raise ValueError("Search window must be larger than the patch")

        return cv2.matchTemplate(window_gray, product_gray, cv2.TM_CCOEFF_NORMED)

    def search_best_patch(
        self,
        product_image: np.ndarray,
        return_image: np.ndarray,
        product_point: Sequence[float],
        return_point: Sequence[float]
    ) -> dict:
        product_patch, product_origin = self.extract_patch(product_image, product_point, self.patch_size)
        search_window, search_origin = self.extract_patch(return_image, return_point, self.window_size)

        product_height, product_width = product_patch.shape[:2]
        window_height, window_width = search_window.shape[:2]

        candidate_rows = window_height - product_height + 1
        candidate_cols = window_width - product_width + 1

        if candidate_rows <= 0 or candidate_cols <= 0:
            raise ValueError("Search window must be larger than the patch")

        similarity_scores = np.zeros((candidate_rows, candidate_cols), dtype=np.float32)
        best_similarity = -np.inf
        best_x = 0
        best_y = 0
        best_patch = product_patch.copy()
        best_similarity_map = np.zeros((product_height, product_width), dtype=np.float32)
        similarity_method = "ssim"

        ncc_map = self._compute_ncc_map(product_patch, search_window)
        flat_ncc = ncc_map.reshape(-1)
        top_candidate_count = min(self._refine_candidates, flat_ncc.size)
        top_candidate_indices = np.argpartition(-flat_ncc, top_candidate_count - 1)[:top_candidate_count]
        top_candidate_indices = top_candidate_indices[np.argsort(-flat_ncc[top_candidate_indices])]
        top_candidate_set = set(int(index) for index in top_candidate_indices.tolist())

        for y in range(candidate_rows):
            for x in range(candidate_cols):
                candidate_patch = search_window[y:y + product_height, x:x + product_width]
                candidate_index = y * candidate_cols + x

                if candidate_index in top_candidate_set:
                    score, similarity_map, method = self.compute_similarity(product_patch, candidate_patch)
                else:
                    score = float(ncc_map[y, x])
                    similarity_map = np.full((product_height, product_width), score, dtype=np.float32)
                    method = "ncc"

                similarity_scores[y, x] = score

                if score > best_similarity:
                    best_similarity = score
                    best_x = x
                    best_y = y
                    best_patch = candidate_patch.copy()
                    best_similarity_map = similarity_map.copy()
                    similarity_method = method

        best_top_left = (search_origin[0] + best_x, search_origin[1] + best_y)
        best_point = (
            best_top_left[0] + product_width // 2,
            best_top_left[1] + product_height // 2
        )

        matched_point = (int(round(float(return_point[0]))), int(round(float(return_point[1]))))
        offset_x = float(best_point[0] - matched_point[0])
        offset_y = float(best_point[1] - matched_point[1])
        pixel_offset = float(np.sqrt(offset_x ** 2 + offset_y ** 2))

        product_patch_gray = self._to_gray(product_patch)
        best_patch_gray = self._to_gray(best_patch)
        patch_difference = cv2.absdiff(product_patch_gray, best_patch_gray)
        patch_difference = cv2.normalize(patch_difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        patch_difference = cv2.applyColorMap(patch_difference, cv2.COLORMAP_TURBO)

        weight, accepted, weight_label = self._score_ssim_weight(best_similarity)

        return {
            "product_point": (int(round(float(product_point[0]))), int(round(float(product_point[1])))),
            "matched_point": matched_point,
            "best_point": best_point,
            "ssim": float(best_similarity),
            "weight": float(weight),
            "accepted": bool(accepted),
            "weight_label": weight_label,
            "best_similarity": float(best_similarity),
            "pixel_offset": pixel_offset,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "best_top_left": best_top_left,
            "search_origin": search_origin,
            "product_patch": product_patch,
            "search_window": search_window,
            "best_patch": best_patch,
            "patch_difference": patch_difference,
            "similarity_scores": similarity_scores,
            "similarity_method": similarity_method,
            "best_similarity_map": best_similarity_map,
        }

    def build_similarity_heatmap(self, similarity_scores: np.ndarray) -> np.ndarray:
        if similarity_scores.size == 0:
            return np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)

        normalized = cv2.normalize(similarity_scores, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        heatmap = cv2.resize(
            normalized,
            (self.window_size, self.window_size),
            interpolation=cv2.INTER_CUBIC
        )
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    def build_similarity_heatmap_from_map(self, similarity_map: np.ndarray) -> np.ndarray:
        if similarity_map.size == 0:
            return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)

        normalized = cv2.normalize(similarity_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    def aggregate_damage(self, local_matches: List[dict], image_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, dict]:
        """Build a pixel-accurate edge-aware damage map from local matches.
        Also returns step-by-step diagnostic information and masks.
        """
        height, width = image_shape[:2]
        damage_map = np.zeros((height, width), dtype=np.uint8)

        patch_area = self.patch_size * self.patch_size
        per_patch_damage_counts: List[int] = []
        per_patch_intensity_counts: List[int] = []
        per_patch_gradient_counts: List[int] = []
        per_patch_final_counts: List[int] = []
        patch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combination_mode = "AND"

        # Diagnostic cumulative masks
        raw_intensity_mask = np.zeros((height, width), dtype=np.uint8)
        raw_gradient_mask = np.zeros((height, width), dtype=np.uint8)
        combined_mask_and = np.zeros((height, width), dtype=np.uint8)
        combined_mask_before_morphology = np.zeros((height, width), dtype=np.uint8)
        
        # Spectral differences to pass to validator
        global_intensity_diff = np.zeros((height, width), dtype=np.uint8)
        global_gradient_diff = np.zeros((height, width), dtype=np.uint8)
        
        # Threshold-filtered masks (active/simulated sequence)
        feature_filtered_mask = np.zeros((height, width), dtype=np.uint8)
        ssim_filtered_mask = np.zeros((height, width), dtype=np.uint8)

        def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
            grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
            magnitude = cv2.magnitude(grad_x, grad_y)
            if float(magnitude.max()) <= 1e-6:
                return np.zeros_like(gray, dtype=np.uint8)
            normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            return normalized.astype(np.uint8)

        print("\n=== Patch Debug Details ===")

        for patch_id, match in enumerate(local_matches):
            product_patch = match.get("product_patch")
            best_patch = match.get("best_patch")

            if product_patch is None or best_patch is None:
                continue

            if product_patch.shape[:2] != best_patch.shape[:2]:
                continue

            # --- Intensity path: blur → illumination normalize → absdiff ---
            product_gray = cv2.GaussianBlur(
                self._to_gray(product_patch), (3, 3), 0
            )
            best_gray = cv2.GaussianBlur(
                self._to_gray(best_patch), (3, 3), 0
            )

            product_mean = float(product_gray.mean())
            product_std = float(product_gray.std())
            best_mean = float(best_gray.mean())
            best_std = float(best_gray.std())

            best_float = best_gray.astype(np.float32)
            if best_std < 1e-6:
                best_normalized = np.full_like(best_float, product_mean, dtype=np.float32)
            else:
                scale = product_std / best_std
                best_normalized = (best_float - best_mean) * scale + product_mean
            best_normalized = np.clip(best_normalized, 0, 255).astype(np.uint8)

            intensity_diff = cv2.absdiff(product_gray, best_normalized)
            intensity_mask = (intensity_diff > self.diff_threshold).astype(np.uint8)
            intensity_damage_px = int(np.count_nonzero(intensity_mask))

            # --- Gradient path: blur → Scharr magnitude → absdiff ---
            product_grad = _gradient_magnitude(product_gray)
            best_grad = _gradient_magnitude(best_gray)
            gradient_diff = cv2.absdiff(product_grad, best_grad)
            gradient_mask = (gradient_diff > self.diff_threshold).astype(np.uint8)
            gradient_damage_px = int(np.count_nonzero(gradient_mask))

            # --- Combine: AND confirms both surface + structural change ---
            and_mask = cv2.bitwise_and(intensity_mask, gradient_mask)
            and_px = int(np.count_nonzero(and_mask))
            min_channel_px = min(intensity_damage_px, gradient_damage_px)
            retention_ratio = and_px / min_channel_px if min_channel_px > 0 else 1.0

            if and_px > 0 and (min_channel_px == 0 or retention_ratio >= 0.10):
                damage_mask_patch = and_mask
                patch_combination = "AND"
            elif intensity_damage_px > 0 and gradient_damage_px > 0:
                grad_support = cv2.dilate(gradient_mask, patch_kernel, iterations=1)
                damage_mask_patch = cv2.bitwise_and(intensity_mask, grad_support)
                patch_combination = "intensity_AND_dilated_gradient"
            else:
                damage_mask_patch = and_mask
                patch_combination = "AND"

            final_damage_px = int(np.count_nonzero(damage_mask_patch))
            if patch_combination != "AND":
                combination_mode = patch_combination

            per_patch_intensity_counts.append(intensity_damage_px)
            per_patch_gradient_counts.append(gradient_damage_px)
            per_patch_final_counts.append(final_damage_px)

            # Combined patch mask BEFORE morphological opening
            combined_patch_and = damage_mask_patch.copy()
            damage_count = final_damage_px
            per_patch_damage_counts.append(damage_count)

            # Retrieve thresholds & determine pass/fail
            ssim = float(match.get("ssim", match.get("best_similarity", 0.0)))
            feat_sim = float(match.get("feature_similarity", 0.0))

            intensity_status = "PASS" if intensity_damage_px > 0 else "FAIL"
            gradient_status = "PASS" if gradient_damage_px > 0 else "FAIL"
            
            # Rejection thresholds
            pass_feature = feat_sim < self.feature_threshold
            pass_ssim = ssim < self.ssim_threshold
            
            feat_status = "PASS" if pass_feature else "FAIL"
            ssim_status = "PASS" if pass_ssim else "FAIL"
            
            final_status = "KEEP" if (intensity_damage_px > 0 or gradient_damage_px > 0) and pass_feature and pass_ssim else "REJECT"

            print(f"Patch {patch_id:3d}: Intensity = {intensity_status:4s} (px: {intensity_damage_px:3d}), "
                  f"Gradient = {gradient_status:4s} (px: {gradient_damage_px:3d}), "
                  f"Feature Similarity = {feat_sim:.4f} ({feat_status:4s}), "
                  f"SSIM = {ssim:.4f} ({ssim_status:4s}), "
                  f"Final = {final_status}")

            if damage_count == 0:
                continue

            # Severity based on SSIM only (geometry, no weight)
            severity = int(np.clip((1.0 - ssim) * 255.0, 1, 255))

            # Resolve placement bounds in image coordinates
            top_left_x, top_left_y = match["best_top_left"]
            patch_h, patch_w = product_patch.shape[:2]

            x0 = max(0, top_left_x)
            y0 = max(0, top_left_y)
            x1 = min(width, top_left_x + patch_w)
            y1 = min(height, top_left_y + patch_h)

            if x0 >= x1 or y0 >= y1:
                continue

            dx0 = x0 - top_left_x
            dy0 = y0 - top_left_y
            
            # Crop to matching valid bounds
            local_intensity = intensity_mask[dy0: dy0 + (y1 - y0), dx0: dx0 + (x1 - x0)]
            local_gradient = gradient_mask[dy0: dy0 + (y1 - y0), dx0: dx0 + (x1 - x0)]
            local_and = combined_patch_and[dy0: dy0 + (y1 - y0), dx0: dx0 + (x1 - x0)]
            local_intensity_diff = intensity_diff[dy0: dy0 + (y1 - y0), dx0: dx0 + (x1 - x0)]
            local_gradient_diff = gradient_diff[dy0: dy0 + (y1 - y0), dx0: dx0 + (x1 - x0)]

            # Place onto global diagnostic masks
            raw_intensity_mask[y0:y1, x0:x1] = np.maximum(raw_intensity_mask[y0:y1, x0:x1], (local_intensity > 0).astype(np.uint8) * 255)
            raw_gradient_mask[y0:y1, x0:x1] = np.maximum(raw_gradient_mask[y0:y1, x0:x1], (local_gradient > 0).astype(np.uint8) * 255)
            combined_mask_and[y0:y1, x0:x1] = np.maximum(combined_mask_and[y0:y1, x0:x1], (local_and > 0).astype(np.uint8) * 255)
            combined_mask_before_morphology[y0:y1, x0:x1] = np.maximum(combined_mask_before_morphology[y0:y1, x0:x1], (local_and > 0).astype(np.uint8) * 255)

            # Accumulate global difference maps
            global_intensity_diff[y0:y1, x0:x1] = np.maximum(global_intensity_diff[y0:y1, x0:x1], local_intensity_diff)
            global_gradient_diff[y0:y1, x0:x1] = np.maximum(global_gradient_diff[y0:y1, x0:x1], local_gradient_diff)

            # Placements for sequential threshold filtering
            if pass_feature:
                feature_filtered_mask[y0:y1, x0:x1] = np.maximum(feature_filtered_mask[y0:y1, x0:x1], (local_and > 0).astype(np.uint8) * 255)
            if pass_feature and pass_ssim:
                ssim_filtered_mask[y0:y1, x0:x1] = np.maximum(ssim_filtered_mask[y0:y1, x0:x1], (local_and > 0).astype(np.uint8) * 255)

            # Active path: accumulate into damage_map ONLY if patch passed both filters
            if pass_feature and pass_ssim:
                current_region = damage_map[y0:y1, x0:x1]
                update_mask = local_and.astype(bool)
                current_region[update_mask] = np.maximum(
                    current_region[update_mask], severity
                )
                damage_map[y0:y1, x0:x1] = current_region

        # Global morphology (Opening then Closing) of active path and diagnostic path
        combined_mask_after_morphology = ssim_filtered_mask.copy()

        if np.count_nonzero(damage_map) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Active path morphology (Opening then Closing)
            damage_map = cv2.morphologyEx(damage_map, cv2.MORPH_OPEN, kernel, iterations=1)
            damage_map = cv2.morphologyEx(damage_map, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Diagnostic path global morphology
            combined_mask_after_morphology = cv2.morphologyEx(ssim_filtered_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            combined_mask_after_morphology = cv2.morphologyEx(combined_mask_after_morphology, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Connected Components and Validation Stage (Semantic Component Validation)
        # Instantiate ComponentValidator with current thresholds
        validator = ComponentValidator(
            min_area=self.min_area_threshold,
            feature_threshold=self.feature_threshold,
            ssim_threshold=self.ssim_threshold,
            gradient_threshold=self.diff_threshold,
            compactness_threshold=0.05,
            min_patch_support=1
        )
        
        # Run CC analysis on active path and validate
        components = validator.compute_statistics(
            (damage_map > 0).astype(np.uint8) * 255,
            global_intensity_diff,
            global_gradient_diff,
            local_matches
        )
        accepted, rejected = validator.validate(components)
        
        # Apply validation decisions to the active path
        diff_area_before_val = int(np.count_nonzero(damage_map))
        for comp in rejected:
            damage_map[comp["mask"] > 0] = 0
            
        diff_area_after_val = int(np.count_nonzero(damage_map))
        pixels_removed_by_val = diff_area_before_val - diff_area_after_val

        # Diagnostic tracker for statistics reporting
        components_diag = validator.compute_statistics(
            combined_mask_after_morphology,
            global_intensity_diff,
            global_gradient_diff,
            local_matches
        )
        accepted_diag, rejected_diag = validator.validate(components_diag)

        total_patches = len(local_matches)
        total_damage_pixels = int(np.count_nonzero(damage_map))

        # Log patch statistics
        rejected_by_feature = sum(1 for m in local_matches if m.get("feature_similarity", 0.0) >= self.feature_threshold)
        rejected_by_ssim = sum(1 for m in local_matches if m.get("ssim", 0.0) >= self.ssim_threshold)
        accepted_patches = sum(1 for m in local_matches if m.get("feature_similarity", 0.0) < self.feature_threshold and m.get("ssim", 0.0) < self.ssim_threshold)
        rejected_patches = len(local_matches) - accepted_patches

        # Pixel counts
        intensity_px_count = int(np.count_nonzero(raw_intensity_mask))
        gradient_px_count = int(np.count_nonzero(raw_gradient_mask))
        and_px_count = int(np.count_nonzero(combined_mask_and))
        before_feature_px_count = int(np.count_nonzero(combined_mask_before_morphology))
        after_feature_px_count = int(np.count_nonzero(feature_filtered_mask))
        after_ssim_px_count = int(np.count_nonzero(ssim_filtered_mask))
        after_morph_px_count = int(np.count_nonzero(combined_mask_after_morphology))

        print("\n=== Validation Statistics ===")
        print(f"Components Before Validation               : {len(components)}")
        print(f"Components After Validation                : {len(accepted)}")
        print(f"Pixels Removed By Validation               : {pixels_removed_by_val}")
        print(f"Pixels Remaining                           : {diff_area_after_val}")
        print(f"Difference Area Before Validation          : {diff_area_before_val}")
        print(f"Difference Area After Validation           : {diff_area_after_val}")

        print("\n=== Pixel Statistics & Diagnostic Summary ===")
        print(f"Intensity pixels:                             {intensity_px_count}")
        print(f"Gradient pixels:                              {gradient_px_count}")
        print(f"Rejected by Feature (Patches):                {rejected_by_feature}")
        print(f"Rejected by SSIM (Patches):                   {rejected_by_ssim}")
        print(f"Accepted Patches:                             {accepted_patches}")
        print(f"Rejected Patches:                             {rejected_patches}")
        print(f"Difference Area Before Feature Filter:        {before_feature_px_count}")
        print(f"Difference Area After Feature Filter:         {after_feature_px_count}")
        print(f"Difference Area After SSIM Filter:            {after_ssim_px_count}")
        print(f"Difference Area After Morphology:             {after_morph_px_count}")
        print(f"Final Difference Area:                        {total_damage_pixels}")

        debug_info = {
            "intensity_pixels": intensity_px_count,
            "gradient_pixels": gradient_px_count,
            "rejected_by_feature_patches": rejected_by_feature,
            "rejected_by_ssim_patches": rejected_by_ssim,
            "accepted_patches": accepted_patches,
            "rejected_patches": rejected_patches,
            "diff_area_before_feature": before_feature_px_count,
            "diff_area_after_feature": after_feature_px_count,
            "diff_area_after_ssim": after_ssim_px_count,
            "diff_area_after_morphology": after_morph_px_count,
            "final_difference_area": total_damage_pixels,
            "components_before_validation": len(components),
            "components_after_validation": len(accepted),
            "pixels_removed_by_validation": pixels_removed_by_val,
            "pixels_remaining": diff_area_after_val,
            "diff_area_before_validation": diff_area_before_val,
            "diff_area_after_validation": diff_area_after_val,
            "accepted_components": accepted_diag,
            "rejected_components": rejected_diag,
            "all_components": components_diag,
            "masks": {
                "raw_intensity_mask": raw_intensity_mask,
                "raw_gradient_mask": raw_gradient_mask,
                "raw_feature_mask": feature_filtered_mask,
                "raw_ssim_mask": ssim_filtered_mask,
                "combined_mask_before_morphology": combined_mask_before_morphology,
                "combined_mask_after_morphology": combined_mask_after_morphology,
                "final_damage_mask": (damage_map > 0).astype(np.uint8) * 255,
                "feature_reject_mask": combined_mask_before_morphology & ~feature_filtered_mask,
                "ssim_reject_mask": feature_filtered_mask & ~ssim_filtered_mask,
            }
        }

        # Severity calculations
        if per_patch_final_counts:
            avg_intensity = float(np.mean(per_patch_intensity_counts))
            avg_gradient = float(np.mean(per_patch_gradient_counts))
            avg_final = float(np.mean(per_patch_final_counts))
            reduction_ratio = 1.0 - (avg_final / avg_intensity) if avg_intensity > 0 else 0.0
            damage_pixel_ratio = avg_final / patch_area if patch_area > 0 else 0.0
            self._log_step(
                f"aggregate_damage summary: "
                f"patches={total_patches}, "
                f"combination_mode={combination_mode}, "
                f"patch_size={self.patch_size}\u00d7{self.patch_size}={patch_area}px, "
                f"diff_threshold={self.diff_threshold}, "
                f"avg_intensity_damage_px={avg_intensity:.1f}, "
                f"avg_gradient_damage_px={avg_gradient:.1f}, "
                f"avg_final_damage_px={avg_final:.1f}, "
                f"reduction_ratio={reduction_ratio:.4f}, "
                f"total_damage_px(after_morphology)={total_damage_pixels}, "
                f"damage_pixel_ratio={damage_pixel_ratio:.4f}"
            )

        return damage_map, debug_info

    def _build_damage_map_from_local_matches(self, local_matches: List[dict], image_shape: Tuple[int, int, int]) -> np.ndarray:

        damage_map, _ = self.aggregate_damage(local_matches, image_shape)

    def build_local_matching_overlay(self, product_image: np.ndarray, aligned_return_image: np.ndarray, damage_map: np.ndarray) -> np.ndarray:
        product_red = np.zeros_like(product_image)
        aligned_green = np.zeros_like(aligned_return_image)

        product_gray = self._to_gray(product_image)
        aligned_gray = self._to_gray(aligned_return_image)

        product_red[:, :, 2] = product_gray
        aligned_green[:, :, 1] = aligned_gray

        overlay = cv2.addWeighted(product_red, 0.5, aligned_green, 0.5, 0)

        if np.count_nonzero(damage_map) > 0:
            damage_colored = cv2.applyColorMap(
                cv2.normalize(damage_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_HOT
            )
            overlay = cv2.addWeighted(overlay, 0.78, damage_colored, 0.22, 0)

        return overlay

    def _build_ssim_histogram(self, similarities: np.ndarray) -> np.ndarray:

        histogram_width = 640
        histogram_height = 360
        margin_left = 60
        margin_right = 20
        margin_top = 30
        margin_bottom = 50
        plot_width = histogram_width - margin_left - margin_right
        plot_height = histogram_height - margin_top - margin_bottom

        canvas = np.full((histogram_height, histogram_width, 3), 255, dtype=np.uint8)

        if similarities.size == 0:
            cv2.putText(canvas, "No SSIM values", (margin_left, histogram_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            return canvas

        bins = np.linspace(0.0, 1.0, 11)
        counts, edges = np.histogram(similarities, bins=bins)
        max_count = max(1, int(np.max(counts)))
        bin_width = plot_width // len(counts)

        for index, count in enumerate(counts):
            bar_height = int(round((count / max_count) * plot_height))
            x0 = margin_left + index * bin_width + 4
            x1 = margin_left + (index + 1) * bin_width - 4
            y1 = histogram_height - margin_bottom
            y0 = y1 - bar_height
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 120, 240), -1)
            cv2.putText(
                canvas,
                str(int(count)),
                (x0, max(margin_top + 15, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        for threshold, color in ((0.75, (0, 140, 255)), (0.80, (0, 180, 0)), (0.85, (0, 0, 255))):
            x = margin_left + int(round(threshold * plot_width))
            cv2.line(canvas, (x, margin_top), (x, histogram_height - margin_bottom), color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{threshold:.2f}",
                (x - 18, histogram_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

        cv2.rectangle(canvas, (margin_left, margin_top), (histogram_width - margin_right, histogram_height - margin_bottom), (0, 0, 0), 1)
        cv2.putText(canvas, "SSIM distribution", (margin_left, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "0.0", (margin_left - 10, histogram_height - margin_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "1.0", (histogram_width - margin_right - 20, histogram_height - margin_bottom + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas

    def run(
        self,
        product_image: np.ndarray,
        aligned_return_image: np.ndarray,
        product_keypoints: Sequence[Sequence[float]],
        return_keypoints: Sequence[Sequence[float]],
        inlier_matches: Sequence,
        product_descriptors: Optional[Sequence] = None,
        return_descriptors: Optional[Sequence] = None,
    ) -> dict:
        local_matches = []
        search_windows = 0

        # L2-normalize descriptors if available for robust cosine similarity calculation
        product_descriptors_np = None
        return_descriptors_np = None
        if product_descriptors is not None and len(product_descriptors) > 0:
            product_descriptors_np = np.asarray(product_descriptors, dtype=np.float32)
        if return_descriptors is not None and len(return_descriptors) > 0:
            return_descriptors_np = np.asarray(return_descriptors, dtype=np.float32)

        for match in inlier_matches:
            if match.queryIdx >= len(product_keypoints) or match.trainIdx >= len(return_keypoints):
                continue

            search_windows += 1
            product_point = product_keypoints[match.queryIdx]
            return_point = return_keypoints[match.trainIdx]
            local_match = self.search_best_patch(product_image, aligned_return_image, product_point, return_point)
            
            # Compute descriptor cosine similarity (normalize inside check for safety)
            feature_similarity = 0.0
            if product_descriptors_np is not None and return_descriptors_np is not None:
                q_idx = match.queryIdx
                t_idx = match.trainIdx
                if q_idx < len(product_descriptors_np) and t_idx < len(return_descriptors_np):
                    desc_p = product_descriptors_np[q_idx]
                    desc_r = return_descriptors_np[t_idx]
                    norm_p = np.linalg.norm(desc_p)
                    norm_r = np.linalg.norm(desc_r)
                    if norm_p > 1e-8 and norm_r > 1e-8:
                        feature_similarity = float(np.dot(desc_p, desc_r) / (norm_p * norm_r))
            
            local_match["feature_similarity"] = feature_similarity
            local_matches.append(local_match)

        if local_matches:
            best_match = min(local_matches, key=lambda item: item["best_similarity"])
        else:
            best_match = {
                "product_patch": np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8),
                "search_window": np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8),
                "best_patch": np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8),
                "patch_difference": np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8),
                "similarity_scores": np.zeros((self.window_size - self.patch_size + 1, self.window_size - self.patch_size + 1), dtype=np.float32),
                "best_top_left": (0, 0),
            }

        damage_map, debug_info = self.aggregate_damage(local_matches, aligned_return_image.shape)
        similarity_heatmap = self.build_similarity_heatmap(best_match["similarity_scores"])
        best_patch_heatmap = self.build_similarity_heatmap_from_map(best_match["best_similarity_map"])

        similarities = np.asarray([match["ssim"] for match in local_matches], dtype=np.float32) if local_matches else np.asarray([0.0], dtype=np.float32)
        ssim_histogram = self._build_ssim_histogram(similarities)

        search_window_debug = best_match["search_window"].copy()
        best_top_left = best_match.get("best_top_left", (0, 0))
        cv2.rectangle(
            search_window_debug,
            (best_top_left[0] - best_match["search_origin"][0], best_top_left[1] - best_match["search_origin"][1]),
            (
                best_top_left[0] - best_match["search_origin"][0] + self.patch_size,
                best_top_left[1] - best_match["search_origin"][1] + self.patch_size,
            ),
            (0, 255, 0),
            2
        )

        summary = self._build_summary(local_matches, damage_map)
        overlay = self.build_local_matching_overlay(product_image, aligned_return_image, damage_map)

        weights = np.asarray([match["weight"] for match in local_matches], dtype=np.float32) if local_matches else np.asarray([0.0], dtype=np.float32)
        accepted_matches = int(np.count_nonzero(weights > 0.0))
        rejected_matches = int(len(local_matches) - accepted_matches)
        filtered_local_matches = accepted_matches
        weight_counts = {
            1.0: int(np.count_nonzero(np.isclose(weights, 1.0))),
            0.7: int(np.count_nonzero(np.isclose(weights, 0.7))),
            0.4: int(np.count_nonzero(np.isclose(weights, 0.4))),
        }

        self._log_step(
            "Local Matching: "
            f"input_matches={len(inlier_matches)}, "
            f"search_windows={search_windows}, "
            f"local_matches={len(local_matches)}, "
            f"filtered_local_matches={filtered_local_matches}, "
            f"accepted_matches={accepted_matches}, "
            f"rejected_matches={rejected_matches}, "
            f"ssim_min={float(np.min(similarities)):.3f}, "
            f"ssim_max={float(np.max(similarities)):.3f}, "
            f"ssim_mean={float(np.mean(similarities)):.3f}, "
            f"ssim_median={float(np.median(similarities)):.3f}, "
            f"avg_weight={float(np.mean(weights)):.3f}, "
            f"min_weight={float(np.min(weights)):.3f}, "
            f"max_weight={float(np.max(weights)):.3f}, "
            f"weight=1.0:{weight_counts[1.0]}, "
            f"weight=0.7:{weight_counts[0.7]}, "
            f"weight=0.4:{weight_counts[0.4]}"
        )

        return {
            "local_matches": local_matches,
            "damage_map": damage_map,
            "summary": summary,
            "debug_images": {
                "patch_product": best_match["product_patch"],
                "search_window": search_window_debug,
                "best_patch": best_match["best_patch"],
                "similarity_heatmap": similarity_heatmap,
                "best_patch_heatmap": best_patch_heatmap,
                "patch_difference": best_match["patch_difference"],
                "damage_map": cv2.cvtColor(damage_map, cv2.COLOR_GRAY2BGR),
                "local_matching_overlay": overlay,
                "ssim_histogram": ssim_histogram,
            },
            "representative_match": best_match,
            "debug_info": debug_info,
        }

    def _build_summary(self, local_matches: List[dict], damage_map: np.ndarray) -> dict:
        if local_matches:
            similarities = np.asarray([match["ssim"] for match in local_matches], dtype=np.float32)
            weights = np.asarray([match["weight"] for match in local_matches], dtype=np.float32)
            offsets = np.asarray([match["pixel_offset"] for match in local_matches], dtype=np.float32)
        else:
            similarities = np.asarray([0.0], dtype=np.float32)
            weights = np.asarray([0.0], dtype=np.float32)
            offsets = np.asarray([0.0], dtype=np.float32)

        weight_sum = float(np.sum(weights))
        weighted_similarity = float(np.sum(weights * similarities) / weight_sum) if weight_sum > 0 else 0.0
        accepted_matches = int(np.count_nonzero(weights > 0.0))
        rejected_matches = int(len(local_matches) - accepted_matches)

        return {
            "total_matches": int(len(local_matches)),
            "filtered_matches": int(accepted_matches),
            "accepted_matches": int(accepted_matches),
            "rejected_matches": int(rejected_matches),
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
            "average_offset": float(np.mean(offsets)),
            "maximum_offset": float(np.max(offsets)),
            "patch_size": int(self.patch_size),
            "search_window_size": int(self.window_size),
            "similarity_threshold": float(self.similarity_threshold),
            "damage_pixels": int(np.count_nonzero(damage_map)),
        }
