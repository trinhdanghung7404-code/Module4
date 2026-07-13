from pathlib import Path
import re
import shutil

path = Path("mesh_damage_detector.py")
text = path.read_text(encoding="utf-8")

backup = Path("mesh_damage_detector.py.bak")
shutil.copy2(path, backup)

version = 'MESH_DAMAGE_DETECTOR_VERSION = "pixel_level_no_component_validator_v3"'

if "MESH_DAMAGE_DETECTOR_VERSION" not in text:
    text = text.replace(
        "from triangle_refiner import TriangleRefiner\n",
        "from triangle_refiner import TriangleRefiner\n\n" + version + "\n",
        1,
    )

debug_method = r'''
    def _build_mesh_debug_info(
        self,
        damage_map: np.ndarray,
        suspected_mask: np.ndarray,
        refined_overlay: np.ndarray,
        triangle_results: Sequence[ScoredTriangle],
    ) -> dict:
        final_mask = (damage_map > 0).astype(np.uint8) * 255
        suspected_binary = (suspected_mask > 0).astype(np.uint8) * 255

        return {
            "intensity_pixels": int(np.count_nonzero(final_mask)),
            "gradient_pixels": int(np.count_nonzero(final_mask)),
            "rejected_by_feature_patches": sum(
                1 for item in triangle_results
                if item.feature_similarity >= self.feature_threshold
            ),
            "rejected_by_ssim_patches": sum(
                1 for item in triangle_results
                if item.similarity >= self.ssim_threshold
            ),
            "accepted_patches": sum(1 for item in triangle_results if item.is_damaged),
            "rejected_patches": sum(1 for item in triangle_results if not item.is_damaged),
            "diff_area_before_feature": int(np.count_nonzero(suspected_binary)),
            "diff_area_after_feature": int(np.count_nonzero(suspected_binary)),
            "diff_area_after_ssim": int(np.count_nonzero(suspected_binary)),
            "diff_area_after_morphology": int(np.count_nonzero(final_mask)),
            "final_difference_area": int(np.count_nonzero(final_mask)),
            "components_before_validation": 0,
            "components_after_validation": 0,
            "pixels_removed_by_validation": 0,
            "pixels_remaining": int(np.count_nonzero(final_mask)),
            "diff_area_before_validation": int(np.count_nonzero(final_mask)),
            "diff_area_after_validation": int(np.count_nonzero(final_mask)),
            "accepted_components": [],
            "rejected_components": [],
            "all_components": [],
            "triangle_count": len(triangle_results),
            "damaged_triangle_count": sum(1 for item in triangle_results if item.is_damaged),
            "masks": {
                "raw_intensity_mask": final_mask,
                "raw_gradient_mask": final_mask,
                "raw_feature_mask": suspected_binary,
                "raw_ssim_mask": suspected_binary,
                "combined_mask_before_morphology": suspected_binary,
                "combined_mask_after_morphology": final_mask,
                "final_damage_mask": final_mask,
                "feature_reject_mask": np.zeros(damage_map.shape[:2], dtype=np.uint8),
                "ssim_reject_mask": np.zeros(damage_map.shape[:2], dtype=np.uint8),
            },
        }
'''

# Xóa nguyên hàm validation cũ
text, n1 = re.subn(
    r"\n    def _apply_morphology_and_validation\([\s\S]*?\n    def _scored_to_local_matches",
    debug_method + "\n\n    def _scored_to_local_matches",
    text,
    count=1,
)

# Thay đoạn gọi validation cũ trong run()
old_call = '''        # Convert to local matches format
        local_matches = self._scored_to_local_matches(triangle_results)

        # Apply morphology and validation
        damage_map, debug_info = self._apply_morphology_and_validation(
            damage_map,
            triangle_results,
            local_matches,
        )

'''

new_call = '''        # Convert to local matches format
        local_matches = self._scored_to_local_matches(triangle_results)

        # Build debug info WITHOUT ComponentValidator
        debug_info = self._build_mesh_debug_info(
            damage_map,
            suspected_mask,
            refined_overlay,
            triangle_results,
        )

'''

text = text.replace(old_call, new_call)

# Nếu bản file đã xóa call nhưng quên tạo debug_info thì chèn thêm
marker = '''        # Convert to local matches format
        local_matches = self._scored_to_local_matches(triangle_results)

'''
if "debug_info = self._build_mesh_debug_info" not in text and marker in text:
    text = text.replace(marker, new_call, 1)

path.write_text(text, encoding="utf-8")

print("Backup:", backup)
print("Removed validation function:", n1 > 0)
print("Contains ComponentValidator:", "ComponentValidator" in text)
print("Contains _apply_morphology_and_validation:", "_apply_morphology_and_validation" in text)
print("Contains version:", "MESH_DAMAGE_DETECTOR_VERSION" in text)