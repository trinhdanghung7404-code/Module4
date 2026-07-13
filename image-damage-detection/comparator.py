from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from geometry import GeometryFeature
from mesh_damage_detector import MeshDamageDetector
from component_validator import ComponentValidator
from feature_representation import FeatureRepresentation


class DamageComparator:

    ALIGNMENT_METHOD = "tps"
    TPS_MIN_CONTROL_POINTS = 4

    def __init__(self):
        self.geometry = GeometryFeature()
        self.mesh_damage_detector = MeshDamageDetector()
        self.feature_representation = FeatureRepresentation()

    def _log_step(self, message):

        print(f"[DamageComparator] {message}")

    def _build_keypoint_objects(self, keypoint_points):

        return [cv2.KeyPoint(float(point[0]), float(point[1]), 1.0) for point in keypoint_points]

    def _summarize_match_distances(self, matches):

        if not matches:
            return 0.0, 0.0

        distances = [float(match.distance) for match in matches]
        return float(np.mean(distances)), float(np.median(distances))

    def _create_keypoints_from_points(self, points):

        return [cv2.KeyPoint(float(point[0]), float(point[1]), 1.0) for point in points]

    def _create_match_canvas(self, image_a, image_b):

        height = max(image_a.shape[0], image_b.shape[0])
        width = image_a.shape[1] + image_b.shape[1]

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[: image_a.shape[0], : image_a.shape[1]] = image_a
        canvas[: image_b.shape[0], image_a.shape[1] : image_a.shape[1] + image_b.shape[1]] = image_b

        return canvas, image_a.shape[1]

    def _draw_matches(self, image_a, keypoints_a, image_b, keypoints_b, matches, match_colors):

        canvas, offset_x = self._create_match_canvas(image_a, image_b)

        for index, match in enumerate(matches):
            if match.queryIdx >= len(keypoints_a) or match.trainIdx >= len(keypoints_b):
                continue

            point_a = keypoints_a[match.queryIdx]
            point_b = keypoints_b[match.trainIdx]
            color = match_colors[index] if index < len(match_colors) else (0, 255, 0)

            start = (int(round(point_a[0])), int(round(point_a[1])))
            end = (int(round(point_b[0] + offset_x)), int(round(point_b[1])))

            cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, start, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, end, 3, color, -1, cv2.LINE_AA)

        return canvas

    def _build_keypoint_visualization(self, image, keypoints):

        return cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )

    def _build_difference_visualization(self, product_image, compared_image):

        diff = cv2.absdiff(product_image, compared_image)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        if np.count_nonzero(diff_gray) == 0:
            return cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR)

        diff_normalized = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
        diff_normalized = diff_normalized.astype(np.uint8)

        return cv2.applyColorMap(diff_normalized, cv2.COLORMAP_TURBO)

    def _build_alignment_overlay(self, product_image, aligned_return_image):

        product_red = np.zeros_like(product_image)
        aligned_green = np.zeros_like(aligned_return_image)

        product_gray = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
        aligned_gray = cv2.cvtColor(aligned_return_image, cv2.COLOR_BGR2GRAY)

        product_red[:, :, 2] = product_gray
        aligned_green[:, :, 1] = aligned_gray

        return cv2.addWeighted(product_red, 0.5, aligned_green, 0.5, 0)

    def _fit_tps_inverse_transform(self, product_points, return_points):

        product_points = np.asarray(product_points, dtype=np.float64)
        return_points = np.asarray(return_points, dtype=np.float64)

        if product_points.ndim != 2 or return_points.ndim != 2:
            return None

        if product_points.shape[0] != return_points.shape[0] or product_points.shape[0] < 4:
            return None

        unique_points, unique_indices = np.unique(product_points, axis=0, return_index=True)
        unique_targets = return_points[unique_indices]

        if unique_points.shape[0] < 4:
            return None

        smoothing_values = (1e-6, 1e-4, 1e-3, 1e-2)

        for smoothing in smoothing_values:
            try:
                x_model = RBFInterpolator(
                    unique_points,
                    unique_targets[:, 0],
                    kernel="thin_plate_spline",
                    smoothing=smoothing
                )
                y_model = RBFInterpolator(
                    unique_points,
                    unique_targets[:, 1],
                    kernel="thin_plate_spline",
                    smoothing=smoothing
                )

                def transform(points):
                    points = np.asarray(points, dtype=np.float64)
                    return np.column_stack([x_model(points), y_model(points)])

                return transform

            except (np.linalg.LinAlgError, ValueError):
                continue

        return None

    def _warp_image_with_tps(self, return_image, product_shape, product_points, return_points):

        transform = self._fit_tps_inverse_transform(product_points, return_points)

        if transform is None:
            return None, None

        height, width = product_shape[:2]

        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        mapped_points = transform(grid_points)
        map_x = mapped_points[:, 0].reshape(height, width).astype(np.float32)
        map_y = mapped_points[:, 1].reshape(height, width).astype(np.float32)

        aligned_return = cv2.remap(
            return_image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        predicted_points = transform(product_points)
        rmse = float(np.sqrt(np.mean(np.sum((predicted_points - return_points) ** 2, axis=1))))

        return aligned_return, rmse

    def _collect_superpoint_match_details(self, product_descriptors, return_descriptors):

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = matcher.match(product_descriptors, return_descriptors)
        knn_matches = matcher.knnMatch(product_descriptors, return_descriptors, k=2)

        ratio_matches = []
        ratio_threshold = None
        match_strategy = "insufficient"

        for candidate_threshold in (0.75, 0.85):
            current_ratio_matches = []
            for pair in knn_matches:
                if len(pair) < 2:
                    continue

                best, second_best = pair
                if best.distance < candidate_threshold * second_best.distance:
                    current_ratio_matches.append(best)

            if len(current_ratio_matches) >= 4:
                ratio_matches = current_ratio_matches
                ratio_threshold = candidate_threshold
                match_strategy = "ratio"
                break

        selected_matches = ratio_matches
        if not selected_matches:
            selected_matches = sorted(raw_matches, key=lambda match: match.distance)
            if len(selected_matches) > 40:
                selected_matches = selected_matches[:40]
            if len(selected_matches) >= 4:
                match_strategy = "top_distance"

        self._log_step(
            f"SuperPoint matches: raw={len(raw_matches)}, ratio={len(ratio_matches)}, selected={len(selected_matches)}, ratio_threshold={ratio_threshold}, strategy={match_strategy}"
        )

        return {
            "raw_matches": raw_matches,
            "ratio_matches": ratio_matches,
            "selected_matches": selected_matches,
            "ratio_threshold": ratio_threshold,
            "match_strategy": match_strategy,
        }

    def _collect_superpoint_matches(self, product_descriptors, return_descriptors):

        match_details = self._collect_superpoint_match_details(product_descriptors, return_descriptors)
        return match_details["selected_matches"], match_details["ratio_threshold"], match_details["match_strategy"]

    def _collect_sift_match_details(self, product_descriptors, return_descriptors):

        return self._collect_superpoint_match_details(product_descriptors, return_descriptors)

    def _collect_sift_matches(self, product_descriptors, return_descriptors):

        return self._collect_superpoint_matches(product_descriptors, return_descriptors)

    def _extract_inlier_matches(self, product_feature, return_feature, alignment_result):

        selected_matches = alignment_result.get("selected_matches", [])
        inlier_mask = alignment_result.get("inlier_mask")

        if not selected_matches or inlier_mask is None:
            return []

        inlier_flags = np.asarray(inlier_mask).ravel().astype(bool)
        return [match for match, is_inlier in zip(selected_matches, inlier_flags) if is_inlier]

    def _feature_representation(self, product_feature, return_feature, alignment_result):

        product_descriptors = np.asarray(product_feature.get("texture_feature", {}).get("descriptors", []), dtype=np.float32)
        return_descriptors = np.asarray(return_feature.get("texture_feature", {}).get("descriptors", []), dtype=np.float32)
        inlier_matches = self._extract_inlier_matches(product_feature, return_feature, alignment_result)

        return self.feature_representation.compare(
            product_descriptors,
            return_descriptors,
            inlier_matches
        )

    def _sift_similarity(self, product_texture, return_texture):

        product_descriptors = np.asarray(product_texture.get("descriptors", []), dtype=np.float32)
        return_descriptors = np.asarray(return_texture.get("descriptors", []), dtype=np.float32)

        product_count = int(product_texture.get("keypoint_count", 0) or 0)
        return_count = int(return_texture.get("keypoint_count", 0) or 0)

        if product_descriptors.size == 0 or return_descriptors.size == 0:
            return {
                "product_keypoints": product_count,
                "return_keypoints": return_count,
                "good_matches": 0,
                "match_ratio": 0.0,
                "similarity_score": 0.0
            }

        if len(product_descriptors.shape) != 2 or len(return_descriptors.shape) != 2:
            return {
                "product_keypoints": product_count,
                "return_keypoints": return_count,
                "product_descriptors": int(product_descriptors.shape[0]) if product_descriptors.ndim == 2 else 0,
                "return_descriptors": int(return_descriptors.shape[0]) if return_descriptors.ndim == 2 else 0,
                "good_matches": 0,
                "raw_matches": 0,
                "match_ratio": 0.0,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "mean_good_distance": 0.0,
                "median_good_distance": 0.0,
                "similarity_score": 0.0
            }

        match_details = self._collect_superpoint_match_details(product_descriptors, return_descriptors)
        ratio_matches = match_details["ratio_matches"]
        selected_matches = match_details["selected_matches"]

        mean_good_distance, median_good_distance = self._summarize_match_distances(ratio_matches)

        normalizer = max(1, min(product_count, return_count))
        match_ratio = len(ratio_matches) / normalizer
        match_ratio = max(0.0, min(1.0, match_ratio))

        return {
            "product_keypoints": product_count,
            "return_keypoints": return_count,
            "product_descriptors": int(product_descriptors.shape[0]),
            "return_descriptors": int(return_descriptors.shape[0]),
            "raw_matches": len(match_details["raw_matches"]),
            "good_matches": len(ratio_matches),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "mean_good_distance": mean_good_distance,
            "median_good_distance": median_good_distance,
            "match_ratio": match_ratio,
            "selected_matches": len(selected_matches),
            "similarity_score": match_ratio * 100.0
        }

    def _align_return_image_homography(self, product_image, return_image, product_texture, return_texture):

        product_descriptors = np.asarray(product_texture.get("descriptors", []), dtype=np.float32)
        return_descriptors = np.asarray(return_texture.get("descriptors", []), dtype=np.float32)
        product_keypoints = np.asarray(product_texture.get("keypoints", []), dtype=np.float32)
        return_keypoints = np.asarray(return_texture.get("keypoints", []), dtype=np.float32)

        if (
            product_descriptors.size == 0
            or return_descriptors.size == 0
            or product_keypoints.size == 0
            or return_keypoints.size == 0
            or len(product_descriptors.shape) != 2
            or len(return_descriptors.shape) != 2
        ):
            self._log_step(
                f"Alignment input missing: product_keypoints={len(product_keypoints)}, return_keypoints={len(return_keypoints)}, product_descriptors_shape={product_descriptors.shape}, return_descriptors_shape={return_descriptors.shape}"
            )
            return return_image, {
                "aligned": False,
                "inlier_count": 0,
                "match_strategy": "missing_descriptors",
                "homography_input_matches": 0,
                "inlier_ratio": 0.0,
                "selected_matches": [],
                "inlier_mask": None
            }

        match_details = self._collect_superpoint_match_details(
            product_descriptors,
            return_descriptors
        )

        selected_matches = match_details["selected_matches"]
        ratio_threshold = match_details["ratio_threshold"]
        match_strategy = match_details["match_strategy"]

        self._log_step(
            f"Homography input: product_keypoints={len(product_keypoints)}, return_keypoints={len(return_keypoints)}, selected_matches={len(selected_matches)}, ratio_threshold={ratio_threshold}, strategy={match_strategy}"
        )

        if len(selected_matches) < 4:
            return return_image, {
                "aligned": False,
                "inlier_count": len(selected_matches),
                "match_strategy": match_strategy,
                "ratio_threshold": ratio_threshold,
                "homography_input_matches": len(selected_matches),
                "inlier_ratio": 0.0,
                "selected_matches": selected_matches,
                "inlier_mask": None,
                "aligned_return_keypoints": []
            }

        source_points = np.float32([return_keypoints[m.trainIdx] for m in selected_matches]).reshape(-1, 1, 2)
        target_points = np.float32([product_keypoints[m.queryIdx] for m in selected_matches]).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)

        if homography is None:
            return return_image, {
                "aligned": False,
                "inlier_count": len(selected_matches),
                "match_strategy": match_strategy,
                "ratio_threshold": ratio_threshold,
                "homography_input_matches": len(selected_matches),
                "inlier_ratio": 0.0,
                "selected_matches": selected_matches,
                "inlier_mask": None,
                "aligned_return_keypoints": []
            }

        aligned_return = cv2.warpPerspective(
            return_image,
            homography,
            (product_image.shape[1], product_image.shape[0])
        )

        all_return_keypoints = np.asarray(return_texture.get("keypoints", []), dtype=np.float32)
        aligned_return_keypoints = []
        if all_return_keypoints.size > 0 and len(all_return_keypoints.shape) == 2:
            transformed_keypoints = cv2.perspectiveTransform(all_return_keypoints.reshape(-1, 1, 2), homography)
            aligned_return_keypoints = transformed_keypoints.reshape(-1, 2).tolist()

        inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
        homography_input_matches = len(selected_matches)
        inlier_ratio = inlier_count / max(1, homography_input_matches)

        self._log_step(
            f"RANSAC inliers: {inlier_count}/{homography_input_matches} (ratio={inlier_ratio:.3f})"
        )

        return aligned_return, {
            "aligned": True,
            "alignment_method": "homography",
            "inlier_count": inlier_count,
            "match_strategy": match_strategy,
            "ratio_threshold": ratio_threshold,
            "homography_input_matches": homography_input_matches,
            "inlier_ratio": inlier_ratio,
            "selected_matches": selected_matches,
            "inlier_mask": inlier_mask,
            "tps_control_points": 0,
            "tps_rmse": None,
            "raw_matches": len(match_details["raw_matches"]),
            "good_matches": len(match_details["ratio_matches"]),
            "mean_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[0],
            "median_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[1],
            "aligned_return_keypoints": aligned_return_keypoints,
        }

    def _align_return_image_tps(self, product_image, return_image, product_texture, return_texture):

        product_descriptors = np.asarray(product_texture.get("descriptors", []), dtype=np.float32)
        return_descriptors = np.asarray(return_texture.get("descriptors", []), dtype=np.float32)
        product_keypoints = np.asarray(product_texture.get("keypoints", []), dtype=np.float32)
        return_keypoints = np.asarray(return_texture.get("keypoints", []), dtype=np.float32)

        if (
            product_descriptors.size == 0
            or return_descriptors.size == 0
            or product_keypoints.size == 0
            or return_keypoints.size == 0
            or len(product_descriptors.shape) != 2
            or len(return_descriptors.shape) != 2
        ):
            aligned_return, homography_result = self._align_return_image_homography(
                product_image,
                return_image,
                product_texture,
                return_texture
            )

            homography_result.update({
                "alignment_method": "homography_fallback",
                "tps_control_points": 0,
                "tps_rmse": None,
                "aligned_return_keypoints": [],
            })

            return aligned_return, homography_result

        match_details = self._collect_superpoint_match_details(product_descriptors, return_descriptors)
        selected_matches = match_details["selected_matches"]
        ratio_threshold = match_details["ratio_threshold"]
        match_strategy = match_details["match_strategy"]

        self._log_step(
            f"TPS input: product_keypoints={len(product_keypoints)}, return_keypoints={len(return_keypoints)}, selected_matches={len(selected_matches)}, ratio_threshold={ratio_threshold}, strategy={match_strategy}"
        )

        if len(selected_matches) < 4:
            aligned_return, homography_result = self._align_return_image_homography(
                product_image,
                return_image,
                product_texture,
                return_texture
            )

            homography_result.update({
                "alignment_method": "homography_fallback",
                "tps_control_points": 0,
                "tps_rmse": None,
                "match_strategy": match_strategy,
                "ratio_threshold": ratio_threshold,
                "raw_matches": len(match_details["raw_matches"]),
                "good_matches": len(match_details["ratio_matches"]),
                "mean_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[0],
                "median_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[1],
                "aligned_return_keypoints": [],
            })

            return aligned_return, homography_result

        source_points = np.float32([return_keypoints[m.trainIdx] for m in selected_matches]).reshape(-1, 1, 2)
        target_points = np.float32([product_keypoints[m.queryIdx] for m in selected_matches]).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)

        if homography is None:
            aligned_return, homography_result = self._align_return_image_homography(
                product_image,
                return_image,
                product_texture,
                return_texture
            )

            homography_result.update({
                "alignment_method": "homography_fallback",
                "tps_control_points": 0,
                "tps_rmse": None,
                "match_strategy": match_strategy,
                "ratio_threshold": ratio_threshold,
                "raw_matches": len(match_details["raw_matches"]),
                "good_matches": len(match_details["ratio_matches"]),
                "mean_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[0],
                "median_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[1],
            })

            return aligned_return, homography_result

        inlier_mask = np.asarray(inlier_mask).ravel().astype(bool) if inlier_mask is not None else np.array([], dtype=bool)
        inlier_source_points = source_points.reshape(-1, 2)[inlier_mask]
        inlier_target_points = target_points.reshape(-1, 2)[inlier_mask]

        self._log_step(
            f"TPS RANSAC inliers: {int(inlier_mask.sum())}/{len(selected_matches)}"
        )

        if inlier_source_points.shape[0] < self.TPS_MIN_CONTROL_POINTS:
            aligned_return, homography_result = self._align_return_image_homography(
                product_image,
                return_image,
                product_texture,
                return_texture
            )

            homography_result.update({
                "alignment_method": "homography_fallback",
                "tps_control_points": int(inlier_source_points.shape[0]),
                "tps_rmse": None,
                "match_strategy": match_strategy,
                "ratio_threshold": ratio_threshold,
                "raw_matches": len(match_details["raw_matches"]),
                "good_matches": len(match_details["ratio_matches"]),
                "mean_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[0],
                "median_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[1],
                "aligned_return_keypoints": [],
            })

            return aligned_return, homography_result

        aligned_return, tps_rmse = self._warp_image_with_tps(
            return_image,
            product_image.shape,
            inlier_target_points,
            inlier_source_points
        )

        aligned_return_keypoints = []
        forward_transform = self._fit_tps_inverse_transform(inlier_source_points, inlier_target_points)
        if forward_transform is not None:
            all_return_keypoints = np.asarray(return_texture.get("keypoints", []), dtype=np.float32)
            if all_return_keypoints.size > 0 and len(all_return_keypoints.shape) == 2:
                transformed_keypoints = forward_transform(all_return_keypoints)
                aligned_return_keypoints = transformed_keypoints.tolist()

        if aligned_return is None:
            aligned_return, homography_result = self._align_return_image_homography(
                product_image,
                return_image,
                product_texture,
                return_texture
            )

            homography_result.update({
                "alignment_method": "homography_fallback",
                "tps_control_points": int(inlier_source_points.shape[0]),
                "tps_rmse": None,
                "match_strategy": match_strategy,
                "ratio_threshold": ratio_threshold,
                "raw_matches": len(match_details["raw_matches"]),
                "good_matches": len(match_details["ratio_matches"]),
                "mean_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[0],
                "median_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[1],
                "aligned_return_keypoints": [],
            })

            return aligned_return, homography_result

        selected_count = len(selected_matches)
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / max(1, selected_count)
        mean_match_distance, median_match_distance = self._summarize_match_distances(match_details["ratio_matches"])

        return aligned_return, {
            "aligned": True,
            "alignment_method": "tps",
            "inlier_count": inlier_count,
            "match_strategy": match_strategy,
            "ratio_threshold": ratio_threshold,
            "homography_input_matches": selected_count,
            "inlier_ratio": inlier_ratio,
            "selected_matches": selected_matches,
            "inlier_mask": inlier_mask,
            "tps_control_points": int(inlier_source_points.shape[0]),
            "tps_rmse": tps_rmse,
            "raw_matches": len(match_details["raw_matches"]),
            "good_matches": len(match_details["ratio_matches"]),
            "mean_match_distance": mean_match_distance,
            "median_match_distance": median_match_distance,
            "aligned_return_keypoints": aligned_return_keypoints,
        }

    def _align_return_image(self, product_image, return_image, product_texture, return_texture):

        if self.ALIGNMENT_METHOD.lower() == "homography":
            return self._align_return_image_homography(
                product_image,
                return_image,
                product_texture,
                return_texture
            )

        return self._align_return_image_tps(
            product_image,
            return_image,
            product_texture,
            return_texture
        )

    def _save_superpoint_debug_artifacts(self, product_feature, return_feature, match_details, alignment_result, debug_dir):

        product_image = cv2.imread(product_feature["image_path"])
        return_image = cv2.imread(return_feature["image_path"])

        if product_image is None:
            raise Exception(f"Cannot open product image: {product_feature['image_path']}")

        if return_image is None:
            raise Exception(f"Cannot open return image: {return_feature['image_path']}")

        product_keypoints = self._build_keypoint_objects(product_feature.get("texture_feature", {}).get("keypoints", []))
        return_keypoints = self._build_keypoint_objects(return_feature.get("texture_feature", {}).get("keypoints", []))

        product_keypoints_path = debug_dir / "product_keypoints.png"
        return_keypoints_path = debug_dir / "return_keypoints.png"
        raw_matches_path = debug_dir / "raw_matches.png"
        good_matches_path = debug_dir / "good_matches.png"
        inlier_matches_path = debug_dir / "inlier_matches.png"

        cv2.imwrite(
            str(product_keypoints_path),
            cv2.drawKeypoints(
                product_image,
                product_keypoints,
                None,
                color=(0, 255, 0),
                flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
            )
        )

        cv2.imwrite(
            str(return_keypoints_path),
            cv2.drawKeypoints(
                return_image,
                return_keypoints,
                None,
                color=(0, 255, 0),
                flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
            )
        )

        raw_matches_image = cv2.drawMatches(
            product_image,
            product_keypoints,
            return_image,
            return_keypoints,
            match_details["raw_matches"],
            None,
            matchColor=(255, 0, 0),
            singlePointColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        good_matches_image = cv2.drawMatches(
            product_image,
            product_keypoints,
            return_image,
            return_keypoints,
            match_details["ratio_matches"],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        inlier_matches = []
        inlier_mask = alignment_result.get("inlier_mask")
        selected_matches = alignment_result.get("selected_matches", [])

        if inlier_mask is not None and len(selected_matches) > 0:
            inlier_flags = np.asarray(inlier_mask).ravel().astype(bool)
            inlier_matches = [match for match, is_inlier in zip(selected_matches, inlier_flags) if is_inlier]

        inlier_matches_image = cv2.drawMatches(
            product_image,
            product_keypoints,
            return_image,
            return_keypoints,
            inlier_matches,
            None,
            matchColor=(0, 0, 255),
            singlePointColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imwrite(str(raw_matches_path), raw_matches_image)
        cv2.imwrite(str(good_matches_path), good_matches_image)
        cv2.imwrite(str(inlier_matches_path), inlier_matches_image)

        mean_good_distance, median_good_distance = self._summarize_match_distances(match_details["ratio_matches"])

        return {
            "product_keypoints": len(product_keypoints),
            "return_keypoints": len(return_keypoints),
            "product_descriptors": int(len(product_feature.get("texture_feature", {}).get("descriptors", []))),
            "return_descriptors": int(len(return_feature.get("texture_feature", {}).get("descriptors", []))),
            "raw_matches": len(match_details["raw_matches"]),
            "good_matches": len(match_details["ratio_matches"]),
            "inlier_count": int(len(inlier_matches)),
            "match_ratio": len(match_details["ratio_matches"]) / max(1, min(len(product_keypoints), len(return_keypoints))),
            "inlier_ratio": len(inlier_matches) / max(1, alignment_result.get("homography_input_matches", 0)),
            "mean_good_distance": mean_good_distance,
            "median_good_distance": median_good_distance,
            "files": {
                "product_keypoints": str(product_keypoints_path),
                "return_keypoints": str(return_keypoints_path),
                "raw_matches": str(raw_matches_path),
                "good_matches": str(good_matches_path),
                "inlier_matches": str(inlier_matches_path),
            }
        }

    def _save_alignment_debug_artifacts(self, product_feature, return_feature, aligned_return_image, alignment_result, match_details, debug_dir):

        product_image = cv2.imread(product_feature["image_path"])
        return_image = cv2.imread(return_feature["image_path"])

        if product_image is None:
            raise Exception(f"Cannot open product image: {product_feature['image_path']}")

        if return_image is None:
            raise Exception(f"Cannot open return image: {return_feature['image_path']}")

        product_keypoints = product_feature.get("texture_feature", {}).get("keypoints", [])
        return_keypoints = return_feature.get("texture_feature", {}).get("keypoints", [])

        product_keypoints_cv = self._create_keypoints_from_points(product_keypoints)
        return_keypoints_cv = self._create_keypoints_from_points(return_keypoints)

        inlier_mask = alignment_result.get("inlier_mask")
        selected_matches = alignment_result.get("selected_matches", [])
        inlier_matches = []
        if inlier_mask is not None and len(selected_matches) > 0:
            inlier_flags = np.asarray(inlier_mask).ravel().astype(bool)
            inlier_matches = [match for match, is_inlier in zip(selected_matches, inlier_flags) if is_inlier]

        product_path = debug_dir / "01_product.png"
        return_path = debug_dir / "02_return.png"
        product_keypoints_path = debug_dir / "03_keypoints_product.png"
        return_keypoints_path = debug_dir / "04_keypoints_return.png"
        raw_matches_path = debug_dir / "05_raw_matches.png"
        good_matches_path = debug_dir / "06_good_matches.png"
        inlier_matches_path = debug_dir / "07_inlier_matches.png"
        aligned_return_path = debug_dir / "08_aligned_return.png"
        overlay_alignment_path = debug_dir / "09_overlay_alignment.png"
        diff_before_path = debug_dir / "10_difference_before_alignment.png"
        diff_after_path = debug_dir / "11_difference_after_alignment.png"

        cv2.imwrite(str(product_path), product_image)
        cv2.imwrite(str(return_path), return_image)
        cv2.imwrite(str(product_keypoints_path), self._build_keypoint_visualization(product_image, product_keypoints_cv))
        cv2.imwrite(str(return_keypoints_path), self._build_keypoint_visualization(return_image, return_keypoints_cv))

        raw_matches_image = self._draw_matches(
            product_image,
            product_keypoints,
            return_image,
            return_keypoints,
            match_details["raw_matches"],
            [(255, 0, 0)] * len(match_details["raw_matches"])
        )

        good_match_colors = []
        if len(selected_matches) > 0:
            inlier_flags = np.asarray(inlier_mask).ravel().astype(bool) if inlier_mask is not None else np.zeros(len(selected_matches), dtype=bool)
            good_match_colors = [(0, 255, 0) if flag else (0, 0, 255) for flag in inlier_flags]

        good_matches_image = self._draw_matches(
            product_image,
            product_keypoints,
            return_image,
            return_keypoints,
            selected_matches,
            good_match_colors
        )

        inlier_matches_image = self._draw_matches(
            product_image,
            product_keypoints,
            return_image,
            return_keypoints,
            inlier_matches,
            [(0, 255, 0)] * len(inlier_matches)
        )

        original_return_resized = return_image
        if product_image.shape != return_image.shape:
            original_return_resized = cv2.resize(
                return_image,
                (product_image.shape[1], product_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        difference_before = self._build_difference_visualization(product_image, original_return_resized)
        difference_after = self._build_difference_visualization(product_image, aligned_return_image)
        overlay_alignment = self._build_alignment_overlay(product_image, aligned_return_image)

        cv2.imwrite(str(raw_matches_path), raw_matches_image)
        cv2.imwrite(str(good_matches_path), good_matches_image)
        cv2.imwrite(str(inlier_matches_path), inlier_matches_image)
        cv2.imwrite(str(aligned_return_path), aligned_return_image)
        cv2.imwrite(str(overlay_alignment_path), overlay_alignment)
        cv2.imwrite(str(diff_before_path), difference_before)
        cv2.imwrite(str(diff_after_path), difference_after)

        return {
            "files": {
                "product": str(product_path),
                "return": str(return_path),
                "product_keypoints": str(product_keypoints_path),
                "return_keypoints": str(return_keypoints_path),
                "raw_matches": str(raw_matches_path),
                "good_matches": str(good_matches_path),
                "inlier_matches": str(inlier_matches_path),
                "aligned_return": str(aligned_return_path),
                "overlay_alignment": str(overlay_alignment_path),
                "difference_before_alignment": str(diff_before_path),
                "difference_after_alignment": str(diff_after_path),
            },
            "summary": {
                "product_keypoints": len(product_keypoints),
                "return_keypoints": len(return_keypoints),
                "raw_matches": len(match_details["raw_matches"]),
                "good_matches": len(match_details["ratio_matches"]),
                "inlier_count": len(inlier_matches),
                "inlier_ratio": alignment_result.get("inlier_ratio", 0.0),
                "mean_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[0],
                "median_match_distance": self._summarize_match_distances(match_details["ratio_matches"])[1],
                "tps_control_points": alignment_result.get("tps_control_points", 0),
                "tps_rmse": alignment_result.get("tps_rmse"),
            }
        }

    def _save_local_matching_debug_artifacts(self, local_matching_result, debug_dir, base_name):

        debug_images = local_matching_result["debug_images"]

        patch_product_path = debug_dir / "01_patch_product.png"
        search_window_path = debug_dir / "02_search_window.png"
        best_patch_path = debug_dir / "03_best_patch.png"
        similarity_heatmap_path = debug_dir / "04_similarity_heatmap.png"
        patch_difference_path = debug_dir / "05_patch_difference.png"
        damage_map_path = debug_dir / "06_damage_map.png"
        overlay_path = debug_dir / "07_local_matching_overlay.png"
        ssim_histogram_path = debug_dir / "08_ssim_histogram.png"

        cv2.imwrite(str(patch_product_path), debug_images["patch_product"])
        cv2.imwrite(str(search_window_path), debug_images["search_window"])
        cv2.imwrite(str(best_patch_path), debug_images["best_patch"])
        cv2.imwrite(str(similarity_heatmap_path), debug_images["similarity_heatmap"])
        cv2.imwrite(str(patch_difference_path), debug_images["patch_difference"])
        cv2.imwrite(str(damage_map_path), debug_images["damage_map"])
        cv2.imwrite(str(overlay_path), debug_images["local_matching_overlay"])
        cv2.imwrite(str(ssim_histogram_path), debug_images["ssim_histogram"])

        return {
            "files": {
                "patch_product": str(patch_product_path),
                "search_window": str(search_window_path),
                "best_patch": str(best_patch_path),
                "similarity_heatmap": str(similarity_heatmap_path),
                "patch_difference": str(patch_difference_path),
                "damage_map": str(damage_map_path),
                "local_matching_overlay": str(overlay_path),
                "ssim_histogram": str(ssim_histogram_path),
            },
            "summary": local_matching_result["summary"],
            "representative_match": local_matching_result.get("representative_match"),
            "base_name": base_name,
        }

    def _build_damage_mask(self, product_feature, return_feature):

        product_image = cv2.imread(product_feature["image_path"])
        return_image = cv2.imread(return_feature["image_path"])

        if product_image is None:
            raise Exception(f"Cannot open product image: {product_feature['image_path']}")

        if return_image is None:
            raise Exception(f"Cannot open return image: {return_feature['image_path']}")

        product_texture = product_feature.get("texture_feature", {})
        return_texture = return_feature.get("texture_feature", {})

        self._log_step(
            f"Texture input: product_keypoints={len(product_texture.get('keypoints', []))}, return_keypoints={len(return_texture.get('keypoints', []))}"
        )

        return_image, alignment_result = self._align_return_image(
            product_image,
            return_image,
            product_texture,
            return_texture
        )

        if product_image.shape != return_image.shape:
            return_image = cv2.resize(
                return_image,
                (product_image.shape[1], product_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        product_mask = self.geometry.build_mask(product_image)
        return_mask = self.geometry.build_mask(return_image)

        inlier_matches = self._extract_inlier_matches(product_feature, return_feature, alignment_result)
        aligned_return_keypoints = alignment_result.get("aligned_return_keypoints", [])
        if not aligned_return_keypoints:
            aligned_return_keypoints = return_feature.get("texture_feature", {}).get("keypoints", [])

        local_matching_result = self.mesh_damage_detector.run(
            product_image,
            return_image,
            product_texture.get("keypoints", []),
            aligned_return_keypoints,
            inlier_matches,
            product_descriptors=product_texture.get("descriptors", []),
            return_descriptors=return_texture.get("descriptors", []),
            debug_mesh_dir=Path("debug_mesh") / f"{Path(product_feature['image_path']).stem}_vs_{Path(return_feature['image_path']).stem}",
        )

        self._log_step(
            f"Mesh matching result: received_inliers={len(inlier_matches)}, total_triangles={local_matching_result['summary'].get('total_matches', 0)}"
        )

        damage_mask = local_matching_result["damage_map"]

        return product_image, return_image, product_mask, return_mask, damage_mask, alignment_result, local_matching_result

    def _build_debug_preview(self, product_image, return_image, product_mask, return_mask, damage_mask):

        product_highlight = product_image.copy()
        product_highlight[product_mask > 0] = (0, 255, 0)
        product_overlay = cv2.addWeighted(product_image, 0.75, product_highlight, 0.25, 0)

        return_highlight = return_image.copy()
        return_highlight[return_mask > 0] = (0, 255, 0)
        return_overlay = cv2.addWeighted(return_image, 0.75, return_highlight, 0.25, 0)

        damage_overlay = return_image.copy()
        damage_overlay[damage_mask > 0] = (0, 0, 255)
        damage_overlay = cv2.addWeighted(return_image, 0.75, damage_overlay, 0.25, 0)

        top = np.hstack([product_overlay, return_overlay])
        bottom = np.hstack([cv2.cvtColor(damage_mask, cv2.COLOR_GRAY2BGR), damage_overlay])

        return np.vstack([top, bottom])

    def _calculate_damage_score(self, product_area, difference_area, largest_damage_area, confidence_score=1.0):
        """Compute the final damage score as geometry_score × confidence_score.

        Geometry score is derived purely from difference_area and largest_damage_area.
        Confidence score (average_weight from local matching) is applied as a multiplier
        and must NOT influence damage_map, difference_area, or largest_damage_area.
        """
        if product_area <= 0 or difference_area <= 0:
            return 0.0

        spread = (difference_area - largest_damage_area) / difference_area
        spread = max(0.0, min(1.0, spread))

        difference_ratio = difference_area / product_area
        largest_ratio = largest_damage_area / product_area

        geometry_score = (
            spread * difference_ratio
            + (1.0 - spread) * largest_ratio
        )

        confidence_score = max(0.0, min(1.0, float(confidence_score)))

        return geometry_score * confidence_score * 100.0

    def compare(self, product_feature, return_feature):

        product_image, return_image, product_mask, return_mask, damage_mask, alignment_result, local_matching_result = self._build_damage_mask(
            product_feature,
            return_feature
        )

        original_return_image = cv2.imread(return_feature["image_path"])
        if original_return_image is None:
            raise Exception(f"Cannot open return image: {return_feature['image_path']}")

        product_descriptors = np.asarray(product_feature.get("texture_feature", {}).get("descriptors", []), dtype=np.float32)
        return_descriptors = np.asarray(return_feature.get("texture_feature", {}).get("descriptors", []), dtype=np.float32)

        if (
            product_descriptors.size > 0
            and return_descriptors.size > 0
            and len(product_descriptors.shape) == 2
            and len(return_descriptors.shape) == 2
        ):
            match_details = self._collect_superpoint_match_details(product_descriptors, return_descriptors)
        else:
            match_details = {
                "raw_matches": [],
                "ratio_matches": [],
                "selected_matches": [],
                "ratio_threshold": None,
                "match_strategy": "insufficient",
            }

        difference_area = int(np.count_nonzero(damage_mask))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(damage_mask)

        largest_damage_area = 0
        if num_labels > 1:
            largest_damage_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))

        # Confidence score: average_weight from local matching summary.
        # This is the ONLY place weight influences the final score.
        # damage_map, difference_area, and largest_damage_area are geometry-only
        # and must NOT change when weight changes.
        local_summary = local_matching_result["summary"]
        confidence_score = float(local_summary.get("average_weight", 1.0))

        self._log_step(
            f"Damage metrics: difference_area={difference_area}, "
            f"largest_damage_area={largest_damage_area}, "
            f"confidence_score(avg_weight)={confidence_score:.4f}"
        )

        damage_score = self._calculate_damage_score(
            float(product_feature["object_area"]),
            float(difference_area),
            float(largest_damage_area),
            confidence_score
        )

        product_name = Path(product_feature["image_path"]).stem
        return_name = Path(return_feature["image_path"]).stem
        base_name = f"{product_name}_vs_{return_name}"

        superpoint_result = self._sift_similarity(
            product_feature.get("texture_feature", {}),
            return_feature.get("texture_feature", {})
        )

        feature_result = self._feature_representation(
            product_feature,
            return_feature,
            alignment_result
        )

        debug_dir = Path("debug_outputs")
        debug_dir.mkdir(exist_ok=True)

        superpoint_debug_dir = debug_dir / "superpoint" / base_name
        superpoint_debug_dir.mkdir(parents=True, exist_ok=True)

        alignment_debug_dir = debug_dir / "alignment" / base_name
        alignment_debug_dir.mkdir(parents=True, exist_ok=True)

        local_matching_debug_dir = debug_dir / "local_matching" / base_name
        local_matching_debug_dir.mkdir(parents=True, exist_ok=True)

        damage_mask_path = debug_dir / f"{base_name}_damage_mask.png"
        preview_path = debug_dir / f"{base_name}_preview.png"

        cv2.imwrite(str(damage_mask_path), damage_mask)
        cv2.imwrite(
            str(preview_path),
            self._build_debug_preview(product_image, return_image, product_mask, return_mask, damage_mask)
        )

        # Save diagnostic masks and overlays
        diag_dir = debug_dir / "diagnostics" / base_name
        diag_dir.mkdir(parents=True, exist_ok=True)
        
        debug_info = local_matching_result.get("debug_info", {})
        masks = debug_info.get("masks", {})
        
        # Save intermediate masks
        for mask_name, mask_val in masks.items():
            if mask_val is not None:
                cv2.imwrite(str(diag_dir / f"{mask_name}.png"), mask_val)
                
        # Helper to create red overlays
        def create_red_overlay(image, mask):
            overlay = image.copy()
            if mask is not None and mask.ndim == 2:
                overlay[mask > 0] = (0, 0, 255) # Red highlight
            return cv2.addWeighted(image, 0.75, overlay, 0.25, 0)
            
        overlay_intensity = create_red_overlay(return_image, masks.get("raw_intensity_mask"))
        overlay_gradient = create_red_overlay(return_image, masks.get("raw_gradient_mask"))
        overlay_feature_reject = create_red_overlay(return_image, masks.get("feature_reject_mask"))
        overlay_ssim_reject = create_red_overlay(return_image, masks.get("ssim_reject_mask"))
        overlay_final_damage = create_red_overlay(return_image, masks.get("final_damage_mask"))

        cv2.imwrite(str(diag_dir / "overlay_intensity.png"), overlay_intensity)
        cv2.imwrite(str(diag_dir / "overlay_gradient.png"), overlay_gradient)
        cv2.imwrite(str(diag_dir / "overlay_feature_reject.png"), overlay_feature_reject)
        cv2.imwrite(str(diag_dir / "overlay_ssim_reject.png"), overlay_ssim_reject)
        cv2.imwrite(str(diag_dir / "overlay_final_damage.png"), overlay_final_damage)

        # Generate and save Semantic Component Validation overlays
        validator = ComponentValidator(
            min_area=self.mesh_damage_detector.min_area_threshold,
            feature_threshold=self.mesh_damage_detector.feature_threshold,
            ssim_threshold=self.mesh_damage_detector.ssim_threshold,
            gradient_threshold=self.mesh_damage_detector.diff_threshold,
            compactness_threshold=0.05,
            min_patch_support=1
        )
        accepted_comps = debug_info.get("accepted_components", [])
        rejected_comps = debug_info.get("rejected_components", [])
        all_comps = debug_info.get("all_components", [])
        
        comp_overlays = validator.export_debug(return_image, accepted_comps, rejected_comps, all_comps)
        cv2.imwrite(str(diag_dir / "overlay_accepted_components.png"), comp_overlays["overlay_accepted"])
        cv2.imwrite(str(diag_dir / "overlay_rejected_components.png"), comp_overlays["overlay_rejected"])
        cv2.imwrite(str(diag_dir / "overlay_component_labels.png"), comp_overlays["overlay_labels"])
        cv2.imwrite(str(diag_dir / "overlay_component_heatmap.png"), comp_overlays["overlay_heatmap"])

        superpoint_debug = self._save_superpoint_debug_artifacts(
            product_feature,
            return_feature,
            match_details,
            alignment_result,
            superpoint_debug_dir
        )

        alignment_debug = self._save_alignment_debug_artifacts(
            product_feature,
            return_feature,
            return_image,
            alignment_result,
            match_details,
            alignment_debug_dir
        )

        local_matching_debug = self._save_local_matching_debug_artifacts(
            local_matching_result,
            local_matching_debug_dir,
            base_name
        )

        superpoint_result.update({
            "raw_matches": superpoint_debug["raw_matches"],
            "good_matches": superpoint_debug["good_matches"],
            "inlier_count": superpoint_debug["inlier_count"],
            "inlier_ratio": superpoint_debug["inlier_ratio"],
            "mean_good_distance": superpoint_debug["mean_good_distance"],
            "median_good_distance": superpoint_debug["median_good_distance"],
        })

        alignment_result.update(alignment_debug["summary"])

        return {
            "difference_area": difference_area,
            "largest_damage_area": largest_damage_area,
            "damage_score": damage_score,
            "feature_result": feature_result,
            "sift_result": superpoint_result,
            "superpoint_result": superpoint_result,
            "alignment_result": alignment_result,
            "alignment_debug": alignment_debug,
            "local_matching_result": local_matching_result["summary"],
            "local_matching_debug": local_matching_debug,
            "sift_debug": superpoint_debug,
            "superpoint_debug": superpoint_debug,
            "debug_images": {
                "damage_mask": str(damage_mask_path),
                "preview": str(preview_path)
            },
            "debug_info": debug_info
        }