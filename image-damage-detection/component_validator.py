import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class ComponentValidator:
    """Validator class for connected component filtering.
    Designed to be extensible, making it easy to swap with ML models.
    """
    def __init__(
        self,
        min_area: int = 15,
        feature_threshold: float = 0.95,
        ssim_threshold: float = 0.90,
        gradient_threshold: float = 25.0,
        compactness_threshold: float = 0.05,
        min_patch_support: int = 3,
        confidence_threshold: float = 0.15,
        w1: float = 0.3,
        w2: float = 0.3,
        w3: float = 0.2,
        w4: float = 0.2
    ):
        self.min_area = min_area
        self.feature_threshold = feature_threshold
        self.ssim_threshold = ssim_threshold
        self.gradient_threshold = gradient_threshold
        self.compactness_threshold = compactness_threshold
        self.min_patch_support = min_patch_support
        self.confidence_threshold = confidence_threshold
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def compute_statistics(
        self,
        damage_mask: np.ndarray,
        intensity_diff: np.ndarray,
        gradient_diff: np.ndarray,
        local_matches: List[dict]
    ) -> List[dict]:
        """Compute the structural/spectral features for all connected components in damage_mask."""
        height, width = damage_mask.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(damage_mask)
        
        components = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]
            
            # Aspect ratio
            aspect_ratio = float(w) / float(h) if h > 0 else 0.0
            
            # Binary mask for this component
            comp_mask = (labels == i).astype(np.uint8) * 255
            
            # Perimeter and Convex Hull via contours
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = 0.0
            convex_hull_area = 0.0
            compactness = 0.0
            solidity = 0.0
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                perimeter = float(cv2.arcLength(contour, True))
                hull = cv2.convexHull(contour)
                convex_hull_area = float(cv2.contourArea(hull))
                
                # Solidity
                if convex_hull_area > 0:
                    solidity = area / convex_hull_area
                    
                # Compactness: 4*pi*area / (perimeter^2)
                if perimeter > 0:
                    compactness = (4.0 * np.pi * area) / (perimeter ** 2)

            # Mean Intensity and Gradient Difference inside the component
            mean_intensity = float(cv2.mean(intensity_diff, mask=comp_mask)[0])
            mean_gradient = float(cv2.mean(gradient_diff, mask=comp_mask)[0])
            
            # Mean Feature Similarity, Mean SSIM, and Patch Coverage
            intersecting_sims = []
            intersecting_ssims = []
            coverage = 0
            
            for match in local_matches:
                x0, y0 = match["best_top_left"]
                h_p, w_p = match["product_patch"].shape[:2]
                x1, y1 = x0 + w_p, y0 + h_p
                
                # Check intersection of patch region with component mask
                roi = comp_mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
                if np.any(roi > 0):
                    intersecting_sims.append(match.get("feature_similarity", 0.0))
                    intersecting_ssims.append(match.get("ssim", 0.0))
                    coverage += 1
                    
            mean_feat_sim = float(np.mean(intersecting_sims)) if intersecting_sims else 1.0
            mean_ssim = float(np.mean(intersecting_ssims)) if intersecting_ssims else 1.0
            
            components.append({
                "id": i,
                "area": area,
                "bbox": (x, y, w, h),
                "aspect_ratio": aspect_ratio,
                "compactness": compactness,
                "solidity": solidity,
                "mean_intensity": mean_intensity,
                "mean_gradient": mean_gradient,
                "mean_feature_similarity": mean_feat_sim,
                "mean_ssim": mean_ssim,
                "coverage": coverage,
                "centroid": (cx, cy),
                "mask": comp_mask,
                "intersecting_sims": intersecting_sims,
                "intersecting_ssims": intersecting_ssims
            })
            
        return components

    def validate(self, components: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Validate each component and print a detailed decision table.
        Returns: (accepted_components, rejected_components)
        """
        accepted = []
        rejected = []
        
        # Terminal table header
        print("\n=== Semantic Component Validation Debug ===")
        print(f"{'Comp ID':7s} | {'Area':5s} | {'Feature':7s} | {'SSIM':5s} | {'Gradient':8s} | {'Support':7s} | {'Conf':5s} | {'Decision':8s} | {'Reject Reason'}")
        print("-" * 125)
        
        for comp in components:
            reasons = []
            
            # 1. Standard thresholds
            if comp["area"] < self.min_area:
                reasons.append(f"Area < {self.min_area}")
            if comp["mean_feature_similarity"] >= self.feature_threshold:
                reasons.append(f"High Feature Similarity ({comp['mean_feature_similarity']:.3f} >= {self.feature_threshold})")
            if comp["mean_ssim"] >= self.ssim_threshold:
                reasons.append(f"High SSIM ({comp['mean_ssim']:.3f} >= {self.ssim_threshold})")
            if comp["mean_gradient"] <= self.gradient_threshold:
                reasons.append(f"Low Gradient ({comp['mean_gradient']:.1f} <= {self.gradient_threshold})")
            if comp["coverage"] < self.min_patch_support:
                reasons.append(f"Low Support ({comp['coverage']} < {self.min_patch_support})")
                
            # 2. Contextual Rejection: lies entirely within high similarity region (Feat > 0.92 and SSIM > 0.90)
            sims = comp["intersecting_sims"]
            ssims = comp["intersecting_ssims"]
            if sims and ssims and all(s > 0.92 and ss > 0.90 for s, ss in zip(sims, ssims)):
                reasons.append("High Contextual Similarity (Feat > 0.92 and SSIM > 0.90)")
                
            # 3. Calculate Confidence Score
            # confidence = w1*(1-feature_similarity) + w2*(1-ssim) + w3*normalized_gradient + w4*support_ratio
            norm_grad = comp["mean_gradient"] / 255.0
            support_ratio = min(1.0, comp["coverage"] / 5.0)
            confidence = (
                self.w1 * (1.0 - comp["mean_feature_similarity"]) +
                self.w2 * (1.0 - comp["mean_ssim"]) +
                self.w3 * norm_grad +
                self.w4 * support_ratio
            )
            comp["confidence"] = confidence
            
            if confidence <= self.confidence_threshold:
                reasons.append(f"Low Confidence ({confidence:.3f} <= {self.confidence_threshold})")
                
            if reasons:
                comp["decision"] = "Reject"
                comp["reason"] = ", ".join(reasons)
                rejected.append(comp)
            else:
                comp["decision"] = "Accept"
                comp["reason"] = "All checks passed"
                accepted.append(comp)
                
            print(f"Comp {comp['id']:3d}    | {comp['area']:5d} | {comp['mean_feature_similarity']:7.3f} | {comp['mean_ssim']:5.3f} | {comp['mean_gradient']:8.1f} | {comp['coverage']:7d} | {comp['confidence']:5.3f} | {comp['decision']:8s} | {comp['reason']}")
            
        print("-" * 125)
        return accepted, rejected

    def export_debug(
        self,
        image: np.ndarray,
        accepted: List[dict],
        rejected: List[dict],
        components: List[dict]
    ) -> Dict[str, np.ndarray]:
        """Generate overlay visualizations for accepted components, rejected components, labels, and heatmaps."""
        height, width = image.shape[:2]
        
        # 1. Overlay Accepted Components (Green highlight)
        accepted_overlay = image.copy()
        for comp in accepted:
            accepted_overlay[comp["mask"] > 0] = (0, 255, 0)
        overlay_accepted = cv2.addWeighted(image, 0.75, accepted_overlay, 0.25, 0)
        
        # 2. Overlay Rejected Components (Red highlight)
        rejected_overlay = image.copy()
        for comp in rejected:
            rejected_overlay[comp["mask"] > 0] = (0, 0, 255)
        overlay_rejected = cv2.addWeighted(image, 0.75, rejected_overlay, 0.25, 0)
        
        # 3. Component Labels (Drawn IDs at centroids)
        overlay_labels = image.copy()
        for comp in components:
            cx, cy = comp["centroid"]
            text = f"ID: {comp['id']}"
            color = (0, 255, 0) if comp["decision"] == "Accept" else (0, 0, 255)
            if np.isnan(cx) or np.isnan(cy):
                continue
            pos = (int(cx) - 20, int(cy) + 5)
            cv2.putText(overlay_labels, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            cv2.circle(overlay_labels, (int(cx), int(cy)), 3, color, -1, cv2.LINE_AA)
            
        # 4. Component Heatmap (Colored by patch coverage support)
        heatmap_gray = np.zeros((height, width), dtype=np.uint8)
        max_coverage = max(1, max((comp["coverage"] for comp in components), default=1))
        
        for comp in components:
            val = int((comp["coverage"] / max_coverage) * 255)
            heatmap_gray[comp["mask"] > 0] = val
            
        overlay_heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        # Keep non-component areas unchanged (blend with raw image where heatmap_gray == 0)
        non_comp_mask = (heatmap_gray == 0)
        overlay_heatmap[non_comp_mask] = image[non_comp_mask]
        
        return {
            "overlay_accepted": overlay_accepted,
            "overlay_rejected": overlay_rejected,
            "overlay_labels": overlay_labels,
            "overlay_heatmap": overlay_heatmap
        }
