from feature_extractor import FeatureExtractor
from comparator import DamageComparator
from database import Database


def analyze_return_product():

    db = Database()

    print("\n=== Analyze Return Product ===")

    products = db.get_products()

    if len(products) == 0:
        print("No product found.")
        return

    print("\nAvailable Products")

    for product in products:
        print(f"{product['id']}. {product['image_path']}")

    product_id_text = input("\nSelect Product ID: ").strip()

    if not product_id_text.isdigit():
        print("Invalid Product ID. Please enter a numeric ID.")
        return

    product_id = int(product_id_text)

    product_feature = db.get_product_feature(product_id)

    if product_feature is None:
        print("Product not found.")
        return

    if not product_feature.get("texture_feature", {}).get("keypoints"):
        product_extractor = FeatureExtractor()
        product_feature = product_extractor.extract(product_feature["image_path"])

    image_path = input("Return image path: ").strip()

    extractor = FeatureExtractor()

    return_feature = extractor.extract(image_path)

    return_feature["product_id"] = product_id

    comparator = DamageComparator()

    result = comparator.compare(
        product_feature,
        return_feature
    )

    db.save_return(
        return_feature,
        result
    )

    print("\nAnalysis Completed")
    print(f"Difference Area     : {result['difference_area']}")
    print(f"Largest Damage Area : {result['largest_damage_area']}")
    print(f"Damage Score (%)    : {result['damage_score']:.2f}")

    if "debug_info" in result:
        db_info = result["debug_info"]
        print("\n=== Validation Statistics ===")
        print(f"Components Before Validation               : {db_info.get('components_before_validation', 0)}")
        print(f"Components After Validation                : {db_info.get('components_after_validation', 0)}")
        print(f"Pixels Removed By Validation               : {db_info.get('pixels_removed_by_validation', 0)}")
        print(f"Pixels Remaining                           : {db_info.get('pixels_remaining', 0)}")
        print(f"Difference Area Before Validation          : {db_info.get('diff_area_before_validation', 0)}")
        print(f"Difference Area After Validation           : {db_info.get('diff_area_after_validation', 0)}")

        print("\n=== Pipeline Statistics & Diagnostic Summary ===")
        print(f"Rejected by Feature (Patches)              : {db_info.get('rejected_by_feature_patches', 0)}")
        print(f"Rejected by SSIM (Patches)                 : {db_info.get('rejected_by_ssim_patches', 0)}")
        print(f"Accepted Patches                           : {db_info.get('accepted_patches', 0)}")
        print(f"Rejected Patches                           : {db_info.get('rejected_patches', 0)}")
        print(f"Difference Area Before Feature Filter      : {db_info.get('diff_area_before_feature', 0)}")
        print(f"Difference Area After Feature Filter       : {db_info.get('diff_area_after_feature', 0)}")
        print(f"Difference Area After SSIM Filter          : {db_info.get('diff_area_after_ssim', 0)}")
        print(f"Difference Area After Morphology           : {db_info.get('diff_area_after_morphology', 0)}")
        print(f"Final Difference Area                      : {db_info.get('final_difference_area', 0)}")

    feature_result = result.get("feature_result")
    if feature_result is not None:
        print(f"Feature Matches     : {feature_result['matches']}")
        print(f"Mean Similarity     : {feature_result['mean_similarity']:.3f}")
        print(f"Median Similarity   : {feature_result['median_similarity']:.3f}")
        print(f"Min Similarity      : {feature_result['min_similarity']:.3f}")
        print(f"Max Similarity      : {feature_result['max_similarity']:.3f}")

    superpoint_result = result.get("superpoint_result") or result.get("sift_result")
    if superpoint_result is not None:
        print(f"SuperPoint Keypoints : {superpoint_result['product_keypoints']} vs {superpoint_result['return_keypoints']}")
        print(f"SuperPoint Descriptors : {superpoint_result['product_descriptors']} vs {superpoint_result['return_descriptors']}")
        print(f"Raw Matches          : {superpoint_result['raw_matches']}")
        print(f"Good Matches         : {superpoint_result['good_matches']}")
        print(f"Inlier Count         : {superpoint_result['inlier_count']}")
        print(f"Match Ratio          : {superpoint_result['match_ratio']:.2f}")
        print(f"Inlier Ratio         : {superpoint_result['inlier_ratio']:.2f}")
        print(f"Mean Good Distance   : {superpoint_result['mean_good_distance']:.2f}")
        print(f"Median Good Distance : {superpoint_result['median_good_distance']:.2f}")
        print(f"Similarity Score     : {superpoint_result['similarity_score']:.2f}")

    if "alignment_result" in result:
        alignment_result = result["alignment_result"]
        print("\nAlignment Statistics")
        print(f"Alignment Method    : {alignment_result.get('alignment_method')}")
        print(f"Total Keypoints Product : {alignment_result.get('product_keypoints', 0)}")
        print(f"Total Keypoints Return  : {alignment_result.get('return_keypoints', 0)}")
        print(f"Raw Matches         : {alignment_result.get('raw_matches', 0)}")
        print(f"Good Matches        : {alignment_result.get('good_matches', 0)}")
        print(f"Inliers             : {alignment_result.get('inlier_count', 0)}")
        print(f"Inlier Ratio        : {alignment_result.get('inlier_ratio', 0.0):.2f}")
        print(f"Mean Match Distance  : {alignment_result.get('mean_match_distance', 0.0):.2f}")
        print(f"Median Match Distance : {alignment_result.get('median_match_distance', 0.0):.2f}")
        print(f"TPS Control Points   : {alignment_result.get('tps_control_points', 0)}")
        tps_rmse = alignment_result.get('tps_rmse')
        if tps_rmse is None:
            print("TPS RMSE            : N/A")
        else:
            print(f"TPS RMSE            : {tps_rmse:.2f}")

    superpoint_debug = result.get("superpoint_debug") or result.get("sift_debug")
    if superpoint_debug is not None:
        print("\nSuperPoint Debug Images")
        print(f"Product Keypoints  : {superpoint_debug['files']['product_keypoints']}")
        print(f"Return Keypoints    : {superpoint_debug['files']['return_keypoints']}")
        print(f"Raw Matches         : {superpoint_debug['files']['raw_matches']}")
        print(f"Good Matches        : {superpoint_debug['files']['good_matches']}")
        print(f"Inlier Matches      : {superpoint_debug['files']['inlier_matches']}")

    if "alignment_debug" in result:
        alignment_debug = result["alignment_debug"]
        print("\nAlignment Debug Images")
        print(f"Product            : {alignment_debug['files']['product']}")
        print(f"Return             : {alignment_debug['files']['return']}")
        print(f"Product Keypoints   : {alignment_debug['files']['product_keypoints']}")
        print(f"Return Keypoints    : {alignment_debug['files']['return_keypoints']}")
        print(f"Raw Matches         : {alignment_debug['files']['raw_matches']}")
        print(f"Good Matches        : {alignment_debug['files']['good_matches']}")
        print(f"Inlier Matches      : {alignment_debug['files']['inlier_matches']}")
        print(f"Aligned Return      : {alignment_debug['files']['aligned_return']}")
        print(f"Overlay Alignment   : {alignment_debug['files']['overlay_alignment']}")
        print(f"Difference Before   : {alignment_debug['files']['difference_before_alignment']}")
        print(f"Difference After    : {alignment_debug['files']['difference_after_alignment']}")

    if "local_matching_result" in result:
        local_matching_result = result["local_matching_result"]
        print("\nLocal Matching Statistics")
        print(f"Total Matches       : {local_matching_result.get('total_matches', 0)}")
        print(f"Average SSIM        : {local_matching_result.get('average_ssim', 0.0):.2f}")
        print(f"Median SSIM         : {local_matching_result.get('median_ssim', 0.0):.2f}")
        print(f"Lowest SSIM         : {local_matching_result.get('lowest_ssim', 0.0):.2f}")
        print(f"Highest SSIM        : {local_matching_result.get('highest_ssim', 0.0):.2f}")
        print(f"Average Offset      : {local_matching_result.get('average_offset', 0.0):.2f}")
        print(f"Maximum Offset      : {local_matching_result.get('maximum_offset', 0.0):.2f}")
        print(f"Patch Size          : {local_matching_result.get('patch_size', 0)}")
        print(f"Search Window Size  : {local_matching_result.get('search_window_size', 0)}")

    if "local_matching_debug" in result:
        local_matching_debug = result["local_matching_debug"]
        print("\nLocal Matching Debug Images")
        print(f"Patch Product       : {local_matching_debug['files']['patch_product']}")
        print(f"Search Window       : {local_matching_debug['files']['search_window']}")
        print(f"Best Patch          : {local_matching_debug['files']['best_patch']}")
        print(f"Similarity Heatmap  : {local_matching_debug['files']['similarity_heatmap']}")
        print(f"Patch Difference    : {local_matching_debug['files']['patch_difference']}")
        print(f"Damage Map          : {local_matching_debug['files']['damage_map']}")
        print(f"Local Overlay       : {local_matching_debug['files']['local_matching_overlay']}")

    if "debug_images" in result:
        print("\nDebug Images")
        print(f"Damage Mask : {result['debug_images']['damage_mask']}")
        print(f"Preview     : {result['debug_images']['preview']}")