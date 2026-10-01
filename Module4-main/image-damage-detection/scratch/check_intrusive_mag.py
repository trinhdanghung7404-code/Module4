import sys, os, cv2, numpy as np
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection')
sys.path.insert(0, r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2')
from preprocessing.segmentation import ObjectSegmenter
from preprocessing.normalization import ImageNormalizer
from preprocessing.registration import ImageRegistration
from mesh.mesh_builder import MeshBuilder
from step3_pipeline import canonical_triangle_patch
from detailed_debug import micro_align_patches

def check_edge_mag(p_path, r_path, tids, label):
    p_img = cv2.imread(p_path); r_img = cv2.imread(r_path)
    seg = ObjectSegmenter()
    p_seg = seg.segment(p_img); r_seg = seg.segment(r_img)
    norm = ImageNormalizer()
    r_norm = norm.normalize(r_img, p_img, r_seg['mask'], p_seg['mask'])
    reg = ImageRegistration(max_keypoints=None)
    reg_res = reg.register(p_img, r_norm, p_seg['mask'], r_seg['mask'])
    mb = MeshBuilder()
    vertices, triangles = mb.build(reg_res['product_points'], reg_res['return_points'], np.arange(len(reg_res['product_points'])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    print(f"\n--- {label} ---")
    for tid in tids:
        tri = triangles[tid]
        pts_p = np.array([vertices[idx].product_xy for idx in tri.vertex_indices], dtype=np.float32)
        pts_r = np.array([vertices[idx].return_xy for idx in tri.vertex_indices], dtype=np.float32)
        patch_p, mask_p = canonical_triangle_patch(p_img, pts_p, target_size=72)
        patch_r, mask_r = canonical_triangle_patch(r_norm, pts_r, target_size=72)
        patch_r_aligned = micro_align_patches(patch_p, patch_r, mask_p, max_shift=3)

        p_raw = cv2.cvtColor(patch_p, cv2.COLOR_BGR2GRAY)
        r_raw = cv2.cvtColor(patch_r_aligned, cv2.COLOR_BGR2GRAY)
        p_c = clahe.apply(p_raw)
        r_c = clahe.apply(r_raw)
        valid = mask_p > 0
        glare_px = (r_c > 240) & (p_c > 240)

        p_bhat = cv2.morphologyEx(p_c, cv2.MORPH_BLACKHAT, k_morph)
        p_what = cv2.morphologyEx(p_c, cv2.MORPH_TOPHAT, k_morph)
        p_c_edge = cv2.Canny(p_c, 40, 120); p_c_edge[~valid] = 0
        p_all_edges = ((p_bhat > 15) | (p_what > 15) | (p_c_edge > 0)).astype(np.uint8) * 255
        dist_to_p_all = cv2.distanceTransform(cv2.bitwise_not(p_all_edges), cv2.DIST_L2, 3)

        grad_p_raw = np.sqrt(cv2.Sobel(p_raw, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(p_raw, cv2.CV_32F, 0, 1)**2)
        r_raw_edge = cv2.Canny(r_raw, 40, 120); r_raw_edge[~valid] = 0

        # Gradient magnitude of r_raw
        gx_r_raw = cv2.Sobel(r_raw, cv2.CV_32F, 1, 0)
        gy_r_raw = cv2.Sobel(r_raw, cv2.CV_32F, 0, 1)
        mag_r_raw = np.sqrt(gx_r_raw**2 + gy_r_raw**2)

        intrusive = (r_raw_edge > 0) & (dist_to_p_all > 5.5) & (grad_p_raw < 25.0) & (~glare_px) & valid
        if np.any(intrusive):
            mags = mag_r_raw[intrusive]
            print(f"Tri #{tid:3d}: intrusive pixels={np.sum(intrusive)}, mean_mag={np.mean(mags):.1f}, max_mag={np.max(mags):.1f}, min_mag={np.min(mags):.1f}")
        else:
            print(f"Tri #{tid:3d}: NO intrusive pixels")

if __name__ == '__main__':
    check_edge_mag(r'images\1.jpg', r'images\2.jpg', [11, 12, 13, 83], '1.jpg vs 2.jpg')
    check_edge_mag(r'images\test_nobg.png', r'images\test_nobg - scar.png', [98, 121, 123, 124, 341, 432, 433, 454], 'scar')
