import os, shutil, cv2

art_dir = r'C:\Users\Admin\.gemini\antigravity\brain\d4e14bb1-53b4-4214-92b9-70b8003c37fd'
base_debug = r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2\debug\01_undamaged_vase'

# 1. Copy tri_0605.jpg card
src_card = os.path.join(base_debug, '03_layer1_structure', '02_khac_nhau', 'tri_0605.jpg')
dst_card = os.path.join(art_dir, 'tri_605_card.jpg')
if os.path.exists(src_card):
    shutil.copyfile(src_card, dst_card)
    print(f"Copied {src_card} to {dst_card}")

# 2. Copy fusion defect crop
src_crop = os.path.join(base_debug, '06_defect_crops', 'defect_tri_605_Nut_GayNet_L1+TrocMen_MatMau_L2.jpg')
dst_crop = os.path.join(art_dir, 'tri_605_crop.jpg')
if os.path.exists(src_crop):
    shutil.copyfile(src_crop, dst_crop)
    print(f"Copied {src_crop} to {dst_crop}")

# 3. Crop cheek area around Tri #605 on 06_structure_damage_overlay.jpg
overlay_path = os.path.join(base_debug, '03_layer1_structure', '06_structure_damage_overlay.jpg')
if os.path.exists(overlay_path):
    img = cv2.imread(overlay_path)
    # Tri #605 is around x in [1000, 1150], y in [1020, 1180]
    # Let's crop x in [950, 1200], y in [980, 1220]
    h, w = img.shape[:2]
    x0, y0 = max(0, 950), max(0, 980)
    x1, y1 = min(w, 1200), min(h, 1220)
    crop_cheek = img[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(art_dir, 'rooster_cheek_caught.jpg'), crop_cheek)
    print(f"Saved rooster_cheek_caught.jpg (size {crop_cheek.shape})")

# 4. Copy scar overlay
scar_overlay = r'c:\Users\Admin\Documents\Module4-main\Module4-main\image-damage-detection\v2\debug\02_scar_defect\03_layer1_structure\06_structure_damage_overlay.jpg'
if os.path.exists(scar_overlay):
    shutil.copyfile(scar_overlay, os.path.join(art_dir, 'scar_l1_overlay_verified.jpg'))
    print("Copied scar overlay")
