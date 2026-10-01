import cv2
from comparator import DamageComparator

# Load và pre-process
def load_and_preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    result = img.copy()
    result[binary == 0] = [0, 0, 0]
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Load và create feature dictionaries
product_img = cv2.imread('images/1.jpg')
return_img_raw = cv2.imread('images/2.jpg')

if product_img is None or return_img_raw is None:
    print('ERROR: Cannot load images')
    exit(1)

def create_feature_dict(img, path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    result = img.copy()
    result[binary == 0] = [0, 0, 0]
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return {
        'image': processed,
        'gray': gray,
        'binary': binary,
        'lab': lab,
        'image_path': path,
    }

product = create_feature_dict(product_img, 'images/1.jpg')
return_feature = create_feature_dict(return_img_raw, 'images/2.jpg')

print(f'Loaded: {product["image"].shape} vs {return_feature["image"].shape}')

# Compare with proper feature dictionaries
comparator = DamageComparator()
result = comparator.compare(product['image'], return_feature['image'])

print('='*60)
print(f'damage_score: {result["damage_score"]:.2f}%')
print(f'difference_area: {result["difference_area"]:.0f} px')
print(f'largest_damage_area: {result["largest_damage_area"]:.0f} px')
print('='*60)
