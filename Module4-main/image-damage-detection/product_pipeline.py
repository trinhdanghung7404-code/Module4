from feature_extractor import FeatureExtractor
from database import Database


def add_product():

    print("\n=== Add New Product ===")

    image_path = input("Image path: ").strip()

    extractor = FeatureExtractor()

    feature = extractor.extract(image_path)

    db = Database()

    db.save_product(feature)

    print("Product feature saved successfully.")