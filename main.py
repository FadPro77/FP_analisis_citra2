from batik_processor import preprocess_and_segment, extract_hsv_features
from db import insert_feature_to_db
import os

def process_image(image_path, label):
    masked_hsv = preprocess_and_segment(image_path)
    features = extract_hsv_features(masked_hsv)
    insert_feature_to_db(os.path.basename(image_path), label, features)
    print(f"{image_path} selesai diproses. Fitur HSV: {features}")

# Contoh pemanggilan
if __name__ == "__main__":
    process_image("dataset/parang/parang1.jpg", "parang")
