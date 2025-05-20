from batik_processor import preprocess_and_segment, extract_hsv_features
from knn_classifier import classify_hsv_feature
import sys

def predict_image(image_path):
    masked_hsv = preprocess_and_segment(image_path)
    features = extract_hsv_features(masked_hsv)
    result = classify_hsv_feature(features)
    print(f"Prediksi untuk {image_path} → {result}")

# Contoh pemanggilan
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gunakan: python predict.py path_to_image.jpg")
    else:
        predict_image(sys.argv[1])
