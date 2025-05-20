import cv2
import numpy as np
from sklearn.cluster import KMeans


def preprocess_and_segment(image_path, k=3):
    # Baca gambar dan ubah ke HSV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan atau rusak: {image_path}")

    img_resized = cv2.resize(img, (256, 256))
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    flat_hsv = hsv_img.reshape((-1, 3))

    # KMeans clustering pada HSV
    kmeans = KMeans(n_clusters=k, random_state=42).fit(flat_hsv)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Ambil label dengan jumlah piksel terbesar → diasumsikan background
    counts = np.bincount(labels)
    background_label = np.argmax(counts)

    # Buat mask yang hanya menyisakan motif (selain background)
    mask = (labels != background_label).astype(np.uint8)
    mask = mask.reshape((256, 256))

    # Terapkan mask ke HSV
    masked_hsv = hsv_img.copy()
    masked_hsv[mask == 0] = [0, 0, 0]  # hapus background

    return masked_hsv


def extract_hsv_features(masked_hsv):
    # Ambil hanya piksel non-background
    hsv_nonzero = masked_hsv[masked_hsv[:, :, 2] > 0]
    if hsv_nonzero.size == 0:
        return [0, 0, 0]

    mean_hsv = np.mean(hsv_nonzero, axis=0)
    return mean_hsv.tolist()
