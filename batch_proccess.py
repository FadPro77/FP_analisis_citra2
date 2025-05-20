import os
from main import process_image  # dari script yang sudah dibuat

def process_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            print(f"[INFO] Processing: {image_path}")  # Log di sini
            process_image(image_path, label)

# Jalankan semua folder
base_path = os.path.join(os.path.dirname(__file__), "dataset")

for motif in os.listdir(base_path):
    motif_path = os.path.join(base_path, motif)
    if os.path.isdir(motif_path):
        process_folder(motif_path, motif)
