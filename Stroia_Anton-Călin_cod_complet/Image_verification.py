from PIL import Image
import os

def verify_images(directory):
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if not file_name.lower().endswith('.png'):
                print(f"[AVERTISMENT] Fișier cu extensie greșită: {file_path}")
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"[EROARE] Imagine invalidă: {file_path} - {e}")

verify_images("dataset/training")
verify_images("dataset/testing")
