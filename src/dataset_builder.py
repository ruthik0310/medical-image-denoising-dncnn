import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "../brain_tumor_mri"
IMG_SIZE = 128

def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):  # walks inside subfolders too
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    images.append(img)
    return images

print(f"[INFO] Loading dataset from: {DATASET_PATH}")
X = load_images_from_folder(DATASET_PATH)
X = np.array(X, dtype="float32") / 255.0
X = np.expand_dims(X, axis=-1)  # add channel

print(f"[INFO] Dataset shape: {X.shape}")

# split into train/test
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# save dataset
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)

print("[INFO] Dataset saved as X_train.npy and X_test.npy")
