import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_NOISY = os.path.join(DATASET_DIR, "train", "Dataset_in")
TRAIN_CLEAN = os.path.join(DATASET_DIR, "train", "Dataset_gt")

TEST_NOISY = os.path.join(DATASET_DIR, "test", "Dataset_in")
TEST_CLEAN = os.path.join(DATASET_DIR, "test", "Dataset_gt")

# Make directories
for d in [TRAIN_NOISY, TRAIN_CLEAN, TEST_NOISY, TEST_CLEAN]:
    os.makedirs(d, exist_ok=True)

# Load your clean dataset (already preprocessed to 128x128 grayscale)
X = np.load("X_train.npy", allow_pickle=True)

print(f"[INFO] Loaded dataset with shape {X.shape}")

# Split into train/test
split_idx = int(0.8 * len(X))
train_data, test_data = X[:split_idx], X[split_idx:]

def add_noise(img):
    """Add Gaussian noise to a clean image"""
    noise = np.random.normal(0, 0.1, img.shape)
    noisy_img = np.clip(img + noise, 0., 1.)
    return noisy_img

def save_images(data, noisy_path, clean_path, prefix="train"):
    for i, img in enumerate(data):
        clean = img.squeeze()
        noisy = add_noise(clean)

        # Save .npy (used for training in model_trainer.py)
        np.save(os.path.join(noisy_path, f"{prefix}_in_{i}.npy"), noisy)
        np.save(os.path.join(clean_path, f"{prefix}_gt_{i}.npy"), clean)

    print(f"[INFO] Saved {len(data)} {prefix} images")

# Save train and test datasets
save_images(train_data, TRAIN_NOISY, TRAIN_CLEAN, prefix="train")
save_images(test_data, TEST_NOISY, TEST_CLEAN, prefix="test")

print("[INFO] Dataset creation complete ✅")
