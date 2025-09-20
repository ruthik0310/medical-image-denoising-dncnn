import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ---------------------------
# Dataset Loader
# ---------------------------
def load_dataset(input_dir, gt_dir):
    input_images = []
    gt_images = []

    for fname in os.listdir(input_dir):
        in_path = os.path.join(input_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if os.path.exists(gt_path):
            in_img = tf.keras.utils.load_img(in_path, color_mode="grayscale", target_size=(128, 128))
            gt_img = tf.keras.utils.load_img(gt_path, color_mode="grayscale", target_size=(128, 128))

            in_arr = tf.keras.utils.img_to_array(in_img) / 255.0
            gt_arr = tf.keras.utils.img_to_array(gt_img) / 255.0

            input_images.append(in_arr)
            gt_images.append(gt_arr)

    return np.array(input_images), np.array(gt_images)

# ---------------------------
# Model Definition
# ---------------------------
def build_model(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = models.Model(inputs, decoded)
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model

# ---------------------------
# Main Training Script
# ---------------------------
if __name__ == "__main__":
    print("🔹 Loading datasets...")

    # ✅ Load train and test
    train_in, train_gt = load_dataset(
        r"C:\Users\ASUS\Downloads\Med-Noise-Cleanse\Med-Noise-Cleanse-main\src\src\dataset\train\Dataset_in",
        r"C:\Users\ASUS\Downloads\Med-Noise-Cleanse\Med-Noise-Cleanse-main\src\src\dataset\train\Dataset_gt"
    )

    test_in, test_gt = load_dataset(
        r"C:\Users\ASUS\Downloads\Med-Noise-Cleanse\Med-Noise-Cleanse-main\src\src\dataset\test\Dataset_in",
        r"C:\Users\ASUS\Downloads\Med-Noise-Cleanse\Med-Noise-Cleanse-main\src\src\dataset\test\Dataset_gt"
    )

    # ✅ Split train into train + val
    train_in, val_in, train_gt, val_gt = train_test_split(
        train_in, train_gt, test_size=0.2, random_state=42
    )

    print(f"✅ Train: {train_in.shape}, Val: {val_in.shape}, Test: {test_in.shape}")

    # Build model
    model = build_model(input_shape=(128, 128, 1))
    model.summary()

    # Train
    history = model.fit(
        train_in, train_gt,
        validation_data=(val_in, val_gt),
        epochs=50,
        batch_size=16
    )

    # Save model
    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/denoising_model.h5")
    print("✅ Training completed & model saved at: saved_model/denoising_model.h5")
