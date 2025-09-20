import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------------------------
# Dataset Loader
# ---------------------------
def load_dataset(input_dir, gt_dir):
    input_images = []
    gt_images = []
    file_names = []

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
            file_names.append(fname)

    return np.array(input_images), np.array(gt_images), file_names


# ---------------------------
# Load Model & Test Data
# ---------------------------
print("🔹 Loading trained model...")
model = tf.keras.models.load_model(
    "saved_model/denoising_model.h5",
    compile=False
)


print("🔹 Loading test dataset...")
test_in, test_gt, test_files = load_dataset(
    r"C:\Users\ASUS\Downloads\Med-Noise-Cleanse\Med-Noise-Cleanse-main\src\src\dataset\test\Dataset_in",
    r"C:\Users\ASUS\Downloads\Med-Noise-Cleanse\Med-Noise-Cleanse-main\src\src\dataset\test\Dataset_gt"
)

print(f"✅ Test dataset loaded: {test_in.shape}")

# ---------------------------
# Run Predictions
# ---------------------------
print("🔹 Running denoising on test images...")
preds = model.predict(test_in, batch_size=8)

# ---------------------------
# Save Results + Compute Metrics
# ---------------------------
os.makedirs("results", exist_ok=True)

psnr_scores = []
ssim_scores = []

for i in range(len(test_in)):
    noisy = test_in[i].squeeze()
    denoised = preds[i].squeeze()
    gt = test_gt[i].squeeze()

    # Compute PSNR and SSIM
    psnr_val = psnr(gt, denoised, data_range=1.0)
    ssim_val = ssim(gt, denoised, data_range=1.0)

    psnr_scores.append(psnr_val)
    ssim_scores.append(ssim_val)

    # Plot side-by-side
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(noisy, cmap="gray")
    plt.title("Noisy Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(denoised, cmap="gray")
    plt.title(f"Denoised\nPSNR={psnr_val:.2f}, SSIM={ssim_val:.3f}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join("results", f"result_{test_files[i]}")
    plt.savefig(save_path)
    plt.close()

# ---------------------------
# Print Average Metrics
# ---------------------------
print("\n📊 Evaluation Metrics on Test Set:")
print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
print(f"Average SSIM: {np.mean(ssim_scores):.4f}")

print("\n✅ Denoised results saved in 'results/' folder.")
