import os
import torch
import torch.nn as nn
import streamlit as st
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config safe path + reliable model URL
# -----------------------------
MODEL_DIR = r"C:\MedNoise\models"
MODEL_FILENAME = "dncnn_sigma2_gray.pth"  # grayscale denoiser
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_gray.pth"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Download if missing
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.warning(" Model not found locally. Downloading pretrained weights...")
    try:
        resp = requests.get(MODEL_URL)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(resp.content)
        st.success(" Model downloaded successfully!")
    except Exception as e:
        st.error(f" Download failed: {e}")

# -----------------------------
# Define the DnCNN model
# -----------------------------
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(channels, 64, 3, padding=1, bias=False), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers += [nn.Conv2d(64, 64, 3, padding=1, bias=False),
                       nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(64, channels, 3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)

# -----------------------------
# Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN().to(device)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    st.success(" Model loaded!")
else:
    st.error(" Model file missing and couldn't be loaded.")

# -----------------------------
# Streamlit UI: Upload & Denoise
# -----------------------------
st.title(" MRI  Image Denoising with Dncnn")
uploaded = st.file_uploader("Upload a grayscale medical image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor).cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Noisy Input")
    axes[0].axis("off")
    axes[1].imshow(output, cmap="gray")
    axes[1].set_title("Denoised Output")
    axes[1].axis("off")
    st.pyplot(fig)