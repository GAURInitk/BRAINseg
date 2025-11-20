# 🧠 BRAINseg — 3D Brain Tumor Segmentation

BRAINseg is a deep learning–based system for **multimodal 3D brain tumor segmentation**.  
It uses a **3D CNN encoder with residual blocks**, an **attention-guided decoder**, and a **VAE auxiliary decoder**.  
The project provides a **FastAPI backend** for inference and a **Streamlit frontend** for visualization.

---

## ✨ Features

- 🧩 **3D CNN** for volumetric feature extraction  
- 🔗 **Residual encoder** with skip connections  
- 🎯 **Attention Gates** applied to skip features  
- 🔄 **VAE auxiliary decoder** for latent regularization  
- 🚀 **FastAPI backend** for serving model predictions  
- 🌐 **Streamlit web app** for user-friendly interaction  

---

## 🧩 Model Architecture

### 🔷 Encoder
- 3D convolution layers  
- Batch Normalization + ReLU  
- Residual blocks  
- Downsampling using MaxPool3d  
- Multi-scale skip connections  

### 🔶 Attention-Guided Decoder
- Transposed convolutions for upsampling  
- Attention Gate on each skip connection  
- Channel projection for skip alignment  
- Produces final **4-channel segmentation mask**  

### 🔵 VAE Decoder
- Dense layers to compute **mean** and **log-variance**  
- Reparameterization:  

3D upsampling decoder  
Reconstructs MRI volume to stabilize encoder features  

📉 Loss Functions Used

-Dice loss
-Cross entropy
-KL divergence (VAE)
-Reconstruction loss (VAE)
