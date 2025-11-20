BRAINseg — 3D Brain Tumor Segmentation

BRAINseg is a deep learning–based system for multimodal 3D brain tumor segmentation.
It uses a 3D CNN encoder with residual blocks, an attention-guided decoder, and a VAE auxiliary decoder.
The project includes a FastAPI backend for inference and a Streamlit frontend for visualization.

Features

3D CNN for volumetric feature extraction

Residual encoder with skip connections

Attention Gates applied on skip features

VAE auxiliary decoder for latent regularization

FastAPI backend for serving model predictions

Streamlit web app for user-friendly interaction

Model Architecture
Encoder

3D convolution layers

Batch Normalization + ReLU

Residual blocks

Downsampling using MaxPool3d

Multi-scale skip connections saved for decoder

Attention-Guided Decoder

Transposed convolutions for upsampling

Attention Gate applied to each skip connection

Channel projection for skip alignment

Produces final 4-channel segmentation mask

VAE Decoder

Dense layers to produce mean and log-variance

Reparameterization: z = mu + sigma * epsilon

3D upsampling decoder

Reconstructs MRI volume to stabilize encoder features

Installation
git clone https://github.com/GAURInitk/BRAINseg
cd BRAINseg
pip install -r requirements.txt


(If using GPU, install the correct PyTorch version from https://pytorch.org
)

Running the Application
Start the FastAPI backend
cd api
uvicorn main:app --reload

Start the Streamlit frontend
cd frontend
streamlit run app.py


Access the UI at:

http://localhost:8501


Upload a 3D MRI volume and view the predicted segmentation.

Training (Optional)
python train.py --data_path /path/to/brats


Loss functions used:

Dice loss

Cross entropy

KL divergence (VAE)

Reconstruction loss (VAE)

Example Results (Placeholder)
Region	Dice Score
ET	0.75
TC	0.70
WT	0.80

(Add your own results here.)

Technologies Used

PyTorch

FastAPI

Streamlit

NumPy, nibabel

CUDA (optional)

Contributing

Issues and pull requests are welcome.

License

MIT License


