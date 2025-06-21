import streamlit as st
import requests
import nibabel as nib
import numpy as np
import imageio
import io
import matplotlib.pyplot as plt

st.title("Brain Tumor Segmentation Demo")

st.write("Upload 4 NIfTI (.nii) files (shape 128x128x128):")

uploaded_files = []
for i in range(1, 5):
    uploaded_files.append(st.file_uploader(f"Upload file {i}", type=["nii"], key=f"file{i}"))

if all(uploaded_files):
    if st.button("Run Segmentation"):
        files = {
            f'file{i+1}': (f.name, f, "application/octet-stream")
            for i, f in enumerate(uploaded_files)
        }
        with st.spinner("Sending data to backend and running inference..."):
            response = requests.post("http://127.0.0.1:8000/predict/", files=files)
        if response.status_code == 200:
            # Save the received .npy file
            with open("segmentation_result.npy", "wb") as f:
                f.write(response.content)
            seg = np.load("segmentation_result.npy", allow_pickle=True)
            seg = np.squeeze(seg)  # shape: (128, 128, 128, 4)
            # Convert one-hot to label map
            label_map = np.argmax(seg, axis=-1)  # shape: (128, 128, 128)

            st.success("Segmentation complete!")

            # Show input files as GIFs
            st.header("Input Files (Middle Slices as GIFs)")
            for i, f in enumerate(uploaded_files):
                f.seek(0)
                try:
                    # Try direct BytesIO loading
                    nii = nib.load(io.BytesIO(f.read()))
                except TypeError:
                    # Fallback for uncompressed .nii using file_map
                    f.seek(0)
                    file_map = nib.Nifti1Image.make_file_map()
                    file_map['image'].fileobj = io.BytesIO(f.read())
                    nii = nib.Nifti1Image.from_file_map(file_map)
                arr = nii.get_fdata()
                arr = np.asarray(arr, dtype=np.float32)
                # Normalize for display
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                # Make GIF of axial slices
                images = [(arr[:, :, z] * 255).astype(np.uint8) for z in range(0, arr.shape[2], 4)]
                gif_bytes = io.BytesIO()
                imageio.mimsave(gif_bytes, images, format='GIF', duration=0.1)
                st.image(gif_bytes.getvalue(), caption=f"Input {i+1} GIF", use_column_width=True)

            # Show output segmentation as GIF with colored labels
            st.header("Output Segmentation (GIF with Labels)")
            # Define a color map for up to 4 labels
            colors = np.array([
                [0, 0, 0],        # background - black
                [255, 0, 0],      # label 1 - red
                [0, 255, 0],      # label 2 - green
                [0, 0, 255],      # label 3 - blue
            ], dtype=np.uint8)
            # Make GIF of colored label slices
            label_gif = []
            for z in range(0, label_map.shape[2], 4):
                rgb = colors[label_map[:, :, z]]
                label_gif.append(rgb)
            gif_bytes = io.BytesIO()
            imageio.mimsave(gif_bytes, label_gif, format='GIF', duration=0.1)
            st.image(gif_bytes.getvalue(), caption="Segmentation Output GIF", use_column_width=True)
        else:
            st.error(f"Backend error: {response.text}")
else:
    st.info("Please upload all 4 input files.")