import streamlit as st
from diffusers import UnCLIPPipeline, UnCLIPImageVariationPipeline
import torch
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    c = False
    if not isinstance(v0,np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1,np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    if c:
        res = torch.from_numpy(v2).to("cuda")
    else:
        res = v2
    return res

@st.cache_resource
def load_model():
    pipe = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    return pipe

pipe = load_model()

st.title('Image Mesher')

file_uploader_columns = st.columns(2, gap="medium")

with file_uploader_columns[0]:
    image_1_file = st.file_uploader(
        label="Upload the first image", 
        type=["png", "jpg", "bmp"], 
        accept_multiple_files=False, 
        key=None, 
        help=None, 
        on_change=None, 
        args=None, 
        kwargs=None, 
        disabled=False, 
        label_visibility="visible"
    )

with file_uploader_columns[1]:
    image_2_file = st.file_uploader(
        label="Upload the second image", 
        type=["png", "jpg", "bmp"], 
        accept_multiple_files=False, 
        key=None, 
        help=None, 
        on_change=None, 
        args=None, 
        kwargs=None, 
        disabled=False, 
        label_visibility="visible"
    )

submit_btn = st.button(
    label="Do your thang!", 
    key=None, 
    help=None, 
    on_click=None, 
    args=None, 
    kwargs=None, 
    type="secondary", 
    disabled=False, 
    use_container_width=False
)

if submit_btn:
    if image_1_file is not None and image_2_file is not None:
        image_1 = Image.open(image_1_file).resize((256, 256))
        image_2 = Image.open(image_2_file).resize((256, 256))

        image_cols = st.columns(2, gap="large")
        with image_cols[0]:
            st.header("Start Image")
            st.image(image_1)
        with image_cols[1]:
            st.header("Finish Image")
            st.image(image_2)

        img_1_features = pipe.feature_extractor(image_1, return_tensors="pt").pixel_values
        img_1_features = img_1_features.to(device="cuda", dtype=torch.float16)
        image_1_embeddings = pipe.image_encoder(img_1_features).image_embeds

        img_2_features = pipe.feature_extractor(image_2, return_tensors="pt").pixel_values
        img_2_features = img_2_features.to(device="cuda", dtype=torch.float16)
        image_2_embeddings = pipe.image_encoder(img_2_features).image_embeds

        # diff_image_embeddings = (0.9 * image_1_embeddings) + (0.1 * image_2_embeddings)
        # diff_image_embeddings = 0.1 * image_1_embeddings + 0.9 * image_2_embeddings

        interpolated_images = []
        for interp_val_unscaled in range(0, 12, 2):
            interp_val = interp_val_unscaled / 10
            interp_image_embeddings = slerp(interp_val, image_1_embeddings, image_2_embeddings)
            outputs = pipe(image_embeddings=interp_image_embeddings, num_images_per_prompt=2)
            interpolated_images.append(outputs.images)

        image_height = 256
        image_width = 256
        num_rows = len(interpolated_images[0]) # num images per interp val
        num_cols = len(interpolated_images) # num different interp vals

        interp_label_cols = st.columns(num_cols, gap="small")
        col_idx = 0 
        for interp_val_unscaled in range(0, 12, 2):
            with interp_label_cols[col_idx]:
                st.write(f"interpolation value - {interp_val_unscaled / 10}")
            col_idx += 1


        # Use nested loops to paste images onto the canvas
        for i in range(num_rows):
            cols = st.columns(num_cols, gap="small")
            for j in range(num_cols):
                with cols[j]:
                    st.image(interpolated_images[j][i])

    else:
        st.write("One of Image 1 or Image 2 is not uplaoded. Both images need to be uploaded.")