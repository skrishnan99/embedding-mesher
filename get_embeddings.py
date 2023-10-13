from diffusers import UnCLIPPipeline, UnCLIPImageVariationPipeline
import torch
import numpy as np
from PIL import Image

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

pipe = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch.float16)
pipe = pipe.to('cuda')

if __name__ = "__main__":
    image_1 = Image.open("./good.jpg")
    image_2 = Image.open("./bad.jpg")

    good_img_features = pipe.feature_extractor(image_1, return_tensors="pt").pixel_values
    good_img_features = good_img_features.to(device="cuda", dtype=torch.float16)
    image_1_embeddings = pipe.image_encoder(good_img_features).image_embeds

    bad_img_features = pipe.feature_extractor(image_2, return_tensors="pt").pixel_values
    bad_img_features = bad_img_features.to(device="cuda", dtype=torch.float16)
    image_2_embeddings = pipe.image_encoder(bad_img_features).image_embeds

    # diff_image_embeddings = (0.9 * image_1_embeddings) + (0.1 * image_2_embeddings)
    # diff_image_embeddings = 0.1 * image_1_embeddings + 0.9 * image_2_embeddings

    interpolated_images = []
    for interp_val_unscaled in range(0, 10, 2):
        interp_val = interp_val_unscaled / 10
        diff_image_embeddings = slerp(interp_val, image_2_embeddings, image_1_embeddings)
        outputs = pipe(image_embeddings=diff_image_embeddings, num_images_per_prompt=2)
        interpolated_images.append(outputs.images)


    image_height = 256
    image_width = 256
    num_rows = len(interpolated_images[0]) # num images per interp val
    num_cols = len(interpolated_images) # num different interp vals

    grid_width = num_cols * image_width
    grid_height = num_rows * image_height
    grid_img = Image.new('RGB', (grid_width, grid_height), 'white')

    # Use nested loops to paste images onto the canvas
    for i in range(num_rows):
        for j in range(num_cols):
            grid_img.paste(interpolated_images[j][i], (j*image_width, i*image_height))


    grid_img.save("interp_grid.jpg")


