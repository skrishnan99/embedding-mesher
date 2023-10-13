from diffusers import UnCLIPPipeline, UnCLIPImageVariationPipeline
import torch
from PIL import Image

# pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
# pipe = pipe.to('cuda')

# prompt = "a portrait of an old monk, highly detailed."

# text_embeddings = pipe(prompt)

# image = pipe([prompt]).images[0]

# image.save("./frog.png")

# print([attr for attr in dir(pipe) if callable(getattr(pipe, attr)) and not attr.startswith("__")])


pipe = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch.float16)
pipe = pipe.to('cuda')

image = Image.open("./bad.jpg")
img_features = pipe.feature_extractor(image, return_tensors="pt").pixel_values
img_features = img_features.to(device="cuda", dtype=torch.float16)
image_embeddings = pipe.image_encoder(img_features).image_embeds

# outputs = pipe(image=image)
outputs = pipe(image_embeddings=image_embeddings)

image = outputs.images[0]
image.save("./bad-variation.jpg")
