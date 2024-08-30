# import torch
# from diffusers import FluxPipeline

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# prompt = "A cat holding a sign that says hello world"
# image = pipe(
#     prompt,
#     guidance_scale=0.0,
#     num_inference_steps=4,
#     max_sequence_length=256,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save("flux-schnell.png")

#%%
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    print("MPS is available.")
else:
    print("MPS is not available.")

# Check if MPS is built
if torch.backends.mps.is_built():
    print("MPS is built and supported.")
else:
    print("MPS is not built.")


#%%

import torch
from diffusers import StableDiffusionPipeline

# Replace the model_id with the Waifu Diffusion model ID
model_id = "hakurei/waifu-diffusion"
# Check if CUDA (NVIDIA GPUs) is available and set the device to GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Optional: Print the device to confirm
print(f"Using device: {device}")

# Load the Waifu Diffusion model with full precision
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to(device)

# Enable attention slicing and memory-efficient attention (optional)
pipe.enable_attention_slicing("max")
pipe.enable_vae_slicing()

# Safety checker modification to avoid black images
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker

# Set the prompt and generate the anime-style image
prompt = (
    "A rural village with humble homes, dusty roads, and people living in poverty, "
    "with a serene yet melancholic atmosphere, surrounded by fields and simple landscapes."
)
###num_inference_steps
#Description: This parameter determines the number of diffusion steps the model takes to generate the image.
#Example: num_inference_steps=50
#Role: The more steps, the more gradual and refined the generation process. A higher number typically results in better image quality, as the model has more opportunities to refine the image, but it also takes longer to generate the image. For example, 50 steps is often a good compromise between speed and quality.
##guidance_scale
#Description: The guidance_scale controls how strongly the generated image adheres to the given prompt.
#Example: guidance_scale=7.5
#Role: A higher guidance scale means the model will focus more on the prompt and less on randomness during the image generation process. This can lead to images that are more aligned with the prompt. However, if the scale is too high, it might distort the image or result in less natural visuals. A typical range is between 5.0 and 10.0, where 7.5 is a balanced choice.
image = pipe(prompt, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, output_type="pil").images[0]

# Check if the image is not black by verifying its pixel values
if image.getbbox():
    image.save("anime_girl.png")
else:
    print("Generated a black image. Try a different prompt or settings.")


# %%

from transformers.utils import logging
from pathlib import Path

# Set up logging to see detailed information
logging.set_verbosity_info()

# Print the default cache directory
cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
print(f"Cache directory: {cache_dir}")

# Alternatively, print the environment variable if set
import os
hf_cache = os.getenv('HF_HOME', cache_dir)
print(f"Hugging Face cache directory: {hf_cache}")


# %%
