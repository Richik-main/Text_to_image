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

model_id = "stabilityai/stable-diffusion-2-1-base"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the model with full precision
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to(device)

# Enable attention slicing and memory-efficient attention (optional)
pipe.enable_attention_slicing("max")
pipe.enable_vae_slicing()

# Safety checker modification to avoid black images
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker

# Set the prompt and generate the image
prompt = "A planet"
image = pipe(prompt, height=256, width=256, num_inference_steps=20, guidance_scale=7.5, output_type="pil").images[0]

# Check if the image is not black by verifying its pixel values
if image.getbbox():
    image.save("dancing_man.png")
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
