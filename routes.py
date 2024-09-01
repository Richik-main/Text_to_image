# routes.py
from flask import Blueprint, render_template, request, url_for
import torch
from diffusers import StableDiffusionPipeline

# Create a Blueprint for the routes
main = Blueprint('main', __name__)

# Device setup
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

# Load the Waifu Diffusion model
model_id = "hakurei/waifu-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to(device)

# Enable attention slicing and memory-efficient attention (optional)
pipe.enable_attention_slicing("max")
pipe.enable_vae_slicing()

# Safety checker modification to avoid black images
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']  # Get the prompt from the form input
    else:
        prompt = (
            "A rural village with humble homes, dusty roads, and people living in poverty, "
            "with a serene yet melancholic atmosphere, surrounded by fields and simple landscapes."
        )

    # Generate the image
    image = pipe(prompt, height=264, width=264, num_inference_steps=10, guidance_scale=7.5, output_type="pil").images[0]

    # Save the generated image
    image_path = "static/images/anime_girl.png"  # Use a relative path within the Flask static directory
    if image.getbbox():
        image.save(image_path)
        image_url = url_for('static', filename='images/anime_girl.png')
    else:
        print("Generated a black image. Try a different prompt or settings.")
        image_url = None  # Handle this case if needed

    return render_template('index.html', image_url=image_url, prompt=prompt)
