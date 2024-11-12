import torch
from diffusers import StableDiffusionPipeline

# Load the model with float32 precision (default for CPU)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cpu")  # Ensure the model is moved to CPU

# Text prompt for image generation
prompt = "A girl flying"

# Generate image (no need for 'autocast' since we are using CPU)
image = pipe(prompt).images[0]

# Save or display the image
image.save("generated_image.png")
image.show()

# Save the model
pipe.save_pretrained("D:/Text-To-Image-Generation-Using-Stable-Diffusion/models")
