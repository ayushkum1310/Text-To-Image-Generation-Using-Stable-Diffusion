import torch
from diffusers import StableDiffusionPipeline

# Load the model from the saved directory
pipe = StableDiffusionPipeline.from_pretrained("D:\Text-To-Image-Generation-Using-Stable-Diffusion\models", torch_dtype=torch.float32).to("cpu")

# Text prompt for image generation
prompt = "A girl flying"

# Generate image
with torch.no_grad():  # Disable gradient calculations for inference
    image = pipe(prompt).images[0]

# Save or display the image
image.save("generated_image_cpu.png")
image.show()
