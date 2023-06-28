import torch    # Tested with 2.0.1+cu118
from diffusers import StableDiffusionPipeline    # <3


# Model location in HF 
model = "https://huggingface.co/vluz/Generalis_V1/blob/main/Generalis_v1.safetensors"

# Create pipe
pipe = StableDiffusionPipeline.from_ckpt(model, 
    torch_dtype=torch.float16, 
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,)

# Cleanup
del pipe.vae.encoder
torch.cuda.empty_cache()

# Send to GPU
pipe = pipe.to("cuda")

# Optimize for low vram use and clear cache again
pipe.enable_vae_tiling()
pipe.enable_attention_slicing("max")
pipe.enable_xformers_memory_efficient_attention(attention_op=None)
pipe.unet.to(memory_format=torch.channels_last)
pipe.enable_sequential_cpu_offload()
torch.cuda.empty_cache()

# Set a prompt
prompt = "a cat"

# Generate image based on prompt
image = pipe(prompt).images[0]

# Save result image to disk
image.save("cat.png")
