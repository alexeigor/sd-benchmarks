import torch 

torch.set_float32_matmul_precision("high")

from diffusers import StableDiffusion3Pipeline
import time

id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipeline = StableDiffusion3Pipeline.from_pretrained(
    id, 
    torch_dtype=torch.float16
).to("cuda")
pipeline.set_progress_bar_config(disable=True)

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)
pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

prompt = "A cat holding a sign that says hello world"
for _ in range(3):
    _ = pipeline(
        prompt=prompt,
        num_inference_steps=50, 
        guidance_scale=5.0,
        generator=torch.manual_seed(1),
    )

start = time.time()
for _ in range(10):
    _ = pipeline(
        prompt=prompt,
        num_inference_steps=50, 
        guidance_scale=5.0,
        generator=torch.manual_seed(1),
    )
end = time.time()
avg_inference_time = (end - start) / 10
print(f"Average inference time: {avg_inference_time:.3f} seconds.")

image = pipeline(
    prompt=prompt,
    num_inference_steps=50, 
    guidance_scale=5.0,
    generator=torch.manual_seed(1),
).images[0]
filename = "_".join(prompt.split(" "))
image.save(f"diffusers_{filename}.png")