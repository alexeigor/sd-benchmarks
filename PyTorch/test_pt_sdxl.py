import os
os.environ['HF_HOME']='/workspace/.cache/huggingface'

import time
import numpy as np

import torch
from diffusers import DiffusionPipeline

sd_args = {"width": 768, "height": 768, "guidance_scale": 7.5}

# @torch.inference_mode()
def benchmark_func(pipe, compiled, prompt):
    for _ in range(5):
        _ =  pipe(prompt, **sd_args)
    # Start benchmark.
    torch.cuda.synchronize()

    # Timed run
    n_runs = 10
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        if not compiled:
            with torch.inference_mode():
                _ = pipe(prompt, **sd_args)
        else:
            _ = pipe(prompt, **sd_args)
        torch.cuda.synchronize()
        end = time.perf_counter_ns() - start
        latencies.append(end)
    
    time_avg_s = np.average(latencies)
    return int(time_avg_s / 1000000.0)

run_compile = True  # Set True / False

def main():
    # load both base & refiner
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8

    prompt = "A majestic lion jumping from a big stone at night"

    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    image.save("image_sdxl.jpg")

if __name__ == "__main__":
    main()
