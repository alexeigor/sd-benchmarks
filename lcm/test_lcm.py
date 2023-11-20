import os

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

import torch
import torch.utils.benchmark as benchmark
import argparse
from diffusers import DiffusionPipeline, LCMScheduler

PROMPT = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_ID = "latent-consistency/lcm-lora-sdxl"


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


def load_pipeline(standard_sdxl=False):
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, variant="fp16")
    if not standard_sdxl:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(LORA_ID)

    pipe.to(device="cuda", dtype=torch.float16)
    return pipe


def call_pipeline(pipe, batch_size, num_inference_steps, guidance_scale):
    images = pipe(
        prompt=PROMPT,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=batch_size,
        guidance_scale=guidance_scale,
    ).images[0]


def main():
    standard_sdxl = False
    batch_size = 1

    pipeline = load_pipeline(standard_sdxl)
    if standard_sdxl:
        num_inference_steps = 25
        guidance_scale = 5
    else:
        num_inference_steps = 4
        guidance_scale = 1

    time = benchmark_fn(call_pipeline, pipeline, batch_size, num_inference_steps, guidance_scale)

    print(f"Batch size: {batch_size} in {time/1e6:.3f} seconds")

if __name__ == "__main__":
    main()
