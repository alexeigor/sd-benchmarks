import os
os.environ['HF_HOME']='/workspace/.cache/huggingface'

import time
import numpy as np

import torch
from diffusers import DiffusionPipeline

# @torch.inference_mode()
def benchmark_func(pipe, compiled, prompt):
    for _ in range(5):
        _ =  pipe(prompt)
    # Start benchmark.
    torch.cuda.synchronize()

    # Timed run
    n_runs = 10
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if not compiled:
            with torch.inference_mode():
                _ = pipe(prompt)
        else:
            _ = pipe(prompt)
        torch.cuda.synchronize()
        end = time.perf_counter() - start
        latencies.append(end)
    
    time_avg_s = np.average(latencies)
    return time_avg_s

run_compile = False  # Set True / False

def main():
    prompt = "a photo of an astronaut riding a horse on mars"

    # model = "runwayml/stable-diffusion-v1-5"
    model = "stabilityai/stable-diffusion-2-1"
    pipe_base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half).to("cuda")
    if run_compile:
        print("Run torch compile")
        pipe_base.unet = torch.compile(pipe_base.unet, mode="reduce-overhead", fullgraph=True)

    baseline_image = pipe_base(prompt, guidance_scale=7.5).images[0]
    baseline_image.save(f"baseline.png")

    prompt = "a beautiful photograph of Mt. Fuji during cherry blossom"

    latency_ms = benchmark_func(pipe_base, run_compile, prompt)

    print(f"Pipeline latency: {latency_ms:.2f}")

if __name__ == "__main__":
    main()
