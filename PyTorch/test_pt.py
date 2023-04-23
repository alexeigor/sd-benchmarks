import torch
import os

from diffusers import DiffusionPipeline

from time import perf_counter
import numpy as np


def measure_latency(pipe, prompt):
    latencies = []
    # warm up
    # pipe.set_progress_bar_config(disable=True)
    for _ in range(2):
        _ =  pipe(prompt)
    # Timed run
    for _ in range(10):
        start_time = perf_counter()
        with torch.inference_mode():
            _ = pipe(prompt)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_s = np.mean(latencies)
    time_std_s = np.std(latencies)
    time_p95_s = np.percentile(latencies,95)
    return f"P95 latency (seconds) - {time_p95_s:.2f}; Average latency (seconds) - {time_avg_s:.2f} +\- {time_std_s:.2f};", time_p95_s

def main():
    prompt = "a dog on a rocket"

    model = "runwayml/stable-diffusion-v1-5"
    pipe_base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half).to("cuda")

    baseline_image = pipe_base(prompt, guidance_scale=7.5).images[0]
    baseline_image.save(f"baseline.png")

    prompt = "a photo of an astronaut riding a horse on mars"

    vanilla_results = measure_latency(pipe_base, prompt)

    print(f"Vanilla pipeline: {vanilla_results[0]}")

if __name__ == "__main__":
    main()