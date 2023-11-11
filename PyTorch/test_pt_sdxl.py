import os
os.environ['HF_HOME']='/workspace/.cache/huggingface'

import time
import numpy as np

import torch
from diffusers import StableDiffusionXLPipeline

sd_args = {"width": 1024, "height": 1024, "guidance_scale": 7.5, "num_inference_steps": 50}

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
    pipe = StableDiffusionXLPipeline.from_pretrained(
        # "segmind/SSD-1B",
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")

    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt, **sd_args).images[0]
 
    image.save("image_sdxl.jpg")

    batch_size = 1
    prompt = ["A majestic lion jumping from a big stone at night"] * batch_size

    latency_ms = benchmark_func(pipe, run_compile, prompt)

    print("Pipeline latency:", latency_ms, "ms")

if __name__ == "__main__":
    main()
