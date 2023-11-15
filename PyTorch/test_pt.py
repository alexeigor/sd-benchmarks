import os
os.environ['HF_HOME']='/workspace/.cache/huggingface'

import time
import numpy as np

import torch
from diffusers import DiffusionPipeline

sd_args_v15 = {"_model_id_": "runwayml/stable-diffusion-v1-5", "width": 512, "height": 512, "guidance_scale": 7.5, "num_inference_steps": 50}
sd_args_v21 = {"_model_id_": "stabilityai/stable-diffusion-2-1", "width": 768, "height": 768, "guidance_scale": 7.5, "num_inference_steps": 50}
sd_args_xl = {"_model_id_": "stabilityai/stable-diffusion-xl-base-1.0", "width": 1024, "height": 1024, "guidance_scale": 7.5, "num_inference_steps": 50}


sd_args = sd_args_v15

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
    batch_size = 1
    prompt = ["A majestic lion jumping from a big stone at night"] * batch_size

    pipe_base = DiffusionPipeline.from_pretrained(
            sd_args["_model_id_"],
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")
    
    if run_compile:
        print("Run torch compile")
        pipe_base.unet = torch.compile(pipe_base.unet, mode="reduce-overhead", fullgraph=True)

    baseline_image = pipe_base(prompt, **sd_args).images
    for idx, im in enumerate(baseline_image):
        im.save(f"{idx:06}.jpg")

    prompt = ["A majestic lion jumping from a big stone at night"] * batch_size

    latency_ms = benchmark_func(pipe_base, run_compile, prompt)

    print("Pipeline latency:", latency_ms, "ms")
    print(sd_args)

if __name__ == "__main__":
    main()
