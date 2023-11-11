import os
os.environ['HF_HOME']='/workspace/.cache/huggingface'

from onediff.infer_compiler import oneflow_compile
from onediff.optimization import rewrite_self_attention
from diffusers import StableDiffusionPipeline
import oneflow as flow
import torch

from time import perf_counter
import numpy as np

sd_args = {"width": 512, "height": 512, "guidance_scale": 7.5, "num_inference_steps": 50}


def measure_latency(pipe, prompt):
    # warm up
    # pipe.set_progress_bar_config(disable=True)
    for _ in range(2):
        _ =  pipe(prompt, **sd_args)
    flow._oneflow_internal.eager.Sync()

    # Timed run
    latencies = []
    for _ in range(10):
        start_time = perf_counter()
        with flow.autocast("cuda"):
            _ = pipe(prompt, **sd_args)
        flow._oneflow_internal.eager.Sync()
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_s = np.mean(latencies)
    time_std_s = np.std(latencies)
    time_p95_s = np.percentile(latencies,95)
    return f"P95 latency (seconds) - {time_p95_s:.2f}; Average latency (seconds) - {time_avg_s:.2f} +\- {time_std_s:.2f};", time_p95_s

def main():
    # model_id = "stabilityai/stable-diffusion-2-1"
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    rewrite_self_attention(pipe.unet)
    pipe.unet = oneflow_compile(pipe.unet)

    prompt = "a photo of an astronaut riding a horse on mars"
    with flow.autocast("cuda"):
        images = pipe(prompt, **sd_args).images
        for i, image in enumerate(images):
            image.save(f"{prompt}-of-{i}.png")

    vanilla_results = measure_latency(pipe, prompt)

    print(f"Vanilla pipeline: {vanilla_results[0]}")

if __name__ == "__main__":
    main()