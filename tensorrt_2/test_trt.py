import os
os.environ['HF_HOME']='/workspace/.cache/huggingface'

import time
import numpy as np

import torch
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

# @torch.inference_mode()
def benchmark_func(pipe, prompt):
    for _ in range(5):
        _ =  pipe(prompt)
    # Start benchmark.
    torch.cuda.synchronize()

    # Timed run
    n_runs = 10
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        _ = pipe(prompt)
        torch.cuda.synchronize()
        end = time.perf_counter_ns() - start
        latencies.append(end)
    
    time_avg_s = np.average(latencies)
    return time_avg_s

def main():
    # Use the DDIMScheduler scheduler here instead
    scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",
                                                    custom_pipeline="stable_diffusion_tensorrt_txt2img",
                                                    revision='fp16',
                                                    torch_dtype=torch.float16,
                                                    scheduler=scheduler,)

    # re-use cached folder to save ONNX models and TensorRT Engines
    pipe.set_cached_folder("stabilityai/stable-diffusion-2-1", revision='fp16',)

    pipe = pipe.to("cuda")

    prompt = "a beautiful photograph of Mt. Fuji during cherry blossom"
    image = pipe(prompt).images[0]
    image.save('tensorrt_mt_fuji.png')

    prompt = ["a beautiful photograph of Mt. Fuji during cherry blossom"] * 1

    latency_ms = benchmark_func(pipe, prompt)

    print("Pipeline latency:", latency_ms)


if __name__ == "__main__":
    main()
