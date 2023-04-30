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
    import torch
    import os
    import functools

    import torch._dynamo.config
    torch._dynamo.config.suppress_errors = True

    import hidet

    # more search 
    hidet.torch.dynamo_config.search_space(2)
    # automatically transform the model to use float16 data type
    hidet.torch.dynamo_config.use_fp16(True)
    # use float16 data type as the accumulate data type in operators with reduction
    hidet.torch.dynamo_config.use_fp16_reduction(True)
    # use tensorcore
    hidet.torch.dynamo_config.use_tensor_core()

    torch.backends.cudnn.benchmark = True

    from diffusers import DiffusionPipeline
    from diffusers import DPMSolverMultistepScheduler


    prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
    prompt = "a grey cat sitting on a chair in the kitchen, animated"

    model = "runwayml/stable-diffusion-v1-5"

    generator = torch.Generator(device="cuda").manual_seed(21)
    dpm = DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler")
    pipe_base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half, scheduler=dpm).to("cuda")
    # pipe_base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half).to("cuda")

    unet = pipe_base.unet
    unet.eval()
    # unet.to(memory_format=torch.channels_last)  # use channels_last memory format
    # unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

    pipe_base.unet = torch.compile(pipe_base.unet, backend='hidet')

    baseline_image = pipe_base(prompt, guidance_scale=7.5, generator=generator, num_inference_steps=120).images[0]
    baseline_image.save(f"baseline.png")

    prompt = "a photo of an astronaut riding a horse on mars"

    vanilla_results = measure_latency(pipe_base, prompt)

    print(f"Vanilla pipeline: {vanilla_results[0]}")

if __name__ == "__main__":
    main()