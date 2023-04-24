import oneflow as flow
flow.mock_torch.enable()

from onediff import OneFlowStableDiffusionPipeline

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
        with flow.autocast("cuda"):
            _ = pipe(prompt)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_s = np.mean(latencies)
    time_std_s = np.std(latencies)
    time_p95_s = np.percentile(latencies,95)
    return f"P95 latency (seconds) - {time_p95_s:.2f}; Average latency (seconds) - {time_avg_s:.2f} +\- {time_std_s:.2f};", time_p95_s

def main():
    pipe = OneFlowStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=flow.float16,
    ).to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    with flow.autocast("cuda"):
        images = pipe(prompt).images
        for i, image in enumerate(images):
            image.save(f"{prompt}-of-{i}.png")

    vanilla_results = measure_latency(pipe, prompt)

    print(f"Vanilla pipeline: {vanilla_results[0]}")

if __name__ == "__main__":
    main()