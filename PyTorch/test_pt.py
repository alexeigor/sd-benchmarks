import torch
import os

from diffusers import DiffusionPipeline

@torch.inference_mode()
def benchmark_func(pipe, prompt):
    latencies = []
    for _ in range(5):
        _ =  pipe(prompt)

    # Start benchmark.
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # Timed run
    n_runs = 10
    for _ in range(n_runs):
        _ = pipe(prompt)
    torch.cuda.synchronize()
    end_event.record()
    # in ms
    time_avg_s = (start_event.elapsed_time(end_event)) / n_runs
    return time_avg_s


def main():
    prompt = "a photo of an astronaut riding a horse on mars"

    model = "runwayml/stable-diffusion-v1-5"
    pipe_base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half).to("cuda")

    baseline_image = pipe_base(prompt, guidance_scale=7.5).images[0]
    baseline_image.save(f"baseline.png")

    latency_ms = benchmark_func(pipe_base, prompt)

    print(f"Pipeline latency: {latency_ms:.2f}")

if __name__ == "__main__":
    main()