import torch
import os

from diffusers import DiffusionPipeline

# @torch.inference_mode()
def benchmark_func(pipe, compiled, prompt):
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
        if not compiled:
            with torch.inference_mode():
                _ = pipe(prompt)
        else:
            _ = pipe(prompt)
    torch.cuda.synchronize()
    end_event.record()
    # in ms
    time_avg_s = (start_event.elapsed_time(end_event)) / n_runs
    return time_avg_s

run_compile = False  # Set True / False

def main():
    prompt = "a photo of an astronaut riding a horse on mars"

    model = "runwayml/stable-diffusion-v1-5"
    pipe_base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.half).to("cuda")
    if run_compile:
        print("Run torch compile")
        pipe_base.unet = torch.compile(pipe_base.unet, mode="reduce-overhead", fullgraph=True)

    baseline_image = pipe_base(prompt, guidance_scale=7.5).images[0]
    baseline_image.save(f"baseline.png")

    latency_ms = benchmark_func(pipe_base, run_compile, prompt)

    print(f"Pipeline latency: {latency_ms:.2f}")

if __name__ == "__main__":
    main()