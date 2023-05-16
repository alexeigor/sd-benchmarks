import os
os.environ['HF_HOME']='/workspace/.cache/huggingface'

import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from diffusers import FlaxStableDiffusionPipeline

import time

os.environ['XLA_FLAGS']='--xla_dump_to=/workspace/xla_dump/'

def benchmark_func(pipeline, prompts, p_params, rng):
    for _ in range(5):
        rng = jax.random.split(rng[0], jax.device_count())
        _ =  pipeline(prompts, p_params, rng, jit=True, height=512, width=512, num_inference_steps=50, guidance_scale=7.5)

    # Start benchmark.

    # Timed run
    n_runs = 10
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        rng = jax.random.split(rng[0], jax.device_count())
        _ =  pipeline(prompts, p_params, rng, jit=True, height=512, width=512, num_inference_steps=50, guidance_scale=7.5)
        end = time.perf_counter() - start
        latencies.append(end)

    # in ms
    time_avg_s = np.average(latencies)
    return time_avg_s

    
def main():
    num_devices = jax.device_count()
    device_type = jax.devices()[0].device_kind

    print(f"Found {num_devices} JAX devices of type {device_type}.")

    dtype = jnp.bfloat16
    
    model = "runwayml/stable-diffusion-v1-5"
    # model = "stabilityai/stable-diffusion-2-1"

    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        model,
        revision="bf16",
        dtype=dtype,
    )

    prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
    prompt = [prompt] * jax.device_count()
    prompt_ids = pipeline.prepare_inputs(prompt)
    
    p_params = replicate(params)
    prompt_ids = shard(prompt_ids)

    def create_key(seed=0):
        return jax.random.PRNGKey(seed)
    rng = create_key(0)
    rng = [rng]
    rng = jax.random.split(rng[0], jax.device_count())

    images = pipeline(prompt_ids, p_params, rng, jit=True).images
    images = images.reshape((images.shape[0] * images.shape[1], ) + images.shape[-3:])
    images = pipeline.numpy_to_pil(images)
    images[0].save('example.png')

    latency_ms = benchmark_func(pipeline, prompt_ids, p_params, rng)

    print(f"Pipeline latency: {latency_ms:.2f}")

if __name__ == "__main__":
    main()