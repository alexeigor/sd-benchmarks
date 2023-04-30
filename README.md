## Stable Diffusion inference benchmarks

model: https://huggingface.co/runwayml/stable-diffusion-v1-5

batch size = 1, image size 512x512, fp16

| GPU                    | PT2.0,fp16,xformers   | AITemplate,fp16 | DeepSpeed,fp16   | Oneflow,fp16     |
| :---                   | :---                  | :---            | :---             | :---             |
| A100-sxm4-40gb         | 1.96 s (4.54gb VRAM)  | 1.01 s (4.06 gb)| 1.28 s (4.97 gb) | 0.98 s (5.62 gb) |
| V100, 16gb             | 2.96 s                |                 |                  |                  |
| T4, 16gb               | 7.83 s                |                 |                  |                  |


## How to run
VM setup https://gist.github.com/alexeigor/b4c21b5e1fe62d670c433d4ac8c9fd83 (Ubuntu, Debian)
```
docker build . -t test_engine
docker run -it -v ${PWD}:${PWD} -w ${PWD} --gpus all test_engine
```

## Optimizations
1. xFormers
2. Channels last https://huggingface.co/docs/diffusers/main/en/optimization/fp16#using-channels-last-memory-format


## References:
- https://github.com/microsoft/DeepSpeed-MII/tree/main/examples/benchmark/txt2img
- https://github.com/Oneflow-Inc/oneflow
- https://github.com/Oneflow-Inc/diffusers
- https://github.com/facebookincubator/AITemplate
- https://arxiv.org/abs/2304.11267
- https://github.com/dbolya/tomesd
- https://huggingface.co/docs/diffusers/main/en/optimization/fp16
- https://github.com/hidet-org/hidet
- https://github.com/stochasticai/x-stable-diffusion
