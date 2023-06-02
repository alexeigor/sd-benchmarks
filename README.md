## Stable Diffusion inference benchmarks

model: 

https://huggingface.co/runwayml/stable-diffusion-v1-5 - sd1.5

https://huggingface.co/stabilityai/stable-diffusion-2-1 - sd2.1

batch size = 1, image size 512x512, 50 iterations

#### A100-SXM, 40GB
| Engine                 | Time, sd1.5           | Time, sd2.1           | Time, sd2.1, 768x768  | 
| :---                   | :---                  | :---                  | :---                  |
| PT2.0,fp16             | 1.96 s (4.54gb VRAM)  |                       |                       |
| PT2.0,fp16 + compile   | 1.41 s (5.96 gb)      |                       |                       |
| AITemplate,fp16        | 1.01 s (4.06 gb)      |                       |                       |
| DeepSpeed,fp16         | 1.28 s (4.97 gb)      |                       |                       |
| Oneflow,fp16           | 0.98 s (5.62 gb)      |                       |                       |
| TensorRT 8.6.1, fp16   | 0.98 s                | 0.90 s                | 1.98 s                |
| Onnxruntime,fp16,CUDA  | 1.05 s                | 1.00 s                |                       |
| Jax,XLA,bf16           | 1.58 s                | 1.35 s                |                       |

#### H100-PCIe, 80GB

| Engine                 | Time, sd1.5           | Time, sd2.1           | 
| :---                   | :---                  | :---                  |
| PT2.0,fp16             | 1.44 s                |                       |
| PT2.0,fp16,compile     | 1.11 s                |                       |
| TensorRT 8.6.1,fp16    | 0.75 s                | 0.68 s                |
| Jax,XLA,bf16           | 1.18 s                |                       |

#### H100-SXM, 80GB
| Engine                 | Time, sd1.5           | Time, sd2.1           | 
| :---                   | :---                  | :---                  |
| PT2.0,fp16             | 1.00 s                | 1.72 s                |
| PT2.0,fp16,compile     | 0.89 s                | 1.44 s                |
| TensorRT 8.6.1,fp16    | 0.60 s                | 0.58 s                |
| Jax,XLA,bf16           | 1.00 s                | 0.79 s                |

#### RTX 4090, 24GB
| Engine                 | Time, sd1.5           | Time, sd2.1           | Time, sd2.1, 768x768  | 
| :---                   | :---                  | :---                  | :---                  |
| TensorRT 8.6.1, fp16   | 0.745 s               | 0.681 s               | 1.524 s               |

#### V100, T4
| GPU                    | PT2.0,fp16,xformers   | 
| :---                   | :---                  | 
| V100, 16gb             | 2.96 s                | 
| T4, 16gb               | 7.83 s                | 

## How to run
Ubuntu, Debian VM setup https://gist.github.com/alexeigor/b4c21b5e1fe62d670c433d4ac8c9fd83
```bash
docker build . --network=host -t test_engine
```

```bash
docker run -it --network=host -v ${PWD}/workspace:/workspace -w /workspace --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 test_engine
```

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
- https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers/models/stable_diffusion
- https://github.com/microsoft/Olive
- https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion (kudos to Denis Timonin https://www.linkedin.com/in/denistimonin/)
- https://medium.com/microsoftazure/accelerating-stable-diffusion-inference-with-onnx-runtime-203bd7728540
