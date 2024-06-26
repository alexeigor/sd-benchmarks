## Stable Diffusion XL inference benchmarks

model: 

https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

batch size = 1, image size 1024x1024, 50 iterations

#### A100-SXM, 40GB
| Engine                      | Time                  | 
| :---                        | :---                  |
| PT2.0,fp16 + compile        | 5.35 s                |
| Onnxruntime,fp16,ORT_CUDA   | 4.28 s                |

#### RTX 4090, 24GB
| Engine                      | Time                  | 
| :---                        | :---                  |
| PT2.0,fp16 + compile        | 6.02 s                |

#### RTX 6000 Ada, 48GB
| Engine                      | Time                  | 
| :---                        | :---                  |
| PT2.0,fp16 + compile        | 9.07 s                |

## Stable Diffusion inference benchmarks

model: 

https://huggingface.co/runwayml/stable-diffusion-v1-5 - sd1.5

https://huggingface.co/stabilityai/stable-diffusion-2-1 - sd2.1

batch size = 1, image size 512x512, 50 iterations

#### A100-SXM, 40GB
| Engine                     | Time, sd1.5           | Time, sd2.1, 512x512  | Time, sd2.1, 768x768  | 
| :---                       | :---                  | :---                  | :---                  |
| PT2.0,fp16                 | 1.96 s (4.54gb VRAM)  |                       |                       |
| PT2.0,fp16 + compile       | 1.36 s (5.96 gb)      |                       | 2.37 s                |
| AITemplate,fp16            | 1.01 s (4.06 gb)      |                       |                       |
| DeepSpeed,fp16             | 1.18 s                |                       | 2.28 s                |
| Oneflow,fp16               | 0.98 s (5.62 gb)      |                       |                       |
| TensorRT 8.6.1, fp16       | 0.98 s                | 0.81 s                | 1.88 s                |
| TensorRT 10.0, fp16        |                       |                       | 1.57 s                |
| Onnxruntime,fp16,ORT_CUDA  | 0.85 s                | 0.76 s                | 1.63 s                |
| Jax,XLA,bf16               | 1.58 s                | 1.35 s                | 3.61 s                |

#### H100-PCIe, 80GB

| Engine                 | Time, sd1.5           | Time, sd2.1           | 
| :---                   | :---                  | :---                  |
| PT2.0,fp16             | 1.44 s                |                       |
| PT2.0,fp16,compile     | 1.11 s                |                       |
| TensorRT 8.6.1,fp16    | 0.75 s                | 0.68 s                |
| Jax,XLA,bf16           | 1.18 s                |                       |

#### H100-SXM, 80GB
| Engine                 | Time, sd1.5           | Time, sd2.1           | Time, sd2.1, 768x768  | 
| :---                   | :---                  | :---                  | :---                  |
| PT2.0,fp16,compile     | 0.83 s                | 0.70 s                | 1.39 s                |
| TensorRT 8.6.1,fp16    | 0.49 s                | 0.48 s                | 1.05 s                |
| Jax,XLA,bf16           | 1.00 s                | 0.79 s                |                       |

#### RTX 4090, 24GB
| Engine                 | Time, sd1.5           | Time, sd2.1           | Time, sd2.1, 768x768  | 
| :---                   | :---                  | :---                  | :---                  |
| PT2.0,fp16,compile     | 1.17 s                |                       | 2.26 s                |
| TensorRT 8.6.1, fp16   | 0.74 s                | 0.68 s                | 1.52 s               |

#### L40, 48GB
| Engine                 | Time, sd1.5           | Time, sd2.1           | Time, sd2.1, 768x768  | 
| :---                   | :---                  | :---                  | :---                  |
| PT2.0,fp16,compile     | 2.09 s                |                       | 3.08 s                |
| TensorRT 8.6.2,fp16    | 0.91 s                |                       | 2.19 s                |

#### RTX 6000 Ada, 48GB
| Engine                 | Time, sd1.5           | Time, sd2.1           | Time, sd2.1, 768x768  | 
| :---                   | :---                  | :---                  | :---                  |
| PT2.0,fp16,compile     | 1.28 s                |                       | 2.77 s                |
| TensorRT 8.6.2,fp16    | 0.90 s                |                       | 2.25 s                |

#### V100, T4
| GPU                    | PT2.0,fp16,xformers   | 
| :---                   | :---                  | 
| V100, 16gb             | 2.96 s                | 
| T4, 16gb               | 7.83 s                | 

## How to run
Ubuntu, Debian VM setup https://gist.github.com/alexeigor/b4c21b5e1fe62d670c433d4ac8c9fd83
```bash
docker build -f ./Dockerfile --network=host --build-arg HF_TOKEN=xxxxx -t test_pt .
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
- https://huggingface.co/blog/sdxl_jax
- https://huggingface.co/blog/simple_sdxl_optimizations
- https://pytorch.org/blog/accelerating-generative-ai-3/


