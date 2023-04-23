# ds-benchmarks

## Stable Diffusion benchmarks

batch size = 1

| GPU                    | PT2.0,fp16,xformers | PT,fp16 | DeepSpeed,fp16 |
| :--------------------- | :-----------        | :-------| :-----         |
| A100, SXM, 40gb        | 2.03 s              | 2.8 s   | 1.28 s         |
| V100, 16gb             | 2.96 s              |         |                |
| T4, 16gb               | 7.83 s              |         |                |
