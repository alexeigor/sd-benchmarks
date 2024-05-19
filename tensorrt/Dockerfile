FROM ubuntu:22.04
# nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip python3-virtualenv cmake \
    # && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Create a virtualenv for dependencies. This isolates these packages from
# system-level packages.
# Use -p python3 or -p python3.7 to select python version. Default is version 2.
RUN virtualenv /env

# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

# !! release/10.0 , please check https://github.com/NVIDIA/TensorRT/releases
RUN git clone https://github.com/NVIDIA/TensorRT.git && cd TensorRT && git checkout release/10.0

RUN pip install tensorrt

RUN pip install -r TensorRT/demo/Diffusion/requirements.txt

CMD mkdir -p /workspace/onnx /workspace/engine /workspace/output && \
python3 /code/TensorRT/demo/Diffusion/demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \ 
--batch-size 1 --build-static-batch --use-cuda-graph --num-warmup-runs 5 \
--version 2.1 --onnx-dir /workspace/onnx --engine-dir /workspace/engine --output-dir /workspace/output -v \
--height 768 --width 768 \
--denoising-steps 50
