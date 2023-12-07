FROM nvcr.io/nvidia/tensorrt:23.11-py3
# switch to https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# !! release/9.1 , please check https://github.com/NVIDIA/TensorRT/releases
RUN git clone https://github.com/NVIDIA/TensorRT.git && cd TensorRT && git checkout release/9.2

# RUN pip install --upgrade tensorrt --extra-index-url https://pypi.nvidia.com

RUN pip install -r TensorRT/demo/Diffusion/requirements.txt

CMD mkdir -p /workspace/onnx /workspace/engine /workspace/output && \
python3 /code/TensorRT/demo/Diffusion/demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \ 
--batch-size 1 --build-static-batch --use-cuda-graph --num-warmup-runs 5 \
--version 2.1 --onnx-dir /workspace/onnx --engine-dir /workspace/engine --output-dir /workspace/output -v \
--height 768 --width 768 \
--denoising-steps 50
