FROM nvcr.io/nvidia/pytorch:23.10-py3
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN git clone https://github.com/rajeevsrao/TensorRT.git && cd TensorRT && git checkout release/8.6

RUN pip install --upgrade tensorrt

RUN pip install -r TensorRT/demo/Diffusion/requirements.txt

CMD mkdir -p /workspace/onnx /workspace/engine /workspace/output && \
python3 /code/TensorRT/demo/Diffusion/demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \ 
--build-static-batch --use-cuda-graph --num-warmup-runs 1 \
--version 2.1 --onnx-dir /workspace/onnx --engine-dir /workspace/engine --output-dir /workspace/output -v \
--height 768 --width 768
