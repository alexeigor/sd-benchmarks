FROM nvcr.io/nvidia/pytorch:23.10-py3
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html

RUN apt-get update && apt-get install -y \
    && apt-get install --no-install-recommends -y git curl python3-pip \
    && pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN git clone https://github.com/NVIDIA/TensorRT.git

RUN pip install --upgrade tensorrt

RUN pip install -r TensorRT/demo/Diffusion/requirements.txt

# RUN pip install --extra-index-url https://pypi.ngc.nvidia.com colored cuda-python ftfy matplotlib nvtx onnx==1.13.1 onnxruntime==1.14.1 onnx-graphsurgeon==0.3.26 polygraphy==0.44.2 scipy
# RUN pip install accelerate diffusers==0.14.0 transformers==4.26.1

CMD mkdir -p /workspace/onnx /workspace/engine /workspace/output && \
python3 /code/TensorRT/demo/Diffusion/demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" \ 
--version 2.1 --onnx-dir /workspace/onnx --engine-dir /workspace/engine --output-dir /workspace/output -v
