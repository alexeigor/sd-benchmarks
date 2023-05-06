FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y curl && \
    apt-get -y install git && \
    apt-get install -y python3-pip && \
    pip install --upgrade pip

WORKDIR /code

RUN git clone https://github.com/NVIDIA/TensorRT.git

# RUN pip install -r TensorRT/demo/Diffusion/requirements.txt
# RUN mkdir -p onnx engine
RUN pip install tensorrt

RUN pip install --extra-index-url https://pypi.ngc.nvidia.com colored cuda-python ftfy matplotlib nvtx onnx onnx-graphsurgeon==0.3.26 polygraphy==0.44.2 scipy
RUN pip install accelerate "torch<2.0.0" diffusers transformers

CMD python3 /code/TensorRT/demo/Diffusion/demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --onnx-dir ./onnx --engine-dir ./engine --output-dir ./output