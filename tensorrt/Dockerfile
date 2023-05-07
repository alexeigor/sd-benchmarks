FROM nvcr.io/nvidia/pytorch:23.02-py3
# custom torch 1.14 build that supports H100

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y curl && \
    apt-get -y install git && \
    apt-get install -y python3-pip && \
    pip install --upgrade pip

WORKDIR /code

RUN git clone https://github.com/NVIDIA/TensorRT.git

RUN pip install --upgrade tensorrt

RUN pip install -r TensorRT/demo/Diffusion/requirements.txt

# RUN pip install --extra-index-url https://pypi.ngc.nvidia.com colored cuda-python ftfy matplotlib nvtx onnx==1.13.1 onnxruntime==1.14.1 onnx-graphsurgeon==0.3.26 polygraphy==0.44.2 scipy
# RUN pip install accelerate diffusers==0.14.0 transformers==4.26.1

CMD python3 /code/TensorRT/demo/Diffusion/demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --version 1.5 --onnx-dir ./onnx --engine-dir ./engine --output-dir ./output -v
